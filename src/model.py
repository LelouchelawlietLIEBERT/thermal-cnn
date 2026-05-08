"""
model.py
--------
Full thermal defect detection model.

Novel contributions vs. baseline:
  - First conv layer widened from 3 → 5 input channels
    (pre-trained ImageNet weights preserved for channels 0-2;
     gradient channels 3-4 initialised to zero)
  - Flat Linear head replaced with a 2-layer GCN classifier head
  - Optional loading of SimCLR pre-trained backbone
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from gcn import GCNHead, build_adjacency_matrix


class ThermalDefectModel(nn.Module):
    """
    EfficientNet-B0 backbone with 5-channel input and GCN label classifier.

    Args:
        num_labels      : number of defect labels (8)
        in_channels     : input channels (5 for [T, dx, dy, |∇T|, ∇²T])
        hidden_dim      : GCN node feature dimension (256)
        adj_matrix      : (num_labels, num_labels) float32 ndarray;
                          computed from training co-occurrence + domain priors.
                          If None, uses identity (no graph structure).
        backbone_weights: ImageNet weights to initialise backbone;
                          pass None to skip (e.g. when loading SimCLR weights)
    """

    def __init__(
        self,
        num_labels: int = 8,
        in_channels: int = 5,
        hidden_dim: int = 256,
        adj_matrix: np.ndarray = None,
        backbone_weights=EfficientNet_B0_Weights.DEFAULT,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        base = efficientnet_b0(weights=backbone_weights)

        # Widen first conv: 3 → in_channels
        # features[0][0] is Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            # Copy pre-trained RGB weights into channels 0-2
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Zero-init thermal-gradient channels (3, 4, ...)
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = 0.0

        base.features[0][0] = new_conv

        # Remove original classifier (keep pooling intact)
        in_features = base.classifier[1].in_features   # 1280
        base.classifier = nn.Identity()
        self.backbone = base

        # ── GCN Classifier Head ───────────────────────────────────────────────
        self.gcn_head = GCNHead(
            in_features=in_features,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            adj_matrix=adj_matrix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 5, 224, 224)
        Returns:
            logits : (batch, num_labels)  — raw logits (before sigmoid)
        """
        features = self.backbone(x)      # (batch, 1280)
        return self.gcn_head(features)   # (batch, num_labels)

    def load_simclr_backbone(self, checkpoint_path: str,
                              device: torch.device = None) -> None:
        """
        Load a SimCLR pre-trained backbone from a checkpoint saved by simclr.py.

        The checkpoint must contain a 'backbone' key with an EfficientNet-B0
        state_dict (already widened to 5 input channels).
        """
        if device is None:
            device = next(self.parameters()).device

        ckpt = torch.load(checkpoint_path, map_location=device)
        backbone_state = ckpt.get('backbone', ckpt)   # handle both formats

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        print(f"  SimCLR backbone loaded from '{checkpoint_path}'")
        if missing:
            print(f"    Missing keys  : {len(missing)}")
        if unexpected:
            print(f"    Unexpected    : {len(unexpected)}")