"""
simclr.py
---------
SimCLR self-supervised pre-training for the thermal defect backbone.

Novel contribution:
  - Contrastive pre-training on ALL thermal CSVs (labelled + unlabelled)
    before supervised fine-tuning — adapts backbone to thermal texture
    without requiring labels.
  - Thermal-appropriate augmentations: temperature jitter, sensor noise,
    random masking — not standard RGB augmentations.
  - NT-Xent (InfoNCE) loss over 5-channel representations.

Usage
-----
    python3 src/simclr.py --csv_dir dataset_csv --epochs 100

Reference: Chen et al., "A Simple Framework for Contrastive Learning
           of Visual Representations", ICML 2020.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import (load_csv_matrix, compute_thermal_channels,
                     normalize_channels, resize_matrix, _random_crop_resize)


# ── Thermal Augmentation ──────────────────────────────────────────────────────

class ThermalAugmentation:
    """
    Stochastic augmentation pipeline on raw 2-D temperature matrices.
    Used to generate the two views required by SimCLR.
    """

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        # 1. Random crop + resize (spatial invariance)
        matrix = _random_crop_resize(matrix, self.target_size, min_scale=0.6)

        # 2. Flips
        if np.random.rand() > 0.5:
            matrix = np.fliplr(matrix).copy()
        if np.random.rand() > 0.5:
            matrix = np.flipud(matrix).copy()

        # 3. Temperature jitter (ambient drift + emissivity variation)
        if np.random.rand() > 0.2:
            matrix = matrix + float(np.random.uniform(-3.0, 3.0))
            matrix = matrix * float(np.random.uniform(0.85, 1.15))

        # 4. Gaussian sensor noise
        if np.random.rand() > 0.2:
            std = np.random.uniform(0, 1.2)
            matrix = (matrix
                      + np.random.normal(0, std, matrix.shape).astype(np.float32))

        # 5. Random pixel masking (simulate occlusion / sensor dropout)
        if np.random.rand() > 0.5:
            mask = np.random.rand(*matrix.shape) < 0.08
            matrix[mask] = float(matrix.mean())

        return matrix.astype(np.float32)


# ── SimCLR Dataset ────────────────────────────────────────────────────────────

class SimCLRDataset(Dataset):
    """
    Returns two independently augmented views of each thermal CSV.
    Accepts ALL *.csv files in csv_dir (labelled and unlabelled).
    """

    def __init__(self, csv_dir: str, target_size: int = 224):
        self.csv_paths = sorted([
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if f.lower().endswith('.csv')
        ])
        if not self.csv_paths:
            raise RuntimeError(f"No CSV files found in '{csv_dir}'.")
        self.augment = ThermalAugmentation(target_size)

    def __len__(self) -> int:
        return len(self.csv_paths)

    def __getitem__(self, idx: int):
        matrix = load_csv_matrix(self.csv_paths[idx])

        v1 = self.augment(matrix.copy())
        v2 = self.augment(matrix.copy())

        t1 = normalize_channels(torch.from_numpy(compute_thermal_channels(v1)))
        t2 = normalize_channels(torch.from_numpy(compute_thermal_channels(v2)))
        return t1, t2


# ── NT-Xent Loss ──────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy loss.

    For a batch of size N, the 2N representations are compared.
    Each positive pair (z1_i, z2_i) is pulled together; all other 2N-2 pairs
    are treated as negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2 : (batch, proj_dim) — L2-normalised projection vectors
        """
        batch = z1.size(0)
        z = torch.cat([z1, z2], dim=0)           # (2B, D)

        # Pairwise cosine similarities scaled by temperature
        sim = F.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=2
        ) / self.temperature                       # (2B, 2B)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
        sim  = sim.masked_fill(mask, -1e9)

        # Positive pairs: z1[i] ↔ z2[i]
        # In the concatenated tensor: z1 at [0..B-1], z2 at [B..2B-1]
        labels = torch.cat([
            torch.arange(batch, 2 * batch, device=z.device),  # for z1 rows
            torch.arange(0,     batch,     device=z.device),  # for z2 rows
        ])                                         # (2B,)

        return F.cross_entropy(sim, labels)


# ── SimCLR Model (backbone + projection head) ─────────────────────────────────

class SimCLRModel(nn.Module):
    """EfficientNet-B0 backbone (5-channel input) + projection head."""

    def __init__(self, in_channels: int = 5, proj_dim: int = 128):
        super().__init__()

        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Widen first conv: 3 → 5 channels
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0
        base.features[0][0] = new_conv

        in_features     = base.classifier[1].in_features  # 1280
        base.classifier = nn.Identity()
        self.backbone   = base

        # Projection head (discarded after pre-training; only backbone is kept)
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


# ── Training Loop ─────────────────────────────────────────────────────────────

def pretrain_simclr(
    csv_dir: str,
    output_path: str,
    epochs: int      = 100,
    batch_size: int  = 16,
    lr: float        = 3e-4,
    temperature: float = 0.07,
    device: torch.device = None,
) -> str:
    """
    Run SimCLR contrastive pre-training and save the backbone weights.

    Args:
        csv_dir     : directory containing *.csv thermal matrices
        output_path : where to save the checkpoint (.pth)
        epochs      : number of training epochs
        batch_size  : number of samples per batch (each sample = 2 views)
        lr          : Adam learning rate
        temperature : NT-Xent temperature τ
        device      : torch device (auto-detected if None)

    Returns:
        output_path on success.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSimCLR pre-training")
    print(f"  device={device}  epochs={epochs}  batch={batch_size}  τ={temperature}")

    dataset = SimCLRDataset(csv_dir)
    print(f"  Found {len(dataset)} CSV files in '{csv_dir}'")

    # Guard: batch_size must be ≤ dataset size
    if len(dataset) < batch_size:
        batch_size = max(2, len(dataset) // 2)
        print(f"  Reduced batch_size to {batch_size} (dataset smaller than original batch).")

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    model     = SimCLRModel(in_channels=5).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for v1, v2 in tqdm(loader, leave=False, desc=f"Epoch {epoch}/{epochs}"):
            v1, v2 = v1.to(device), v2.to(device)
            z1     = model(v1)
            z2     = model(v2)
            loss   = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  NT-Xent loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'backbone': model.backbone.state_dict(),
                'epoch':    epoch,
                'loss':     best_loss,
            }, output_path)

    print(f"\nSimCLR complete. Best loss: {best_loss:.4f}")
    print(f"Backbone checkpoint → {output_path}")
    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SimCLR thermal backbone pre-training")
    parser.add_argument("--csv_dir",     default="dataset_csv",
                        help="Directory of raw *.csv thermal matrices")
    parser.add_argument("--output",      default="outputs/models/simclr_backbone.pth",
                        help="Output path for backbone checkpoint")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    pretrain_simclr(
        csv_dir=args.csv_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
    )