"""
inference.py
------------
Monte Carlo Dropout inference engine for thermal defect detection.

Novel contribution:
  - Keeps Dropout active during inference (MC Dropout)
  - Runs N forward passes per sample → mean prediction + epistemic uncertainty
  - Parts with high uncertainty are flagged for human review (human-in-the-loop)
  - BatchNorm remains in eval mode (uses running stats, not batch stats)

Usage (CLI)
-----------
    python3 src/inference.py path/to/sample.csv

Usage (API)
-----------
    from inference import ThermalInferenceEngine

    engine = ThermalInferenceEngine(
        model_path="outputs/models/best_model.pth",
        thresholds_path="outputs/models/best_thresholds.npy",
        adj_path="outputs/models/adjacency_matrix.npy",
    )
    result = engine.predict("dataset_csv/TDI_xxx.csv")
    # result["flag_for_review"] → True/False
    # result["uncertain_labels"] → list of label names
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from dataset import (load_csv_matrix, compute_thermal_channels,
                     normalize_channels, resize_matrix, LABEL_COLS)
from gcn import build_adjacency_matrix
from model import ThermalDefectModel
from utils import compute_ece


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """
    Set only Dropout layers to train mode (leaves BatchNorm in eval mode).

    This is the standard MC Dropout approach:
      - Dropout active   → stochastic forward pass → uncertainty estimate
      - BatchNorm eval   → uses running statistics  → stable predictions
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ── Inference Engine ──────────────────────────────────────────────────────────

class ThermalInferenceEngine:
    """
    Load a trained ThermalDefectModel and run MC Dropout inference.

    Args:
        model_path           : path to best_model.pth
        thresholds_path      : path to best_thresholds.npy
        adj_path             : path to adjacency_matrix.npy (optional)
        n_passes             : number of stochastic forward passes (default 50)
        uncertainty_threshold: std above which a label is flagged uncertain
        num_labels           : number of defect labels (8)
        in_channels          : number of input channels (5)
        hidden_dim           : GCN hidden dimension (must match training)
        device               : torch device (auto-detected if None)
    """

    def __init__(
        self,
        model_path: str,
        thresholds_path: str,
        adj_path: str = None,
        n_passes: int = 50,
        uncertainty_threshold: float = 0.15,
        num_labels: int = 8,
        in_channels: int = 5,
        hidden_dim: int = 256,
        device: torch.device = None,
    ):
        self.device               = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_passes             = n_passes
        self.uncertainty_threshold = uncertainty_threshold

        # Load adjacency matrix (None → identity fallback inside model)
        adj_matrix = None
        if adj_path and os.path.exists(adj_path):
            adj_matrix = np.load(adj_path)
        elif adj_path:
            print(f"Warning: adjacency matrix '{adj_path}' not found — "
                  "using identity (no graph structure).")

        # Build and load model
        self.model = ThermalDefectModel(
            num_labels=num_labels,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            adj_matrix=adj_matrix,
            backbone_weights=None,       # weights loaded from checkpoint below
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        # Per-label decision thresholds
        self.thresholds = np.load(thresholds_path)

        print(
            f"Inference engine ready  |  device={self.device}  "
            f"|  MC passes={n_passes}  |  unc_thresh={uncertainty_threshold}"
        )

    def _preprocess(self, csv_path: str) -> torch.Tensor:
        """Load CSV → 5-channel normalised tensor → (1, 5, 224, 224)."""
        matrix   = load_csv_matrix(csv_path)
        matrix   = resize_matrix(matrix, 224)
        channels = compute_thermal_channels(matrix)
        tensor   = normalize_channels(torch.from_numpy(channels))
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, csv_path: str) -> dict:
        """
        Run MC Dropout inference on a single thermal CSV file.

        Returns a dict with:
            predictions      : dict[label → bool]   (thresholded mean prob)
            mean_probs       : dict[label → float]  (mean over N passes)
            uncertainty      : dict[label → float]  (std  over N passes)
            flag_for_review  : bool  (True if any label uncertainty > threshold)
            uncertain_labels : list[str]
        """
        tensor = self._preprocess(csv_path)

        # eval() disables dropout; then _enable_dropout re-activates it only
        self.model.eval()
        _enable_dropout(self.model)

        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                logits = self.model(tensor)                  # (1, num_labels)
                probs  = torch.sigmoid(logits).cpu().numpy() # (1, num_labels)
                all_probs.append(probs[0])

        all_probs  = np.stack(all_probs, axis=0)  # (N, num_labels)
        mean_probs = all_probs.mean(axis=0)        # (num_labels,)
        std_probs  = all_probs.std(axis=0)         # (num_labels,)

        preds = (mean_probs >= self.thresholds).tolist()

        uncertain_labels = [
            LABEL_COLS[i] for i in range(len(LABEL_COLS))
            if std_probs[i] > self.uncertainty_threshold
        ]

        return {
            "predictions":      {l: bool(p) for l, p in zip(LABEL_COLS, preds)},
            "mean_probs":       {l: round(float(p), 4)
                                 for l, p in zip(LABEL_COLS, mean_probs)},
            "uncertainty":      {l: round(float(s), 4)
                                 for l, s in zip(LABEL_COLS, std_probs)},
            "flag_for_review":  len(uncertain_labels) > 0,
            "uncertain_labels": uncertain_labels,
        }

    def predict_batch(self, csv_paths: list) -> list:
        """Run inference on a list of CSV paths. Returns list of result dicts."""
        return [self.predict(p) for p in csv_paths]

    def evaluate_calibration(self, csv_paths: list,
                              label_matrix: np.ndarray) -> float:
        """
        Compute ECE over a labelled set to assess uncertainty calibration.

        Args:
            csv_paths    : list of CSV file paths
            label_matrix : (N, num_labels) binary ground-truth array

        Returns:
            ece : scalar float (lower is better)
        """
        all_mean_probs = []
        for path in csv_paths:
            result = self.predict(path)
            all_mean_probs.append(list(result["mean_probs"].values()))

        mean_probs_arr = np.array(all_mean_probs)   # (N, num_labels)
        ece = compute_ece(mean_probs_arr, label_matrix)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        return ece


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MC Dropout inference on a thermal CSV matrix"
    )
    parser.add_argument("csv_path",
                        help="Path to the raw IR CSV temperature matrix")
    parser.add_argument("--model",      default="outputs/models/best_model.pth")
    parser.add_argument("--thresholds", default="outputs/models/best_thresholds.npy")
    parser.add_argument("--adj",        default="outputs/models/adjacency_matrix.npy")
    parser.add_argument("--passes",     type=int,   default=50,
                        help="Number of MC Dropout forward passes")
    parser.add_argument("--unc_thresh", type=float, default=0.15,
                        help="Std threshold for flagging uncertain labels")
    args = parser.parse_args()

    engine = ThermalInferenceEngine(
        model_path=args.model,
        thresholds_path=args.thresholds,
        adj_path=args.adj,
        n_passes=args.passes,
        uncertainty_threshold=args.unc_thresh,
    )

    result = engine.predict(args.csv_path)
    print(json.dumps(result, indent=2))