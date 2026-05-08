"""
utils.py
--------
Evaluation metrics and helper functions.

  compute_f1            : macro + micro F1 from logits
  find_best_thresholds  : per-label threshold search on validation set
  compute_ece           : Expected Calibration Error for uncertainty evaluation
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_f1(logits: torch.Tensor, targets: torch.Tensor,
               threshold=0.3) -> tuple:
    """
    Compute macro and micro F1 from raw logits.

    Args:
        logits    : (N, num_labels) raw model output (before sigmoid)
        targets   : (N, num_labels) binary ground-truth labels
        threshold : float applied to all labels, or
                    np.ndarray shape (num_labels,) for per-label thresholds
    Returns:
        (macro_f1, micro_f1)
    """
    probs   = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()

    if isinstance(threshold, (int, float)):
        preds = (probs >= threshold).astype(int)
    else:
        preds = (probs >= np.array(threshold).reshape(1, -1)).astype(int)

    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    return macro, micro


def find_best_thresholds(logits: torch.Tensor,
                          targets: torch.Tensor) -> np.ndarray:
    """
    Search threshold in [0.10, 0.90] independently per label on the
    validation set to maximise per-label F1.

    Returns:
        best_thresholds : (num_labels,) float32 array
    """
    probs    = torch.sigmoid(logits).cpu().numpy()
    targets  = targets.cpu().numpy()
    n_labels = targets.shape[1]

    best_thresholds = np.full(n_labels, 0.3, dtype=np.float32)

    for i in range(n_labels):
        best_f1, best_t = 0.0, 0.3
        for t in np.arange(0.10, 0.91, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1    = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best_thresholds[i] = best_t

    return best_thresholds


def compute_ece(mean_probs: np.ndarray, targets: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error — measures whether MC Dropout uncertainty
    is calibrated (lower is better; 0.0 = perfectly calibrated).

    Aggregates over all label-sample pairs.

    Args:
        mean_probs : (N, num_labels) mean sigmoid probabilities from MC passes
        targets    : (N, num_labels) binary ground-truth labels
        n_bins     : number of confidence bins

    Returns:
        ece : scalar float
    """
    probs_flat   = mean_probs.flatten()
    targets_flat = targets.flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs_flat)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs_flat[mask].mean()
        bin_acc  = targets_flat[mask].mean()
        ece     += (mask.sum() / total) * abs(bin_conf - bin_acc)

    return float(ece)