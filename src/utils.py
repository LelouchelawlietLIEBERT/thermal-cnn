import numpy as np
import torch
from sklearn.metrics import f1_score

def compute_f1(logits: torch.Tensor, targets: torch.Tensor, threshold=0.3):
    """
    Args:
        logits:    raw model output (before sigmoid)
        targets:   ground truth binary labels
        threshold: float (applied to all labels) or
                   np.ndarray of shape (num_labels,) for per-label thresholds
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


def find_best_thresholds(logits: torch.Tensor, targets: torch.Tensor):
    """
    Search [0.1, 0.9] per label on the validation set.
    Returns the threshold array that maximises macro-F1.
    """
    probs   = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()
    n_labels = targets.shape[1]

    best_thresholds = np.full(n_labels, 0.3)
    for i in range(n_labels):
        best_f1, best_t = 0.0, 0.3
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t

    return best_thresholds