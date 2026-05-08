"""
train.py
--------
Supervised fine-tuning of the novel thermal defect detection model.

Novel contributions vs. baseline:
  - Reads raw CSV matrices via the 5-channel ThermalDefectDataset
  - Builds GCN adjacency matrix from training-split labels + domain priors
  - Optionally loads SimCLR pre-trained backbone before fine-tuning
  - AdamW with CosineAnnealingWarmRestarts (not plain Adam + CosineAnnealingLR)
  - Saves adjacency matrix alongside model weights for reproducible inference

Usage
-----
    # Without SimCLR pre-training:
    python3 src/train.py

    # With SimCLR pre-training:
    python3 src/simclr.py --csv_dir dataset_csv
    python3 src/train.py --simclr outputs/models/simclr_backbone.pth
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from dataset import ThermalDefectDataset, LABEL_COLS
from gcn import build_adjacency_matrix
from model import ThermalDefectModel
from utils import compute_f1, find_best_thresholds


# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH   = "dataset_V2.parquet"
CSV_DIR        = "dataset_csv"
OUTPUT_DIR     = "outputs/models"
NUM_LABELS     = 8
BATCH_SIZE     = 8
NUM_EPOCHS     = 80
LR             = 1e-4
VAL_SPLIT      = 0.2
SEED           = 42
GCN_HIDDEN_DIM = 256
IN_CHANNELS    = 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


def build_pos_weight(labels_array: np.ndarray) -> torch.Tensor:
    """Per-label pos_weight = neg_count / pos_count for BCEWithLogitsLoss."""
    labels = torch.tensor(labels_array)
    pos    = labels.sum(dim=0).clamp(min=1)
    neg    = (labels.shape[0] - pos).clamp(min=1)
    return (neg / pos).clamp(max=20.0).to(DEVICE)

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for tensors, targets in tqdm(loader, leave=False):
            tensors = tensors.to(DEVICE)
            targets = targets.to(DEVICE)
            logits  = model(tensors)
            loss    = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * tensors.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    avg_loss    = total_loss / len(loader.dataset)
    macro_f1, micro_f1 = compute_f1(all_logits, all_targets)
    return avg_loss, macro_f1, micro_f1


def main(simclr_path: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Two separate instances so augmentations only apply to training samples
    train_full = ThermalDefectDataset(PARQUET_PATH, CSV_DIR, train=True)
    val_full   = ThermalDefectDataset(PARQUET_PATH, CSV_DIR, train=False)

    indices = list(range(len(train_full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=SEED
    )

    train_ds     = Subset(train_full, train_idx)
    val_ds       = Subset(val_full,   val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # ── GCN Adjacency Matrix ──────────────────────────────────────────────────
    # Built from training labels only — prevents data leakage into graph structure
    train_labels = train_full.labels[train_idx]
    adj_matrix   = build_adjacency_matrix(train_labels, num_labels=NUM_LABELS)
    adj_save     = os.path.join(OUTPUT_DIR, "adjacency_matrix.npy")
    np.save(adj_save, adj_matrix)
    print(f"Adjacency matrix saved → {adj_save}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ThermalDefectModel(
        num_labels=NUM_LABELS,
        in_channels=IN_CHANNELS,
        hidden_dim=GCN_HIDDEN_DIM,
        adj_matrix=adj_matrix,
    ).to(DEVICE)

    # Load SimCLR pre-trained backbone if provided
    if simclr_path and os.path.exists(simclr_path):
        print(f"Loading SimCLR backbone from '{simclr_path}'...")
        model.load_simclr_backbone(simclr_path, device=DEVICE)
    elif simclr_path:
        print(f"Warning: SimCLR checkpoint '{simclr_path}' not found — "
              "falling back to ImageNet init.")

    # ── Training setup ────────────────────────────────────────────────────────
    pos_weight = build_pos_weight(train_full.labels[train_idx])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential learning rates: backbone << head (protects pre-trained features)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": LR * 0.1, "weight_decay": 1e-4},
        {"params": model.gcn_head.parameters(),  "lr": LR,       "weight_decay": 1e-4},
    ])
    # Warm restarts encourage escaping local minima on small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_f1     = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nDevice     : {DEVICE}")
    print(f"Parameters : {total_params:,} total | {trainable_p:,} trainable")
    print(f"Samples    : {len(train_full)} total | "
          f"{len(train_idx)} train | {len(val_idx)} val\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_mac, tr_mic = run_epoch(
            model, train_loader, criterion, optimizer, train=True
        )
        vl_loss, vl_mac, vl_mic = run_epoch(
            model, val_loader, criterion, optimizer, train=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"Train loss {tr_loss:.4f}  macro-F1 {tr_mac:.3f}  micro-F1 {tr_mic:.3f}  |  "
            f"Val   loss {vl_loss:.4f}  macro-F1 {vl_mac:.3f}  micro-F1 {vl_mic:.3f}"
        )

        if vl_mac > best_val_f1:
            best_val_f1 = vl_mac
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val macro-F1 = {best_val_f1:.3f})")

    # ── Per-label threshold tuning ────────────────────────────────────────────
    print("\nRunning per-label threshold tuning on validation set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for tensors, targets in val_loader:
            all_logits.append(model(tensors.to(DEVICE)).cpu())
            all_targets.append(targets)

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    best_thresholds  = find_best_thresholds(all_logits, all_targets)
    thresh_save_path = os.path.join(OUTPUT_DIR, "best_thresholds.npy")
    np.save(thresh_save_path, best_thresholds)

    macro_fixed, _ = compute_f1(all_logits, all_targets, threshold=0.3)
    macro_tuned, _ = compute_f1(all_logits, all_targets, threshold=best_thresholds)

    print("\nPer-label thresholds:")
    for label, t in zip(LABEL_COLS, best_thresholds):
        print(f"  {label:<22} → {t:.2f}")

    print(f"\nVal macro-F1 @ fixed 0.3  : {macro_fixed:.3f}")
    print(f"Val macro-F1 @ tuned      : {macro_tuned:.3f}")
    print(f"\nBest val macro-F1         : {best_val_f1:.3f}")
    print(f"Model      → {best_model_path}")
    print(f"Thresholds → {thresh_save_path}")
    print(f"Adjacency  → {adj_save}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train novel thermal defect detection model"
    )
    parser.add_argument(
        "--simclr", default=None,
        help="Path to SimCLR pre-trained backbone checkpoint "
             "(e.g. outputs/models/simclr_backbone.pth)"
    )
    args = parser.parse_args()
    main(simclr_path=args.simclr)