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
from model import ThermalDefectModel
from utils import compute_f1, find_best_thresholds


# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "dataset_V2.parquet"
IMAGE_DIR    = "dataset_images"
OUTPUT_DIR   = "outputs/models"
NUM_LABELS   = 8
BATCH_SIZE   = 8
NUM_EPOCHS   = 40
LR           = 1e-4
VAL_SPLIT    = 0.2
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


def build_pos_weight(dataset: ThermalDefectDataset) -> torch.Tensor:
    labels = torch.tensor(dataset.labels)
    pos = labels.sum(dim=0).clamp(min=1)
    neg = (labels.shape[0] - pos).clamp(min=1)
    return (neg / pos).to(DEVICE)


def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            logits = model(images)
            loss   = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    avg_loss    = total_loss / len(loader.dataset)
    macro_f1, micro_f1 = compute_f1(all_logits, all_targets)
    return avg_loss, macro_f1, micro_f1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    train_full = ThermalDefectDataset(PARQUET_PATH, IMAGE_DIR, train=True)
    val_full   = ThermalDefectDataset(PARQUET_PATH, IMAGE_DIR, train=False)

    indices = list(range(len(train_full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=SEED
    )

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model      = ThermalDefectModel(num_labels=NUM_LABELS).to(DEVICE)
    pos_weight = build_pos_weight(train_full)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Fine-tune everything end-to-end: backbone at lower LR, head at full LR
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(),   "lr": LR * 0.1},
        {"params": model.classifier.parameters(), "lr": LR},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_f1     = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    print(f"Device : {DEVICE}")
    print(f"Samples: {len(train_full)} total | {len(train_idx)} train | {len(val_idx)} val\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_mac, tr_mic = run_epoch(model, train_loader, criterion, optimizer, train=True)
        vl_loss, vl_mac, vl_mic = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train loss {tr_loss:.4f}  macro-F1 {tr_mac:.3f}  micro-F1 {tr_mic:.3f} | "
            f"Val loss {vl_loss:.4f}  macro-F1 {vl_mac:.3f}  micro-F1 {vl_mic:.3f}"
        )

        if vl_mac > best_val_f1:
            best_val_f1 = vl_mac
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val macro-F1 = {best_val_f1:.3f})")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print("\nRunning threshold tuning...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            all_logits.append(model(images.to(DEVICE)).cpu())
            all_targets.append(targets)

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    best_thresholds = find_best_thresholds(all_logits, all_targets)
    np.save(os.path.join(OUTPUT_DIR, "best_thresholds.npy"), best_thresholds)

    macro_fixed, _ = compute_f1(all_logits, all_targets, threshold=0.3)
    macro_tuned, _ = compute_f1(all_logits, all_targets, threshold=best_thresholds)

    print("\nPer-label thresholds:")
    for label, t in zip(LABEL_COLS, best_thresholds):
        print(f"  {label:<22} → {t:.2f}")

    print(f"\nVal macro-F1 @ 0.3   : {macro_fixed:.3f}")
    print(f"Val macro-F1 @ tuned : {macro_tuned:.3f}")
    print(f"\nBest val macro-F1    : {best_val_f1:.3f}")
    print(f"Model  → {best_model_path}")
    print(f"Thresh → {os.path.join(OUTPUT_DIR, 'best_thresholds.npy')}")


if __name__ == "__main__":
    main()