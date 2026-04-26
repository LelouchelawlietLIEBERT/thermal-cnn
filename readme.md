# Thermal Defect Detection — Injection Moulding

Multi-label defect classification from thermal (IR) images of injection-moulded plastic parts,
using a fine-tuned EfficientNet-B0 CNN.

---

## Problem

During injection moulding, deviations in process parameters produce surface and structural
defects. A robot positions each ejected part in front of an IR camera, which captures the
surface temperature distribution as a raw CSV matrix. This project predicts 8 defect types
simultaneously from that thermal image — before any human inspector sees the part.

| Label | Defect |
|---|---|
| `LBL_SinkMarks` | Surface depression from insufficient material packing |
| `LBL_SprueCircle` | Circular mark around the sprue gate |
| `LBL_Underfilled` | Cavity not fully filled with melt |
| `LBL_OldGranulate` | Contamination from previous material batch |
| `LBL_StreaksLevel1` | Light flow/moisture streaks |
| `LBL_StreaksLevel2` | Moderate streaks |
| `LBL_StreaksLevel3` | Severe streaks |
| `LBL_NOK` | Overall reject flag — 1 if SinkMarks or Underfilled present |

---

## Dataset

- **Source:** ProBayes research project — SKZ (German Plastics Centre) + Fraunhofer IPA
- **Machine:** KraussMaffei 160-750PX injection moulding machine
- **Material:** Polypropylene (PP) — March 2022 Taguchi experimental plan
- **Size:** 156 labelled parts across 13 experiment series (A20–A32), 12 parts each
- **Labels:** `dataset_V2.parquet` — one row per injection moulding cycle
- **Images:** `dataset_images/` — PNG thermal images (pre-converted from raw IR CSV matrices)

### Image–Label Mapping

```
IR_Image1Name in parquet : TDI_000020220317 06_24_53.736A1.csv
Corresponding image      : dataset_images/TDI_000020220317 06_24_53.736A1.png
```

The `.csv` extension is replaced with `.png`. Any parquet row whose image does not
exist on disk is dropped silently at dataset load time.

### Label Distribution Note

Defects are rare by design — most experimental series use near-optimal settings.
`LBL_NOK`, `LBL_SinkMarks`, and `LBL_Underfilled` have the most positive samples.
`LBL_OldGranulate` and `LBL_SprueCircle` are very sparse. Per-label `pos_weight`
in the loss function compensates for this imbalance.

---

## Pipeline

```
dataset_images/*.png          ← thermal images (CSV→PNG conversion already done)
        ↓
  ThermalDefectDataset
  ├── Train: Resize(256) → RandomResizedCrop(224) → Flip → Rotate → ColorJitter → Normalize
  └── Val:   Resize(224) → Normalize
        ↓
  EfficientNet-B0 backbone  (pretrained ImageNet weights)
  + Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→8)
        ↓
  BCEWithLogitsLoss with per-label pos_weight
  Adam — backbone LR: 1e-5, classifier LR: 1e-4
  CosineAnnealingLR over 40 epochs
  Gradient clipping (max norm = 1.0)
        ↓
  Per-label threshold tuning on validation set
        ↓
  Multi-label defect prediction
```

---

## Project Structure

```
thermal-cnn/
├── dataset_csv/          # raw IR CSV matrices (not used at training time)
├── dataset_images/       # PNG thermal images  ← model reads from here
├── dataset_V2.parquet    # labels + metadata
├── src/
│   ├── dataset.py        # PyTorch Dataset + transforms
│   ├── model.py          # EfficientNet-B0 classifier head
│   ├── train.py          # training loop + threshold tuning
│   └── utils.py          # F1 metric + per-label threshold search
├── outputs/
│   └── models/
│       ├── best_model.pth         # saved on best val macro-F1
│       └── best_thresholds.npy   # per-label decision thresholds
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. CPU-only training is supported (no GPU required).

---

## Training

Run from the project root (`thermal-cnn/`):

```bash
python3 src/train.py
```

Training prints per-epoch loss and F1 for both splits. At the end, per-label
threshold tuning runs on the validation set and results are saved automatically.

Expected output at end of training:

```
Per-label thresholds:
  LBL_SinkMarks          → 0.30
  LBL_SprueCircle        → 0.70
  LBL_Underfilled        → 0.80
  LBL_OldGranulate       → 0.15
  ...

Val macro-F1 @ 0.3   : 0.125
Val macro-F1 @ tuned : 0.307
```

---

## Evaluation Metrics

| Metric | Why |
|---|---|
| **Macro F1** | Treats each label equally — penalises poor performance on rare defects |
| **Micro F1** | Aggregates across all label-sample pairs |
| **Per-label threshold** | Replaces fixed 0.5 cutoff with a threshold tuned per defect class |

Accuracy is not reported — it is misleading in imbalanced multi-label settings where
predicting "no defect" for everything yields artificially high scores.

---

## Results

| Setting | Val macro-F1 |
|---|---|
| Fixed threshold (0.3) | 0.125 |
| Per-label tuned threshold | 0.307 |

Results are limited by dataset size (156 samples, single material, single test day).
The gap between fixed and tuned thresholds reflects significant class imbalance — rare
defect labels require much lower or higher thresholds than the 0.3 default.

---

## Configuration

All hyperparameters are at the top of `src/train.py`:

```python
BATCH_SIZE  = 8
NUM_EPOCHS  = 40
LR          = 1e-4      # classifier LR; backbone runs at LR * 0.1
VAL_SPLIT   = 0.2
```

---

## Key Engineering Decisions

**Differential learning rates** — backbone updates at `1e-5`, classifier head at `1e-4`.
Prevents destroying pretrained ImageNet features while still adapting to thermal image
characteristics.

**Separate dataset instances for train/val** — `train_test_split` on indices, with
`Subset(train_full, train_idx)` and `Subset(val_full, val_idx)` pointing to separate
`ThermalDefectDataset` objects. This ensures augmentation applies only to training
samples — a common bug when using `random_split` on a shared dataset object.

**Per-label threshold tuning** — after training, `find_best_thresholds()` in `utils.py`
searches `[0.1, 0.9]` independently per label on the validation set. Thresholds are
saved to `outputs/models/best_thresholds.npy` for reproducible inference.

**`pos_weight` in BCEWithLogitsLoss** — computed from training label frequencies.
Upweights the loss contribution of positive (defect) samples to prevent the model
from collapsing to predicting all-negative.