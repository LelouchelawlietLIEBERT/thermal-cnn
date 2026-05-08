"""
dataset.py
----------
PyTorch Dataset for thermal defect detection.

Novel contributions vs. baseline:
  - Loads raw float32 CSV temperature matrices instead of quantised PNGs
  - Constructs 5-channel input tensor per sample:
      Ch0: raw temperature T
      Ch1: horizontal gradient dT/dx  (Sobel)
      Ch2: vertical gradient dT/dy    (Sobel)
      Ch3: gradient magnitude |∇T|
      Ch4: Laplacian ∇²T
  - Thermal-appropriate augmentations (Gaussian noise, temp jitter)
    instead of RGB-biased ColorJitter
  - Per-channel z-score normalisation per sample
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LABEL_COLS = [
    "LBL_SinkMarks",
    "LBL_SprueCircle",
    "LBL_Underfilled",
    "LBL_OldGranulate",
    "LBL_StreaksLevel1",
    "LBL_StreaksLevel2",
    "LBL_StreaksLevel3",
    "LBL_NOK",
]

# Sobel kernels (3×3)
_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float32)

_LAP_K   = np.array([[ 0,  1,  0],
                      [ 1, -4,  1],
                      [ 0,  1,  0]], dtype=np.float32)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def load_csv_matrix(path: str) -> np.ndarray:
    """Load a raw IR temperature matrix from CSV → float32 2-D array.
    Handles semicolon-separated, comma-decimal, UTF-8 BOM format."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = []
            for v in line.split(";"):
                v = v.strip().replace(",", ".")
                if v:
                    try:
                        row.append(float(v))
                    except ValueError:
                        pass
            if row:
                rows.append(row)
    min_cols = min(len(r) for r in rows)
    rows = [r[:min_cols] for r in rows]
    arr = np.array(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def resize_matrix(matrix: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize a 2-D float32 matrix to (target_size × target_size) bilinearly."""
    h, w = matrix.shape
    if h == target_size and w == target_size:
        return matrix
    t = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    t = F.interpolate(t, size=(target_size, target_size),
                      mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def compute_thermal_channels(matrix: np.ndarray) -> np.ndarray:
    """
    Build the 5-channel physics-informed representation.

    Returns ndarray shape (5, H, W):
        [T, dT/dx, dT/dy, |∇T|, ∇²T]
    """
    try:
        from scipy.ndimage import convolve
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    T   = matrix.astype(np.float32)
    dx  = convolve(T, _SOBEL_X, mode='constant', cval=0.0)
    dy  = convolve(T, _SOBEL_Y, mode='constant', cval=0.0)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    lap = convolve(T, _LAP_K,   mode='constant', cval=0.0)
    return np.stack([T, dx, dy, mag, lap], axis=0)  # (5, H, W)


def normalize_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score normalisation. Shape in/out: (C, H, W)."""
    out = torch.empty_like(tensor)
    for c in range(tensor.shape[0]):
        ch  = tensor[c]
        mu  = ch.mean()
        std = ch.std().clamp(min=1e-6)
        out[c] = (ch - mu) / std
    return out


# ── Augmentations ─────────────────────────────────────────────────────────────

def _random_crop_resize(matrix: np.ndarray, target: int = 224,
                        min_scale: float = 0.7) -> np.ndarray:
    H, W    = matrix.shape
    scale   = np.random.uniform(min_scale, 1.0)
    crop_h  = max(int(H * scale), 4)
    crop_w  = max(int(W * scale), 4)
    top     = np.random.randint(0, max(H - crop_h + 1, 1))
    left    = np.random.randint(0, max(W - crop_w + 1, 1))
    cropped = matrix[top:top + crop_h, left:left + crop_w].copy()
    return resize_matrix(cropped, target)


def augment_train(matrix: np.ndarray, target: int = 224) -> np.ndarray:
    """Thermal-appropriate training augmentations on raw temperature matrix."""
    matrix = _random_crop_resize(matrix, target)

    # Random flips
    if np.random.rand() > 0.5:
        matrix = np.fliplr(matrix).copy()
    if np.random.rand() > 0.5:
        matrix = np.flipud(matrix).copy()

    # Rotation (multiples of 90°)
    k = np.random.randint(0, 4)
    if k:
        matrix = np.rot90(matrix, k).copy()

    # Temperature offset — ambient drift simulation
    matrix = matrix + float(np.random.uniform(-2.0, 2.0))

    # Temperature scale — emissivity variation
    matrix = matrix * float(np.random.uniform(0.9, 1.1))

    # Gaussian sensor noise
    matrix = (matrix
              + np.random.normal(0, np.random.uniform(0, 0.8),
                                 matrix.shape).astype(np.float32))
    return matrix.astype(np.float32)


def augment_val(matrix: np.ndarray, target: int = 224) -> np.ndarray:
    """Validation transform: centre-crop to square then resize."""
    H, W    = matrix.shape
    side    = min(H, W)
    top     = (H - side) // 2
    left    = (W - side) // 2
    cropped = matrix[top:top + side, left:left + side]
    return resize_matrix(cropped, target)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ThermalDefectDataset(Dataset):
    """
    Loads raw IR CSV matrices and returns (5-channel tensor, label vector).

    Args:
        parquet_path : path to dataset_V2.parquet
        csv_dir      : directory containing raw *.csv temperature matrices
        train        : True → training augmentations; False → val transform
        target_size  : spatial resolution after resize/crop (default 224)
    """

    def __init__(self, parquet_path: str, csv_dir: str,
                 train: bool = True, target_size: int = 224):
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=["IR_Image1Name"] + LABEL_COLS)

        # IR_Image1Name stores the bare filename (e.g. TDI_xxx.csv); map to csv_dir
        df["csv_path"] = df["IR_Image1Name"].apply(
            lambda f: os.path.join(csv_dir, os.path.basename(f))
        )
        df = df[df["csv_path"].apply(os.path.exists)].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(
                f"No matching CSVs found in '{csv_dir}'. "
                "Check that csv_dir contains the raw *.csv matrix files "
                "whose names match IR_Image1Name in the parquet."
            )

        self.csv_paths   = df["csv_path"].tolist()
        self.labels      = df[LABEL_COLS].values.astype("float32")
        self.train       = train
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.csv_paths)

    def __getitem__(self, idx: int):
        matrix = load_csv_matrix(self.csv_paths[idx])

        if self.train:
            matrix = augment_train(matrix, self.target_size)
        else:
            matrix = augment_val(matrix, self.target_size)

        channels = compute_thermal_channels(matrix)               # (5, H, W)
        tensor   = normalize_channels(torch.from_numpy(channels)) # (5, H, W)

        return tensor, self.labels[idx]