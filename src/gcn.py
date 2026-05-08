"""
gcn.py
------
Graph Convolutional Network classifier head for multi-label defect detection.

Novel contribution:
  - 2-layer GCN over 8 defect-label nodes
  - Adjacency matrix built from two sources:
      1. Statistical co-occurrence from training labels
      2. Hard-coded domain priors (SinkMarks/Underfilled → NOK; Streaks chain)
  - Models inter-label dependencies that a flat Linear head ignores

Reference: ML-GCN (Chen et al., CVPR 2019) — adapted for small label sets
           without word-embedding features.
"""

import numpy as np
import torch
import torch.nn as nn

# ── Label index reference ──────────────────────────────────────────────────────
# 0: SinkMarks  1: SprueCircle  2: Underfilled  3: OldGranulate
# 4: Streaks1   5: Streaks2     6: Streaks3     7: NOK


# ── Graph Convolution Layer ───────────────────────────────────────────────────

class GraphConvolution(nn.Module):
    """
    Single GCN layer: H' = σ(Ã_norm H W)

    Args:
        in_features  : input feature dimension per node
        out_features : output feature dimension per node
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, num_nodes, in_features)
            adj : (num_nodes, num_nodes) — normalised adjacency (on same device)
        Returns:
            (batch, num_nodes, out_features)
        """
        # Linear transform: (batch, N, in) × (in, out) → (batch, N, out)
        support = torch.matmul(x, self.weight)
        # Graph aggregation: (1, N, N) × (batch, N, out) → (batch, N, out)
        out = torch.matmul(adj.unsqueeze(0), support)
        if self.bias is not None:
            out = out + self.bias
        return out


# ── Adjacency Matrix Construction ─────────────────────────────────────────────

def build_adjacency_matrix(label_matrix: np.ndarray,
                            num_labels: int = 8) -> np.ndarray:
    """
    Build a row-normalised adjacency matrix from:
      1. Conditional co-occurrence P(label_j=1 | label_i=1) from training data
      2. Hard-coded domain-prior edges

    Domain priors
    -------------
    SinkMarks(0) ↔ NOK(7)       : deterministic (NOK = SinkMarks OR Underfilled)
    Underfilled(2) ↔ NOK(7)     : deterministic
    StreaksL1(4) ↔ StreaksL2(5) : ordered severity chain  (weight 0.8)
    StreaksL2(5) ↔ StreaksL3(6) : ordered severity chain  (weight 0.8)

    Args:
        label_matrix : (N_train, num_labels) binary float array
        num_labels   : number of label nodes (default 8)

    Returns:
        A_norm : (num_labels, num_labels) float32 row-normalised adjacency
    """
    A = np.zeros((num_labels, num_labels), dtype=np.float32)

    # 1. Statistical co-occurrence
    for i in range(num_labels):
        pos_i = float(label_matrix[:, i].sum())
        if pos_i > 0:
            for j in range(num_labels):
                A[i, j] = float((label_matrix[:, i] * label_matrix[:, j]).sum()) / pos_i

    # 2. Domain prior edges (take max to avoid downgrading statistical weights)
    A[0, 7] = max(A[0, 7], 1.0);  A[7, 0] = max(A[7, 0], 1.0)   # SinkMarks ↔ NOK
    A[2, 7] = max(A[2, 7], 1.0);  A[7, 2] = max(A[7, 2], 1.0)   # Underfilled ↔ NOK
    A[4, 5] = max(A[4, 5], 0.8);  A[5, 4] = max(A[5, 4], 0.8)   # L1 ↔ L2
    A[5, 6] = max(A[5, 6], 0.8);  A[6, 5] = max(A[6, 5], 0.8)   # L2 ↔ L3

    # Self-loops (each node attends to itself)
    np.fill_diagonal(A, 1.0)

    # Row-normalise: Ã_norm = D̃⁻¹ Ã
    row_sums = A.sum(axis=1, keepdims=True).clip(min=1e-6)
    return (A / row_sums).astype(np.float32)


# ── GCN Classifier Head ───────────────────────────────────────────────────────

class GCNHead(nn.Module):
    """
    2-layer GCN classifier head.

    Architecture (per sample):
        image_features (in_features,)
          → Linear(in_features, num_labels × hidden_dim) + ReLU + Dropout
          → reshape to (num_labels, hidden_dim)
          → GCN Layer 1: ReLU( Ã H W₁ )   → (num_labels, hidden_dim//2)
          → GCN Layer 2:       Ã H' W₂      → (num_labels, 1)
          → squeeze                           → (num_labels,)  logits

    Args:
        in_features : backbone output dimension (1280 for EfficientNet-B0)
        num_labels  : number of defect labels (8)
        hidden_dim  : GCN node feature dimension (256)
        adj_matrix  : (num_labels, num_labels) float32 ndarray; if None → identity
    """

    def __init__(self, in_features: int, num_labels: int = 8,
                 hidden_dim: int = 256, adj_matrix: np.ndarray = None):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Project global image features to per-label initial node features
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels * hidden_dim),
        )

        self.gc1  = GraphConvolution(hidden_dim, hidden_dim // 2)
        self.gc2  = GraphConvolution(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

        # Adjacency as a non-trainable buffer (moves with .to(device))
        if adj_matrix is not None:
            adj_t = torch.from_numpy(adj_matrix).float()
        else:
            adj_t = torch.eye(num_labels)
        self.register_buffer('adj', adj_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_features)
        Returns:
            logits : (batch, num_labels)
        """
        batch = x.size(0)
        h = self.input_proj(x)                             # (B, L*D)
        h = h.view(batch, self.num_labels, self.hidden_dim) # (B, L, D)

        h = self.relu(self.gc1(h, self.adj))               # (B, L, D//2)
        h = self.gc2(h, self.adj)                          # (B, L, 1)
        return h.squeeze(-1)                               # (B, L)

    def update_adjacency(self, label_matrix: np.ndarray) -> None:
        """Recompute adjacency from new training labels (e.g. after split)."""
        A = build_adjacency_matrix(label_matrix, self.num_labels)
        self.adj = torch.from_numpy(A).float().to(self.adj.device)