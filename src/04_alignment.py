"""
Phase 4 -- Dynamic Programming Alignment.

Smith-Waterman (local) and Needleman-Wunsch (global) alignment
on ESM-2 cosine similarity matrices.
"""

import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F

def compute_similarity_matrix(emb_a, emb_b):
    """Cosine similarity matrix [M, N] from embeddings [M,D] and [N,D]."""
    return torch.mm(F.normalize(emb_a, dim=1), F.normalize(emb_b, dim=1).T)

EMB_DIR = Path(__file__).resolve().parent.parent / "embeddings"


def smith_waterman(S: np.ndarray, gap_penalty: float = -1.0):
    """
    Local alignment on similarity matrix S [M, N].
    Returns (max_score, path) where path is list of (i, j) tuples.
    """
    M, N = S.shape
    F  = np.zeros((M + 1, N + 1), dtype=np.float32)
    TB = np.zeros((M + 1, N + 1), dtype=np.int8)
    # Traceback: 0=stop, 1=diagonal, 2=up, 3=left
    max_score, max_pos = 0.0, (0, 0)

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            diag = F[i-1, j-1] + S[i-1, j-1]
            up   = F[i-1, j]   + gap_penalty
            left = F[i,   j-1] + gap_penalty
            best = max(0.0, diag, up, left)
            F[i, j] = best

            if   best == 0:    TB[i, j] = 0
            elif best == diag: TB[i, j] = 1
            elif best == up:   TB[i, j] = 2
            else:              TB[i, j] = 3

            if best > max_score:
                max_score, max_pos = best, (i, j)

    # Traceback
    path = []
    i, j = max_pos
    while TB[i, j] != 0:
        path.append((i - 1, j - 1))
        t = TB[i, j]
        if   t == 1: i -= 1; j -= 1
        elif t == 2: i -= 1
        else:        j -= 1

    return float(max_score), path[::-1]


def needleman_wunsch(S: np.ndarray, gap_penalty: float = -1.0):
    """
    Global alignment on similarity matrix S [M, N].
    Returns (score, path).
    """
    M, N = S.shape
    F  = np.full((M + 1, N + 1), -np.inf, dtype=np.float32)
    TB = np.zeros((M + 1, N + 1), dtype=np.int8)
    F[0, 0] = 0.0
    for i in range(1, M + 1): F[i, 0] = i * gap_penalty
    for j in range(1, N + 1): F[0, j] = j * gap_penalty

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            diag = F[i-1, j-1] + S[i-1, j-1]
            up   = F[i-1, j]   + gap_penalty
            left = F[i,   j-1] + gap_penalty
            best = max(diag, up, left)
            F[i, j] = best
            if   best == diag: TB[i, j] = 1
            elif best == up:   TB[i, j] = 2
            else:              TB[i, j] = 3

    # Traceback from bottom-right
    path, i, j = [], M, N
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = TB[i, j]
        if   t == 1: i -= 1; j -= 1
        elif t == 2: i -= 1
        else:        j -= 1

    return float(F[M, N]), path[::-1]


def align_proteins(id_a: str, id_b: str,
                   mode: str = "local",
                   gap_penalty: float = -1.0):
    """
    End-to-end: load embeddings -> similarity matrix -> alignment.
    Returns (score, path).
    """
    emb_a = torch.load(EMB_DIR / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(EMB_DIR / f"{id_b}.pt", weights_only=True)
    S     = compute_similarity_matrix(emb_a, emb_b).numpy()

    if mode == "local":
        return smith_waterman(S, gap_penalty)
    else:
        return needleman_wunsch(S, gap_penalty)


# ─── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    pairs = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "test_pairs.csv"
    )

    print("Testing Smith-Waterman on first 5 pairs:\n")
    for _, row in pairs.head(5).iterrows():
        score, path = align_proteins(row["id_a"], row["id_b"], mode="local")
        label = "HOMOLOG" if row["label"] == 1 else "NON-HOMOLOG"
        print(f"  {row['id_a']} vs {row['id_b']}  [{label}]")
        print(f"    SW score: {score:.2f}   path length: {len(path)}")
