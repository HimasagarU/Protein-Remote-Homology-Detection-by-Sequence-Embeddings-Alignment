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


from numba import njit

@njit
def _smith_waterman_numba(S, gap_open=-1.0, gap_extend=-0.1):
    """
    Numba-optimized SW with Affine Gaps.
    gap_open: Penalty for opening a gap (charged once)
    gap_extend: Penalty for extending an already open gap
    """
    M, N = S.shape
    # F: Match/Mismatch, IX: Gap in row, IY: Gap in column
    F  = np.zeros((M + 1, N + 1), dtype=np.float32)
    IX = np.zeros((M + 1, N + 1), dtype=np.float32)
    IY = np.zeros((M + 1, N + 1), dtype=np.float32)
    TB = np.zeros((M + 1, N + 1), dtype=np.int8) # 0=stop, 1=diag, 2=up, 3=left
    
    max_score, max_pos = 0.0, (0, 0)
    
    # Initialize IX/IY with very small values to avoid opening gaps at 0
    # Actually for local alignment, 0 is the floor.
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            # Diag match
            diag = F[i-1, j-1] + S[i-1, j-1]
            
            # Update IX (gap in B / deletion in A)
            # IX[i,j] can come from Match Matrix (open) or from IX itself (extend)
            ix_open   = F[i-1, j] + gap_open + gap_extend
            ix_extend = IX[i-1, j] + gap_extend
            IX[i, j]  = max(ix_open, ix_extend)
            
            # Update IY (gap in A / insertion in A)
            iy_open   = F[i, j-1] + gap_open + gap_extend
            iy_extend = IY[i, j-1] + gap_extend
            IY[i, j]  = max(iy_open, iy_extend)
            
            best = max(0.0, diag, IX[i, j], IY[i, j])
            F[i, j] = best
            
            if   best == 0.0: TB[i, j] = 0
            elif best == diag: TB[i, j] = 1
            elif best == IX[i, j]: TB[i, j] = 2
            else:              TB[i, j] = 3
            
            if best > max_score:
                max_score, max_pos = best, (i, j)
                
    # Traceback
    path = []
    curr_i, curr_j = max_pos
    while TB[curr_i, curr_j] != 0:
        path.append((curr_i - 1, curr_j - 1))
        t = TB[curr_i, curr_j]
        if   t == 1: curr_i -= 1; curr_j -= 1
        elif t == 2: curr_i -= 1
        else:        curr_j -= 1
            
    return float(max_score), path[::-1]


def smith_waterman(S: np.ndarray, gap_penalty: float = -1.0):
    """
    Compatibility wrapper. Interprets gap_penalty as gap_open+gap_extend.
    Defaulting gap_extend to gap_penalty/10 if not provided separately.
    """
    return _smith_waterman_numba(S, gap_open=gap_penalty, gap_extend=gap_penalty*0.1)


@njit
def _needleman_wunsch_numba(S, gap_penalty):
    M, N = S.shape
    F  = np.full((M + 1, N + 1), -1e9, dtype=np.float32)
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
    
    path, curr_i, curr_j = [], M, N
    while curr_i > 0 or curr_j > 0:
        path.append((curr_i - 1, curr_j - 1))
        t = TB[curr_i, curr_j]
        if   t == 1: curr_i -= 1; curr_j -= 1
        elif t == 2: curr_i -= 1
        else:        curr_j -= 1
    return float(F[M, N]), path[::-1]


def needleman_wunsch(S: np.ndarray, gap_penalty: float = -1.0):
    return _needleman_wunsch_numba(S, gap_penalty)



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
