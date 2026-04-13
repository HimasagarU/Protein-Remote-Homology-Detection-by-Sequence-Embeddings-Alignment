"""
Phase 4 -- Dynamic programming alignment on ESM-2 similarity matrices.

Smith-Waterman uses affine gaps and returns a residue-residue match path.
Needleman-Wunsch uses affine gaps for full-length alignment.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit

from pipeline_common import DATA_DIR, load_embedding, normalize_version


NEG_INF = np.float32(-1e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align protein pairs with Smith-Waterman or Needleman-Wunsch.")
    parser.add_argument("--version", choices=["v1", "v3"], default="v1")
    parser.add_argument("--mode", choices=["local", "global"], default="local")
    parser.add_argument("--pairs", type=int, default=5, help="Number of test pairs to preview.")
    parser.add_argument("--gap-open", type=float, default=-1.0)
    parser.add_argument("--gap-extend", type=float, default=-0.1)
    return parser.parse_args()


def compute_similarity_matrix(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    norm_a = F.normalize(emb_a, dim=1)
    norm_b = F.normalize(emb_b, dim=1)
    return torch.mm(norm_a, norm_b.T)


@njit
def _smith_waterman_affine_numba(S, gap_open, gap_extend):
    M, N = S.shape
    H = np.zeros((M + 1, N + 1), dtype=np.float32)
    E = np.full((M + 1, N + 1), NEG_INF, dtype=np.float32)
    G = np.full((M + 1, N + 1), NEG_INF, dtype=np.float32)

    H_tb = np.zeros((M + 1, N + 1), dtype=np.int8)  # 0 stop, 1 diag, 2 E, 3 G
    E_tb = np.zeros((M + 1, N + 1), dtype=np.int8)  # 1 open from H, 2 extend E
    G_tb = np.zeros((M + 1, N + 1), dtype=np.int8)  # 1 open from H, 2 extend G

    max_score = 0.0
    max_i = 0
    max_j = 0

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            open_e = H[i - 1, j] + gap_open
            extend_e = E[i - 1, j] + gap_extend
            if open_e >= extend_e:
                E[i, j] = open_e
                E_tb[i, j] = 1
            else:
                E[i, j] = extend_e
                E_tb[i, j] = 2

            open_g = H[i, j - 1] + gap_open
            extend_g = G[i, j - 1] + gap_extend
            if open_g >= extend_g:
                G[i, j] = open_g
                G_tb[i, j] = 1
            else:
                G[i, j] = extend_g
                G_tb[i, j] = 2

            diag = H[i - 1, j - 1] + S[i - 1, j - 1]
            best = diag
            tb = 1
            if E[i, j] > best:
                best = E[i, j]
                tb = 2
            if G[i, j] > best:
                best = G[i, j]
                tb = 3
            if best < 0.0:
                best = 0.0
                tb = 0

            H[i, j] = best
            H_tb[i, j] = tb

            if best > max_score:
                max_score = best
                max_i = i
                max_j = j

    path = []
    state = 0  # 0 = H, 1 = E, 2 = G
    i = max_i
    j = max_j

    while i > 0 and j > 0:
        if state == 0:
            tb = H_tb[i, j]
            if tb == 0:
                break
            if tb == 1:
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif tb == 2:
                state = 1
            else:
                state = 2
        elif state == 1:
            tb = E_tb[i, j]
            i -= 1
            if tb == 1:
                state = 0
        else:
            tb = G_tb[i, j]
            j -= 1
            if tb == 1:
                state = 0

    path.reverse()
    return float(max_score), path


@njit
def _needleman_wunsch_affine_numba(S, gap_open, gap_extend):
    M, N = S.shape
    H = np.full((M + 1, N + 1), NEG_INF, dtype=np.float32)
    E = np.full((M + 1, N + 1), NEG_INF, dtype=np.float32)
    G = np.full((M + 1, N + 1), NEG_INF, dtype=np.float32)

    H_tb = np.zeros((M + 1, N + 1), dtype=np.int8)
    E_tb = np.zeros((M + 1, N + 1), dtype=np.int8)
    G_tb = np.zeros((M + 1, N + 1), dtype=np.int8)

    H[0, 0] = 0.0
    for i in range(1, M + 1):
        E[i, 0] = gap_open if i == 1 else E[i - 1, 0] + gap_extend
        H[i, 0] = E[i, 0]
        H_tb[i, 0] = 2
        E_tb[i, 0] = 1 if i == 1 else 2
    for j in range(1, N + 1):
        G[0, j] = gap_open if j == 1 else G[0, j - 1] + gap_extend
        H[0, j] = G[0, j]
        H_tb[0, j] = 3
        G_tb[0, j] = 1 if j == 1 else 2

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            open_e = H[i - 1, j] + gap_open
            extend_e = E[i - 1, j] + gap_extend
            if open_e >= extend_e:
                E[i, j] = open_e
                E_tb[i, j] = 1
            else:
                E[i, j] = extend_e
                E_tb[i, j] = 2

            open_g = H[i, j - 1] + gap_open
            extend_g = G[i, j - 1] + gap_extend
            if open_g >= extend_g:
                G[i, j] = open_g
                G_tb[i, j] = 1
            else:
                G[i, j] = extend_g
                G_tb[i, j] = 2

            diag = H[i - 1, j - 1] + S[i - 1, j - 1]
            best = diag
            tb = 1
            if E[i, j] > best:
                best = E[i, j]
                tb = 2
            if G[i, j] > best:
                best = G[i, j]
                tb = 3

            H[i, j] = best
            H_tb[i, j] = tb

    path = []
    state = 0
    i = M
    j = N

    while i > 0 or j > 0:
        if state == 0:
            tb = H_tb[i, j]
            if tb == 1 and i > 0 and j > 0:
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif tb == 2 and i > 0:
                state = 1
            elif tb == 3 and j > 0:
                state = 2
            else:
                break
        elif state == 1:
            tb = E_tb[i, j]
            i -= 1
            if tb == 1:
                state = 0
        else:
            tb = G_tb[i, j]
            j -= 1
            if tb == 1:
                state = 0

    path.reverse()
    return float(H[M, N]), path


def smith_waterman(S: np.ndarray, gap_open: float = -1.0, gap_extend: float = -0.1):
    return _smith_waterman_affine_numba(S, gap_open, gap_extend)


def needleman_wunsch(S: np.ndarray, gap_open: float = -1.0, gap_extend: float = -0.1):
    return _needleman_wunsch_affine_numba(S, gap_open, gap_extend)


def align_proteins(
    id_a: str,
    id_b: str,
    version: str = "v1",
    mode: str = "local",
    gap_open: float = -1.0,
    gap_extend: float = -0.1,
):
    emb_a = load_embedding(id_a, version)
    emb_b = load_embedding(id_b, version)
    S = compute_similarity_matrix(emb_a, emb_b).numpy()

    if mode == "local":
        return smith_waterman(S, gap_open=gap_open, gap_extend=gap_extend)
    return needleman_wunsch(S, gap_open=gap_open, gap_extend=gap_extend)


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    import pandas as pd

    pairs = pd.read_csv(DATA_DIR / "test_pairs.csv")
    print(f"Testing {args.mode} alignment on the first {args.pairs} pairs ({version}):\n")
    for _, row in pairs.head(args.pairs).iterrows():
        score, path = align_proteins(
            row["id_a"],
            row["id_b"],
            version=version,
            mode=args.mode,
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
        )
        label = "HOMOLOG" if row["label"] == 1 else "NON-HOMOLOG"
        print(f"  {row['id_a']} vs {row['id_b']} [{label}]")
        print(f"    score: {score:.2f}   matched residues: {len(path)}")
