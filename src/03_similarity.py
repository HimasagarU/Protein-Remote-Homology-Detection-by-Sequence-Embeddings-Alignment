"""
Phase 3 -- Dynamic Cosine Similarity Matrix.

Computes pairwise cosine similarity between per-residue embeddings
of two proteins.  Result is an [M, N] matrix where M and N are the
sequence lengths of proteins A and B respectively.
"""

import torch
import torch.nn.functional as F
from pathlib import Path

EMB_DIR = Path(__file__).resolve().parent.parent / "embeddings"


def compute_similarity_matrix(emb_a: torch.Tensor,
                               emb_b: torch.Tensor) -> torch.Tensor:
    """
    emb_a: [M, D]   emb_b: [N, D]
    Returns: [M, N] cosine similarity matrix, values in [-1, 1].
    """
    norm_a = F.normalize(emb_a, dim=1)      # [M, D]
    norm_b = F.normalize(emb_b, dim=1)      # [N, D]
    return torch.mm(norm_a, norm_b.T)       # [M, N]


def load_and_compare(id_a: str, id_b: str) -> torch.Tensor:
    """Load two embeddings by protein ID and return their similarity matrix."""
    emb_a = torch.load(EMB_DIR / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(EMB_DIR / f"{id_b}.pt", weights_only=True)
    return compute_similarity_matrix(emb_a, emb_b)


# ─── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    pairs = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "test_pairs.csv"
    )

    # Test on first 3 pairs
    for _, row in pairs.head(3).iterrows():
        S = load_and_compare(row["id_a"], row["id_b"])
        label = "HOMOLOG" if row["label"] == 1 else "NON-HOMOLOG"
        print(f"{row['id_a']} vs {row['id_b']}  [{label}]")
        print(f"  Matrix shape : {tuple(S.shape)}")
        print(f"  Mean sim     : {S.mean():.4f}")
        print(f"  Max sim      : {S.max():.4f}")
        print(f"  Min sim      : {S.min():.4f}")
        print()
