"""
Phase 3 -- Residue-residue cosine similarity matrices.
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from pipeline_common import DATA_DIR, load_embedding, normalize_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute residue-level cosine similarity matrices.")
    parser.add_argument("--version", choices=["v1", "v3"], default="v1")
    parser.add_argument("--pairs", type=int, default=3, help="Number of test pairs to preview.")
    return parser.parse_args()


def compute_similarity_matrix(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """
    emb_a: [M, D], emb_b: [N, D]
    returns: [M, N] cosine similarity matrix
    """
    norm_a = F.normalize(emb_a, dim=1)
    norm_b = F.normalize(emb_b, dim=1)
    return torch.mm(norm_a, norm_b.T)


def load_and_compare(id_a: str, id_b: str, version: str = "v1") -> torch.Tensor:
    emb_a = load_embedding(id_a, version)
    emb_b = load_embedding(id_b, version)
    return compute_similarity_matrix(emb_a, emb_b)


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    pairs = pd.read_csv(DATA_DIR / "test_pairs.csv")

    for _, row in pairs.head(args.pairs).iterrows():
        S = load_and_compare(row["id_a"], row["id_b"], version=version)
        label = "HOMOLOG" if row["label"] == 1 else "NON-HOMOLOG"
        print(f"{row['id_a']} vs {row['id_b']} [{label}] ({version})")
        print(f"  Matrix shape : {tuple(S.shape)}")
        print(f"  Mean sim     : {S.mean():.4f}")
        print(f"  Max sim      : {S.max():.4f}")
        print(f"  Min sim      : {S.min():.4f}")
        print()
