from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EMBEDDING_DIRS = {
    "v1": ROOT / "embeddings",
    "v3": ROOT / "embeddings_v3",
}

RESULTS_DIRS = {
    "v1": ROOT / "results_v1_baseline",
    "v3": ROOT / "results",
}


def normalize_version(version: str) -> str:
    version = version.lower().strip()
    if version not in EMBEDDING_DIRS:
        raise ValueError(f"Unsupported version '{version}'. Use 'v1' or 'v3'.")
    return version


def get_embedding_dir(version: str, create: bool = False) -> Path:
    emb_dir = EMBEDDING_DIRS[normalize_version(version)]
    if create:
        emb_dir.mkdir(exist_ok=True)
    return emb_dir


def get_results_dir(version: str, create: bool = False) -> Path:
    results_dir = RESULTS_DIRS[normalize_version(version)]
    if create:
        results_dir.mkdir(exist_ok=True)
    return results_dir


def load_embedding(protein_id: str, version: str) -> torch.Tensor:
    emb_dir = get_embedding_dir(version, create=False)
    emb_path = emb_dir / f"{protein_id}.pt"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing embedding: {emb_path}. Run src/02_embed.py for version '{version}' first."
        )
    return torch.load(emb_path, weights_only=True)
