"""
Phase 7 -- Alignment visualization for either supported pipeline version.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_common import get_results_dir, load_embedding, normalize_version


_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alignment)
compute_similarity_matrix = _alignment.compute_similarity_matrix
smith_waterman = _alignment.smith_waterman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one positive and one negative alignment example.")
    parser.add_argument("--version", choices=["v1", "v3"], default="v3")
    parser.add_argument("--gap-open", type=float, default=-1.0)
    parser.add_argument("--gap-extend", type=float, default=-0.1)
    return parser.parse_args()


def visualize_pair(
    id_a: str,
    id_b: str,
    version: str,
    ax,
    title_prefix: str,
    is_homolog: bool,
    gap_open: float,
    gap_extend: float,
):
    emb_a = load_embedding(id_a, version)
    emb_b = load_embedding(id_b, version)
    S = compute_similarity_matrix(emb_a, emb_b).numpy()
    score, path = smith_waterman(S, gap_open=gap_open, gap_extend=gap_extend)

    im = ax.imshow(S, cmap="viridis", aspect="auto", origin="upper", vmin=-0.5, vmax=1.0)
    if path:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color="red", linewidth=2.0, label="Matched residues")

    status = "Remote homolog" if is_homolog else "Non-homolog"
    ax.set_title(f"{title_prefix}: {id_a} vs {id_b}\n[{status}] score={score:.2f}")
    ax.set_ylabel(f"Protein A ({len(emb_a)} residues)")
    ax.set_xlabel(f"Protein B ({len(emb_b)} residues)")
    ax.legend(loc="upper right")
    return im


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    results_dir = get_results_dir(version, create=True)

    pairs_to_plot = [
        {"id_a": "d1nhya1", "id_b": "d2hrkb_", "is_homolog": True, "prefix": "Positive pair"},
        {"id_a": "d2g3ra2", "id_b": "d7lvsb_", "is_homolog": False, "prefix": "Negative pair"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, pair in zip(axes, pairs_to_plot):
        im = visualize_pair(
            id_a=pair["id_a"],
            id_b=pair["id_b"],
            version=version,
            ax=axis,
            title_prefix=pair["prefix"],
            is_homolog=pair["is_homolog"],
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
        )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine similarity")
    plt.subplots_adjust(right=0.9)

    out_path = results_dir / "alignment_visualization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved alignment visualization -> {out_path}")
