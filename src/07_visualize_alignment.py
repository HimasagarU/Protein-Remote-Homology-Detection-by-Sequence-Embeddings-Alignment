"""
Phase 7 -- Visualization for Presentation.

This script generates a high-quality visualization of the Smith-Waterman
alignment path overlaid on the continuous embedding similarity matrix.
It compares a known remote homolog pair (positive) against a non-homolog pair (negative).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_al)
smith_waterman = _al.smith_waterman
compute_similarity_matrix = _al.compute_similarity_matrix

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "embeddings"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def visualize_pair(id_a: str, id_b: str, ax, title_prefix: str, is_homolog: bool):
    """Computes similarity matrix and alignment path, then plots it on the given axes."""
    # 1. Load embeddings
    emb_a = torch.load(EMB_DIR / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(EMB_DIR / f"{id_b}.pt", weights_only=True)
    
    # 2. Compute similarity matrix S
    # S is [M, N] where M is length of Protein A (Y-axis), N is length of Protein B (X-axis)
    S = compute_similarity_matrix(emb_a, emb_b).numpy()
    
    # 3. Compute optimal structure alignment path
    score, path = smith_waterman(S, gap_penalty=-1.0)
    
    # 4. Plot Heatmap
    im = ax.imshow(S, cmap="viridis", aspect="auto", origin="upper", vmin=-0.5, vmax=1.0)
    
    # 5. Overlay Alignment Path
    if len(path) > 0:
        # path is a list of (y, x) tuples corresponding to (row in A, col in B)
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color="red", linewidth=2.5, label="Alignment Path")
    
    # 6. Formatting
    status = "Remote Homolog" if is_homolog else "Non-Homolog"
    ax.set_title(f"{title_prefix}: {id_a} vs {id_b}\n[{status}] - Score: {score:.2f}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Protein A ({id_a}) Length", fontsize=10)
    ax.set_xlabel(f"Protein B ({id_b}) Length", fontsize=10)
    ax.legend(loc="upper right")
    
    return im

def generate_presentation_plot():
    print("Generating visualization for presentation...")
    
    # Selected a strong positive and a clear negative from the dataset
    pairs_to_plot = [
        {"id_a": "d1nhya1", "id_b": "d2hrkb_", "is_homolog": True,  "prefix": "Positive Pair"},
        {"id_a": "d2g3ra2", "id_b": "d7lvsb_", "is_homolog": False, "prefix": "Negative Pair"}
    ]
    
    # Set up matplotlib figure (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, pair in enumerate(pairs_to_plot):
        im = visualize_pair(
            id_a=pair["id_a"], 
            id_b=pair["id_b"], 
            ax=axes[i], 
            title_prefix=pair["prefix"], 
            is_homolog=pair["is_homolog"]
        )
        
    # Add a global colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    
    # Save the output image
    out_path = RESULTS / "alignment_visualization.png"
    plt.subplots_adjust(right=0.9)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully saved high-quality visualization to: {out_path}")

if __name__ == "__main__":
    generate_presentation_plot()
