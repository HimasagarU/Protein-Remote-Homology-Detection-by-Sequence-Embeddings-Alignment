"""
Phase 8 -- Dendrogram of globally pooled protein embeddings.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_common import DATA_DIR, get_results_dir, load_embedding, normalize_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a dendrogram of pooled protein embeddings.")
    parser.add_argument("--version", choices=["v1", "v3"], default="v3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    results_dir = get_results_dir(version, create=True)
    labels_df = pd.read_csv(DATA_DIR / "scop_labels.csv")

    target_superfamilies = [
        "a.1.1",
        "a.4.5",
        "a.22.1",
        "a.24.9",
        "a.25.1",
    ]

    sampled = []
    for superfamily in target_superfamilies:
        sf_df = labels_df[labels_df["superfamily"] == superfamily]
        sampled.append(sf_df.sample(n=min(7, len(sf_df)), random_state=42))
    subset_df = pd.concat(sampled, ignore_index=True)

    global_embeddings = []
    labels = []
    for _, row in subset_df.iterrows():
        tensor = load_embedding(row["id"], version)
        global_embeddings.append(tensor.mean(dim=0).numpy())
        labels.append(f"{row['id']} ({row['superfamily']})")

    global_embeddings = np.asarray(global_embeddings)
    dist_matrix = pdist(global_embeddings, metric="cosine")
    linkage_matrix = linkage(dist_matrix, method="average")

    fig, ax = plt.subplots(figsize=(14, 8))
    set_link_color_palette(["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"])
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0.45,
        above_threshold_color="grey",
    )

    ax.set_title(f"Protein Embedding Clustering Dendrogram ({version})")
    ax.set_ylabel("Cosine distance")
    ax.set_xlabel("Proteins labeled by superfamily")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    out_path = results_dir / "embedding_tree_clustering.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved tree visualization -> {out_path}")
