"""
Phase 8 -- Tree Visualization for Presentation.

Generates a Hierarchical Clustering Dendrogram (Tree) of protein embeddings.
This visually proves to your professor that the pre-trained ESM-2 model inherently 
groups structurally related proteins together in continuous vector space, *even without* Smith-Waterman.
"""

import pandas as pd
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "embeddings"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def generate_tree_plot():
    print("Generating Hierarchical Clustering Tree (Dendrogram)...")
    
    # 1. Load Labels
    df = pd.read_csv(ROOT / "data" / "scop_labels.csv")
    
    # 2. Select 5 diverse superfamilies to show clear clustering
    target_superfamilies = [
        "a.1.1",    # Globin-like
        "a.4.5",    # Cytochrome c
        "a.22.1",   # EF-hand
        "a.24.9",   # 4-helical cytokines
        "a.25.1"    # Ferritin-like
    ]
    
    # Sample 7 proteins from each
    sampled_dfs = []
    for sf in target_superfamilies:
        sf_df = df[df["superfamily"] == sf].head(7)
        sampled_dfs.append(sf_df)
    
    subset_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    print(f"Sampled {len(subset_df)} proteins across {len(target_superfamilies)} superfamilies.")

    # 3. Load global average embeddings
    global_embeddings = []
    labels = []
    
    for _, row in subset_df.iterrows():
        prot_id = row['id']
        sf_label = row['superfamily']
        
        emb_path = EMB_DIR / f"{prot_id}.pt"
        if not emb_path.exists():
            print(f"Missing {prot_id}.pt, skipping...")
            continue
            
        # Load and mean-pool the [L, 1280] sequence to a generic global [1280] structural vector
        tensor = torch.load(emb_path, weights_only=True)
        global_vec = tensor.mean(dim=0).numpy()
        
        global_embeddings.append(global_vec)
        labels.append(f"{prot_id} ({sf_label})")
        
    global_embeddings = np.array(global_embeddings)
    
    # 4. Compute Pairwise Cosine Distances natively
    # pdist computes the condensed distance matrix
    dist_matrix = pdist(global_embeddings, metric='cosine')
    
    # 5. Compute Hierarchical Clustering Linkage
    # Using 'average' linkage on cosine distance
    Z = linkage(dist_matrix, method='average')
    
    # 6. Plotting the Dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Custom color palette for the 5 clusters
    set_link_color_palette(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231'])
    
    dendro = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0.45, # Tweak threshold to highlight our specific superfamily branches visually
        above_threshold_color='grey'
    )
    
    ax.set_title("Protein Embedding Clustering Dendrogram\n(Global Mean-Pooled ESM-2 Vectors)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cosine Distance", fontsize=12)
    ax.set_xlabel("Proteins labeled by (Superfamily)", fontsize=12)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Save Plot
    out_path = RESULTS / "embedding_tree_clustering.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Successfully saved tree dendrogram to: {out_path}")

if __name__ == "__main__":
    generate_tree_plot()
