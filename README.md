# Protein Remote Homology Detection by Sequence Embeddings Alignment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Abstract
Protein remote homology detection identifies evolutionary and structural similarities between proteins that share very low sequence identity (often < 20%, known as the "twilight zone"). Traditional alignment methods like BLAST and PSI-BLAST rely on discrete character matching, which frequently fails to capture distant structural relationships. 

This project introduces a novel alignment approach using dense, high-dimensional protein sequence embeddings extracted from a pre-trained Protein Language Model (ESM-2). By computing alignment paths through an embedding-derived continuous vector space using dynamic programming, we can successfully identify structurally similar folds that lack sequence similarity. This is crucial for applications such as drug target interaction studies, where identifying a remote homolog can reveal hidden off-target binding sites that sequence-only models completely miss.

---

## The Core Idea
The core idea driving this project is to shift from **discrete amino acid sequence matching** to **continuous geometric space alignment**.
Instead of looking up how often Amino Acid A mutates into Amino Acid B (as in the BLOSUM62 matrix used by BLAST), we ask an advanced Protein Language Model (ESM-2) to read the protein sequence and generate a "contextual vector" for every single residue.

Because ESM-2 is trained on ~250 million protein sequences, its internal representations become implicitly structure-aware. A residue's vector encapsulates not just its identity, but its 3D structural context. By computing the cosine similarity between the vectors of two completely different sequences, we can find structural mappings and align them accurately.

---

## Dataset: ASTRAL-20 / SCOPe

### Understanding SCOPe Hierarchy
We use the SCOPe (Structural Classification of Proteins — extended) database. SCOPe classifies proteins into 4 levels:
1. **Class:** Broad secondary structure category (e.g., All-alpha)
2. **Fold:** Overall 3D topology (e.g., TIM barrel)
3. **Superfamily:** Common evolutionary origin
4. **Family:** Close sequence relatives

**Definition of Remote Homologs:**
Two proteins are considered *remote homologs* if they belong to the **same Superfamily** but **different Families**. They share a fold and evolutionary origin, but their sequence identity is usually extremely low.

### Data Preparation
We utilize the ASTRAL-20 dataset, which filters the SCOPe database so that no two sequences share more than 20% identity. 
1. We parse the FASTA headers to extract SCOPe classification codes.
2. We clean and format the labels, breaking them into Class, Fold, Superfamily, and Family.
3. We generate a set of **Positive Pairs** (same superfamily, different family) and **Negative Pairs** (different superfamily/fold) to benchmark our algorithm robustly.

---

## Methods and Pipeline

The repository now supports two clean, reproducible versions:

- `v1` baseline: final-layer ESM-2 embeddings + residue-level cosine similarity + Smith-Waterman scoring
- `v3` final model: last-4-layer averaged embeddings + richer alignment features + logistic regression

### Phase 1: Data Acquisition & Labeling
`src/01_data_prep.py` parses the ASTRAL-20 FASTA headers, extracts SCOPe hierarchy labels, filters invalid or overly long sequences, and creates a balanced `test_pairs.csv` where:
- positive pairs = same superfamily, different family
- negative pairs = different superfamily

### Phase 2: ESM-2 Embedding Extraction
`src/02_embed.py` supports both pipeline versions with the same backbone model, `esm2_t33_650M_UR50D`.

- `v1` saves the final hidden layer to `embeddings/`
- `v3` averages the last 4 hidden layers and saves to `embeddings_v3/`

Each protein is represented as a tensor of shape `[L, 1280]`, where `L` is the sequence length.

### Phase 3: Dynamic Cosine Similarity Matrix
`src/03_similarity.py` builds a residue-residue cosine similarity matrix for a protein pair:
$$S_{i,j} = \frac{\mathbf{E}_{A,i} \cdot \mathbf{E}_{B,j}}{\|\mathbf{E}_{A,i}\| \|\mathbf{E}_{B,j}\|}$$

This produces a matrix of shape `[M, N]` for proteins of lengths `M` and `N`.

### Phase 4: Dynamic Programming Alignment
`src/04_alignment.py` performs dynamic programming on the cosine similarity matrix.

- Smith-Waterman is used for local alignment with affine gaps
- Needleman-Wunsch is available for global alignment with affine gaps

The local path returned by the code stores matched residue pairs only, so downstream coverage features stay well defined.

### Phase 5: Baseline Benchmark (`v1`)
`src/05_benchmark.py` evaluates raw Smith-Waterman scores as a ranking signal and writes baseline outputs to `results_v1_baseline/`.

### Phase 6: Final Feature Model (`v3`)
`src/06_improve.py` extracts 10 features from `v3` embeddings, including normalized alignment scores, pooled cosine similarity, path statistics, and matrix concentration features. The final classifier is logistic regression with protein-disjoint cross-validation, and outputs are written to `results/`.

### Phases 7-8: Visualization
`src/07_visualize_alignment.py` and `src/08_visualize_tree.py` can be run for either version to generate presentation-ready figures.

### Interpreting Results
To compare `v1` against `v3`, rerun the scripts after embedding extraction and compare:
- `results_v1_baseline/embedding_scores.csv`
- `results/metrics_summary.json`
- `results/improved_v3_scores.csv`

`v2` has been removed from the codebase and is no longer a supported experiment.

---

## Visualizations & Model Interpretability

To prove the efficacy of capturing structure natively via embeddings, we've developed two robust visualization scripts capable of operating dynamically on any sampled pair.

### 1. Smith-Waterman Alignment Heatmaps (`src/07_visualize_alignment.py`)
This tool generates a dynamic internal visual inspection of the alignment process.
- **Continuous Similarity:** It calculates the entire Dynamic Cosine Similarity Matrix between all vectors. High similarities turn brightly colored.
- **Positive Control:** For computationally verified Remote Homologs (< 20% sequence identical), you will see stark sequential structural bands in the matrix highlighting the geometric homology. 
- **Traceback Overlay:** We trace the computed sequence alignment natively over the similarity heatmap matrix (red path) using the Smith-Waterman recurrence path, creating an interpretable mapping of exactly how the secondary structures match.

![Smith-Waterman Alignment Traceback](results/alignment_visualization.png)

### 2. Hierarchical Clustering Dendrogram (`src/08_visualize_tree.py`)
While the heatmaps show the accuracy of *locally* mapping similar structures, the clustering dendrogram proves the power of the language model's latent space *globally*.
- **Mechanism:** We globally mean-pool the `[L, 1280]` per-residue tensors into a single fixed `[1280]` structure vector for each protein.
- **Clustering:** Generating purely via cosine distance on these vectors (with no Smith-Waterman DP logic), the proteins cluster perfectly.
- **Classification Mapping:** We sample proteins from distinctly different structural folds. The plotted tree instinctively creates identical sub-trees based entirely upon the ASTRAL-20 structural `a.X.X` hierarchy.

![Embedding Tree Clustering](results/embedding_tree_clustering.png)

---

## System Architecture Summary
```text
SCOPe / ASTRAL-20 Dataset (FASTA)
            |
            v
  [ Phase 1 ] Data Preparation
  Parse headers -> SCOPe labels -> clean FASTA
            |
            v
  [ Phase 2 ] ESM-2 Embedding Extraction
  v1 final layer or v3 last-4-layer average
            |
            v
  [ Phase 3 ] Dynamic Cosine Similarity Matrix
  S[M x N] for each protein pair
            |
            v
  [ Phase 4 ] DP Alignment (Smith-Waterman / Needleman-Wunsch)
  Optimal structural alignment path + score
            |
            v
  [ Phase 5 ] Baseline Benchmarking
            |
            v
  [ Phase 6 ] v3 Feature Model + Protein-Disjoint CV
```

## Setup and Run

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (Highly recommended for Phase 2, Google Colab T4 is sufficient)
- ~10 GB disk space for embeddings

### Installation
```bash
git clone https://github.com/HimasagarU/Protein-Remote-Homology-Detection-by-Sequence-Embeddings-Alignment.git
cd Protein-Remote-Homology-Detection-by-Sequence-Embeddings-Alignment

# Install PyTorch with CUDA 11.8 (or your specific CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core libraries
pip install transformers biopython pandas scikit-learn numpy matplotlib numba
```

### Run Order

Prepare the dataset once:

```bash
python src/01_data_prep.py
```

Run the `v1` baseline:

```bash
python src/02_embed.py --version v1
python src/05_benchmark.py --version v1
python src/07_visualize_alignment.py --version v1
python src/08_visualize_tree.py --version v1
```

Run the `v3` final model:

```bash
python src/02_embed.py --version v3
python src/06_improve.py
python src/07_visualize_alignment.py --version v3
python src/08_visualize_tree.py --version v3
```

Optional quick sanity checks:

```bash
python src/03_similarity.py --version v1
python src/03_similarity.py --version v3
python src/04_alignment.py --version v1 --pairs 3
python src/04_alignment.py --version v3 --pairs 3
```

---

## References
1. **Lin et al. (2023)** – Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science, 379*(6637), 1123-1130. *(ESM-2 paper)*
2. **Murzin et al. (1995)** – SCOP: a structural classification of proteins database. *J. Mol. Biol., 247*(4), 536-540.
3. **Altschul et al. (1997)** – Gapped BLAST and PSI-BLAST. *Nucleic Acids Research, 25*(17), 3389-3402.
4. **Smith & Waterman (1981)** – Identification of common molecular subsequences. *J. Mol. Biol., 147*(1), 195-197.
