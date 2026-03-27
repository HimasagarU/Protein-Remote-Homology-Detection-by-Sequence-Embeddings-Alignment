"""
Phase 5 -- Benchmarking and Evaluation.

Runs Smith-Waterman alignment on all test pairs, computes ROC-AUC,
precision/recall/F1, and generates ROC curve plot.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, average_precision_score
)

import importlib.util

# Load 04_alignment module (module name starts with digit, can't use normal import)
_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alignment)
align_proteins = _alignment.align_proteins

ROOT        = Path(__file__).resolve().parent.parent
PAIRS_CSV   = ROOT / "data" / "test_pairs.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def score_all_pairs(pairs_df: pd.DataFrame, gap_penalty: float = -1.0):
    """Run SW alignment on every pair. Returns DataFrame with scores."""
    results = []
    total = len(pairs_df)

    for idx, row in pairs_df.iterrows():
        score, path = align_proteins(row["id_a"], row["id_b"],
                                     mode="local", gap_penalty=gap_penalty)
        results.append({
            "id_a":       row["id_a"],
            "id_b":       row["id_b"],
            "label":      row["label"],
            "sw_score":   score,
            "path_len":   len(path),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] scored")

    return pd.DataFrame(results)


def compute_metrics(df: pd.DataFrame, score_col: str = "sw_score"):
    """Compute ROC-AUC, PR-AUC, and top-k metrics."""
    y_true  = df["label"].values
    y_score = df[score_col].values

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)

    # Threshold at median score for binary metrics
    threshold = np.median(y_score)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "ROC-AUC":   roc_auc,
        "PR-AUC":    pr_auc,
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "F1":         f1_score(y_true, y_pred, zero_division=0),
    }


def plot_roc(df: pd.DataFrame, score_col: str = "sw_score", out_path: Path = None):
    """Generate and save ROC curve."""
    y_true  = df["label"].values
    y_score = df[score_col].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ESM-2 + SW (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Remote Homology Detection: ROC Curve", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path or RESULTS_DIR / "roc_curve.png"), dpi=200)
    plt.close()
    print(f"  ROC curve saved -> {out_path or RESULTS_DIR / 'roc_curve.png'}")


def plot_score_distribution(df: pd.DataFrame, out_path: Path = None):
    """Plot score distributions for homolog vs non-homolog pairs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = df[df["label"] == 1]["sw_score"]
    neg = df[df["label"] == 0]["sw_score"]
    ax.hist(neg, bins=30, alpha=0.6, color="tomato", label="Non-homolog")
    ax.hist(pos, bins=30, alpha=0.6, color="steelblue", label="Remote homolog")
    ax.set_xlabel("Smith-Waterman Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution: Homolog vs Non-Homolog", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path or RESULTS_DIR / "score_distribution.png"), dpi=200)
    plt.close()
    print(f"  Score distribution saved -> {out_path or RESULTS_DIR / 'score_distribution.png'}")


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5 -- Benchmarking")
    print("=" * 60)

    pairs_df = pd.read_csv(str(PAIRS_CSV))
    print(f"\nScoring {len(pairs_df)} pairs with Smith-Waterman (gap=-1.0)...\n")

    t0 = time.time()
    results_df = score_all_pairs(pairs_df, gap_penalty=-1.0)
    elapsed = time.time() - t0

    # Save raw scores
    scores_path = RESULTS_DIR / "embedding_scores.csv"
    results_df.to_csv(str(scores_path), index=False)
    print(f"\n  Scores saved -> {scores_path.name}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(pairs_df):.2f}s/pair)\n")

    # Metrics
    metrics = compute_metrics(results_df)
    print("  Metrics:")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    # Score stats by class
    print("\n  Score stats:")
    print(results_df.groupby("label")["sw_score"].describe().to_string())

    # Plots
    print()
    plot_roc(results_df)
    plot_score_distribution(results_df)

    print("\n[OK] Phase 5 complete.")
