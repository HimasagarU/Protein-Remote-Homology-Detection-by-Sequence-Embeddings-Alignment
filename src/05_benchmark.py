"""
Phase 5 -- Baseline benchmark for v1 or v3 alignment scores.

This script evaluates Smith-Waterman alignment scores without a downstream classifier.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from pipeline_common import DATA_DIR, get_results_dir, normalize_version


matplotlib.use("Agg")
import matplotlib.pyplot as plt


_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alignment)
align_proteins = _alignment.align_proteins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark raw Smith-Waterman alignment scores.")
    parser.add_argument("--version", choices=["v1", "v3"], default="v1")
    parser.add_argument("--gap-open", type=float, default=-1.0)
    parser.add_argument("--gap-extend", type=float, default=-0.1)
    return parser.parse_args()


def score_all_pairs(
    pairs_df: pd.DataFrame,
    version: str,
    gap_open: float,
    gap_extend: float,
) -> pd.DataFrame:
    results = []
    total = len(pairs_df)

    for idx, row in pairs_df.iterrows():
        score, path = align_proteins(
            row["id_a"],
            row["id_b"],
            version=version,
            mode="local",
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        results.append(
            {
                "id_a": row["id_a"],
                "id_b": row["id_b"],
                "label": row["label"],
                "sw_score": score,
                "matched_residues": len(path),
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] scored")

    return pd.DataFrame(results)


def compute_ranking_metrics(df: pd.DataFrame, score_col: str = "sw_score") -> dict:
    y_true = df["label"].values
    y_score = df[score_col].values
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
    }

    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    positives = max(int(ranked["label"].sum()), 1)
    for k in (10, 25, 50, 100):
        top_k = ranked.head(min(k, len(ranked)))
        metrics[f"Precision@{k}"] = top_k["label"].mean()
        metrics[f"Recall@{k}"] = top_k["label"].sum() / positives
    return metrics


def plot_roc(df: pd.DataFrame, out_path: Path, score_col: str = "sw_score") -> None:
    y_true = df["label"].values
    y_score = df[score_col].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"SW score (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Remote Homology Detection ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_score_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = df[df["label"] == 1]["sw_score"]
    neg = df[df["label"] == 0]["sw_score"]
    ax.hist(neg, bins=30, alpha=0.6, color="tomato", label="Non-homolog")
    ax.hist(pos, bins=30, alpha=0.6, color="steelblue", label="Remote homolog")
    ax.set_xlabel("Smith-Waterman score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    results_dir = get_results_dir(version, create=True)
    pairs_df = pd.read_csv(DATA_DIR / "test_pairs.csv")

    print("=" * 60)
    print(f"Phase 5 -- Baseline Benchmark ({version})")
    print("=" * 60)
    print(f"\nScoring {len(pairs_df)} pairs with Smith-Waterman...\n")

    t0 = time.time()
    results_df = score_all_pairs(
        pairs_df,
        version=version,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
    )
    elapsed = time.time() - t0

    scores_path = results_dir / "embedding_scores.csv"
    results_df.to_csv(scores_path, index=False)
    print(f"\nScores saved -> {scores_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed / len(pairs_df):.2f}s/pair)\n")

    metrics = compute_ranking_metrics(results_df)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:12s}: {value:.4f}")

    metrics_path = results_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({key: round(value, 4) for key, value in metrics.items()}, handle, indent=2)

    print("\nScore stats:")
    print(results_df.groupby("label")["sw_score"].describe().to_string())

    plot_roc(results_df, results_dir / "roc_curve.png")
    plot_score_distribution(results_df, results_dir / "score_distribution.png")
    print(f"\nPlots saved in {results_dir}")
    print(f"Metrics saved -> {metrics_path}")
