"""
Phase 6 -- v3 feature-based model on top of multi-layer embeddings.

This script is the supported final pipeline:
- embeddings averaged over the last 4 ESM-2 layers
- richer geometric and alignment-derived features
- protein-disjoint cross-validation
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pipeline_common import DATA_DIR, get_embedding_dir, get_results_dir


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alignment)
smith_waterman = _alignment.smith_waterman
compute_similarity_matrix = _alignment.compute_similarity_matrix


FEATURE_COLS = [
    "sw_raw",
    "sw_norm_geom",
    "sw_norm_min",
    "global_sim",
    "coverage",
    "mean_path_sim",
    "max_path_sim",
    "diag_ratio",
    "top_k_mean",
    "len_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the v3 remote-homology model.")
    parser.add_argument("--gap-open", type=float, default=-0.5)
    parser.add_argument("--gap-extend", type=float, default=-0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def extract_features(id_a: str, id_b: str, emb_dir: Path, gap_open: float, gap_extend: float) -> dict:
    emb_a = torch.load(emb_dir / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(emb_dir / f"{id_b}.pt", weights_only=True)

    len_a, len_b = emb_a.shape[0], emb_b.shape[0]
    min_len = max(min(len_a, len_b), 1)
    geom_len = max(float(np.sqrt(len_a * len_b)), 1.0)

    S = compute_similarity_matrix(emb_a, emb_b).numpy()
    raw_score, path = smith_waterman(S, gap_open=gap_open, gap_extend=gap_extend)

    global_sim = F.cosine_similarity(
        emb_a.mean(dim=0, keepdim=True),
        emb_b.mean(dim=0, keepdim=True),
    ).item()

    path_len = len(path)
    coverage = path_len / min_len
    if path_len > 0:
        path_sims = np.array([S[i, j] for i, j in path], dtype=np.float32)
        mean_path_sim = float(path_sims.mean())
        max_path_sim = float(path_sims.max())
    else:
        mean_path_sim = 0.0
        max_path_sim = 0.0

    positive_S = np.clip(S, a_min=0.0, a_max=None)
    diag_width = max(1, min(5, min(S.shape) // 4))
    diag_energy = 0.0
    for offset in range(-diag_width, diag_width + 1):
        diag_energy += float(np.trace(positive_S, offset=offset))
    total_energy = float(positive_S.sum())
    diag_ratio = diag_energy / total_energy if total_energy > 0 else 0.0

    flat_S = S.ravel()
    top_k = min(50, flat_S.size)
    if top_k == 0:
        top_k_mean = 0.0
    else:
        top_slice = np.partition(flat_S, flat_S.size - top_k)[-top_k:]
        top_k_mean = float(top_slice.mean())

    return {
        "sw_raw": raw_score,
        "sw_norm_geom": raw_score / geom_len,
        "sw_norm_min": raw_score / min_len,
        "global_sim": global_sim,
        "path_len": path_len,
        "coverage": coverage,
        "mean_path_sim": mean_path_sim,
        "max_path_sim": max_path_sim,
        "diag_ratio": diag_ratio,
        "top_k_mean": top_k_mean,
        "len_ratio": min_len / max(len_a, len_b),
    }


def build_feature_table(
    pairs_df: pd.DataFrame,
    emb_dir: Path,
    gap_open: float,
    gap_extend: float,
) -> pd.DataFrame:
    rows = []
    total = len(pairs_df)

    for idx, row in pairs_df.iterrows():
        feats = extract_features(
            row["id_a"],
            row["id_b"],
            emb_dir=emb_dir,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        feats["id_a"] = row["id_a"]
        feats["id_b"] = row["id_b"]
        feats["label"] = row["label"]
        rows.append(feats)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] feature rows ready")

    return pd.DataFrame(rows)


def build_protein_disjoint_folds(
    pairs_df: pd.DataFrame,
    n_splits: int,
    random_state: int,
):
    proteins = sorted(set(pairs_df["id_a"]) | set(pairs_df["id_b"]))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (_, val_protein_idx) in enumerate(splitter.split(proteins), start=1):
        val_proteins = {proteins[i] for i in val_protein_idx}
        val_mask = pairs_df["id_a"].isin(val_proteins) & pairs_df["id_b"].isin(val_proteins)
        train_mask = ~pairs_df["id_a"].isin(val_proteins) & ~pairs_df["id_b"].isin(val_proteins)

        train_idx = pairs_df.index[train_mask].to_numpy()
        val_idx = pairs_df.index[val_mask].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(f"Fold {fold_idx} is empty. Increase pair coverage before running v3.")
        if pairs_df.loc[train_idx, "label"].nunique() < 2 or pairs_df.loc[val_idx, "label"].nunique() < 2:
            raise RuntimeError(f"Fold {fold_idx} does not contain both classes.")

        folds.append((fold_idx, train_idx, val_idx))
    return folds


def compute_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    metrics = {
        "ROC-AUC": roc_auc_score(labels, scores),
        "PR-AUC": average_precision_score(labels, scores),
    }

    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    positives = max(int(ranked_labels.sum()), 1)
    for k in (10, 25, 50, 100):
        top_k = ranked_labels[: min(k, ranked_labels.size)]
        metrics[f"Precision@{k}"] = float(top_k.mean())
        metrics[f"Recall@{k}"] = float(top_k.sum() / positives)
    return metrics


if __name__ == "__main__":
    args = parse_args()
    emb_dir = get_embedding_dir("v3", create=False)
    results_dir = get_results_dir("v3", create=True)
    pairs_df = pd.read_csv(DATA_DIR / "test_pairs.csv")

    print("=" * 60)
    print("Phase 6 -- v3 Feature Model")
    print("=" * 60)
    print(f"\nUsing embeddings from {emb_dir}")
    print(f"Scoring {len(pairs_df)} pairs with multi-layer features...\n")

    features_df = build_feature_table(
        pairs_df,
        emb_dir=emb_dir,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
    )

    print("\nIndividual feature AUCs:")
    for col in FEATURE_COLS:
        auc = roc_auc_score(features_df["label"], features_df[col])
        print(f"  {col:16s}: {auc:.4f}")

    folds = build_protein_disjoint_folds(
        features_df[["id_a", "id_b", "label"]],
        n_splits=args.folds,
        random_state=args.random_state,
    )

    X = features_df[FEATURE_COLS].values
    y = features_df["label"].values
    cv_scores = np.full(len(features_df), np.nan, dtype=np.float32)
    fold_metrics = []

    print(f"\nProtein-disjoint cross-validation ({args.folds} folds):")
    for fold_idx, train_idx, val_idx in folds:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        cv_scores[val_idx] = val_probs
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_metrics.append(fold_auc)
        print(
            f"  Fold {fold_idx}: AUC={fold_auc:.4f} "
            f"(train pairs={len(train_idx)}, val pairs={len(val_idx)})"
        )

    eval_mask = ~np.isnan(cv_scores)
    eval_labels = y[eval_mask]
    eval_scores = cv_scores[eval_mask]
    summary_metrics = compute_ranking_metrics(eval_labels, eval_scores)
    summary_metrics["Evaluated pairs"] = int(eval_mask.sum())
    summary_metrics["Total pairs"] = int(len(features_df))

    print("\nProtein-disjoint CV metrics:")
    for key, value in summary_metrics.items():
        if isinstance(value, int):
            print(f"  {key:16s}: {value}")
        else:
            print(f"  {key:16s}: {value:.4f}")

    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf_full.fit(X_full, y)

    features_df["combined_prob_cv"] = cv_scores
    features_df["combined_prob_full"] = clf_full.predict_proba(X_full)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(eval_labels, eval_scores)
    auc = roc_auc_score(eval_labels, eval_scores)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"v3 protein-disjoint CV (AUC={auc:.3f})")
    for col, color, label in [
        ("global_sim", "forestgreen", "Global cosine"),
        ("sw_norm_geom", "steelblue", "SW / geom"),
        ("max_path_sim", "purple", "Max path sim"),
    ]:
        fpr_col, tpr_col, _ = roc_curve(features_df["label"], features_df[col])
        auc_col = roc_auc_score(features_df["label"], features_df[col])
        ax.plot(fpr_col, tpr_col, color=color, lw=1.5, alpha=0.8, label=f"{label} ({auc_col:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(eval_labels, eval_scores)
    ap = average_precision_score(eval_labels, eval_scores)
    ax.plot(recall, precision, color="forestgreen", lw=2, label=f"PR curve (AP={ap:.3f})")
    ax.axhline(y=eval_labels.mean(), color="gray", linestyle="--", label="Class balance")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    pos_scores = eval_scores[eval_labels == 1]
    neg_scores = eval_scores[eval_labels == 0]
    ax.hist(neg_scores, bins=20, alpha=0.6, color="tomato", label="Non-homolog", density=True)
    ax.hist(pos_scores, bins=20, alpha=0.6, color="steelblue", label="Remote homolog", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Validation Score Distribution")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    importances = np.abs(clf_full.coef_[0])
    order = np.argsort(importances)
    ax.barh([FEATURE_COLS[i] for i in order], importances[order], color="teal")
    ax.set_xlabel("|Logistic coefficient|")
    ax.set_title("Feature Importance")

    plt.tight_layout()
    dashboard_path = results_dir / "v3_dashboard.png"
    plt.savefig(dashboard_path, dpi=200)
    plt.close()

    scores_path = results_dir / "improved_v3_scores.csv"
    features_df.to_csv(scores_path, index=False)

    metrics_path = results_dir / "metrics_summary.json"
    metrics_out = {k: (round(v, 4) if not isinstance(v, int) else v) for k, v in summary_metrics.items()}
    metrics_out["fold_aucs"] = [round(x, 4) for x in fold_metrics]
    metrics_out["embedding_type"] = "multi-layer (last 4)"
    metrics_out["feature_names"] = FEATURE_COLS
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, indent=2)

    print(f"\nSaved feature table -> {scores_path}")
    print(f"Saved metrics      -> {metrics_path}")
    print(f"Saved dashboard    -> {dashboard_path}")
