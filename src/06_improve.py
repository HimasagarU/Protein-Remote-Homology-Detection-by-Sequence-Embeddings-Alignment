"""
Tier 3 Improvements — Multi-layer embeddings, richer features, proper cross-validation.

Uses embeddings_v2/ (multi-layer averaged) if available, falls back to embeddings/.
Extracts 6 features per pair and uses 5-fold stratified CV for the LR combiner.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Prefer multi-layer embeddings if available
EMB_DIR_V2 = ROOT / "embeddings_v2"
EMB_DIR_V1 = ROOT / "embeddings"

def get_emb_dir():
    """Check which embedding directory to use."""
    # Check if v2 has enough files
    if EMB_DIR_V2.exists():
        v2_count = len(list(EMB_DIR_V2.glob("*.pt")))
        if v2_count > 100:
            print(f"  Using multi-layer embeddings ({v2_count} files in embeddings_v2/)")
            return EMB_DIR_V2
    print(f"  Using single-layer embeddings (embeddings/)")
    return EMB_DIR_V1

# Import alignment
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_al)
smith_waterman_numba = _al._smith_waterman_numba
compute_similarity_matrix = _al.compute_similarity_matrix


def extract_features(id_a, id_b, emb_dir, gap_open=-0.5, gap_extend=-0.1):
    """
    Extract a rich feature vector for a protein pair.
    Returns dict with 6+ features.
    """
    emb_a = torch.load(emb_dir / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(emb_dir / f"{id_b}.pt", weights_only=True)
    
    len_a, len_b = emb_a.shape[0], emb_b.shape[0]
    
    # 1. Cosine similarity matrix
    S = compute_similarity_matrix(emb_a, emb_b).numpy()
    
    # 2. Smith-Waterman alignment
    raw_score, path = smith_waterman_numba(S, gap_open=gap_open, gap_extend=gap_extend)
    
    # 3. Global mean-pooled cosine similarity
    global_sim = F.cosine_similarity(
        emb_a.mean(dim=0, keepdim=True),
        emb_b.mean(dim=0, keepdim=True)
    ).item()
    
    # 4. Path-based features
    path_len = len(path)
    min_len = min(len_a, len_b)
    geom_len = np.sqrt(len_a * len_b)
    
    # Coverage: what fraction of the shorter protein is aligned
    coverage = path_len / min_len if min_len > 0 else 0
    
    if path_len > 0:
        # Mean similarity along the alignment path
        path_sims = [S[i, j] for i, j in path if 0 <= i < S.shape[0] and 0 <= j < S.shape[1]]
        mean_path_sim = np.mean(path_sims) if path_sims else 0
        max_path_sim = np.max(path_sims) if path_sims else 0
    else:
        mean_path_sim = 0
        max_path_sim = 0
    
    # 5. Matrix statistics (diagonal band energy)
    # How much energy is concentrated near the diagonal vs scattered
    diag_width = min(5, min(S.shape) // 4)
    diag_energy = 0
    total_energy = S.sum()
    for k in range(-diag_width, diag_width + 1):
        diag_energy += np.trace(S, offset=k)
    diag_ratio = diag_energy / (total_energy + 1e-8)
    
    # 6. Top-K max similarities in the matrix
    flat_S = S.flatten()
    top_k = min(50, len(flat_S))
    top_k_mean = np.sort(flat_S)[-top_k:].mean()
    
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


FEATURE_COLS = [
    "sw_raw", "sw_norm_geom", "sw_norm_min", "global_sim",
    "coverage", "mean_path_sim", "max_path_sim",
    "diag_ratio", "top_k_mean", "len_ratio"
]


if __name__ == "__main__":
    print("=" * 60)
    print("Tier 3: Multi-Layer + Rich Features + 5-Fold CV")
    print("=" * 60)
    
    emb_dir = get_emb_dir()
    pairs_df = pd.read_csv(str(ROOT / "data" / "test_pairs.csv"))
    
    # ── Step 1: Extract rich features ──────────────────────────────
    print(f"\n[1/5] Extracting {len(FEATURE_COLS)} features for {len(pairs_df)} pairs...")
    
    rows = []
    total = len(pairs_df)
    for idx, row in pairs_df.iterrows():
        try:
            feats = extract_features(row["id_a"], row["id_b"], emb_dir)
            feats["id_a"] = row["id_a"]
            feats["id_b"] = row["id_b"]
            feats["label"] = row["label"]
            rows.append(feats)
        except FileNotFoundError:
            continue
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"    [{idx+1}/{total}]")
    
    results_df = pd.DataFrame(rows)
    print(f"  Completed: {len(results_df)} pairs scored")
    
    X = results_df[FEATURE_COLS].values
    y = results_df["label"].values
    
    # ── Step 2: Individual feature AUCs ────────────────────────────
    print("\n[2/5] Individual Feature AUCs:")
    for col in FEATURE_COLS:
        auc = roc_auc_score(y, results_df[col])
        print(f"    {col:20s}: {auc:.4f}")
    
    # ── Step 3: 5-Fold Stratified CV ───────────────────────────────
    print("\n[3/5] 5-Fold Stratified Cross-Validation...")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
        clf.fit(X_train_s, y_train)
        
        val_probs = clf.predict_proba(X_val_s)[:, 1]
        oof_preds[val_idx] = val_probs
        
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"    Fold {fold+1}: AUC = {fold_auc:.4f}")
    
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\n    CV Mean AUC:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"    OOF AUC:      {cv_auc:.4f}")
    
    results_df["combined_prob_cv"] = oof_preds
    
    # Also train on full data for feature importance
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
    clf_full.fit(X_full_s, y)
    results_df["combined_prob_full"] = clf_full.predict_proba(X_full_s)[:, 1]
    
    # ── Step 4: Comprehensive metrics ──────────────────────────────
    print("\n[4/5] Comprehensive Metrics (on OOF predictions):")
    
    metrics = {
        "ROC-AUC (CV)": cv_auc,
        "PR-AUC (MAP)": average_precision_score(y, oof_preds),
    }
    
    # Precision/Recall@K
    for k in [10, 25, 50, 100]:
        sorted_idx = np.argsort(oof_preds)[::-1][:k]
        prec_k = y[sorted_idx].mean()
        rec_k = y[sorted_idx].sum() / y.sum()
        metrics[f"Precision@{k}"] = prec_k
        metrics[f"Recall@{k}"] = rec_k
    
    for k, v in metrics.items():
        print(f"    {k:20s}: {v:.4f}")
    
    # ── Step 5: Visualizations ─────────────────────────────────────
    print("\n[5/5] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 5a. ROC Curve comparison (all versions)
    ax = axes[0, 0]
    # Current V3
    fpr, tpr, _ = roc_curve(y, oof_preds)
    ax.plot(fpr, tpr, 'darkorange', lw=2, label=f'V3 LR-CV (AUC={cv_auc:.3f})')
    # Individual features
    for col, color, name in [
        ("global_sim", "forestgreen", "Global Cosim"),
        ("sw_norm_geom", "steelblue", "SW Norm"),
        ("mean_path_sim", "purple", "Path Similarity"),
    ]:
        fpr_f, tpr_f, _ = roc_curve(y, results_df[col])
        auc_f = roc_auc_score(y, results_df[col])
        ax.plot(fpr_f, tpr_f, color=color, lw=1.5, alpha=0.7, label=f'{name} ({auc_f:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves — Feature Comparison')
    ax.legend(fontsize=8, loc='lower right')
    
    # 5b. Precision-Recall curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y, oof_preds)
    ap = average_precision_score(y, oof_preds)
    ax.plot(rec, prec, 'forestgreen', lw=2, label=f'PR (AP={ap:.3f})')
    ax.axhline(y=y.mean(), color='gray', linestyle='--', label=f'Baseline ({y.mean():.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(fontsize=9)
    
    # 5c. Score distribution violin-style
    ax = axes[1, 0]
    pos_scores = oof_preds[y == 1]
    neg_scores = oof_preds[y == 0]
    ax.hist(neg_scores, bins=30, alpha=0.6, color='tomato', label='Non-homolog', density=True)
    ax.hist(pos_scores, bins=30, alpha=0.6, color='steelblue', label='Remote homolog', density=True)
    ax.set_xlabel('Combined Score (OOF)'); ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(fontsize=9)
    
    # 5d. Feature importance
    ax = axes[1, 1]
    importances = np.abs(clf_full.coef_[0])
    sorted_idx = np.argsort(importances)
    ax.barh([FEATURE_COLS[i] for i in sorted_idx], importances[sorted_idx], color='teal')
    ax.set_xlabel('|LR Coefficient|')
    ax.set_title('Feature Importance (Logistic Regression)')
    
    plt.tight_layout()
    plt.savefig(str(RESULTS / "v3_dashboard.png"), dpi=200)
    plt.close()
    print(f"  Dashboard saved -> results/v3_dashboard.png")
    
    # Save all results
    results_df.to_csv(str(RESULTS / "improved_v3_scores.csv"), index=False)
    
    # Save metrics summary
    import json
    metrics_out = {k: round(v, 4) for k, v in metrics.items()}
    metrics_out["cv_fold_aucs"] = [round(x, 4) for x in fold_aucs]
    metrics_out["embedding_type"] = "multi-layer (last 4)" if emb_dir == EMB_DIR_V2 else "single-layer (last)"
    metrics_out["n_features"] = len(FEATURE_COLS)
    metrics_out["feature_names"] = FEATURE_COLS
    
    with open(str(RESULTS / "metrics_summary.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved -> results/metrics_summary.json")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: ROC-AUC (5-fold CV) = {cv_auc:.4f}")
    print(f"{'='*60}")
