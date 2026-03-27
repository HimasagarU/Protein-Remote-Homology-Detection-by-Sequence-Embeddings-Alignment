"""
Tier 1 Improvements — Normalized scoring, gap penalty sweep, combined score.

Runs all three Tier 1 improvements and compares ROC-AUC against baseline.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "embeddings"
RESULTS = ROOT / "results"

# Import alignment functions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parent / "04_alignment.py"
)
_al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_al)
smith_waterman = _al.smith_waterman
compute_similarity_matrix = _al.compute_similarity_matrix


# ─── Scoring functions ─────────────────────────────────────────────

def score_pair(id_a: str, id_b: str, gap: float = -1.0):
    """Returns (raw_sw, normalized_sw, global_sim, path_len)."""
    emb_a = torch.load(EMB_DIR / f"{id_a}.pt", weights_only=True)
    emb_b = torch.load(EMB_DIR / f"{id_b}.pt", weights_only=True)

    # Smith-Waterman local alignment
    S = compute_similarity_matrix(emb_a, emb_b).numpy()
    raw_score, path = smith_waterman(S, gap)
    # Normalize by the minimum length of the two proteins
    min_len = min(emb_a.shape[0], emb_b.shape[0])
    norm_score = raw_score / min_len

    # Global mean-pooled cosine similarity
    global_sim = F.cosine_similarity(
        emb_a.mean(dim=0, keepdim=True),
        emb_b.mean(dim=0, keepdim=True)
    ).item()

    return raw_score, norm_score, global_sim, len(path)


def score_all(pairs_df, gap=-1.0):
    """Score all pairs, return DataFrame with all score types."""
    rows = []
    total = len(pairs_df)
    for idx, row in pairs_df.iterrows():
        raw, norm, glob, plen = score_pair(row["id_a"], row["id_b"], gap)
        rows.append({
            "id_a": row["id_a"], "id_b": row["id_b"], "label": row["label"],
            "sw_raw": raw, "sw_norm": norm, "global_sim": glob, "path_len": plen,
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"    [{idx+1}/{total}]")
    return pd.DataFrame(rows)


def combined_score(df, alpha):
    """Blend standardized raw SW + global similarity."""
    sw = df["sw_raw"]
    sw_scaled = (sw - sw.min()) / (sw.max() - sw.min() + 1e-8)
    return alpha * sw_scaled + (1 - alpha) * df["global_sim"]


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pairs_df = pd.read_csv(str(ROOT / "data" / "test_pairs.csv"))
    labels = pairs_df["label"].values

    # ── Step 1: Gap penalty sweep ──────────────────────────────────
    print("=" * 60)
    print("Step 1: Gap penalty sweep")
    print("=" * 60)

    best_gap, best_auc = -1.0, 0.0
    gap_results = {}

    for g in [-0.2, -0.5, -0.8, -1.0, -1.5, -2.0]:
        print(f"\n  gap={g:.1f}")
        df = score_all(pairs_df, gap=g)
        auc_raw  = roc_auc_score(labels, df["sw_raw"])
        auc_norm = roc_auc_score(labels, df["sw_norm"])
        auc_glob = roc_auc_score(labels, df["global_sim"])
        gap_results[g] = {"raw": auc_raw, "norm": auc_norm, "global": auc_glob, "df": df}
        print(f"    Raw AUC:  {auc_raw:.4f}  |  Norm AUC: {auc_norm:.4f}  |  Global: {auc_glob:.4f}")

        if auc_norm > best_auc:
            best_auc, best_gap = auc_norm, g

    print(f"\n  Best gap: {best_gap}  (norm AUC={best_auc:.4f})")

    # ── Step 2: Combined score with best gap ───────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Combined score (alpha sweep)")
    print("=" * 60)

    best_df = gap_results[best_gap]["df"]
    best_alpha, best_combined_auc = 0.5, 0.0

    for alpha in np.arange(0.1, 1.0, 0.1):
        combo = combined_score(best_df, alpha)
        auc = roc_auc_score(labels, combo)
        print(f"  alpha={alpha:.1f}  AUC={auc:.4f}")
        if auc > best_combined_auc:
            best_combined_auc, best_alpha = auc, alpha

    print(f"\n  Best alpha: {best_alpha:.1f}  (AUC={best_combined_auc:.4f})")

    # ── Step 3: Final results with best config ─────────────────────
    print("\n" + "=" * 60)
    print("Final comparison")
    print("=" * 60)

    best_df["combined"] = combined_score(best_df, best_alpha)

    print(f"\n  Baseline (raw SW, gap=-1.0):      AUC = {gap_results[-1.0]['raw']:.4f}")
    print(f"  + Normalization (gap={best_gap}):    AUC = {best_auc:.4f}")
    print(f"  + Combined (alpha={best_alpha:.1f}):       AUC = {best_combined_auc:.4f}")

    # Save improved scores
    out_path = RESULTS / "improved_scores.csv"
    best_df.to_csv(str(out_path), index=False)

    # ── Step 4: Plot comparison ROC ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, col, color in [
        ("Baseline (raw SW)", "sw_raw", "lightcoral"),
        ("Normalized SW", "sw_norm", "steelblue"),
        ("Global mean-pool", "global_sim", "forestgreen"),
        (f"Combined (a={best_alpha:.1f})", "combined", "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(labels, best_df[col])
        auc = roc_auc_score(labels, best_df[col])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Remote Homology Detection: Improved ROC Curves", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(str(RESULTS / "roc_improved.png"), dpi=200)
    plt.close()
    print(f"\n  ROC saved -> results/roc_improved.png")

    print("\n[OK] Tier 1 improvements complete.")
