"""
Phase 1 — Data Preparation for Protein Remote Homology Detection.

Parses ASTRAL-20 FASTA, extracts SCOPe hierarchy labels,
cleans sequences, and generates balanced test pairs.
"""

import random
from pathlib import Path
from itertools import combinations

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# ─── Paths ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_FASTA   = ROOT / "data" / "astral_20.fa"
CLEAN_FASTA = ROOT / "data" / "astral_20_clean.fasta"
LABELS_CSV  = ROOT / "data" / "scop_labels.csv"
PAIRS_CSV   = ROOT / "data" / "test_pairs.csv"

# ─── Config ────────────────────────────────────────────────────────
MIN_SEQ_LEN   = 30    # drop very short fragments
MAX_SEQ_LEN   = 1022  # ESM-2 limit (1024 − 2 special tokens)
PAIR_SAMPLE_N = 500   # number of sequences to sample for pair generation
SEED          = 42

random.seed(SEED)


# ─── 1. Parse SCOPe headers ───────────────────────────────────────

def parse_scop_header(description: str) -> dict | None:
    """
    SCOPe ASTRAL header format:
      >d2bkma_ a.1.1.1 (A:) automated matches {Geobacillus ...}

    Returns dict with id, scop_code, class, fold, superfamily, family
    or None if the header can't be parsed.
    """
    parts = description.strip().split()
    if len(parts) < 2:
        return None

    domain_id = parts[0]
    scop_code = parts[1]
    levels = scop_code.split(".")

    if len(levels) != 4:
        return None

    return {
        "id":          domain_id,
        "scop_code":   scop_code,
        "class":       levels[0],
        "fold":        f"{levels[0]}.{levels[1]}",
        "superfamily": f"{levels[0]}.{levels[1]}.{levels[2]}",
        "family":      scop_code,
    }


# ─── 2. Clean sequences and build label table ─────────────────────

def clean_and_label(raw_path: Path, clean_path: Path, labels_path: Path):
    """Read raw FASTA, filter bad sequences, write clean FASTA + labels CSV."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    records_out, labels = [], []
    skipped = 0

    for record in SeqIO.parse(str(raw_path), "fasta"):
        info = parse_scop_header(record.description)
        if info is None:
            skipped += 1
            continue

        seq = str(record.seq).upper()

        # Filter: length and valid amino acids
        if len(seq) < MIN_SEQ_LEN or len(seq) > MAX_SEQ_LEN:
            skipped += 1
            continue
        if not set(seq).issubset(valid_aa):
            skipped += 1
            continue

        # Build clean record
        clean_rec = SeqRecord(record.seq, id=info["id"], description=info["scop_code"])
        records_out.append(clean_rec)

        info["length"] = len(seq)
        labels.append(info)

    SeqIO.write(records_out, str(clean_path), "fasta")
    df = pd.DataFrame(labels)
    df.to_csv(str(labels_path), index=False)

    print(f"[OK] Saved {len(records_out)} clean sequences -> {clean_path.name}")
    print(f"  Skipped {skipped} (bad header / short / long / non-standard AA)")
    print(f"  Unique superfamilies: {df['superfamily'].nunique()}")
    print(f"  Unique families:      {df['family'].nunique()}")
    return df


# ─── 3. Generate balanced test pairs ──────────────────────────────

def generate_test_pairs(labels_df: pd.DataFrame, pairs_path: Path, n: int = PAIR_SAMPLE_N):
    """
    From a random sample of n sequences, enumerate all pairs and label them:
      1 = remote homolog (same superfamily, different family)
      0 = non-homolog (different superfamily)

    Outputs a balanced CSV with equal positive and negative counts.
    """
    df = labels_df.set_index("id")

    # Sample a subset to keep pair count manageable
    ids = df.index.tolist()
    if len(ids) > n:
        ids = sorted(random.sample(ids, n))
    else:
        ids = sorted(ids)

    positive, negative = [], []

    for id_a, id_b in combinations(ids, 2):
        a, b = df.loc[id_a], df.loc[id_b]
        same_sf = a["superfamily"] == b["superfamily"]
        diff_fam = a["family"] != b["family"]

        if same_sf and diff_fam:
            positive.append((id_a, id_b, 1))
        elif not same_sf:
            negative.append((id_a, id_b, 0))
        # Skip same-family pairs (trivial homologs)

    # Balance the two classes
    k = min(len(positive), len(negative))
    if k == 0:
        print("[WARN] No positive pairs found -- try a larger sample or check labels.")
        return

    random.shuffle(negative)
    pairs = positive[:k] + negative[:k]
    random.shuffle(pairs)

    pairs_df = pd.DataFrame(pairs, columns=["id_a", "id_b", "label"])
    pairs_df.to_csv(str(pairs_path), index=False)

    print(f"\n[OK] Test pairs -> {pairs_path.name}")
    print(f"  Positive (remote homologs): {k}")
    print(f"  Negative (non-homologs):    {k}")
    print(f"  Total:                      {2 * k}")


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 — Data Preparation")
    print("=" * 60)

    labels_df = clean_and_label(RAW_FASTA, CLEAN_FASTA, LABELS_CSV)
    generate_test_pairs(labels_df, PAIRS_CSV)

    print("\n[OK] Phase 1 complete.")
