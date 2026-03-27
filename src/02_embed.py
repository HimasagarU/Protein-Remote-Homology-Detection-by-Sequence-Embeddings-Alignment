"""
Phase 2 -- ESM-2 Embedding Extraction.

Loads the pre-trained ESM-2 model and extracts per-residue embeddings
for every cleaned protein sequence.  Each embedding is saved as a
.pt file of shape [L, D] where L = sequence length, D = embedding dim.
"""

import os
import torch
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

# ─── Config ────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
FASTA_PATH = ROOT / "data" / "astral_20_clean.fasta"
EMB_DIR    = ROOT / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"   # 650M params, dim=1280
MAX_LEN    = 1022                               # 1024 - 2 special tokens
BATCH_LOG  = 50                                 # print progress every N

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load model ────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("Model ready.\n")


@torch.no_grad()
def extract_embedding(sequence: str) -> torch.Tensor:
    """
    Returns per-residue embedding: Tensor [L, D].
    Truncates to MAX_LEN if needed.  Strips CLS/EOS tokens.
    """
    seq    = sequence[:MAX_LEN].upper()
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    hidden = model(**inputs).last_hidden_state          # [1, L+2, D]
    return hidden[0, 1:-1, :].cpu()                     # [L, D]


# ─── Main loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    records = list(SeqIO.parse(str(FASTA_PATH), "fasta"))
    total   = len(records)
    done    = 0

    print(f"Extracting embeddings for {total} sequences ...\n")

    for i, record in enumerate(records):
        out_path = EMB_DIR / f"{record.id}.pt"

        # Skip if already extracted (resume-safe)
        if out_path.exists():
            done += 1
            continue

        emb = extract_embedding(str(record.seq))
        torch.save(emb, str(out_path))
        done += 1

        if (i + 1) % BATCH_LOG == 0 or (i + 1) == total:
            print(f"  [{done}/{total}]  {record.id}  shape={tuple(emb.shape)}")

    print(f"\nDone. {done}/{total} embeddings in {EMB_DIR}")
