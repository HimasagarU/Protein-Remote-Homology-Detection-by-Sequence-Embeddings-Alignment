"""
Phase 2b -- Multi-Layer ESM-2 Embedding Extraction.

Extracts per-residue embeddings by averaging the last 4 hidden layers
of ESM-2, which captures a richer mix of structural information than
using only the final layer.

Output: embeddings_v2/<domain_id>.pt  shape [L, 1280]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

# ─── Config ────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
FASTA_PATH = ROOT / "data" / "astral_20_clean.fasta"
EMB_DIR    = ROOT / "embeddings_v2"
EMB_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"   # 650M params, 33 layers, dim=1280
MAX_LEN    = 1022                               # 1024 - 2 special tokens
BATCH_LOG  = 100
N_LAYERS_AVG = 4                                # Average last N layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load model ────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"Model ready. Will average last {N_LAYERS_AVG} layers.\n")


@torch.no_grad()
def extract_multilayer_embedding(sequence: str) -> torch.Tensor:
    """
    Returns per-residue embedding by averaging last N hidden layers.
    Output: Tensor [L, D] where D=1280
    """
    seq    = sequence[:MAX_LEN].upper()
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    
    outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of (n_layers+1) tensors, each [1, L+2, D]
    # Layer 0 = embedding layer, layers 1..33 = transformer layers
    hidden_states = outputs.hidden_states
    
    # Average the last N_LAYERS_AVG layers
    last_n = torch.stack(hidden_states[-N_LAYERS_AVG:], dim=0)  # [N, 1, L+2, D]
    averaged = last_n.mean(dim=0)                                # [1, L+2, D]
    
    # Strip CLS and EOS tokens
    return averaged[0, 1:-1, :].cpu()                            # [L, D]


# ─── Main loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    records = list(SeqIO.parse(str(FASTA_PATH), "fasta"))
    total   = len(records)
    done    = 0
    skipped = 0

    print(f"Extracting multi-layer embeddings for {total} sequences ...\n")

    for i, record in enumerate(records):
        out_path = EMB_DIR / f"{record.id}.pt"

        # Skip if already extracted (resume-safe)
        if out_path.exists():
            skipped += 1
            done += 1
            continue

        emb = extract_multilayer_embedding(str(record.seq))
        torch.save(emb, str(out_path))
        done += 1

        if (i + 1) % BATCH_LOG == 0 or (i + 1) == total:
            print(f"  [{done}/{total}]  {record.id}  shape={tuple(emb.shape)}  (skipped {skipped} existing)")

    print(f"\nDone. {done}/{total} embeddings saved to {EMB_DIR}")
