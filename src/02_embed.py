"""
Phase 2 -- ESM-2 embedding extraction for the supported pipelines.

`v1`: save the final hidden layer to `embeddings/`
`v3`: average the last 4 hidden layers and save to `embeddings_v3/`
"""

import argparse
import os
from pathlib import Path

import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

from pipeline_common import DATA_DIR, get_embedding_dir, normalize_version


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FASTA_PATH = DATA_DIR / "astral_20_clean.fasta"
MAX_LEN = 1022  # ESM-2 max tokens minus CLS/EOS
BATCH_LOG = 100
V3_LAYER_COUNT = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("Model ready.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for v1 or v3.")
    parser.add_argument(
        "--version",
        choices=["v1", "v3"],
        default="v3",
        help="Embedding variant to extract.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if the output file already exists.",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_embedding(sequence: str, version: str) -> torch.Tensor:
    """
    Return a per-residue embedding tensor with shape [L, 1280].
    """
    seq = sequence[:MAX_LEN].upper()
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    if version == "v1":
        residue_states = hidden_states[-1]
    else:
        residue_states = torch.stack(hidden_states[-V3_LAYER_COUNT:], dim=0).mean(dim=0)

    return residue_states[0, 1:-1, :].cpu()


if __name__ == "__main__":
    args = parse_args()
    version = normalize_version(args.version)
    emb_dir = get_embedding_dir(version, create=True)
    records = list(SeqIO.parse(str(FASTA_PATH), "fasta"))
    total = len(records)
    done = 0
    skipped = 0

    print(f"Extracting {version} embeddings for {total} sequences -> {Path(emb_dir).name}\n")

    for i, record in enumerate(records, start=1):
        out_path = emb_dir / f"{record.id}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            done += 1
            continue

        emb = extract_embedding(str(record.seq), version=version)
        torch.save(emb, str(out_path))
        done += 1

        if i % BATCH_LOG == 0 or i == total:
            print(
                f"  [{done}/{total}] {record.id} shape={tuple(emb.shape)} "
                f"(skipped {skipped} existing)"
            )

    print(f"\nDone. Saved {done - skipped if not args.overwrite else done} embeddings to {emb_dir}")
