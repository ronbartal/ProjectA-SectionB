"""Audit MiniLM token lengths of chunks (truncation check + E1 preview).

Tokenization only (no embedding / no GPU), so this is cheap to run locally.
Use --chunk-words / --chunk-overlap to preview how a candidate E1 chunk size
would tokenize -- without rebuilding the index -- to see how much truncation
(beyond the model's 256-token cap) a config would cause.

    python scripts/audit_tokens.py                 # current utils defaults
    python scripts/audit_tokens.py --chunk-words 120 --chunk-overlap 30
    python scripts/audit_tokens.py --chunk-words 150 --chunk-overlap 33
    python scripts/audit_tokens.py --entries-dir data/sample_corpus --max-pages 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from chunk import chunk_entry
from utils import CHUNK_OVERLAP, CHUNK_WORDS, EMBEDDING_MODEL_NAME, iter_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--batch", type=int, default=2000, help="Tokenizer batch size"
    )
    parser.add_argument(
        "--entries-dir",
        type=Path,
        default=None,
        help="Corpus directory (default: data/Wikipedia Entries)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Cap pages processed (useful for local sample runs)",
    )
    args = parser.parse_args()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    max_len = model.max_seq_length
    tok = model.tokenizer
    print(f"model={EMBEDDING_MODEL_NAME}  max_seq_length={max_len}")
    print(f"chunk_words={args.chunk_words}  chunk_overlap={args.chunk_overlap}")

    records = list(iter_entries(args.entries_dir))
    if args.max_pages is not None:
        records = records[: args.max_pages]
    texts = []
    for rec in records:
        for c in chunk_entry(
            rec,
            chunk_words=args.chunk_words,
            chunk_overlap=args.chunk_overlap,
        ):
            texts.append(c.text)
    print(f"pages={len(records)}  chunks={len(texts)}")

    lengths = []
    for i in range(0, len(texts), args.batch):
        enc = tok(
            texts[i : i + args.batch],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])

    lengths = np.array(lengths)
    n = len(lengths)
    over = int((lengths > max_len).sum())

    print("\n=== token length distribution (incl. special tokens) ===")
    print(
        f"min={lengths.min()}  median={int(np.median(lengths))}  "
        f"mean={lengths.mean():.1f}"
    )
    print(
        f"p90={int(np.percentile(lengths, 90))}  "
        f"p95={int(np.percentile(lengths, 95))}  "
        f"p99={int(np.percentile(lengths, 99))}  max={lengths.max()}"
    )
    print(f"\nchunks_exceeding_{max_len}_tokens={over}  ({100 * over / n:.1f}%)")
    print("-> these have their tail SILENTLY TRUNCATED at encode time")

    if over:
        lost = lengths[lengths > max_len] - max_len
        print(
            f"truncated_tokens_lost: median={int(np.median(lost))}  "
            f"mean={lost.mean():.1f}  max={int(lost.max())}"
        )


if __name__ == "__main__":
    main()
