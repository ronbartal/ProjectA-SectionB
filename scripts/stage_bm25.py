"""Add BM25 artifacts to an existing dense-only index dir (no re-embed)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk import chunk_corpus
from lexical import build_bm25_artifacts
from utils import iter_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 into an existing artifacts dir.")
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--chunk-words", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    parser.add_argument("--prefix-title", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # Stream records straight into chunking; keeping the full records list
    # alive alongside chunks would double the corpus text in memory.
    chunks = chunk_corpus(
        iter_entries(),
        chunk_words=args.chunk_words,
        chunk_overlap=args.chunk_overlap,
        prefix_title=args.prefix_title,
    )
    build_bm25_artifacts(
        chunks,
        args.artifacts_dir,
        chunk_words=args.chunk_words,
        chunk_overlap=args.chunk_overlap,
        prefix_title=args.prefix_title,
    )
    print(f"BM25 built: {len(chunks)} chunks -> {args.artifacts_dir}")


if __name__ == "__main__":
    main()
