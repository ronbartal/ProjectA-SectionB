"""Add BM25 artifacts to an existing dense-only index dir (no re-embed).

Chunking parameters are read from the target dir's index_meta.json (not from
CLI flags) so the BM25 rows are guaranteed to be built with the same config as
the dense index. After chunking, the regenerated (page_id, chunk_id) sequence
is checked against index_meta.json and the script aborts before writing
anything if the ordering does not align row-for-row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk import chunk_corpus
from lexical import build_bm25_artifacts
from utils import iter_entries

# Defined locally to avoid importing index.py (which pulls in the embedding
# stack); must stay in sync with index.INDEX_META_NAME.
INDEX_META_NAME = "index_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build BM25 into an existing artifacts dir. Chunking params are "
            "taken from the dir's index_meta.json to guarantee row alignment "
            "with the dense index."
        )
    )
    parser.add_argument("artifacts_dir", type=Path)
    args = parser.parse_args()

    meta_path = args.artifacts_dir / INDEX_META_NAME
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found: BM25 can only be staged into a dir that "
            "already holds a dense index (build it with scripts/build_index.py)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_words = int(meta["chunk_words"])
    chunk_overlap = int(meta["chunk_overlap"])
    prefix_title = bool(meta["prefix_title"])

    # Stream records straight into chunking; keeping the full records list
    # alive alongside chunks would double the corpus text in memory.
    chunks = chunk_corpus(
        iter_entries(),
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
        prefix_title=prefix_title,
    )

    # Row i of the BM25 CSR matrix must be the same chunk as dense vector
    # row i (E4 fusion scores BM25 by FAISS row index). Comparing the full
    # (page_id, chunk_id) sequences catches both parameter and corpus drift.
    expected_pages = [int(p) for p in meta["page_ids"]]
    expected_chunks = [int(c) for c in meta["chunk_ids"]]
    got_pages = [c.page_id for c in chunks]
    got_chunks = [c.chunk_id for c in chunks]
    if got_pages != expected_pages or got_chunks != expected_chunks:
        raise SystemExit(
            f"Chunk ordering mismatch vs {meta_path}: regenerated "
            f"{len(chunks)} chunks, dense index has {len(expected_pages)}. "
            "Has the corpus changed since the dense index was built? "
            "Aborting without writing BM25 artifacts."
        )

    build_bm25_artifacts(
        chunks,
        args.artifacts_dir,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
        prefix_title=prefix_title,
    )
    print(f"BM25 built: {len(chunks)} chunks -> {args.artifacts_dir}")


if __name__ == "__main__":
    main()
