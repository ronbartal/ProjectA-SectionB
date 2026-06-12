"""Add chunk_texts.npy to an existing dense index dir (no re-embed) — E6.

Chunking parameters are read from the target dir's index_meta.json (not from
CLI flags) so the regenerated passages are guaranteed to use the same config
as the dense index. The regenerated (page_id, chunk_id) sequence is checked
against index_meta.json and the script aborts before writing anything if the
ordering does not align row-for-row — chunk_texts[i] must be the exact string
that produced index_vectors row i.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import numpy as np

from chunk import chunk_corpus
from utils import iter_entries

# Defined locally to avoid importing index.py (which pulls in the embedding
# stack); must stay in sync with index.INDEX_META_NAME / index.CHUNK_TEXTS_NAME.
INDEX_META_NAME = "index_meta.json"
CHUNK_TEXTS_NAME = "chunk_texts.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Persist chunk passage texts into an existing artifacts dir. "
            "Chunking params are taken from the dir's index_meta.json to "
            "guarantee row alignment with the dense index."
        )
    )
    parser.add_argument("artifacts_dir", type=Path)
    args = parser.parse_args()

    meta_path = args.artifacts_dir / INDEX_META_NAME
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found: chunk texts can only be staged into a dir "
            "that already holds a dense index (build with scripts/build_index.py)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_words = int(meta["chunk_words"])
    chunk_overlap = int(meta["chunk_overlap"])
    prefix_title = bool(meta["prefix_title"])

    chunks = chunk_corpus(
        iter_entries(),
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
        prefix_title=prefix_title,
    )

    # Row i of chunk_texts must be the same chunk as dense vector row i
    # (the reranker fetches passages by FAISS row index). Comparing the full
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
            "Aborting without writing chunk_texts.npy."
        )

    texts = np.asarray([c.text for c in chunks], dtype=object)
    out_path = args.artifacts_dir / CHUNK_TEXTS_NAME
    np.save(out_path, texts)
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"chunk_texts staged: {len(texts)} passages -> {out_path} "
        f"({size_mb:.0f} MB)"
    )


if __name__ == "__main__":
    main()
