"""Offline index build and load (not timed at grading).

Persists three artifacts:
  - index_vectors.npy : float32 (n_chunks x 384) L2-normalized chunk embeddings.
  - index_meta.json   : per-chunk page_id / chunk_id maps + build parameters.
  - index.faiss       : a FAISS IndexFlatIP over the chunk vectors (exact cosine).

retrieve.py searches the FAISS index and aggregates chunk hits back to pages.
The numpy vectors are kept as a fallback in case FAISS is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:  # FAISS is part of the allowed dependency set but optional locally.
    import faiss
except Exception:  # pragma: no cover - exercised only when faiss is missing.
    faiss = None  # type: ignore[assignment]

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import (
    ARTIFACTS_DIR,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    EMBEDDING_MODEL_NAME,
    ensure_artifacts_dir,
    iter_entries,
)

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
INDEX_FAISS_NAME = "index.faiss"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Chunk the corpus, embed every chunk, and persist vectors + meta + FAISS.

    Returns (vectors, page_ids) where row i of `vectors` corresponds to the
    page in page_ids[i] (chunks, not unique pages).
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    # Diagnostics: pages with no body are still indexed (by title), and pages
    # with neither title nor body produce an empty chunk worth flagging.
    title_only = 0
    fully_empty = 0
    for rec in records:
        has_title = bool(str(rec.get("title", "")).strip())
        has_content = bool(str(rec.get("content", "")).strip())
        if not has_content and has_title:
            title_only += 1
        elif not has_content and not has_title:
            fully_empty += 1

    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": EMBEDDING_MODEL_NAME,
        "num_vectors": len(page_ids),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if faiss is not None and vectors.shape[0] > 0:
        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)  # cosine sim since vectors are normalized
        index.add(vectors)
        faiss.write_index(index, str(out_dir / INDEX_FAISS_NAME))

    print(
        f"build_index: {len(records)} pages -> {len(chunks)} chunks "
        f"({title_only} title-only, {fully_empty} fully-empty)"
    )

    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int], Optional["faiss.Index"]]:
    """
    Load chunk vectors, the chunk->page_id map, and the FAISS index.

    The FAISS index is returned when both the library and the artifact are
    present; otherwise it is None and callers fall back to the numpy vectors.
    """
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]

    index = None
    faiss_path = root / INDEX_FAISS_NAME
    if faiss is not None:
        if not faiss_path.exists():
            # FAISS is available but the artifact is missing: the build is
            # incomplete, so fail loud rather than silently degrading.
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. Re-run "
                "scripts/build_index.py to rebuild artifacts."
            )
        index = faiss.read_index(str(faiss_path))
    # If faiss could not be imported, index stays None and retrieve.py falls
    # back to the numpy brute-force path using the loaded vectors.

    return vectors, page_ids, index
