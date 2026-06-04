"""Query-time retrieval (timed portion includes query embedding).

Pipeline: embed queries -> FAISS search top chunks -> max-pool chunk scores to
their page_id -> return the top-`top_k` distinct pages per query.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL, TOP_CHUNKS


def _rank_pages_from_chunks(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: List[int],
    top_k: int,
) -> List[int]:
    """Max-pool chunk scores per page and return the top_k distinct page_ids."""
    best: dict[int, float] = {}
    for idx, score in zip(chunk_indices, chunk_scores):
        if idx < 0:  # FAISS pads with -1 when fewer than requested are found.
            continue
        pid = page_ids[int(idx)]
        prev = best.get(pid)
        if prev is None or score > prev:
            best[pid] = float(score)
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered[:top_k]]


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    top_chunks: int = TOP_CHUNKS,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Uses a FAISS IndexFlatIP over chunk embeddings when available, falling back
    to a brute-force numpy dot product otherwise. Chunk hits are aggregated to
    pages via max-pooling, which rewards a single strongly-matching passage.
    """
    corpus_vectors, page_ids, index = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    n_chunks = len(page_ids)
    k = min(top_chunks, n_chunks) if n_chunks else 0
    if k == 0:
        return [[] for _ in queries]

    if index is not None:
        scores, indices = index.search(query_vectors, k)
    else:
        sim = query_vectors @ corpus_vectors.T
        indices = np.argsort(-sim, axis=1)[:, :k]
        scores = np.take_along_axis(sim, indices, axis=1)

    ranked: List[List[int]] = []
    for row_idx in range(query_vectors.shape[0]):
        ranked.append(
            _rank_pages_from_chunks(
                indices[row_idx], scores[row_idx], page_ids, top_k
            )
        )
    return ranked
