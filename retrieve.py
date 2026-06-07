"""Query-time retrieval (timed portion includes query embedding).

Pipeline: embed queries -> FAISS search top chunks -> aggregate chunk scores to
their page_id -> (optionally) fuse with a BM25 lexical ranking -> return the
top-`top_k` distinct pages per query.

Dense aggregation (utils.AGG_SCOPE / utils.PAGE_POOL_K):
  - "window": a page is scored from only the chunks inside the retrieved window.
  - "page":   two-stage rerank -- the window selects candidate pages, then each
              candidate is rescored over ALL of its chunks. Score is the MEAN of
              the page's top-`PAGE_POOL_K` chunk cosines (0 -> all chunks).

E4 lexical fusion (utils.FUSION): when "rrf", a BM25 page ranking is fused with
the dense ranking via Reciprocal Rank Fusion. BM25 page score = `BM25_PAGE_AGG`
over the page's chunk BM25 scores, computed over the chunks in the dense window
(BM25_SCOPE="window") or all page chunks ("page").

PRF (utils.PRF): when enabled (page scope), a first dense pass expands the query
via Rocchio page-level pseudo-relevance feedback, then a second pass ranks with
the expanded query. See `_prf_expand_query`.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from embed import embed_queries
from index import load_index
from lexical import Bm25Index, bm25_score_row, load_bm25, tokenize
from utils import (
    AGG_SCOPE,
    BM25_PAGE_AGG,
    BM25_SCOPE,
    FUSION,
    K_EVAL,
    PAGE_POOL_K,
    PRF,
    PRF_ALPHA,
    PRF_PAGE_REPR,
    PRF_TOPN,
    RRF_K,
    TOP_CHUNKS,
)


def _build_page_to_chunks(page_ids: List[int]) -> Dict[int, np.ndarray]:
    """Map each page_id to the array of chunk row-indices that belong to it."""
    tmp: Dict[int, List[int]] = defaultdict(list)
    for ci, pid in enumerate(page_ids):
        tmp[int(pid)].append(ci)
    return {pid: np.asarray(idxs, dtype=np.int64) for pid, idxs in tmp.items()}


def _topk_mean(scores: np.ndarray, pool_k: int) -> float:
    """Mean of the top-`pool_k` scores (all of them when pool_k <= 0)."""
    if pool_k and scores.shape[0] > pool_k:
        scores = np.partition(scores, -pool_k)[-pool_k:]
    return float(scores.mean())


def _agg_scores(values: Sequence[float], agg: str) -> float:
    arr = np.asarray(values) if len(values) else np.zeros(1)
    if agg == "sum":
        return float(arr.sum())
    if agg == "mean":
        return float(arr.mean())
    return float(arr.max())


def _ranks(scores: Dict[int, float]) -> Dict[int, int]:
    """page_id -> 1-based rank (best first)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {pid: r for r, (pid, _) in enumerate(ordered, start=1)}


def _rrf_fuse(
    dense_scores: Dict[int, float],
    bm25_scores: Dict[int, float],
    rrf_k: int,
) -> List[int]:
    """Reciprocal Rank Fusion of the dense and BM25 page rankings (best first)."""
    rd = _ranks(dense_scores)
    rb = _ranks(bm25_scores)
    fused = {
        pid: 1.0 / (rrf_k + rd[pid]) + 1.0 / (rrf_k + rb.get(pid, 10 ** 9))
        for pid in dense_scores
    }
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered]


def _collect_candidates(
    chunk_indices: np.ndarray,
    page_ids: List[int],
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Distinct candidate pages (first-seen order) + their in-window chunk rows."""
    seen: set[int] = set()
    candidates: List[int] = []
    windowed: Dict[int, List[int]] = defaultdict(list)
    for idx in chunk_indices:
        i = int(idx)
        if i < 0:  # FAISS pads with -1 when fewer than requested are found.
            continue
        pid = page_ids[i]
        if pid not in seen:
            seen.add(pid)
            candidates.append(pid)
        windowed[pid].append(i)
    return candidates, windowed


def _rank_pages_window(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: List[int],
    top_k: int,
    pool_k: int = PAGE_POOL_K,
) -> List[int]:
    """Window scope: score each page by the mean of its top-`pool_k` in-window
    chunks. pool_k <= 0 averages every in-window chunk of the page.
    """
    totals: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for idx, score in zip(chunk_indices, chunk_scores):
        if idx < 0:
            continue
        pid = page_ids[int(idx)]
        c = counts.get(pid, 0)
        if pool_k <= 0 or c < pool_k:
            totals[pid] = totals.get(pid, 0.0) + float(score)
            counts[pid] = c + 1
    scored = {pid: totals[pid] / counts[pid] for pid in totals}
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered[:top_k]]


def _page_dense_scores(
    query_vec: np.ndarray,
    candidates: Sequence[int],
    page_to_chunks: Dict[int, np.ndarray],
    corpus_vectors: np.ndarray,
    pool_k: int,
) -> Dict[int, float]:
    """Dense page score = mean of each candidate's top-`pool_k` chunk cosines
    over ALL of its chunks (pool_k <= 0 -> all chunks)."""
    return {
        pid: _topk_mean(corpus_vectors[page_to_chunks[pid]] @ query_vec, pool_k)
        for pid in candidates
    }


def _page_bm25_scores(
    query_terms: Sequence[str],
    candidates: Sequence[int],
    windowed: Dict[int, List[int]],
    page_to_chunks: Dict[int, np.ndarray],
    bm25: Bm25Index,
    *,
    agg: str = BM25_PAGE_AGG,
    scope: str = BM25_SCOPE,
) -> Dict[int, float]:
    """BM25 page score = `agg` over the page's chunk BM25 scores. `scope`
    "window" only scores the page's in-window chunks (fast); "page" scores all
    of its chunks. Each chunk's BM25 is computed once and cached per query.
    """
    cache: Dict[int, float] = {}

    def bm25_of(chunk: int) -> float:
        v = cache.get(chunk)
        if v is None:
            v = bm25_score_row(
                bm25.data, bm25.indices, bm25.indptr, chunk,
                query_terms, bm25.idf, bm25.vocab, bm25.avg_dl, bm25.k1, bm25.b,
            )
            cache[chunk] = v
        return v

    scores: Dict[int, float] = {}
    for pid in candidates:
        rows = windowed[pid] if scope == "window" else page_to_chunks[pid].tolist()
        scores[pid] = _agg_scores([bm25_of(int(c)) for c in rows], agg)
    return scores


def _prf_expand_query(
    query_vec: np.ndarray,
    chunk_indices: np.ndarray,
    page_ids: List[int],
    page_to_chunks: Dict[int, np.ndarray],
    corpus_vectors: np.ndarray,
    *,
    alpha: float = PRF_ALPHA,
    top_n: int = PRF_TOPN,
    page_repr: str = PRF_PAGE_REPR,
) -> np.ndarray:
    """Rocchio dense query expansion from page-level pseudo-relevance feedback.

    The first-pass `chunk_indices` select candidate pages; the top-`top_n` by
    page-scope mean cosine are treated as pseudo-relevant. Each is represented
    by its mean chunk vector ("mean") or single best-matching chunk ("best");
    their centroid expands the query: q' = normalize(alpha*q + (1-alpha)*c).
    Page-level feedback de-duplicates so one page can't dominate the centroid.
    """
    candidates, _ = _collect_candidates(chunk_indices, page_ids)
    if not candidates:
        return query_vec
    dense = _page_dense_scores(query_vec, candidates, page_to_chunks, corpus_vectors, 0)
    top_pages = sorted(dense, key=dense.get, reverse=True)[:top_n]
    reps = []
    for pid in top_pages:
        chunks = page_to_chunks[pid]
        if page_repr == "best":
            chunks = chunks[[int(np.argmax(corpus_vectors[chunks] @ query_vec))]]
        reps.append(corpus_vectors[chunks].mean(axis=0))
    if not reps:
        return query_vec
    centroid = np.mean(reps, axis=0)
    expanded = alpha * query_vec + (1.0 - alpha) * centroid
    norm = np.linalg.norm(expanded)
    return expanded / norm if norm > 0 else query_vec


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    top_chunks: int = TOP_CHUNKS,
    artifacts_dir: Optional[Path] = None,
    scope: str = AGG_SCOPE,
    pool_k: int = PAGE_POOL_K,
    fusion: str = FUSION,
    prf: bool = PRF,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Dense retrieval uses a FAISS IndexFlatIP over chunk embeddings (numpy
    fallback otherwise). Chunk hits are aggregated to pages per `scope`. When
    `fusion == "rrf"` (and page scope), a BM25 page ranking is fused with the
    dense ranking via Reciprocal Rank Fusion -- exact-term matching that the
    embedding misses, which lifts in-candidate relevant pages into the top 10.

    When `prf` (page scope), a first pass expands each query via Rocchio
    pseudo-relevance feedback (see `_prf_expand_query`) and a second pass ranks
    with the expanded query; BM25 keeps the original query terms.
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

    use_fusion = scope == "page" and fusion == "rrf"
    use_prf = scope == "page" and prf
    page_to_chunks = (
        _build_page_to_chunks(page_ids) if scope == "page" else {}
    )
    bm25 = load_bm25(artifacts_dir) if use_fusion else None

    def _search(qv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if index is not None:
            return index.search(qv, k)
        sim = qv @ corpus_vectors.T
        idx = np.argsort(-sim, axis=1)[:, :k]
        return np.take_along_axis(sim, idx, axis=1), idx

    scores, indices = _search(query_vectors)

    if use_prf:
        expanded = np.stack([
            _prf_expand_query(
                query_vectors[i], indices[i], page_ids, page_to_chunks,
                corpus_vectors,
            )
            for i in range(query_vectors.shape[0])
        ]).astype(np.float32)
        query_vectors = np.ascontiguousarray(expanded)
        scores, indices = _search(query_vectors)

    ranked: List[List[int]] = []
    for row_idx in range(query_vectors.shape[0]):
        if scope != "page":
            ranked.append(
                _rank_pages_window(
                    indices[row_idx], scores[row_idx], page_ids, top_k, pool_k
                )
            )
            continue

        candidates, windowed = _collect_candidates(indices[row_idx], page_ids)
        if not candidates:
            ranked.append([])
            continue
        dense_scores = _page_dense_scores(
            query_vectors[row_idx], candidates, page_to_chunks,
            corpus_vectors, pool_k,
        )
        if use_fusion:
            bm25_scores = _page_bm25_scores(
                tokenize(queries[row_idx]), candidates, windowed,
                page_to_chunks, bm25,
            )
            ranked.append(_rrf_fuse(dense_scores, bm25_scores, RRF_K)[:top_k])
        else:
            ordered = sorted(
                dense_scores.items(), key=lambda kv: kv[1], reverse=True
            )
            ranked.append([pid for pid, _ in ordered[:top_k]])
    return ranked
