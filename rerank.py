"""Cross-encoder reranking of the fused page shortlist (E6).

Option A — rerank only, after retrieval: the fused (dense + BM25 RRF) ranking
selects the top-`RERANK_POOL` candidate pages; a cross-encoder then scores
(query, passage) for each page, where the passage is the page's best in-window
chunk text (`chunk_texts.npy`). The final order blends both signals:

    final = RERANK_ALPHA * ce_minmax + (1 - RERANK_ALPHA) * fused_rank_norm

Why a blend and not the raw CE order: in the 2026-06-12 A/B (real text,
29 queries) pure CE order gained +0.009 k-fold but was split-half UNSTABLE
(one half -0.053); the alpha=0.3 blend gained MORE (+0.0115) with both halves
at-or-above baseline. Light CE influence mirrors the PRF lesson (light
expansion wins). Pages below the shortlist keep their fused order.

Course rule: additional pretrained models are allowed for RERANKING ONLY —
the cross-encoder lives in this module and never touches indexing or
first-stage retrieval.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from utils import RERANK_ALPHA, RERANK_MODEL_NAME, RERANK_POOL

_CE_CACHE: dict = {}


def _get_cross_encoder(model_name: str):
    """Lazy singleton — the CE model loads once per process, first use only."""
    ce = _CE_CACHE.get(model_name)
    if ce is None:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(model_name)
        _CE_CACHE[model_name] = ce
    return ce


def rerank_pages(
    query: str,
    fused_pages: Sequence[int],
    windowed: Dict[int, List[int]],
    chunk_sim: Dict[int, float],
    chunk_texts: np.ndarray,
    *,
    pool: int = RERANK_POOL,
    alpha: float = RERANK_ALPHA,
    model_name: str = RERANK_MODEL_NAME,
) -> List[int]:
    """Reorder the top-`pool` fused pages by the CE/fused blend (best first).

    `windowed` maps page_id -> in-window chunk rows; `chunk_sim` maps chunk
    row -> dense cosine (used to pick the page's best passage). Pages with no
    usable passage, and everything beyond the pool, keep their fused order.
    """
    shortlist = list(fused_pages[:pool])
    pids: List[int] = []
    pairs: List[tuple] = []
    for pid in shortlist:
        rows = windowed.get(pid)
        if not rows:
            continue
        best_row = max(rows, key=lambda c: chunk_sim.get(int(c), float("-inf")))
        passage = str(chunk_texts[int(best_row)])
        if passage.strip():
            pids.append(pid)
            pairs.append((query, passage))
    if not pairs:
        return list(fused_pages)

    ce = _get_cross_encoder(model_name)
    scores = np.asarray(
        ce.predict(pairs, batch_size=64, show_progress_bar=False),
        dtype=np.float64,
    )

    lo, hi = float(scores.min()), float(scores.max())
    ce_norm = {
        pid: ((s - lo) / (hi - lo) if hi > lo else 0.5)
        for pid, s in zip(pids, scores)
    }
    in_ce = [p for p in fused_pages if p in ce_norm]
    fused_norm = {
        p: 1.0 - r / max(len(in_ce) - 1, 1) for r, p in enumerate(in_ce)
    }
    blended = sorted(
        pids,
        key=lambda p: -(alpha * ce_norm[p] + (1.0 - alpha) * fused_norm[p]),
    )
    tail = [p for p in fused_pages if p not in ce_norm]
    return blended + tail
