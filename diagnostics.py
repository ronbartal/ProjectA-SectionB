"""Portable diagnostic harness for Section B retrieval experiments.

Scores the pipeline against public queries with set-aware page-level metrics,
chunk-level diagnostics, k-fold cross-validation, and data-quality checks.
Depends only on artifacts/ and data/public_queries.json (no raw corpus).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from embed import embed_queries
from eval import K_EVAL, load_query_file, ndcg_at_k
from index import load_chunk_texts, load_index
from lexical import Bm25Index, bm25_score_row, load_bm25, tokenize
from utils import (
    AGG_SCOPE,
    ARTIFACTS_DIR,
    BM25_PAGE_AGG,
    BM25_SCOPE,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    FUSION,
    GRADING_QUERY_TIME_LIMIT_S,
    PAGE_POOL_K,
    PRF,
    PRF_ALPHA,
    PRF_PAGE_REPR,
    PRF_TOPN,
    PUBLIC_QUERIES_PATH,
    RERANK,
    RERANK_ALPHA,
    RERANK_MODEL_NAME,
    RERANK_POOL,
    RRF_K,
    TOP_CHUNKS,
)

K_FOLD_SEED = 42
RECALL_KS = (10, 50, 100)


@dataclass
class QueryDiagnostics:
    query_id: str
    query: str
    n_relevant: int
    ndcg_at_10: float
    mrr: float
    recall_at: Dict[str, float]
    relevant_in_top10: int
    page_ranks: Dict[str, int]
    chunk_best_ranks: Dict[str, int]
    chunk_in_top_k: Dict[str, bool]
    missing_relevant_ids: List[int]


@dataclass
class BucketSummary:
    n_relevant: int
    count: int
    mean_ndcg_at_10: float
    mean_mrr: float
    mean_recall_at_10: float


@dataclass
class DiagnosticReport:
    tag: str
    timestamp: str
    config: Dict[str, Any]
    summary: Dict[str, Any]
    bucket_summaries: List[BucketSummary]
    kfold: Dict[str, Any]
    data_quality: Dict[str, Any]
    sanity: Dict[str, Any]
    query_timing: Dict[str, Any]
    per_query: List[QueryDiagnostics]


def _max_pool_aggregate(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: Sequence[int],
) -> Tuple[List[int], Dict[int, float]]:
    """Max-pool chunk scores per page; return full ranked page list + scores."""
    best: Dict[int, float] = {}
    for idx, score in zip(chunk_indices, chunk_scores):
        if idx < 0:
            continue
        pid = int(page_ids[int(idx)])
        prev = best.get(pid)
        if prev is None or float(score) > prev:
            best[pid] = float(score)
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [pid for pid, _ in ordered]
    scores = {pid: sc for pid, sc in ordered}
    return ranked, scores


def aggregate_top_k_pages(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: Sequence[int],
    *,
    top_k: int = K_EVAL,
) -> List[int]:
    """Mirror retrieve max-pool aggregation (kept for back-compat)."""
    ranked, _ = _max_pool_aggregate(chunk_indices, chunk_scores, page_ids)
    return ranked[:top_k]


def _ranks(scores: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {pid: r for r, (pid, _) in enumerate(ordered, start=1)}


def _bm25_page_scores(
    query_terms: Sequence[str],
    candidates: Sequence[int],
    windowed: Dict[int, List[int]],
    page_to_chunks: Dict[int, np.ndarray],
    bm25: Bm25Index,
    agg: str,
    scope: str,
) -> Dict[int, float]:
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

    out: Dict[int, float] = {}
    for pid in candidates:
        rows = windowed[pid] if scope == "window" else page_to_chunks[pid].tolist()
        vals = np.asarray([bm25_of(int(c)) for c in rows]) if rows else np.zeros(1)
        if agg == "sum":
            out[pid] = float(vals.sum())
        elif agg == "mean":
            out[pid] = float(vals.mean())
        else:
            out[pid] = float(vals.max())
    return out


def aggregate_to_pages(
    sim_row: np.ndarray,
    chunk_order: np.ndarray,
    page_ids: Sequence[int],
    page_to_chunks: Dict[int, np.ndarray],
    *,
    scope: str = AGG_SCOPE,
    pool_k: int = PAGE_POOL_K,
    k_window: int = TOP_CHUNKS,
    fusion: str = "none",
    bm25: Optional[Bm25Index] = None,
    query_terms: Optional[Sequence[str]] = None,
    rrf_k: int = RRF_K,
    bm25_agg: str = BM25_PAGE_AGG,
    bm25_scope: str = BM25_SCOPE,
    rerank: bool = False,
    query: Optional[str] = None,
    chunk_texts: Optional[np.ndarray] = None,
) -> List[int]:
    """Aggregate chunk similarities to a full ranked page list (best first).

    Mirrors retrieve.search_batch exactly:
      - "window": score a page by the mean of its top-`pool_k` chunks inside the
        top-`k_window` window (pool_k <= 0 -> all in-window chunks).
      - "page":   candidate pages come from the window, then each is scored over
        ALL of its chunks via the full `sim_row` (pool_k <= 0 -> all chunks).
    When `fusion == "rrf"` (page scope), a BM25 page ranking is fused with the
    dense ranking via Reciprocal Rank Fusion. When `rerank` (page scope + rrf),
    the fused shortlist is reordered by the cross-encoder blend (rerank.py).

    `chunk_order` is the full corpus chunk order (best-first) for this query;
    `sim_row` is the query-vs-all-chunks similarity row.
    """
    window_idx = chunk_order[:k_window]
    if scope == "page":
        seen: set[int] = set()
        candidates: List[int] = []
        windowed: Dict[int, List[int]] = defaultdict(list)
        for idx in window_idx:
            i = int(idx)
            pid = int(page_ids[i])
            if pid not in seen:
                seen.add(pid)
                candidates.append(pid)
            windowed[pid].append(i)
        scored: Dict[int, float] = {}
        for pid in candidates:
            scores = sim_row[page_to_chunks[pid]]
            if pool_k > 0 and scores.shape[0] > pool_k:
                scores = np.partition(scores, -pool_k)[-pool_k:]
            scored[pid] = float(scores.mean())
        if fusion == "rrf" and bm25 is not None:
            bm25_scores = _bm25_page_scores(
                query_terms or [], candidates, windowed, page_to_chunks,
                bm25, bm25_agg, bm25_scope,
            )
            rd = _ranks(scored)
            rb = _ranks(bm25_scores)
            fused = {
                pid: 1.0 / (rrf_k + rd[pid]) + 1.0 / (rrf_k + rb.get(pid, 10 ** 9))
                for pid in scored
            }
            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            fused_list = [pid for pid, _ in ordered]
            if rerank and chunk_texts is not None and query is not None:
                from rerank import rerank_pages

                chunk_sim = {
                    int(c): float(sim_row[int(c)])
                    for rows_ in windowed.values()
                    for c in rows_
                }
                fused_list = rerank_pages(
                    query, fused_list, windowed, chunk_sim, chunk_texts,
                )
            return fused_list
    else:
        totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for idx in window_idx:
            pid = int(page_ids[int(idx)])
            c = counts.get(pid, 0)
            if pool_k <= 0 or c < pool_k:
                totals[pid] = totals.get(pid, 0.0) + float(sim_row[int(idx)])
                counts[pid] = c + 1
        scored = {pid: totals[pid] / counts[pid] for pid in totals}
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered]


def _chunk_order_and_scores(
    sim_row: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (chunk_indices, chunk_scores) sorted best-first for one query."""
    order = np.argsort(-sim_row)
    return order, sim_row[order]


def _recall_at_k(ranked_pages: Sequence[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_pages[:k])
    return len(top & relevant) / len(relevant)


def _mrr(ranked_pages: Sequence[int], relevant: Set[int]) -> float:
    for i, pid in enumerate(ranked_pages, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def _page_ranks(ranked_pages: Sequence[int], relevant: Set[int]) -> Dict[int, int]:
    rank_map: Dict[int, int] = {}
    seen: Set[int] = set()
    for i, pid in enumerate(ranked_pages, start=1):
        if pid in seen:
            continue
        seen.add(pid)
        if pid in relevant:
            rank_map[pid] = i
    for pid in relevant:
        rank_map.setdefault(pid, -1)
    return rank_map


def _chunk_best_ranks(
    chunk_order: np.ndarray,
    page_ids_arr: np.ndarray,
    relevant: Set[int],
) -> Dict[int, int]:
    """Best chunk rank (1-based) for each relevant page, -1 if none."""
    best: Dict[int, int] = {pid: -1 for pid in relevant}
    for rank, idx in enumerate(chunk_order, start=1):
        pid = int(page_ids_arr[int(idx)])
        if pid in relevant and best[pid] == -1:
            best[pid] = rank
    return best


def _chunk_in_top_k(
    chunk_order: np.ndarray,
    page_ids_arr: np.ndarray,
    relevant: Set[int],
    top_k: int,
) -> Dict[int, bool]:
    """Whether each relevant page has at least one chunk in the top_k window."""
    top_indices = chunk_order[:top_k]
    pages_in_window = {int(page_ids_arr[int(i)]) for i in top_indices}
    return {pid: pid in pages_in_window for pid in relevant}


def _make_kfold_assignments(n: int, k: int, seed: int = K_FOLD_SEED) -> List[int]:
    """Deterministic fold id per sample index (0 .. k-1)."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_ids = np.empty(n, dtype=int)
    for fold, idx in enumerate(indices):
        fold_ids[idx] = fold % k
    return fold_ids.tolist()


def _check_missing_relevant(
    all_relevant: Sequence[Set[int]],
    corpus_page_ids: Set[int],
) -> Tuple[List[int], int]:
    missing: Set[int] = set()
    for rel in all_relevant:
        for pid in rel:
            if pid not in corpus_page_ids:
                missing.add(pid)
    return sorted(missing), len(missing)


def measure_query_phase_time(
    queries: List[str],
    *,
    artifacts_dir: Optional[Path] = None,
    top_chunks: int = TOP_CHUNKS,
    scope: str = AGG_SCOPE,
    pool_k: int = PAGE_POOL_K,
    fusion: str = FUSION,
    prf: bool = PRF,
    rerank: bool = RERANK,
    time_limit_s: float = GRADING_QUERY_TIME_LIMIT_S,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Time the graded query path: embed queries + retrieve (same as main.run).

    Mirrors scripts/eval_public.py wall-clock measurement.
    """
    from retrieve import search_batch

    root = artifacts_dir or ARTIFACTS_DIR
    t0 = time.perf_counter()
    ranked = search_batch(
        queries, top_k=K_EVAL, top_chunks=top_chunks, artifacts_dir=root,
        scope=scope, pool_k=pool_k, fusion=fusion, prf=prf, rerank=rerank,
    )
    elapsed = time.perf_counter() - t0
    timing = {
        "query_phase_time_s": float(elapsed),
        "time_limit_s": float(time_limit_s),
        "within_budget": elapsed <= time_limit_s,
        "num_queries": len(queries),
        "num_ranked": len(ranked),
    }
    return ranked, timing


def run_diagnostics(
    *,
    artifacts_dir: Optional[Path] = None,
    queries_path: Optional[Path] = None,
    kfold: int = 5,
    top_chunks: int = TOP_CHUNKS,
    scope: str = AGG_SCOPE,
    pool_k: int = PAGE_POOL_K,
    fusion: str = FUSION,
    prf: bool = PRF,
    rerank: bool = RERANK,
    tag: str = "baseline",
    sanity_check: bool = True,
    time_run: bool = True,
) -> DiagnosticReport:
    """
    Run the full diagnostic harness and return a structured report.

    Uses numpy exact search over all chunks; aggregation mirrors
    retrieve.search_batch for the given `scope` / `pool_k` / `fusion` (defaults
    from utils).
    """
    if scope not in ("window", "page"):
        raise ValueError(f"Unsupported scope: {scope!r}")

    root = artifacts_dir or ARTIFACTS_DIR
    qpath = queries_path or PUBLIC_QUERIES_PATH
    rows = load_query_file(qpath)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    vectors, page_ids_list, _index = load_index(root)
    page_ids_arr = np.asarray(page_ids_list, dtype=np.int64)
    corpus_pages = set(int(x) for x in page_ids_list)
    page_to_chunks: Dict[int, np.ndarray] = {}
    if scope == "page":
        _tmp: Dict[int, List[int]] = defaultdict(list)
        for _ci, _pid in enumerate(page_ids_list):
            _tmp[int(_pid)].append(_ci)
        page_to_chunks = {p: np.asarray(v, dtype=np.int64) for p, v in _tmp.items()}

    use_fusion = scope == "page" and fusion == "rrf"
    use_rerank = use_fusion and rerank
    bm25 = load_bm25(root) if use_fusion else None
    chunk_texts = load_chunk_texts(root) if use_rerank else None

    missing_ids, n_missing = _check_missing_relevant(ground_truth, corpus_pages)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        sims = np.zeros((len(queries), vectors.shape[0]), dtype=np.float32)
    else:
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        sims = query_vectors @ vectors.T

    n_chunks = vectors.shape[0]
    k_window = min(top_chunks, n_chunks) if n_chunks else 0

    # PRF: expand each query from the first pass, then rank on the expanded sims.
    # Uses retrieve._prf_expand_query so the harness mirrors production exactly.
    use_prf = scope == "page" and prf and query_vectors.size and k_window
    if use_prf:
        from retrieve import _prf_expand_query

        expanded = np.empty_like(query_vectors)
        for i in range(len(queries)):
            if not queries[i].strip():
                expanded[i] = query_vectors[i]
                continue
            part = np.argpartition(-sims[i], k_window - 1)[:k_window]
            wo = part[np.argsort(-sims[i][part])]
            expanded[i] = _prf_expand_query(
                query_vectors[i], wo, page_ids_list, page_to_chunks, vectors
            )
        query_vectors = np.ascontiguousarray(expanded.astype(np.float32))
        sims = query_vectors @ vectors.T

    per_query: List[QueryDiagnostics] = []
    ranked_top10: List[List[int]] = []
    ndcg_scores: List[float] = []

    for i, row in enumerate(rows):
        rel = ground_truth[i]
        rel_missing = sorted(pid for pid in rel if pid not in corpus_pages)

        if k_window == 0 or not queries[i].strip():
            top10: List[int] = []
            full_ranked: List[int] = []
            chunk_order = np.array([], dtype=int)
        else:
            chunk_order, _chunk_scores = _chunk_order_and_scores(sims[i])
            full_ranked = aggregate_to_pages(
                sims[i], chunk_order, page_ids_list, page_to_chunks,
                scope=scope, pool_k=pool_k, k_window=k_window,
                fusion=fusion, bm25=bm25,
                query_terms=tokenize(queries[i]) if use_fusion else None,
                rerank=use_rerank, query=queries[i], chunk_texts=chunk_texts,
            )
            top10 = full_ranked[:K_EVAL]

        ranked_top10.append(top10)

        ndcg = ndcg_at_k(top10, rel, k=K_EVAL)
        ndcg_scores.append(ndcg)

        page_rank_map = _page_ranks(full_ranked, rel)
        chunk_ranks = (
            _chunk_best_ranks(chunk_order, page_ids_arr, rel)
            if len(chunk_order)
            else {pid: -1 for pid in rel}
        )
        chunk_topk = (
            _chunk_in_top_k(chunk_order, page_ids_arr, rel, k_window)
            if len(chunk_order)
            else {pid: False for pid in rel}
        )

        recall_at = {
            str(k): _recall_at_k(full_ranked, rel, k) for k in RECALL_KS
        }

        per_query.append(
            QueryDiagnostics(
                query_id=row["query_id"],
                query=row["query"],
                n_relevant=len(rel),
                ndcg_at_10=ndcg,
                mrr=_mrr(full_ranked, rel),
                recall_at=recall_at,
                relevant_in_top10=len(set(top10[:K_EVAL]) & rel),
                page_ranks={str(k): v for k, v in page_rank_map.items()},
                chunk_best_ranks={str(k): v for k, v in chunk_ranks.items()},
                chunk_in_top_k={str(k): v for k, v in chunk_topk.items()},
                missing_relevant_ids=rel_missing,
            )
        )

    # Bucket summaries by n_relevant
    buckets: Dict[int, List[QueryDiagnostics]] = defaultdict(list)
    for qd in per_query:
        buckets[qd.n_relevant].append(qd)

    bucket_summaries = []
    for n_rel in sorted(buckets):
        group = buckets[n_rel]
        bucket_summaries.append(
            BucketSummary(
                n_relevant=n_rel,
                count=len(group),
                mean_ndcg_at_10=float(
                    np.mean([g.ndcg_at_10 for g in group])
                ),
                mean_mrr=float(np.mean([g.mrr for g in group])),
                mean_recall_at_10=float(
                    np.mean([g.recall_at["10"] for g in group])
                ),
            )
        )

    # k-fold CV on per-query NDCG
    fold_assign = _make_kfold_assignments(len(rows), kfold)
    fold_ndcg: Dict[int, List[float]] = defaultdict(list)
    for i, fold in enumerate(fold_assign):
        fold_ndcg[fold].append(ndcg_scores[i])
    fold_means = {
        str(f): float(np.mean(scores)) for f, scores in sorted(fold_ndcg.items())
    }
    fold_mean_list = list(fold_means.values())
    kfold_summary = {
        "k": kfold,
        "seed": K_FOLD_SEED,
        "per_fold_mean_ndcg_at_10": fold_means,
        "mean_ndcg_at_10": float(np.mean(fold_mean_list)),
        "std_ndcg_at_10": float(np.std(fold_mean_list, ddof=0)),
    }

    # Graded query path: timed run() equivalent (embed + FAISS retrieve).
    query_timing: Dict[str, Any] = {}
    prod_top10: List[List[int]] = []
    if time_run or sanity_check:
        prod_top10, query_timing = measure_query_phase_time(
            queries,
            artifacts_dir=root,
            top_chunks=top_chunks,
            scope=scope,
            pool_k=pool_k,
            fusion=fusion,
            prf=prf,
            rerank=use_rerank,
        )

    # Sanity: harness top-10 vs retrieve.search_batch (reuses timed call above).
    sanity_result: Dict[str, Any] = {"passed": True, "mismatches": []}
    if sanity_check:
        for i, row in enumerate(rows):
            harness = ranked_top10[i]
            prod = prod_top10[i]
            if harness != prod:
                sanity_result["passed"] = False
                sanity_result["mismatches"].append(
                    {
                        "query_id": row["query_id"],
                        "harness": harness,
                        "retrieve": prod,
                    }
                )

    summary = {
        "num_queries": len(rows),
        "mean_ndcg_at_10": float(np.mean(ndcg_scores)),
        "mean_mrr": float(np.mean([q.mrr for q in per_query])),
        "mean_recall_at_10": float(
            np.mean([q.recall_at["10"] for q in per_query])
        ),
        "mean_recall_at_50": float(
            np.mean([q.recall_at["50"] for q in per_query])
        ),
        "mean_recall_at_100": float(
            np.mean([q.recall_at["100"] for q in per_query])
        ),
        "queries_with_any_relevant_in_top10": sum(
            1 for q in per_query if q.relevant_in_top10 > 0
        ),
        "mean_relevant_in_top10": float(
            np.mean([q.relevant_in_top10 for q in per_query])
        ),
        "query_phase_time_s": query_timing.get("query_phase_time_s"),
        "within_grading_budget": query_timing.get("within_budget"),
    }

    meta_path = root / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    config = {
        "artifacts_dir": str(root),
        "queries_path": str(qpath),
        "scope": scope,
        "pool_k": pool_k,
        "fusion": "rrf" if use_fusion else "none",
        "bm25_page_agg": BM25_PAGE_AGG if use_fusion else None,
        "bm25_scope": BM25_SCOPE if use_fusion else None,
        "rrf_k": RRF_K if use_fusion else None,
        "prf": bool(use_prf),
        "prf_alpha": PRF_ALPHA if use_prf else None,
        "prf_topn": PRF_TOPN if use_prf else None,
        "prf_page_repr": PRF_PAGE_REPR if use_prf else None,
        "rerank": bool(use_rerank),
        "rerank_pool": RERANK_POOL if use_rerank else None,
        "rerank_alpha": RERANK_ALPHA if use_rerank else None,
        "rerank_model": RERANK_MODEL_NAME if use_rerank else None,
        "top_chunks": top_chunks,
        "k_eval": K_EVAL,
        "chunk_words": meta.get("chunk_words", CHUNK_WORDS),
        "chunk_overlap": meta.get("chunk_overlap", CHUNK_OVERLAP),
        "num_chunks": int(vectors.shape[0]),
        "num_corpus_pages": len(corpus_pages),
    }

    data_quality = {
        "missing_relevant_ids": missing_ids,
        "n_missing_relevant_ids": n_missing,
    }

    return DiagnosticReport(
        tag=tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        summary=summary,
        bucket_summaries=bucket_summaries,
        kfold=kfold_summary,
        data_quality=data_quality,
        sanity=sanity_result,
        query_timing=query_timing,
        per_query=per_query,
    )


def report_to_dict(report: DiagnosticReport) -> Dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    return {
        "tag": report.tag,
        "timestamp": report.timestamp,
        "config": report.config,
        "summary": report.summary,
        "bucket_summaries": [asdict(b) for b in report.bucket_summaries],
        "kfold": report.kfold,
        "data_quality": report.data_quality,
        "sanity": report.sanity,
        "query_timing": report.query_timing,
        "per_query": [asdict(q) for q in report.per_query],
    }


def save_report(report: DiagnosticReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_to_dict(report), indent=2), encoding="utf-8"
    )


def print_report_summary(report: DiagnosticReport) -> None:
    """Print a concise console summary."""
    s = report.summary
    kf = report.kfold
    dq = report.data_quality

    print(f"tag={report.tag}")
    print(f"artifacts={report.config['artifacts_dir']}")
    print(f"num_queries={s['num_queries']}  num_chunks={report.config['num_chunks']}")
    print(
        f"aggregation: scope={report.config.get('scope')}  "
        f"pool_k={report.config.get('pool_k')} (0=all)  "
        f"top_chunks={report.config.get('top_chunks')}"
    )
    if report.config.get("fusion") == "rrf":
        print(
            f"fusion: rrf(k={report.config.get('rrf_k')})  "
            f"bm25_agg={report.config.get('bm25_page_agg')}  "
            f"bm25_scope={report.config.get('bm25_scope')}"
        )
    if report.config.get("prf"):
        print(
            f"prf: alpha={report.config.get('prf_alpha')}  "
            f"top_n={report.config.get('prf_topn')}  "
            f"page_repr={report.config.get('prf_page_repr')}"
        )
    if report.config.get("rerank"):
        print(
            f"rerank: pool={report.config.get('rerank_pool')}  "
            f"alpha={report.config.get('rerank_alpha')}  "
            f"model={report.config.get('rerank_model')}"
        )
    print()
    print("=== Scoreboard (matches eval_public.py NDCG math) ===")
    print(f"mean_ndcg@10={s['mean_ndcg_at_10']:.4f}")
    print(f"mean_mrr={s['mean_mrr']:.4f}")
    print(f"mean_recall@{{10,50,100}}={s['mean_recall_at_10']:.4f}, "
          f"{s['mean_recall_at_50']:.4f}, {s['mean_recall_at_100']:.4f}")
    print(f"queries_with_any_relevant_in_top10={s['queries_with_any_relevant_in_top10']}")
    print(f"mean_relevant_pages_in_top10={s['mean_relevant_in_top10']:.2f}")
    print()
    qt = report.query_timing
    if qt:
        budget = "OK" if qt.get("within_budget") else "OVER BUDGET"
        print("=== Graded query timing (main.run / eval_public path) ===")
        print(
            f"query_phase_time={qt['query_phase_time_s']:.2f}s  "
            f"limit={qt['time_limit_s']:.0f}s  [{budget}]"
        )
    print()
    print(f"=== k-fold CV (k={kf['k']}, seed={kf['seed']}) ===")
    for fold, score in kf["per_fold_mean_ndcg_at_10"].items():
        print(f"  fold {fold}: ndcg@10={score:.4f}")
    print(f"  mean={kf['mean_ndcg_at_10']:.4f}  std={kf['std_ndcg_at_10']:.4f}")
    print()
    print("=== Per-bucket (by n_relevant) ===")
    for b in report.bucket_summaries:
        print(
            f"  n_rel={b.n_relevant}  count={b.count}  "
            f"ndcg@10={b.mean_ndcg_at_10:.4f}  "
            f"mrr={b.mean_mrr:.4f}  recall@10={b.mean_recall_at_10:.4f}"
        )
    print()
    print("=== Data quality ===")
    print(f"missing_relevant_ids={dq['n_missing_relevant_ids']}")
    print()
    print("=== Sanity (harness top-10 vs retrieve.search_batch) ===")
    if report.sanity["passed"]:
        print("PASSED")
    else:
        print(f"FAILED ({len(report.sanity['mismatches'])} mismatches)")
        for mm in report.sanity["mismatches"][:5]:
            print(f"  {mm['query_id']}: harness={mm['harness']} retrieve={mm['retrieve']}")


def compare_reports(path_a: Path, path_b: Path) -> None:
    """Print per-query and per-bucket NDCG deltas between two saved reports."""
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))

    print(f"Compare: {path_a.name} -> {path_b.name}")
    print(
        f"mean_ndcg@10: {a['summary']['mean_ndcg_at_10']:.4f} -> "
        f"{b['summary']['mean_ndcg_at_10']:.4f}  "
        f"(delta {b['summary']['mean_ndcg_at_10'] - a['summary']['mean_ndcg_at_10']:+.4f})"
    )

    by_id_a = {q["query_id"]: q for q in a["per_query"]}
    by_id_b = {q["query_id"]: q for q in b["per_query"]}

    print("\nPer-query NDCG@10 deltas (largest improvements first):")
    deltas = []
    for qid in by_id_a:
        da = by_id_a[qid]["ndcg_at_10"]
        db = by_id_b[qid]["ndcg_at_10"]
        deltas.append((db - da, qid, da, db, by_id_a[qid]["n_relevant"]))
    deltas.sort(reverse=True)
    for delta, qid, da, db, n_rel in deltas[:15]:
        sign = "+" if delta >= 0 else ""
        print(f"  {qid}  n_rel={n_rel}  {da:.4f} -> {db:.4f}  ({sign}{delta:.4f})")

    print("\nPer-bucket mean NDCG@10:")
    buckets_a = {row["n_relevant"]: row for row in a["bucket_summaries"]}
    buckets_b = {row["n_relevant"]: row for row in b["bucket_summaries"]}
    for n_rel in sorted(set(buckets_a) | set(buckets_b)):
        ba = buckets_a.get(n_rel, {}).get("mean_ndcg_at_10", float("nan"))
        bb = buckets_b.get(n_rel, {}).get("mean_ndcg_at_10", float("nan"))
        print(f"  n_rel={n_rel}: {ba:.4f} -> {bb:.4f}  (delta {bb - ba:+.4f})")
