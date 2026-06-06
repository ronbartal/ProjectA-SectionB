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
from eval import K_EVAL, dcg_at_k, load_query_file, ndcg_at_k
from index import load_index
from utils import (
    ARTIFACTS_DIR,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    GRADING_QUERY_TIME_LIMIT_S,
    PUBLIC_QUERIES_PATH,
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
    """Mirror retrieve._rank_pages_from_chunks (max-pool, top_k pages)."""
    ranked, _ = _max_pool_aggregate(chunk_indices, chunk_scores, page_ids)
    return ranked[:top_k]


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


def _duplicate_query_analysis(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Find duplicate query strings with differing relevant sets."""
    by_text: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_text[row["query"]].append(row)

    duplicates = []
    for text, group in by_text.items():
        if len(group) <= 1:
            continue
        sets = [g["relevant_page_ids"] for g in group]
        all_same = all(s == sets[0] for s in sets[1:])
        if not all_same:
            union = set()
            for s in sets:
                union |= s
            duplicates.append(
                {
                    "query": text,
                    "query_ids": [g["query_id"] for g in group],
                    "n_instances": len(group),
                    "n_unique_sets": len({frozenset(s) for s in sets}),
                    "union_size": len(union),
                    "sets": [sorted(s) for s in sets],
                }
            )
    return {
        "duplicate_query_strings": len(
            [t for t, g in by_text.items() if len(g) > 1]
        ),
        "duplicate_with_differing_sets": len(duplicates),
        "details": duplicates,
    }


def _union_oracle_ndcg(
    rows: List[Dict[str, Any]],
    ranked_by_query_id: Dict[str, List[int]],
) -> float:
    """Upper-bound NDCG if each duplicate query string used the union of labels."""
    by_text: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_text[row["query"]].append(row)

    scores: List[float] = []
    for text, group in by_text.items():
        union_rel: Set[int] = set()
        for g in group:
            union_rel |= g["relevant_page_ids"]
        # Same ranking for all instances of this query string.
        ranked = ranked_by_query_id[group[0]["query_id"]]
        scores.append(ndcg_at_k(ranked, union_rel, k=K_EVAL))
    return float(sum(scores) / len(scores)) if scores else 0.0


def measure_query_phase_time(
    queries: List[str],
    *,
    artifacts_dir: Optional[Path] = None,
    top_chunks: int = TOP_CHUNKS,
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
        queries, top_k=K_EVAL, top_chunks=top_chunks, artifacts_dir=root
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
    aggregation: str = "max_pool",
    tag: str = "baseline",
    sanity_check: bool = True,
    time_run: bool = True,
) -> DiagnosticReport:
    """
    Run the full diagnostic harness and return a structured report.

    Uses numpy exact search over all chunks; default aggregation mirrors
    retrieve.py max-pool over the top_chunks window.
    """
    if aggregation != "max_pool":
        raise ValueError(f"Unsupported aggregation: {aggregation!r}")

    root = artifacts_dir or ARTIFACTS_DIR
    qpath = queries_path or PUBLIC_QUERIES_PATH
    rows = load_query_file(qpath)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    vectors, page_ids_list, _index = load_index(root)
    page_ids_arr = np.asarray(page_ids_list, dtype=np.int64)
    corpus_pages = set(int(x) for x in page_ids_list)

    missing_ids, n_missing = _check_missing_relevant(ground_truth, corpus_pages)
    dup_info = _duplicate_query_analysis(rows)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        sims = np.zeros((len(queries), vectors.shape[0]), dtype=np.float32)
    else:
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        sims = query_vectors @ vectors.T

    n_chunks = vectors.shape[0]
    k_window = min(top_chunks, n_chunks) if n_chunks else 0

    per_query: List[QueryDiagnostics] = []
    ranked_top10: List[List[int]] = []
    ranked_full: Dict[str, List[int]] = {}
    ndcg_scores: List[float] = []

    for i, row in enumerate(rows):
        rel = ground_truth[i]
        rel_missing = sorted(pid for pid in rel if pid not in corpus_pages)

        if k_window == 0 or not queries[i].strip():
            top10: List[int] = []
            full_ranked: List[int] = []
            chunk_order = np.array([], dtype=int)
        else:
            chunk_order, chunk_scores = _chunk_order_and_scores(sims[i])
            window_idx = chunk_order[:k_window]
            window_scores = chunk_scores[:k_window]
            full_ranked, _ = _max_pool_aggregate(window_idx, window_scores, page_ids_list)
            top10 = full_ranked[:K_EVAL]
            chunk_order = chunk_order  # full corpus chunk order

        ranked_top10.append(top10)
        ranked_full[row["query_id"]] = full_ranked

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

    union_oracle = _union_oracle_ndcg(rows, ranked_full)

    # Graded query path: timed run() equivalent (embed + FAISS retrieve).
    query_timing: Dict[str, Any] = {}
    prod_top10: List[List[int]] = []
    if time_run or sanity_check:
        prod_top10, query_timing = measure_query_phase_time(
            queries,
            artifacts_dir=root,
            top_chunks=top_chunks,
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
        "aggregation": aggregation,
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
        "duplicate_queries": dup_info,
        "union_oracle_mean_ndcg_at_10": union_oracle,
        "reported_mean_ndcg_at_10": summary["mean_ndcg_at_10"],
        "oracle_minus_reported": union_oracle - summary["mean_ndcg_at_10"],
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
    dup = dq["duplicate_queries"]
    print(
        f"duplicate_query_strings={dup['duplicate_query_strings']}  "
        f"differing_sets={dup['duplicate_with_differing_sets']}"
    )
    print(f"union_oracle_ndcg@10={dq['union_oracle_mean_ndcg_at_10']:.4f}  "
          f"(+{dq['oracle_minus_reported']:.4f} vs reported)")
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
    buckets_a = {b["n_relevant"]: b for b in a["bucket_summaries"]}
    buckets_b = {b["n_relevant"]: b for b in b["bucket_summaries"]}
    for n_rel in sorted(set(buckets_a) | set(buckets_b)):
        ba = buckets_a.get(n_rel, {}).get("mean_ndcg_at_10", float("nan"))
        bb = buckets_b.get(n_rel, {}).get("mean_ndcg_at_10", float("nan"))
        print(f"  n_rel={n_rel}: {ba:.4f} -> {bb:.4f}  (delta {bb - ba:+.4f})")
