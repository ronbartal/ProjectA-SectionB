"""E3 aggregation / TOP_CHUNKS sweep (Yehoraz, query-side only).

Read-only exploration: loads the existing dense index once, embeds the public
queries once, then evaluates every combination of TOP_CHUNKS x aggregation
cheaply in-memory. No artifact rebuild, no edits to retrieve.py.

Aggregation modes:
  max        -> page score = best single chunk (current production behaviour)
  sumN       -> page score = sum of the page's top-N chunk scores in the window
  meanN      -> page score = mean of the page's top-N chunk scores
                (averaged over chunks actually present, i.e. min(N, available);
                 mean1 == max). Rewards pages whose best passages are
                 consistently high without rewarding chunk count.

Scopes (which chunks are averaged in stage 2):
  window     -> only chunks that landed in the FAISS top-TOP_CHUNKS window.
  page       -> two-stage rerank: the window only selects CANDIDATE pages, then
                each candidate is rescored over ALL of its chunks (including
                ones not returned), using its true top-N by query score.

Usage:
  python scripts/sweep_e3.py
  python scripts/sweep_e3.py --scope page --top-chunks 300 500 --modes max mean2 mean3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from eval import K_EVAL, load_query_file, ndcg_at_k
from index import load_index
from utils import PUBLIC_QUERIES_PATH, TOP_CHUNKS

K_FOLD_SEED = 42


def _parse_mode(mode: str) -> Tuple[str, int]:
    """Return (agg_type, N) for a mode label.

    'max' -> ('max', 1); 'sumK' -> ('sum', K); 'meanK' -> ('mean', K).
    """
    if mode == "max":
        return ("max", 1)
    if mode.startswith("sum"):
        return ("sum", int(mode[3:]))
    if mode.startswith("mean"):
        return ("mean", int(mode[4:]))
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def _rank_pages(
    window_idx: np.ndarray,
    window_scores: np.ndarray,
    page_ids: Sequence[int],
    agg_type: str,
    top_n: int,
) -> List[int]:
    """Aggregate window chunk scores to pages.

    'max'  -> best single chunk per page.
    'sum'  -> sum of the page's top-N chunk scores.
    'mean' -> mean of the page's top-N chunk scores (averaged over chunks
              actually present, i.e. min(N, available)).

    window_idx / window_scores are assumed sorted best-first, so the first N
    scores seen for a page are its top-N chunks.
    """
    per_page: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for idx, score in zip(window_idx, window_scores):
        if idx < 0:
            continue
        pid = int(page_ids[int(idx)])
        c = counts.get(pid, 0)
        if c < top_n:
            per_page[pid] = per_page.get(pid, 0.0) + float(score)
            counts[pid] = c + 1
    if agg_type == "mean":
        per_page = {pid: total / counts[pid] for pid, total in per_page.items()}
    ordered = sorted(per_page.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered]


def _rank_pages_full(
    sim_row: np.ndarray,
    window_idx: np.ndarray,
    page_to_chunks: Dict[int, np.ndarray],
    page_ids: Sequence[int],
    agg_type: str,
    top_n: int,
) -> List[int]:
    """Two-stage rerank: window selects candidate pages, then score each over
    ALL of its chunks (page scope).

    Candidate pages are those appearing in `window_idx`. For each candidate, the
    score is the max / sum / mean of its true top-N chunks across the whole page
    (using the full query-vs-chunk similarity row), not just the windowed ones.
    """
    seen: set[int] = set()
    candidates: List[int] = []
    for idx in window_idx:
        if idx < 0:
            continue
        pid = int(page_ids[int(idx)])
        if pid not in seen:
            seen.add(pid)
            candidates.append(pid)

    scored: Dict[int, float] = {}
    for pid in candidates:
        chunk_idx = page_to_chunks[pid]
        scores = sim_row[chunk_idx]
        if top_n > 0 and scores.size > top_n:
            top = np.partition(scores, -top_n)[-top_n:]
        else:
            top = scores
        if agg_type == "mean":
            scored[pid] = float(top.mean())
        elif agg_type == "sum":
            scored[pid] = float(top.sum())
        else:  # max
            scored[pid] = float(scores.max())
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in ordered]


def _make_kfold_assignments(n: int, k: int, seed: int = K_FOLD_SEED) -> List[int]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_ids = np.empty(n, dtype=int)
    for fold, idx in enumerate(indices):
        fold_ids[idx] = fold % k
    return fold_ids.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 aggregation / TOP_CHUNKS sweep.")
    parser.add_argument(
        "--top-chunks", type=int, nargs="+", default=[100, 200, 300, 500]
    )
    parser.add_argument(
        "--modes", type=str, nargs="+",
        default=["max", "mean2", "mean3", "mean5", "mean10"],
    )
    parser.add_argument(
        "--scope", type=str, nargs="+", choices=["window", "page"],
        default=["window", "page"],
        help="window: average windowed chunks; page: rescore over all page chunks",
    )
    parser.add_argument("--kfold", type=int, default=5)
    args = parser.parse_args()

    max_window = max(args.top_chunks)
    mode_to_agg = {m: _parse_mode(m) for m in args.modes}

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    print("Loading index + embedding queries (once)...")
    vectors, page_ids, _index = load_index()
    qv = embed_queries(queries)
    qv = np.ascontiguousarray(qv, dtype=np.float32)
    sims = qv @ vectors.T  # (n_queries, n_chunks)
    n_chunks = vectors.shape[0]
    cap = min(max_window, n_chunks)

    # page_id -> all chunk indices on that page (for page-scope rescoring).
    page_to_chunks: Dict[int, np.ndarray] = {}
    if "page" in args.scope:
        tmp: Dict[int, List[int]] = {}
        for ci, pid in enumerate(page_ids):
            tmp.setdefault(int(pid), []).append(ci)
        page_to_chunks = {pid: np.asarray(idxs) for pid, idxs in tmp.items()}

    # Per query: top `cap` chunk indices + scores, sorted best-first (reused).
    top_idx_all: List[np.ndarray] = []
    top_score_all: List[np.ndarray] = []
    for i in range(sims.shape[0]):
        part = np.argpartition(-sims[i], cap - 1)[:cap]
        order = part[np.argsort(-sims[i][part])]
        top_idx_all.append(order)
        top_score_all.append(sims[i][order])

    fold_assign = _make_kfold_assignments(len(rows), args.kfold)

    results: List[Tuple[str, str, int, float, float, float]] = []
    for scope in args.scope:
        for tc in args.top_chunks:
            for mode in args.modes:
                agg_type, top_n = mode_to_agg[mode]
                ndcgs: List[float] = []
                for i in range(len(rows)):
                    w_idx = top_idx_all[i][:tc]
                    if scope == "page":
                        ranked = _rank_pages_full(
                            sims[i], w_idx, page_to_chunks, page_ids,
                            agg_type, top_n,
                        )
                    else:
                        w_sc = top_score_all[i][:tc]
                        ranked = _rank_pages(w_idx, w_sc, page_ids, agg_type, top_n)
                    ndcgs.append(ndcg_at_k(ranked[:K_EVAL], ground_truth[i], k=K_EVAL))
                ndcgs_arr = np.asarray(ndcgs)
                mean_ndcg = float(ndcgs_arr.mean())
                fold_means = [
                    float(ndcgs_arr[[j for j, f in enumerate(fold_assign) if f == fold]].mean())
                    for fold in range(args.kfold)
                ]
                kfold_mean = float(np.mean(fold_means))
                kfold_std = float(np.std(fold_means))
                results.append((scope, mode, tc, mean_ndcg, kfold_mean, kfold_std))

    print(f"\n=== E3 sweep (baseline = window/max/TOP_CHUNKS={TOP_CHUNKS} = 0.1332; "
          f"current prod = window/mean2/500 = 0.1612) ===")
    print(f"{'scope':<8}{'mode':<8}{'top_chunks':>11}{'ndcg@10':>10}{'kfold':>9}{'+/-':>8}")
    for scope, mode, tc, mean_ndcg, kfm, kfs in sorted(
        results, key=lambda r: r[4], reverse=True
    ):
        flag = ""
        if scope == "window" and mode == "max" and tc == 200:
            flag = "  <-- old baseline"
        elif scope == "window" and mode == "mean2" and tc == TOP_CHUNKS:
            flag = "  <-- current prod"
        print(f"{scope:<8}{mode:<8}{tc:>11}{mean_ndcg:>10.4f}{kfm:>9.4f}{kfs:>8.4f}{flag}")


if __name__ == "__main__":
    main()
