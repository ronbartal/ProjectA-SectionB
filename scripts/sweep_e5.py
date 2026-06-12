"""E5 page-vector fusion sweep (Yehoraz, query-side, read-only exploration).

Blends Ron's page-level embeddings (title + first/last sentence, §4.2.1) into
the live pipeline (PRF + page-scope dense mean-all + BM25-max + RRF, 0.4274).

Per query, every signal is computed ONCE and cached:
  - dense  : page-scope mean-all cosine vs the PRF-expanded query (production)
  - bm25   : max over in-window chunk BM25 scores, original query terms
  - pv_orig: page-vector cosine vs the ORIGINAL query embedding
  - pv_prf : page-vector cosine vs the PRF-EXPANDED query embedding

Fusion configs swept in-memory (all free after phase 1):
  - anchor    : rrf2(dense, bm25)                      -> must reproduce 0.4274
  - rrf3      : rrf(dense, bm25, pv)   [pv = orig|prf]
  - rrf2pv    : rrf(dense, pv)         (no BM25, for information)
  - blend a   : dense' = a*dense + (1-a)*pv, then rrf2(dense', bm25)
                (both are cosines of normalized vectors -> same scale)

Usage:
  python scripts/sweep_e5.py
  python scripts/sweep_e5.py --artifacts-dir artifacts_variants/notitle_150
  python scripts/sweep_e5.py --split-half --configs rrf3_prf blend0.7_prf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from eval import K_EVAL, load_query_file, ndcg_at_k
from index import load_index
from lexical import load_bm25, tokenize
from page_index import load_page_index, page_scores_for_ids
from retrieve import (
    _collect_candidates,
    _page_bm25_scores,
    _page_dense_scores,
    _prf_expand_query,
    _ranks,
)
from utils import ARTIFACTS_DIR, PAGE_POOL_K, PRF, PUBLIC_QUERIES_PATH, RRF_K, TOP_CHUNKS

K_FOLD_SEED = 42
ANCHOR = 0.4274  # live pipeline on the fixed 29-query file


def _kfold_assign(n: int, k: int, seed: int = K_FOLD_SEED) -> List[int]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    out = np.empty(n, dtype=int)
    for f, i in enumerate(idx):
        out[i] = f % k
    return out.tolist()


def _kfold_stats(ndcgs: Sequence[float], fold: Sequence[int], k: int) -> Tuple[float, float]:
    arr = np.asarray(ndcgs)
    means = [float(arr[[j for j, f in enumerate(fold) if f == ff]].mean()) for ff in range(k)]
    return float(np.mean(means)), float(np.std(means))


def _window_order(sim_row: np.ndarray, cap: int) -> np.ndarray:
    part = np.argpartition(-sim_row, cap - 1)[:cap]
    return part[np.argsort(-sim_row[part])]


def _rrf_n(score_dicts: Sequence[Dict[int, float]], k: int = RRF_K) -> List[int]:
    """N-way RRF over the page sets; pages absent from a ranker get no credit."""
    base = score_dicts[0]
    ranks = [_ranks(d) for d in score_dicts]
    fused = {
        pid: sum(1.0 / (k + r.get(pid, 10 ** 9)) for r in ranks)
        for pid in base
    }
    return [p for p, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 page-vector fusion sweep.")
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR,
                    help="Chunk/BM25 index dir (page index always from artifacts/)")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--split-half", action="store_true",
                    help="Also report split-half (even/odd queries) per config")
    ap.add_argument("--no-bm25", action="store_true",
                    help="Skip BM25 (e.g. variant dir with broken bm25_tf.npz); "
                         "sweeps dense vs dense+page-vector only")
    args = ap.parse_args()

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]
    n = len(rows)
    fold = _kfold_assign(n, args.kfold)

    print(f"Loading chunk index ({args.artifacts_dir}), BM25, page index, queries ({n})...")
    vectors, page_ids, _ = load_index(args.artifacts_dir)
    bm25 = None if args.no_bm25 else load_bm25(args.artifacts_dir)
    pages = load_page_index()  # chunk-config independent -> always artifacts/
    qv = np.ascontiguousarray(embed_queries(queries), dtype=np.float32)
    cap = min(TOP_CHUNKS, vectors.shape[0])

    p2c: Dict[int, np.ndarray] = {}
    _t: Dict[int, List[int]] = {}
    for ci, pid in enumerate(page_ids):
        _t.setdefault(int(pid), []).append(ci)
    p2c = {p: np.asarray(v, dtype=np.int64) for p, v in _t.items()}

    # Phase 1: per-query signals (computed once).
    print("Phase 1: dense / bm25 / page-vector signals per query...")
    q_dense: List[Dict[int, float]] = []
    q_bm25: List[Dict[int, float]] = []
    q_pv_orig: List[Dict[int, float]] = []
    q_pv_prf: List[Dict[int, float]] = []
    for i in range(n):
        sim = qv[i] @ vectors.T
        qexp = qv[i]
        if PRF:
            qexp = _prf_expand_query(qv[i], _window_order(sim, cap), page_ids, p2c, vectors)
            sim = qexp @ vectors.T
        wo = _window_order(sim, cap)
        cands, windowed = _collect_candidates(wo, page_ids)
        q_dense.append(_page_dense_scores(qexp, cands, p2c, vectors, PAGE_POOL_K))
        if bm25 is not None:
            q_bm25.append(
                _page_bm25_scores(tokenize(queries[i]), cands, windowed, p2c, bm25)
            )
        q_pv_orig.append(page_scores_for_ids(pages, qv[i], cands))
        q_pv_prf.append(page_scores_for_ids(pages, qexp, cands))

    def eval_cfg(rank_fn: Callable[[int], List[int]]) -> Tuple[float, float, float, float, float]:
        nd = [ndcg_at_k(rank_fn(i)[:K_EVAL], gt[i], k=K_EVAL) for i in range(n)]
        km, ks = _kfold_stats(nd, fold, args.kfold)
        arr = np.asarray(nd)
        half_a = float(arr[0::2].mean())
        half_b = float(arr[1::2].mean())
        return float(arr.mean()), km, ks, half_a, half_b

    configs: Dict[str, Callable[[int], List[int]]] = {}
    if bm25 is not None:
        configs["anchor_rrf2"] = lambda i: _rrf_n([q_dense[i], q_bm25[i]])
    else:
        # No-BM25 mode: dense-only anchor + dense/page-vector fusions.
        configs["anchor_dense"] = lambda i: _rrf_n([q_dense[i]])
    for tag, pv in (("orig", q_pv_orig), ("prf", q_pv_prf)):
        if bm25 is not None:
            configs[f"rrf3_{tag}"] = lambda i, pv=pv: _rrf_n(
                [q_dense[i], q_bm25[i], pv[i]]
            )
        configs[f"rrf2pv_{tag}"] = lambda i, pv=pv: _rrf_n([q_dense[i], pv[i]])
        for a in args.alphas:
            def rank(i, pv=pv, a=a):
                blend = {p: a * q_dense[i][p] + (1 - a) * pv[i].get(p, 0.0)
                         for p in q_dense[i]}
                rankers = [blend] if bm25 is None else [blend, q_bm25[i]]
                return _rrf_n(rankers)
            configs[f"blend{a:.1f}_{tag}"] = rank

    print(f"\n=== E5 sweep on {args.artifacts_dir.name} (anchor should be {ANCHOR}) ===")
    hdr = f"{'config':<18}{'ndcg@10':>9}{'kfold':>9}{'+/-':>8}"
    if args.split_half:
        hdr += f"{'half_A':>9}{'half_B':>9}"
    print(hdr)
    results = []
    for name, fn in configs.items():
        m, km, ks, ha, hb = eval_cfg(fn)
        results.append((name, m, km, ks, ha, hb))
    best_km = max(r[2] for r in results)
    for name, m, km, ks, ha, hb in sorted(results, key=lambda r: r[2], reverse=True):
        line = f"{name:<18}{m:>9.4f}{km:>9.4f}{ks:>8.4f}"
        if args.split_half:
            line += f"{ha:>9.4f}{hb:>9.4f}"
        if km == best_km:
            line += "  <-- best"
        if name == "anchor_rrf2":
            line += "  (anchor)"
        print(line)


if __name__ == "__main__":
    main()
