"""A/B test: current pipeline (A) vs cross-encoder reranking (B).

A (baseline): PRF + page-scope dense + BM25/RRF fusion -> top-10 (live pipeline).
B (rerank)  : same retrieval, then Option A CE rerank on the top-RERANK_POOL
              fused pages (best dense chunk passage per page).

Passage text: uses artifacts/chunk_texts.npy when present (Ron offline build).
Otherwise reconstructs a proxy passage from BM25 tokens (TF-sorted join) --
valid for an architecture A/B test, but production should use real chunk text.

Usage:
  python scripts/sweep_rerank_ab.py
  python scripts/sweep_rerank_ab.py --pools 30 40 --model cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from eval import K_EVAL, load_query_file, ndcg_at_k
from index import load_index
from lexical import Bm25Index, load_bm25, tokenize
from retrieve import (
    _collect_candidates,
    _page_bm25_scores,
    _page_dense_scores,
    _prf_expand_query,
    _rrf_fuse,
)
from utils import (
    BM25_PAGE_AGG,
    BM25_SCOPE,
    PAGE_POOL_K,
    PRF,
    PRF_ALPHA,
    PRF_PAGE_REPR,
    PRF_TOPN,
    PUBLIC_QUERIES_PATH,
    RRF_K,
    TOP_CHUNKS,
)

K_FOLD_SEED = 42
A_BASELINE = 0.4274  # live pipeline on the fixed 29-query file
CHUNK_TEXTS_NAME = "chunk_texts.npy"


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


def _build_page_to_chunks(page_ids: List[int]) -> Dict[int, np.ndarray]:
    tmp: Dict[int, List[int]] = {}
    for ci, pid in enumerate(page_ids):
        tmp.setdefault(int(pid), []).append(ci)
    return {p: np.asarray(v, dtype=np.int64) for p, v in tmp.items()}


def _load_chunk_texts(artifacts_dir: Path) -> Optional[np.ndarray]:
    path = artifacts_dir / CHUNK_TEXTS_NAME
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _bm25_chunk_text(row: int, bm25: Bm25Index) -> str:
    """Proxy passage: BM25 tokens sorted by TF (no word order)."""
    s, e = int(bm25.indptr[row]), int(bm25.indptr[row + 1])
    if s >= e:
        return ""
    cols = bm25.indices[s:e]
    tfs = bm25.data[s:e]
    order = np.argsort(-tfs)
    return " ".join(str(bm25.vocab[int(cols[i])]) for i in order)


class PassageLookup:
    def __init__(self, texts: Optional[np.ndarray], bm25: Bm25Index):
        self._texts = texts
        self._bm25 = bm25
        self._cache: Dict[int, str] = {}
        self.source = "chunk_texts.npy" if texts is not None else "bm25_token_proxy"

    def __call__(self, chunk_idx: int) -> str:
        v = self._cache.get(chunk_idx)
        if v is not None:
            return v
        if self._texts is not None:
            v = str(self._texts[chunk_idx])
        else:
            v = _bm25_chunk_text(chunk_idx, self._bm25)
        self._cache[chunk_idx] = v
        return v


def _best_chunk_in_window(
    pid: int,
    windowed: Dict[int, List[int]],
    sim_row: np.ndarray,
) -> int:
    chunks = windowed[pid]
    return int(chunks[int(np.argmax([sim_row[c] for c in chunks]))])


def _fused_pages_for_query(
    i: int,
    qv: np.ndarray,
    vectors: np.ndarray,
    page_ids: List[int],
    p2c: Dict[int, np.ndarray],
    bm25: Bm25Index,
    query: str,
    *,
    cap: int,
    use_prf: bool,
) -> Tuple[List[int], Dict[int, List[int]], np.ndarray]:
    sim = qv[i] @ vectors.T
    wo = _window_order(sim, cap)
    qvec = qv[i]
    if use_prf:
        qvec = _prf_expand_query(qvec, wo, page_ids, p2c, vectors)
        sim = qvec @ vectors.T
        wo = _window_order(sim, cap)
    cands, windowed = _collect_candidates(wo, page_ids)
    if not cands:
        return [], windowed, sim
    dense = _page_dense_scores(qvec, cands, p2c, vectors, PAGE_POOL_K)
    bm = _page_bm25_scores(tokenize(query), cands, windowed, p2c, bm25)
    fused = _rrf_fuse(dense, bm, RRF_K)
    return fused, windowed, sim


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B: baseline vs CE rerank.")
    ap.add_argument("--pools", type=int, nargs="+", default=[20, 30, 40, 50])
    ap.add_argument(
        "--model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model (rerank stage only)",
    )
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--no-prf", action="store_true")
    ap.add_argument("--artifacts-dir", type=Path, default=STUDENT_ROOT / "artifacts")
    args = ap.parse_args()
    use_prf = PRF and not args.no_prf

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]
    fold = _kfold_assign(len(rows), args.kfold)

    print("Loading index, BM25, embedding queries...")
    vectors, page_ids, _ = load_index(args.artifacts_dir)
    bm25 = load_bm25(args.artifacts_dir)
    texts = _load_chunk_texts(args.artifacts_dir)
    lookup = PassageLookup(texts, bm25)
    print(f"Passage source: {lookup.source}")

    qv = np.ascontiguousarray(embed_queries(queries), dtype=np.float32)
    cap = min(TOP_CHUNKS, vectors.shape[0])
    p2c = _build_page_to_chunks(page_ids)

    print("Phase 1: baseline fused rankings (A)...")
    fused_lists: List[List[int]] = []
    windowed_lists: List[Dict[int, List[int]]] = []
    sim_rows: List[np.ndarray] = []
    for i in range(len(rows)):
        fused, windowed, sim = _fused_pages_for_query(
            i, qv, vectors, page_ids, p2c, bm25, queries[i],
            cap=cap, use_prf=use_prf,
        )
        fused_lists.append(fused)
        windowed_lists.append(windowed)
        sim_rows.append(sim)

    def eval_lists(lists: Sequence[List[int]]) -> Tuple[float, float, float, float, float]:
        nd = [ndcg_at_k(lst[:K_EVAL], gt[i], k=K_EVAL) for i, lst in enumerate(lists)]
        km, ks = _kfold_stats(nd, fold, args.kfold)
        arr = np.asarray(nd)
        return float(arr.mean()), km, ks, float(arr[0::2].mean()), float(arr[1::2].mean())

    a_mean, a_km, a_ks, a_ha, a_hb = eval_lists([f[:K_EVAL] for f in fused_lists])
    print(f"\n=== A: baseline (no CE rerank) ===")
    print(f"  ndcg@10={a_mean:.4f}  kfold={a_km:.4f} +/-{a_ks:.4f}"
          f"  half_A={a_ha:.4f}  half_B={a_hb:.4f}")
    print(f"  reference live score={A_BASELINE}")

    print(f"\nLoading cross-encoder: {args.model} ...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.model)

    print("\n=== B: CE rerank on shortlist (ce = pure CE order; "
          "rrf = RRF(fused, ce); bN = N*ce_norm + (1-N)*fused_norm) ===")
    print(f"{'variant':>12}{'ndcg@10':>10}{'kfold':>9}{'+/-':>8}{'delta':>8}"
          f"{'dHalfA':>9}{'dHalfB':>9}{'ce_s':>8}")
    best = (-1.0, None)
    for pool in args.pools:
        # CE scores computed once per pool; all fusion variants reuse them.
        t0 = time.perf_counter()
        per_q: List[Tuple[List[int], List[int], np.ndarray]] = []
        for i in range(len(rows)):
            fused = fused_lists[i]
            if not fused:
                per_q.append((fused, [], np.empty(0)))
                continue
            short = fused[:pool]
            pairs = []
            pids = []
            for pid in short:
                if pid not in windowed_lists[i]:
                    continue
                cidx = _best_chunk_in_window(pid, windowed_lists[i], sim_rows[i])
                passage = lookup(cidx)
                if passage.strip():
                    pairs.append((queries[i], passage))
                    pids.append(pid)
            scores = (
                ce.predict(pairs, batch_size=64, show_progress_bar=False)
                if pairs else np.empty(0)
            )
            per_q.append((fused, pids, np.asarray(scores, dtype=np.float64)))
        ce_s = time.perf_counter() - t0

        def variant_lists(mode: str, alpha: float = 0.5) -> List[List[int]]:
            out: List[List[int]] = []
            for fused, pids, scores in per_q:
                if not pids:
                    out.append(fused[:K_EVAL])
                    continue
                ce_rank = {p: r for r, (p, _) in enumerate(
                    sorted(zip(pids, scores), key=lambda x: x[1], reverse=True))}
                if mode == "ce":
                    order = sorted(pids, key=lambda p: ce_rank[p])
                elif mode == "rrf":
                    fr = {p: r for r, p in enumerate(fused) if p in ce_rank}
                    order = sorted(
                        pids,
                        key=lambda p: -(1.0 / (RRF_K + fr[p]) + 1.0 / (RRF_K + ce_rank[p])),
                    )
                else:  # score blend on min-max normalized scores
                    lo, hi = float(scores.min()), float(scores.max())
                    ce_n = {p: ((s - lo) / (hi - lo) if hi > lo else 0.5)
                            for p, s in zip(pids, scores)}
                    npids = len(pids)
                    fr = {p: r for r, p in enumerate(fused) if p in ce_n}
                    fu_n = {p: 1.0 - fr[p] / max(npids - 1, 1) for p in ce_n}
                    order = sorted(
                        pids,
                        key=lambda p: -(alpha * ce_n[p] + (1 - alpha) * fu_n[p]),
                    )
                tail = [p for p in fused if p not in ce_rank]
                out.append((order + tail)[:K_EVAL])
            return out

        variants = [("ce", 0.0), ("rrf", 0.0), ("b0.3", 0.3), ("b0.5", 0.5), ("b0.7", 0.7)]
        for vname, alpha in variants:
            mode = vname if vname in ("ce", "rrf") else "blend"
            m, km, ks, ha, hb = eval_lists(variant_lists(mode, alpha))
            delta = km - a_km
            label = f"p{pool}_{vname}"
            flag = ""
            if km > best[0]:
                best = (km, label)
                flag = "  <-- best B"
            print(f"{label:>12}{m:>10.4f}{km:>9.4f}{ks:>8.4f}{delta:>+8.4f}"
                  f"{ha - a_ha:>+9.4f}{hb - a_hb:>+9.4f}{ce_s:>8.1f}{flag}")

    print(f"\nBest B: {best[1]} kfold={best[0]:.4f} (A was {a_km:.4f})")
    if best[0] > a_km + 0.005:
        print("Verdict: reranking likely helps (delta > +0.005 on kfold mean).")
    elif best[0] < a_km - 0.005:
        print("Verdict: reranking likely hurts on this setup.")
    else:
        print("Verdict: reranking within noise (+/-0.005); not clearly better.")


if __name__ == "__main__":
    main()
