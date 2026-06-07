"""Local CPU sanity checks for E2 BM25 artifacts."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import numpy as np

from chunk import chunk_corpus
from lexical import (
    bm25_score_row,
    build_bm25_artifacts,
    load_bm25,
    tokenize,
)
from utils import ENTRIES_DIR, PUBLIC_QUERIES_PATH, load_public_queries, iter_entries


def main() -> None:
    out = STUDENT_ROOT / "artifacts_sanity_e2"
    if out.exists():
        try:
            shutil.rmtree(out)
        except OSError:
            pass
    out.mkdir(parents=True, exist_ok=True)

    records = list(iter_entries())[:500]
    if len(records) < 10:
        print("SKIP: need local corpus under data/Wikipedia Entries/")
        return

    chunks = chunk_corpus(records)
    t0 = time.perf_counter()
    build_bm25_artifacts(chunks, out)
    build_s = time.perf_counter() - t0

    bm25 = load_bm25(out)
    n_chunks = len(chunks)

    assert len(bm25.indptr) == n_chunks + 1, "indptr length"
    assert bm25.n_docs == n_chunks, "n_docs"
    assert bm25.vocab_size == len(bm25.vocab), "vocab_size"
    assert len(bm25.idf) == bm25.vocab_size, "idf map size"

    # avg_dl check
    dls = [len(tokenize(c.text)) for c in chunks]
    expected_avg = sum(dls) / len(dls)
    assert abs(bm25.avg_dl - expected_avg) < 1e-6, "avg_dl mismatch"

    # CSR row count
    assert int(bm25.indptr[-1]) == len(bm25.data), "CSR nnz consistency"

    # Overlap query / chunk scoring
    queries = load_public_queries(PUBLIC_QUERIES_PATH)
    sample_queries = [q["query"] for q in queries[:2]]
    scored_any = False
    for qi, query in enumerate(sample_queries):
        q_terms = tokenize(query)
        for row in range(min(3, n_chunks)):
            s = bm25_score_row(
                bm25.data,
                bm25.indices,
                bm25.indptr,
                row,
                q_terms,
                bm25.idf,
                bm25.vocab,
                bm25.avg_dl,
                bm25.k1,
                bm25.b,
            )
            if s > 0:
                scored_any = True
                print(f"  query[{qi}] row={row} bm25={s:.4f} (overlap OK)")
    assert scored_any, "expected BM25 > 0 for at least one query/chunk pair"

    # Extrapolate full-corpus build time from subset
    pages_local = len(records)
    pages_full = 27074
    est_full_s = build_s * (pages_full / pages_local)
    print(f"\nOK: {n_chunks} chunks, vocab={bm25.vocab_size}, nnz={len(bm25.data)}")
    print(f"build_time={build_s:.1f}s on {pages_local} pages")
    print(f"extrapolated_full_corpus_bm25_build~{est_full_s:.0f}s ({est_full_s/60:.1f} min)")

    try:
        shutil.rmtree(out)
    except OSError:
        print(f"(left temp dir {out} — remove manually if needed)")
    print("sanity_e2: PASSED")


if __name__ == "__main__":
    main()
