"""Diagnostic harness for public queries (set-aware metrics + k-fold CV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from diagnostics import (
    compare_reports,
    print_report_summary,
    run_diagnostics,
    save_report,
)
from utils import (
    AGG_SCOPE,
    ARTIFACTS_DIR,
    FUSION,
    PAGE_POOL_K,
    PRF,
    PUBLIC_QUERIES_PATH,
    TOP_CHUNKS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run set-aware retrieval diagnostics on public queries."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory containing index_vectors.npy and index_meta.json",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=PUBLIC_QUERIES_PATH,
        help="Path to public_queries.json",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=["window", "page"],
        default=AGG_SCOPE,
        help=f"Aggregation scope (default from utils: {AGG_SCOPE})",
    )
    parser.add_argument(
        "--pool-k",
        type=int,
        default=PAGE_POOL_K,
        help=f"Mean of top-K chunks per page; 0=all (default: {PAGE_POOL_K})",
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=TOP_CHUNKS,
        help=f"Candidate window size (default from utils: {TOP_CHUNKS})",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["none", "rrf"],
        default=FUSION,
        help=f"Lexical (BM25) fusion method (default from utils: {FUSION})",
    )
    prf_group = parser.add_mutually_exclusive_group()
    prf_group.add_argument(
        "--prf", dest="prf", action="store_true",
        help="Enable PRF query expansion",
    )
    prf_group.add_argument(
        "--no-prf", dest="prf", action="store_false",
        help="Disable PRF query expansion",
    )
    parser.set_defaults(prf=PRF)
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Label for this run (used in results/diag_<tag>.json)",
    )
    parser.add_argument(
        "--no-sanity",
        action="store_true",
        help="Skip sanity check against retrieve.search_batch",
    )
    parser.add_argument(
        "--no-time-run",
        action="store_true",
        help="Skip graded query-phase timing (main.run path)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        type=Path,
        help="Compare two saved diagnostic JSON reports",
    )
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return

    report = run_diagnostics(
        artifacts_dir=args.artifacts_dir,
        queries_path=args.queries_path,
        kfold=args.kfold,
        top_chunks=args.top_chunks,
        scope=args.scope,
        pool_k=args.pool_k,
        fusion=args.fusion,
        prf=args.prf,
        tag=args.tag,
        sanity_check=not args.no_sanity,
        time_run=not args.no_time_run,
    )

    out_path = STUDENT_ROOT / "results" / f"diag_{args.tag}.json"
    save_report(report, out_path)

    print_report_summary(report)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
