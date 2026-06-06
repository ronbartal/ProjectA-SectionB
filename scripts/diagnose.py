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
from utils import ARTIFACTS_DIR, PUBLIC_QUERIES_PATH


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
