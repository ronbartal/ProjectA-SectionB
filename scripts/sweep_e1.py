"""E1 chunking sweep: build + diagnose three configs sequentially.

Runs on the VM with the full corpus. Each config writes to
artifacts_sweep/<tag>/ and results/diag_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from diagnostics import run_diagnostics, save_report
from index import build_index
from utils import STUDENT_ROOT as ROOT

SWEEP_DIR = ROOT / "artifacts_sweep"
RESULTS_DIR = ROOT / "results"
BASELINE_PATH = RESULTS_DIR / "diag_baseline.json"


@dataclass(frozen=True)
class SweepConfig:
    tag: str
    chunk_words: int
    chunk_overlap: int
    prefix_title: bool


E1_CONFIGS = (
    SweepConfig("notitle_180", chunk_words=180, chunk_overlap=40, prefix_title=False),
    SweepConfig("title_120", chunk_words=120, chunk_overlap=30, prefix_title=True),
    SweepConfig("notitle_120", chunk_words=120, chunk_overlap=30, prefix_title=False),
    SweepConfig("title_150", chunk_words=150, chunk_overlap=33, prefix_title=True),
)


def _load_baseline_summary() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        "tag": "title_180 (baseline)",
        "mean_ndcg": data["summary"]["mean_ndcg_at_10"],
        "kfold_mean": data["kfold"]["mean_ndcg_at_10"],
        "kfold_std": data["kfold"]["std_ndcg_at_10"],
        "recall_at_10": data["summary"]["mean_recall_at_10"],
    }


def _print_table(rows: list[dict]) -> None:
    print("\n=== E1 sweep summary (2x2 vs baseline) ===")
    print(
        f"{'tag':<16} {'words':>5} {'ovlp':>4} {'title':>5} "
        f"{'ndcg@10':>8} {'kfold':>8} {'+/-':>6} {'rec@10':>7}"
    )
    for r in rows:
        print(
            f"{r['tag']:<16} {r['chunk_words']:>5} {r['chunk_overlap']:>4} "
            f"{str(r['prefix_title']):>5} "
            f"{r['mean_ndcg']:>8.4f} {r['kfold_mean']:>8.4f} "
            f"{r['kfold_std']:>6.4f} {r['recall_at_10']:>7.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 chunking sweep (3 configs).")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip build_index; only run diagnostics on existing artifacts_sweep/",
    )
    parser.add_argument(
        "--only",
        metavar="TAG",
        choices=[c.tag for c in E1_CONFIGS],
        help="Run a single config (e.g. title_150, notitle_180, title_120, notitle_120)",
    )
    args = parser.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [c for c in E1_CONFIGS if c.tag == args.only] if args.only else list(E1_CONFIGS)

    rows: list[dict] = []
    if not args.only:
        baseline = _load_baseline_summary()
        if baseline:
            rows.append(
                {
                    "tag": baseline["tag"],
                    "chunk_words": 180,
                    "chunk_overlap": 40,
                    "prefix_title": True,
                    **baseline,
                }
            )

    for cfg in configs:
        out_dir = SWEEP_DIR / cfg.tag
        print(f"\n--- {cfg.tag} ---")
        t0 = time.perf_counter()
        if not args.skip_build:
            build_index(
                artifacts_dir=out_dir,
                chunk_words=cfg.chunk_words,
                chunk_overlap=cfg.chunk_overlap,
                prefix_title=cfg.prefix_title,
            )
        report = run_diagnostics(artifacts_dir=out_dir, tag=cfg.tag)
        save_report(report, RESULTS_DIR / f"diag_{cfg.tag}.json")
        elapsed = time.perf_counter() - t0
        qt = report.query_timing.get("query_phase_time_s", float("nan"))
        budget = report.query_timing.get("within_budget")
        print(
            f"mean_ndcg@10={report.summary['mean_ndcg_at_10']:.4f}  "
            f"kfold={report.kfold['mean_ndcg_at_10']:.4f} "
            f"+/- {report.kfold['std_ndcg_at_10']:.4f}  "
            f"recall@10={report.summary['mean_recall_at_10']:.4f}  "
            f"run_time={qt:.2f}s{' OK' if budget else ' OVER'}  "
            f"({elapsed:.0f}s total)"
        )
        rows.append(
            {
                "tag": cfg.tag,
                "chunk_words": cfg.chunk_words,
                "chunk_overlap": cfg.chunk_overlap,
                "prefix_title": cfg.prefix_title,
                "mean_ndcg": report.summary["mean_ndcg_at_10"],
                "kfold_mean": report.kfold["mean_ndcg_at_10"],
                "kfold_std": report.kfold["std_ndcg_at_10"],
                "recall_at_10": report.summary["mean_recall_at_10"],
            }
        )

    _print_table(rows)
    print(f"\nResults saved under {RESULTS_DIR}/diag_<tag>.json")


if __name__ == "__main__":
    main()
