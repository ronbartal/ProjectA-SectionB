"""Build per-page embeddings (E5) from the full corpus — offline, not timed.

Writes artifacts_variants/page_vectors.npy and artifacts_variants/page_meta.json.
These are experiment-only outputs (run() does not load them), so they live under
artifacts_variants/, not the deployed artifacts/.
Chunk-config independent: run once per corpus, reuse across variant indices.

Usage (VM with corpus):
    python scripts/build_page_index.py
    python scripts/build_page_index.py --artifacts-dir artifacts_variants
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from page_index import build_page_index
from utils import ARTIFACTS_VARIANTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Build page-level MiniLM index (E5).")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_VARIANTS_DIR,
        help="Output directory (default: artifacts_variants/)",
    )
    args = parser.parse_args()
    build_page_index(artifacts_dir=args.artifacts_dir)
    print(f"Done. Page index written under {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
