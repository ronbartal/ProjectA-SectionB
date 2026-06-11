"""Local CPU sanity checks for E5 page index artifacts."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import numpy as np

from embed import embed_queries
from page_index import build_page_index, load_page_index, page_embed_text
from utils import iter_entries


def main() -> None:
    out = STUDENT_ROOT / "artifacts_sanity_e5"
    if out.exists():
        try:
            shutil.rmtree(out)
        except OSError:
            pass
    out.mkdir(parents=True, exist_ok=True)

    records = list(iter_entries())[:200]
    if len(records) < 5:
        print("SKIP: need local corpus under data/Wikipedia Entries/")
        return

    # Spot-check text recipe
    sample = records[0]
    text = page_embed_text(sample)
    assert isinstance(text, str)
    title = str(sample.get("title", "")).strip()
    if title:
        assert title in text or not str(sample.get("content", "")).strip()

    build_page_index(artifacts_dir=out, records=records)
    idx = load_page_index(out)
    assert idx.vectors.shape == (len(records), 384)
    assert len(idx.page_ids) == len(records)

    # Lookup + score
    pid = int(records[0]["page_id"])
    q = embed_queries(["test query"])[0]
    s = idx.score(q, pid)
    assert -1.0 <= s <= 1.0

    norms = np.linalg.norm(idx.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "vectors should be L2-normalized"

    print(f"OK: sanity_e5 passed ({len(records)} pages, sample score={s:.4f})")


if __name__ == "__main__":
    main()
