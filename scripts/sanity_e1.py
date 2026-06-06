"""Local E1 sanity checks (no GPU, no full corpus)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk import chunk_entry
from diagnostics import run_diagnostics
from index import build_index
from utils import ENTRIES_DIR


def _assert_chunk_logic() -> None:
    long_page = {
        "page_id": 1,
        "title": "Test Entity",
        "content": " ".join(f"word{i}" for i in range(500)),
    }
    short_page = {"page_id": 2, "title": "Stub", "content": "one two three"}

    with_title = chunk_entry(long_page, prefix_title=True)[0].text
    no_title = chunk_entry(long_page, prefix_title=False)[0].text
    assert with_title.startswith("Test Entity.")
    assert not no_title.startswith("Test Entity")
    assert no_title.startswith("word")

    n180 = len(chunk_entry(long_page, chunk_words=180, chunk_overlap=40))
    n120 = len(chunk_entry(long_page, chunk_words=120, chunk_overlap=30))
    assert n120 > n180, f"expected more chunks at 120w, got {n120} vs {n180}"

    assert len(chunk_entry(short_page)) == 1
    print("chunk logic: OK")


def _assert_end_to_end() -> None:
    src = ENTRIES_DIR
    if not src.is_dir():
        print("end-to-end: SKIP (no local corpus at data/Wikipedia Entries/)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        mini = Path(tmp) / "mini"
        mini.mkdir()
        for i, path in enumerate(sorted(src.glob("*.json"))[:50]):
            shutil.copy2(path, mini / path.name)
            if i >= 49:
                break

        out = Path(tmp) / "artifacts"
        build_index(
            entries_dir=mini,
            artifacts_dir=out,
            chunk_words=120,
            chunk_overlap=30,
            prefix_title=False,
        )
        meta = json.loads((out / "index_meta.json").read_text(encoding="utf-8"))
        assert meta["chunk_words"] == 120
        assert meta["chunk_overlap"] == 30
        assert meta["prefix_title"] is False

        report = run_diagnostics(artifacts_dir=out, tag="sanity_mini")
        assert report.sanity["passed"], report.sanity.get("mismatches", [])
        print("end-to-end mini build + diagnostics sanity: OK")


def main() -> None:
    _assert_chunk_logic()
    _assert_end_to_end()
    print("\nAll E1 sanity checks passed.")


if __name__ == "__main__":
    main()
