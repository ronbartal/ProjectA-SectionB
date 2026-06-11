"""Per-page MiniLM embeddings for E5 fusion (title + first + last sentence).

Chunk-config independent: build once from the raw corpus, consume at query time
via page_id lookup (no second FAISS index needed).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from embed import embed_texts
from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME, iter_entries

PAGE_VECTORS_NAME = "page_vectors.npy"
PAGE_META_NAME = "page_meta.json"
PAGE_RECIPE_ID = "title_first_last_sentence"

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split body text on sentence boundaries (. ! ? followed by whitespace)."""
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_END.split(text) if p.strip()]


def page_embed_text(record: Dict[str, Any]) -> str:
    """
    Build the page-level embed string: title, first sentence, last sentence.

    Order matches truncation priority on MiniLM (256 tokens): title survives,
    then first sentence, then last sentence if still within the cap.
    """
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    sentences = split_sentences(content)
    first = sentences[0] if sentences else ""
    last = sentences[-1] if len(sentences) > 1 else ""

    parts: List[str] = []
    if title:
        parts.append(title)
    if first:
        parts.append(first)
    if last and last != first:
        parts.append(last)

    if not parts:
        return ""
    return ". ".join(parts)


def build_page_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    records: Optional[List[Dict[str, Any]]] = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Embed one vector per corpus page and persist page_vectors.npy + page_meta.json.

    Returns (vectors, page_ids) with row i aligned to page_ids[i] (sorted by page_id).
    """
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = list(iter_entries(entries_dir))
    records = list(records)
    records.sort(key=lambda r: int(r["page_id"]))

    page_ids = [int(r["page_id"]) for r in records]
    texts = [page_embed_text(r) for r in records]

    empty_text = sum(1 for t in texts if not t.strip())
    vectors = embed_texts(texts)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    np.save(out_dir / PAGE_VECTORS_NAME, vectors)
    meta = {
        "model": EMBEDDING_MODEL_NAME,
        "num_pages": len(page_ids),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0,
        "recipe": PAGE_RECIPE_ID,
        "text_order": ["title", "first_sentence", "last_sentence"],
        "page_ids": page_ids,
        "empty_text_pages": empty_text,
    }
    (out_dir / PAGE_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(
        f"build_page_index: {len(page_ids)} pages -> {PAGE_VECTORS_NAME} "
        f"({empty_text} empty-text pages)"
    )
    return vectors, page_ids


@dataclass(frozen=True)
class PageIndex:
    """Loaded page-level embeddings for query-time lookup by page_id."""

    vectors: np.ndarray
    page_ids: List[int]
    page_id_to_row: Dict[int, int]

    def score(self, query_vector: np.ndarray, page_id: int) -> float:
        """Cosine similarity (dot product) between query and a page vector."""
        row = self.page_id_to_row.get(int(page_id))
        if row is None:
            return 0.0
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        return float(np.dot(q, self.vectors[row]))


def load_page_index(artifacts_dir: Optional[Path] = None) -> PageIndex:
    """Load page_vectors.npy and page_meta.json from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors_path = root / PAGE_VECTORS_NAME
    meta_path = root / PAGE_META_NAME
    missing = [p.name for p in (vectors_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing page index artifact(s) in {root}: {', '.join(missing)}. "
            "Build offline with: python scripts/build_page_index.py"
        )

    vectors = np.load(vectors_path)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    if vectors.shape[0] != len(page_ids):
        raise ValueError(
            f"page_vectors rows ({vectors.shape[0]}) != page_ids "
            f"({len(page_ids)}) in {meta_path}"
        )

    page_id_to_row = {pid: i for i, pid in enumerate(page_ids)}
    return PageIndex(
        vectors=vectors,
        page_ids=page_ids,
        page_id_to_row=page_id_to_row,
    )


def page_scores_for_ids(
    page_index: PageIndex,
    query_vector: np.ndarray,
    page_ids: Sequence[int],
) -> Dict[int, float]:
    """Batch helper: cosine scores for a set of page_ids (deduped)."""
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    out: Dict[int, float] = {}
    for pid in page_ids:
        pid = int(pid)
        if pid in out:
            continue
        row = page_index.page_id_to_row.get(pid)
        if row is not None:
            out[pid] = float(np.dot(q, page_index.vectors[row]))
    return out
