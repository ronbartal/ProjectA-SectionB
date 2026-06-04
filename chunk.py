"""Passage chunking for the corpus.

Each page is split into overlapping word-windows so that a single relevant
sentence is not diluted by the rest of a long article. The page title is
prepended to every chunk because queries usually reference the entity by name
while the supporting fact lives deep in the body.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import CHUNK_OVERLAP, CHUNK_WORDS


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _windows(words: List[str], size: int, overlap: int) -> List[List[str]]:
    """Yield overlapping windows of `size` words advancing by `size - overlap`."""
    if not words:
        return []
    step = max(1, size - overlap)
    if len(words) <= size:
        return [words]
    windows: List[List[str]] = []
    start = 0
    n = len(words)
    while start < n:
        windows.append(words[start : start + size])
        if start + size >= n:
            break
        start += step
    return windows


def chunk_entry(
    record: Dict[str, Any],
    *,
    chunk_words: int = CHUNK_WORDS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split one corpus entry into overlapping passage chunks.

    Long pages yield multiple chunks; short pages yield exactly one. The title
    is prefixed to every chunk so the entity name travels with each passage.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    words = content.split()
    windows = _windows(words, chunk_words, chunk_overlap)
    if not windows:
        # Empty body: still index the title alone so the page is retrievable.
        windows = [[]]

    chunks: List[Chunk] = []
    for chunk_id, window in enumerate(windows):
        body = " ".join(window)
        text = f"{title}. {body}".strip() if title else body
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
