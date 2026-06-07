"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

# Passage-chunking / retrieval parameters (shared by index build and query time).
# E1 winner: title_150 (150w/33, prefix_title=True) — median ~201 tokens, ~2% truncation.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 33
PREFIX_TITLE = True
# How many chunk hits to pull from FAISS before aggregating to distinct pages.
# In page scope this only selects the CANDIDATE page set (then each candidate is
# rescored over all of its chunks), so it just needs to be large enough to catch
# every relevant page; NDCG plateaus by ~200-500.
TOP_CHUNKS = 500
# E3 aggregation scope:
#   "window" -> a page is scored from only the chunks inside the retrieved window
#   "page"   -> two-stage rerank: the window selects candidate pages, then each
#               candidate is scored over ALL of its chunks (incl. ones not
#               returned). Page scope was a large E3 win (0.2476 vs 0.1332).
AGG_SCOPE = "page"
# A page's score is the MEAN of its top-`PAGE_POOL_K` chunk cosine scores against
# the query. PAGE_POOL_K = 0 means use ALL of the page's chunks (the E3 winner;
# NDCG@10 plateaued once K covered the whole page). With PAGE_POOL_K = 1 this is
# equivalent to classic max-pooling.
PAGE_POOL_K = 0

# E4 lexical fusion: combine the dense page ranking with a BM25 page ranking.
#   FUSION = "rrf"  -> Reciprocal Rank Fusion (scale-free, robust): the winner.
#   FUSION = "none" -> dense-only (E3 behaviour).
# BM25 page score = `BM25_PAGE_AGG` over the page's chunk BM25 scores, computed
# only over chunks inside the dense window (BM25_SCOPE="window", ~15x cheaper
# than scoring every page chunk for a statistically-equal result). RRF_K is the
# standard rank-fusion constant (insensitive between ~10-100 here).
FUSION = "rrf"
RRF_K = 60
BM25_PAGE_AGG = "max"   # "max" (best passage) beat "sum"/"mean" in the E4 sweep
BM25_SCOPE = "window"   # "window" (fast) or "page" (all chunks, ~+0.01, slow)

# Pseudo-Relevance Feedback (Rocchio dense query expansion), query-side only.
#   PRF = True -> two-pass: a first dense pass picks the top-`PRF_TOPN` pseudo-
#   relevant PAGES, each represented by its mean chunk vector (PRF_PAGE_REPR);
#   their centroid expands the query  q' = norm(alpha*q + (1-alpha)*centroid),
#   and the second pass ranks with q'. BM25 still uses the original query terms.
# Page-level feedback de-duplicates so one long page can't dominate the centroid
# (chunk-level feedback caused drift and lost vs no-PRF, so it isn't exposed).
# alpha=0.9 (light touch) + N=10 was the stable sweep winner (0.3113 vs 0.2993).
PRF = True
PRF_ALPHA = 0.9
PRF_TOPN = 10
PRF_PAGE_REPR = "mean"  # "mean" (E3-consistent) or "best" (single best chunk)
# Graded query phase budget: one run(queries) call (embed + retrieve), GPU at grading.
GRADING_QUERY_TIME_LIMIT_S = 60.0


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory."""
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected student/data/Wikipedia Entries/ with one JSON file per page."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
