"""BM25 lexical index artifacts for E2 / E4 hybrid retrieval.

Builds per-chunk term-frequency CSR matrix and precomputed IDF at index time.
Yehoraz imports tokenize() and load_bm25() at query time for E4 fusion.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chunk import Chunk
from utils import (
    ARTIFACTS_DIR,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    PREFIX_TITLE,
)

BM25_VOCAB_NAME = "bm25_vocab.json"
BM25_TF_NAME = "bm25_tf.npz"
BM25_META_NAME = "bm25_meta.json"

BM25_K1 = 1.5
BM25_B = 0.75
BM25_MIN_DF = 2
TOKENIZER_ID = "regex_[a-z0-9]+_lower"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization (shared with query-time E4)."""
    return _TOKEN_RE.findall(text.lower())


def _idf(df: int, n_docs: int) -> float:
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


@dataclass(frozen=True)
class Bm25Index:
    """Loaded BM25 artifacts for query-time scoring."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    vocab: np.ndarray
    idf: Dict[str, float]
    avg_dl: float
    n_docs: int
    k1: float
    b: float

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def col_for_token(self, token: str) -> Optional[int]:
        """Return CSR column index for token, or None if OOV."""
        idx = np.searchsorted(self.vocab, token)
        if idx < len(self.vocab) and self.vocab[idx] == token:
            return int(idx)
        return None


def bm25_score_row(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row: int,
    query_terms: Sequence[str],
    idf_map: Dict[str, float],
    vocab: np.ndarray,
    avg_dl: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """Okapi BM25 score for one chunk row (reference scorer for sanity / E4).

    Sums once per distinct query term (no query-TF weighting).
    """
    if avg_dl <= 0:
        return 0.0
    start, end = int(indptr[row]), int(indptr[row + 1])
    row_cols = indices[start:end]
    row_data = data[start:end]
    col_to_tf = {int(c): float(tf) for c, tf in zip(row_cols, row_data)}
    dl = float(sum(row_data))
    score = 0.0
    seen: set[str] = set()
    for term in query_terms:
        if term in seen:
            continue
        seen.add(term)
        idf_val = idf_map.get(term)
        if idf_val is None:
            continue
        idx = np.searchsorted(vocab, term)
        if idx >= len(vocab) or vocab[idx] != term:
            continue
        tf = col_to_tf.get(int(idx), 0.0)
        if tf <= 0:
            continue
        denom = tf + k1 * (1.0 - b + b * dl / avg_dl)
        score += idf_val * (tf * (k1 + 1.0) / denom)
    return score


def build_bm25_artifacts(
    chunks: Sequence[Chunk],
    out_dir: Path,
    *,
    min_df: int = BM25_MIN_DF,
    k1: float = BM25_K1,
    b: float = BM25_B,
    chunk_words: int = CHUNK_WORDS,
    chunk_overlap: int = CHUNK_OVERLAP,
    prefix_title: bool = PREFIX_TITLE,
) -> None:
    """
    Two-pass CPU build: accumulate df, filter vocab, write CSR + IDF + meta.

    Row i of the CSR matrix aligns with chunks[i] and index_meta page_ids[i].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_docs = len(chunks)
    if n_docs == 0:
        raise ValueError("build_bm25_artifacts: empty chunk list")

    # Pass 1: document frequencies and per-chunk token lists (for pass 2).
    df_counts: Dict[str, int] = {}
    chunk_tokens: List[List[str]] = []
    total_dl = 0

    for chunk in chunks:
        tokens = tokenize(chunk.text)
        chunk_tokens.append(tokens)
        total_dl += len(tokens)
        seen: set[str] = set()
        for tok in tokens:
            if tok not in seen:
                df_counts[tok] = df_counts.get(tok, 0) + 1
                seen.add(tok)

    avg_dl = total_dl / n_docs

    # Filter vocabulary and assign column indices (sorted for searchsorted).
    token_list = sorted(t for t, df in df_counts.items() if df >= min_df)
    token_to_col = {t: i for i, t in enumerate(token_list)}
    vocab_size = len(token_list)

    idf_map = {t: _idf(df_counts[t], n_docs) for t in token_list}

    # Pass 2: build CSR.
    data_parts: List[float] = []
    indices_parts: List[int] = []
    indptr = [0]

    for tokens in chunk_tokens:
        tf_counts: Dict[str, int] = {}
        for tok in tokens:
            if tok in token_to_col:
                tf_counts[tok] = tf_counts.get(tok, 0) + 1
        cols = sorted(tf_counts.keys(), key=lambda t: token_to_col[t])
        for tok in cols:
            indices_parts.append(token_to_col[tok])
            data_parts.append(float(tf_counts[tok]))
        indptr.append(len(data_parts))

    data = np.array(data_parts, dtype=np.float32)
    indices = np.array(indices_parts, dtype=np.int32)
    indptr_arr = np.array(indptr, dtype=np.int32)
    vocab_arr = np.array(token_list, dtype=object)

    np.savez(
        out_dir / BM25_TF_NAME,
        data=data,
        indices=indices,
        indptr=indptr_arr,
        vocab=vocab_arr,
    )
    (out_dir / BM25_VOCAB_NAME).write_text(
        json.dumps(idf_map, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "n_docs": n_docs,
        "vocab_size": vocab_size,
        "avg_dl": avg_dl,
        "k1": k1,
        "b": b,
        "min_df": min_df,
        "tokenizer": TOKENIZER_ID,
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap,
        "prefix_title": prefix_title,
    }
    (out_dir / BM25_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"build_bm25: {n_docs} chunks, vocab={vocab_size}, "
        f"avg_dl={avg_dl:.1f}, nnz={len(data)}"
    )


def load_bm25(artifacts_dir: Optional[Path] = None) -> Bm25Index:
    """Load BM25 artifacts from artifacts_dir (default: utils.ARTIFACTS_DIR)."""
    root = artifacts_dir or ARTIFACTS_DIR
    vocab_path = root / BM25_VOCAB_NAME
    tf_path = root / BM25_TF_NAME
    meta_path = root / BM25_META_NAME
    missing = [p.name for p in (vocab_path, tf_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing BM25 artifact(s) in {root}: {', '.join(missing)}. "
            "Build them offline with: python scripts/build_index.py"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idf_map = json.loads(vocab_path.read_text(encoding="utf-8"))
    npz = np.load(tf_path, allow_pickle=True)
    vocab = npz["vocab"]

    if len(vocab) != meta["vocab_size"]:
        raise ValueError(
            f"BM25 vocab size mismatch: npz has {len(vocab)}, "
            f"meta says {meta['vocab_size']}"
        )

    return Bm25Index(
        data=npz["data"],
        indices=npz["indices"],
        indptr=npz["indptr"],
        vocab=vocab,
        idf=idf_map,
        avg_dl=float(meta["avg_dl"]),
        n_docs=int(meta["n_docs"]),
        k1=float(meta.get("k1", BM25_K1)),
        b=float(meta.get("b", BM25_B)),
    )
