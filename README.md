# Section B — Wikipedia-Style Retrieval Pipeline

End-to-end semantic page-level text retrieval over a ~27K-page Wikipedia-style corpus. 

Given a batch of natural-language queries, the graded entry point
`run(queries: list[str]) -> list[list[int]]` returns one ranked list of `page_id`s per query (best
first), scored by **NDCG@10** (binary relevance; only the top 10 per query count). `main.run`
delegates to `retrieve.search_batch`.

Each query is processed in these steps:

1. **Embed:** encode the query with MiniLM.
2. **Retrieve:** FAISS cosine search over the chunk vectors to gather candidate pages.
3. **Expand (PRF):** query expansion based on the top results, then re-search.
4. **Score:** rank the candidates with BM25.
5. **Fuse (RRF):** merge the dense and BM25 rankings.
6. **Rerank (optional):** cross-encoder over the shortlist.

**Presentation video:** _TODO — add the public link here before submission (required)._

---

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Entry point: `run(queries)` (graded) and `build_offline_index()`. |
| `chunk.py` | Page → overlapping word-window passages (title-prefixed). |
| `embed.py` | MiniLM wrapper; lazy model singleton; L2-normalized float32 vectors. |
| `index.py` | Offline chunk index build (embed → FAISS + meta) and load helpers. |
| `lexical.py` | BM25: tokenizer, offline CSR/IDF build, and Okapi scorer. |
| `retrieve.py` | Query-time pipeline (`search_batch`): embed → FAISS → aggregate → RRF → PRF → top-10. |
| `rerank.py` | Optional cross-encoder reranker over the fused shortlist. |
| `page_index.py` | Per-page embedding variant (experiment; not in the active path). |
| `utils.py` | Paths, tunable constants, corpus/query loaders. |
| `eval.py` | **Read-only** NDCG@10 utilities. |
| `scripts/build_index.py` | **Read-only** offline build driver. |
| `scripts/eval_public.py` | **Read-only** self-test → mean NDCG@10 on public queries. |
| `data/public_queries.json` | Labeled public queries. |
| `artifacts/` | Prebuilt index (see Artifacts below). |

---

## Quick start

The corpus index is **prebuilt and shipped under `artifacts/`**.
No need to rebuild it, only import `run()` and load the artifacts from disk. The large artifacts are
stored with **Git LFS**, so you must fetch them after cloning.

```bash
# Requires: Python 3.10+, git-lfs installed (https://git-lfs.com)
git clone https://github.com/ronbartal/ProjectA-SectionB
cd ProjectA-SectionB
git lfs pull                       # REQUIRED: fetches artifacts/index.faiss + index_vectors.npy
pip install -r requirements.txt
python scripts/eval_public.py      # prints mean NDCG@10 on the public queries
```

`eval_public.py` prints, e.g.:
```
public_queries=29
mean_ndcg@10=0.4274
query_phase_time=7.16s
```

### Notes

- **Git LFS:** if `run()` raises `FileNotFoundError` for `artifacts/index.faiss` or
  `artifacts/index_vectors.npy`, Git LFS did not fetch them — run `git lfs install && git lfs pull`.
- **Device:** the embedding model runs on GPU if available, else CPU (auto-detected). The graded
  run assumes a GPU for the <= 60 s budget. On a machine whose PyTorch build doesn't support its
  GPU it falls back to CPU automatically.

---

## Pipeline overview

The system has two phases — an **offline build** and the **online query** call.

**Offline** — run once on your machine (untimed); writes `artifacts/`:

```
corpus pages
  → chunk into overlapping ~150-word passages   (chunk.py)
  → embed each passage with MiniLM, 384-d        (embed.py)
  → build a FAISS cosine index + a BM25 index    (index.py / lexical.py)
  → save to artifacts/
```

**Online** — one `run(queries)` call at grading (≤ 60 s, GPU):

```
query
  → embed with MiniLM
  → FAISS search   →  top-500 candidate chunks
  → PRF: expand the query from the top hits, then search again
  → score each candidate page:  dense cosine  +  BM25
  → fuse the two rankings with RRF
  → (optional) cross-encoder rerank the shortlist
  → return the 10 best page_ids
```

### What `run()` does at query time (defaults from `utils.py`)

1. Embed queries (MiniLM, normalized).
2. **Dense search:** FAISS returns the top `TOP_CHUNKS=500` chunks per query (cosine).
3. **PRF (pseudo-relevance feedback):** treat the top first-pass pages as pseudo-relevant,
   form a page centroid (mean of its chunk vectors), expand the query
   `q' = normalize(0.9*q + 0.1*centroid)` (`PRF_ALPHA=0.9`, `PRF_TOPN=10`), and re-search.
4. **Page aggregation (dense):** each candidate page is scored by the mean cosine over *all*
   its chunks (`AGG_SCOPE="page"`, `PAGE_POOL_K=0`).
5. **Lexical (BM25):** per page, the max BM25 score over its in-window chunks
   (`BM25_PAGE_AGG="max"`, `BM25_SCOPE="window"`; Okapi `k1=1.5, b=0.75`).
6. **Fusion:** Reciprocal Rank Fusion of the dense and BM25 page rankings
   (`FUSION="rrf"`, `RRF_K=60`).
7. Return the top `K_EVAL=10` page_ids per query.
8. **Optional rerank** (`RERANK=False` by default): a cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores the top `RERANK_POOL=20` fused pages and
   blends the CE score with the fused rank. Off by default because it can exceed the 60 s CPU
   budget; enable it (`RERANK=True` in `utils.py`) on a GPU machine.

---

## Artifacts (`artifacts/`, prebuilt and shipped)

`run()` loads these from disk.
Corpus = 27,074 pages → 521,322 chunks, embedding dim 384.

| File | Format | Contents | LFS |
|---|---|---|-----|
| `index_vectors.npy` | `np.save`, float32 `(521322, 384)` | L2-normalized chunk embeddings (row-aligned with the FAISS index). | yes |
| `index.faiss` | FAISS `IndexFlatIP` | Exact cosine index over the chunk vectors. | yes |
| `index_meta.json` | JSON | `page_ids` (len 521322, chunk-row → page_id), `chunk_ids`, `model`, `num_vectors`, `dim=384`, `chunk_words=150`, `chunk_overlap=33`, `prefix_title=true`. | yes |
| `chunk_texts.npy` | `np.save` object `(521322,)` | The passage text per chunk (row-aligned). Used only by the optional reranker. | yes |
| `bm25_tf.npz` | `np.savez` | CSR term-frequency matrix: `data` (f32), `indices` (i32), `indptr` (i32, len 521323), `vocab` (object). | yes |
| `bm25_vocab.json` | JSON `dict[str, float]` | token → IDF (vocab size 319,990). | yes |
| `bm25_meta.json` | JSON | `n_docs`, `vocab_size`, `avg_dl`, `k1=1.5`, `b=0.75`, `min_df=2`, `tokenizer`, chunk params. | no  |
| `page_vectors.npy` | `np.save`, float32 `(27074, 384)` | Per-page vectors (title + first/last sentence). **Built for an experiment; not used by the current `run()`.** | yes |
| `page_meta.json` | JSON | `model`, `num_pages=27074`, `dim=384`, `recipe`, `page_ids`. | no  |


---

## Configuration (`utils.py`)

| Constant | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (384-d). |
| `CHUNK_WORDS`, `CHUNK_OVERLAP` | 150, 33 | Chunk window / overlap. |
| `PREFIX_TITLE` | `True` | Prepend page title to each chunk. |
| `TOP_CHUNKS` | 500 | FAISS candidates per query. |
| `AGG_SCOPE`, `PAGE_POOL_K` | `page`, 0 | Page score = mean cosine over all page chunks. |
| `FUSION`, `RRF_K` | `rrf`, 60 | Reciprocal Rank Fusion of dense + BM25. |
| `PRF`, `PRF_ALPHA`, `PRF_TOPN` | `True`, 0.9, 10 | Pseudo-relevance feedback expansion. |
| `RERANK`, `RERANK_POOL` | `False`, 20 | Optional cross-encoder rerank (enable on GPU). |
| `K_EVAL` | 10 | Ranked IDs scored per query. |

---

## Results (development, public queries, NDCG@10)

Each design step was validated on the public queries before adoption:

| Configuration | NDCG@10 |
|---|---|
| Dense chunk baseline | 0.133 |
| + page-scope aggregation | 0.248 |
| + RRF fusion (dense + BM25) | 0.299 |
| + PRF query expansion (**default**) | **0.311** |
| + cross-encoder rerank (GPU) | 0.439 |
