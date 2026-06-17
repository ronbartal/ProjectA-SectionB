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
6. **Rerank:** a cross-encoder (MiniLM-L-12) rescores the fused shortlist.

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
| `rerank.py` | Cross-encoder reranker over the fused shortlist (enabled by default). |
| `page_index.py` | Per-page embedding variant (experiment; writes to `artifacts_variants/`, not used by `run()`). |
| `utils.py` | Paths, tunable constants, corpus/query loaders. |
| `eval.py` | **Read-only** NDCG@10 utilities. |
| `scripts/build_index.py` | **Read-only** offline build driver. |
| `scripts/eval_public.py` | **Read-only** self-test → mean NDCG@10 on public queries. |
| `data/public_queries.json` | Labeled public queries. |
| `artifacts/` | Prebuilt index loaded by `run()` (see Artifacts below). |

---

## Quick start

The corpus index is **prebuilt and shipped under `artifacts/`** via **Git LFS** (~2.4 GB).
You must have Git LFS installed and fetch the artifacts after cloning — without them `run()`
only sees ~130-byte pointer stubs and returns nothing. No index rebuild is needed.

**Step 1 — install Git LFS** (skip if `git lfs version` already prints a version).
With sudo: `sudo apt-get install git-lfs`  (or `conda install -c conda-forge git-lfs`).
Without sudo, install the static binary into your home directory:

```bash
LFS_VER=$(curl -s https://api.github.com/repos/git-lfs/git-lfs/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
curl -L "https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VER}/git-lfs-linux-amd64-v${LFS_VER}.tar.gz" | tar -xz
mkdir -p ~/.local/bin && cp git-lfs-*/git-lfs ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"      # add this line to ~/.bashrc to persist
git lfs version
```

**Step 2 — clone, fetch artifacts, install deps, run** (Python 3.10+):

```bash
git clone https://github.com/ronbartal/ProjectA-SectionB
cd ProjectA-SectionB
git lfs install                    # one-time: enable the LFS smudge filter for this repo
git lfs pull                       # REQUIRED: downloads the real artifacts/ (~2.4 GB)
ls -lh artifacts/index.faiss       # sanity: ~764 MB, NOT a ~130-byte stub

# Use an isolated venv: a stale torch in the system/user site-packages can shadow a
# good one and break the import (see the register_fake note below). A venv avoids that.
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect True
python scripts/eval_public.py      # prints mean NDCG@10 on the public queries
```

`eval_public.py` prints, e.g.:
```
public_queries=29
mean_ndcg@10=0.4530
query_phase_time=<GPU-dependent>
```
`query_phase_time` covers loading the artifacts + both models and running all queries; it varies
widely by machine (single-digit seconds on a datacenter GPU, tens of seconds on a laptop GPU) and
must stay under the 60 s budget. On the course VM (Tesla M60) a full cold run — including the
first-time download of both models — measured **~21 s**, comfortably inside the budget.

### Notes

- **Git LFS:** if `run()` raises `FileNotFoundError` for `artifacts/index.faiss` or
  `artifacts/index_vectors.npy`, Git LFS did not fetch them — run `git lfs install && git lfs pull`.
- **Device / pinned torch:** the pipeline requires a **GPU**. The graded run — query embedding
  plus the cross-encoder rerank — targets the 60 s GPU budget; the cross-encoder is far too slow
  on CPU. `requirements.txt` pins `torch==2.3.1+cu121` (via the PyTorch cu121 index) and
  `sentence-transformers==3.3.1` because an unpinned install resolves to a CUDA-13 torch that does
  **not** support the course VM's Tesla M60 and silently falls back to CPU. After install, verify
  with `python -c "import torch; print(torch.cuda.is_available())"` → expect `True`.
- **`module 'torch.library' has no attribute 'register_fake'`:** an old `torch` (< 2.4) in your
  system/user site-packages is being imported alongside a newer `torchvision`. Always run inside
  the `.venv` above (it ignores `~/.local`); if it still appears, `pip uninstall -y torchvision`
  (this text pipeline does not need it).
- **`ensurepip is not available` when creating the venv:** a bare course VM may lack the
  `python3.10-venv` package. Either `sudo apt-get install python3.10-venv`, or create the env
  without pip and bootstrap it manually:
  ```bash
  python3 -m venv --without-pip .venv
  source .venv/bin/activate
  curl -sS https://bootstrap.pypa.io/get-pip.py | python
  pip install -r requirements.txt
  ```

---

## Pipeline overview

The system has two phases — an **offline build** and the **online query** call.

**Offline** — run once on your machine (untimed); writes `artifacts/`:

```
corpus pages
  → chunk into overlapping ~150-word passages   (chunk.py)
  → embed each passage with MiniLM, 384-d        (embed.py)
  → build a FAISS cosine index over the vectors  (index.py)
  → build a BM25 lexical index                   (lexical.py)
  → save the chunk passage texts (for rerank)    (chunk_texts.npy)
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
  → cross-encoder rerank the top-50 fused shortlist (MiniLM-L-12)
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
7. **Rerank** (`RERANK=True`): a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`) rescores
   the top `RERANK_POOL=50` fused pages, scoring each page's best in-window chunk as the passage.
   The final order blends the CE score with the fused rank,
   `RERANK_ALPHA=0.5 * ce_minmax + 0.5 * fused_rank_norm`. Requires a GPU (the cross-encoder is too
   slow on CPU for the 60 s budget).
8. Return the top `K_EVAL=10` page_ids per query.

---

## Artifacts (`artifacts/`, prebuilt and shipped)

`run()` loads these from disk.
Corpus = 27,074 pages → 521,322 chunks, embedding dim 384.

| File | Format | Contents | LFS |
|---|---|---|-----|
| `index_vectors.npy` | `np.save`, float32 `(521322, 384)` | L2-normalized chunk embeddings (row-aligned with the FAISS index). | yes |
| `index.faiss` | FAISS `IndexFlatIP` | Exact cosine index over the chunk vectors. | yes |
| `index_meta.json` | JSON | `page_ids` (len 521322, chunk-row → page_id), `chunk_ids`, `model`, `num_vectors`, `dim=384`, `chunk_words=150`, `chunk_overlap=33`, `prefix_title=true`. | yes |
| `chunk_texts.npy` | `np.save` object `(521322,)` | The passage text per chunk (row-aligned). Used by the cross-encoder reranker. | yes |
| `bm25_tf.npz` | `np.savez` | CSR term-frequency matrix: `data` (f32), `indices` (i32), `indptr` (i32, len 521323), `vocab` (object). | yes |
| `bm25_vocab.json` | JSON `dict[str, float]` | token → IDF (vocab size 319,990). | yes |
| `bm25_meta.json` | JSON | `n_docs`, `vocab_size`, `avg_dl`, `k1=1.5`, `b=0.75`, `min_df=2`, `tokenizer`, chunk params. | no  |


---

## Configuration (`utils.py`)

| Constant | Value | Meaning |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (384-d). |
| `CHUNK_WORDS`, `CHUNK_OVERLAP` | 150, 33 | Chunk window / overlap. |
| `PREFIX_TITLE` | `True` | Prepend page title to each chunk. |
| `TOP_CHUNKS` | 500 | FAISS candidates per query. |
| `AGG_SCOPE`, `PAGE_POOL_K` | `page`, 0 | Page score = mean cosine over all page chunks. |
| `FUSION`, `RRF_K` | `rrf`, 60 | Reciprocal Rank Fusion of dense + BM25. |
| `PRF`, `PRF_ALPHA`, `PRF_TOPN` | `True`, 0.9, 10 | Pseudo-relevance feedback expansion. |
| `RERANK`, `RERANK_POOL`, `RERANK_ALPHA` | `True`, 50, 0.5 | Cross-encoder rerank of the fused shortlist (GPU). |
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Reranker cross-encoder model. |
| `K_EVAL` | 10 | Ranked IDs scored per query. |

---

## Results (development, public queries, NDCG@10)

Each design step was validated on the public queries before adoption:

| Configuration | NDCG@10 |
|---|---|
| Dense chunk baseline | 0.133 |
| + page-scope aggregation | 0.248 |
| + RRF fusion (dense + BM25) | 0.299 |
| + PRF query expansion | 0.311 |
| + cross-encoder rerank, MiniLM-L-6 / pool 20 / alpha 0.3 (GPU) | 0.439 |
| + reranker upgrade: MiniLM-L-12 / pool 50 / alpha 0.5 (**default**) | **0.453** |
