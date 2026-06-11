# Section B — Workflow & Task Division

> **Team:** Ron (indexing / corpus side) · Yehoraz (query / ranking side)
>
> **Goal:** Maximize mean NDCG@10 on 50 hidden queries within a 1-week sprint.
>
> **Last updated:** 2026-06-07

---

## 1  Project overview

A semantic retrieval pipeline over **~27 074 Wikipedia pages** (full corpus on Ron's VM; verified in `results/diag_baseline.json`).
The grader calls `main.run(queries)` once with all evaluation queries.
Only the first 10 page_ids per query are scored (NDCG@10, binary relevance).

### Current status (2026-06-07, branch `yehoraz_develop`)

| Layer | Status | NDCG@10 |
|-------|--------|---------|
| Baseline (E1+E2 dense only) | locked on `main` artifacts | 0.1332 |
| **E3** page-scope mean-all | done, not merged | 0.2476 |
| **E4** BM25 + dense RRF | done, not merged | 0.2993 |
| **PRF** Rocchio query expansion | done, not merged | **0.3113** ← **current production config on `yehoraz_develop`** |
| **E6** cross-encoder rerank | A/B tested, **blocked** — needs `chunk_texts.npy` from Ron + GPU timing check | 0.3284 (proxy text; not shippable) |
| **E5** title-vector fusion | blocked on Ron artifact | — |

**Next gate for score improvement:** Ron ships `chunk_texts.npy` (§4.3) → Yehoraz reruns A/B with real text → if stable + within 60s, merge rerank into `retrieve.py`.

### Pipeline stages

```
[OFFLINE — not timed, Ron's VM]
  corpus JSON → chunk → embed (MiniLM) → FAISS + BM25 CSR + chunk_texts
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Ron owns: chunk.py, embed.py, index.py, lexical.py, build_index.py

[QUERY TIME — timed, grader GPU, 60s budget]
  queries
    → MiniLM embed (+ optional PRF expand)          [Yehoraz: retrieve.py]
    → FAISS top-500 chunks → page candidates
    → page-scope dense mean-all + BM25-max + RRF
    → (optional) cross-encoder rerank top-M pages   [blocked until §4.3]
    → top-10 page_ids
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Yehoraz owns: retrieve.py (+ rerank.py when enabled), utils.py constants
```

### Key constraints

- **Indexing embedding model is fixed:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Used for FAISS / dense retrieval only.
- **Additional pretrained models are allowed for reranking only** (e.g. cross-encoder). They must not replace MiniLM for indexing or first-stage retrieval.
- Allowed deps: `numpy`, `sentence-transformers`, `faiss-cpu` (see `requirements.txt`). Cross-encoders load via `sentence-transformers.CrossEncoder`.
- Staff do **not** rebuild the index — committed `artifacts/` are graded as-is.
- `eval.py` is **read-only** (do not modify).
- Query-phase budget: **60s** (`utils.GRADING_QUERY_TIME_LIMIT_S`). Current stack ~15–25s CPU; CE rerank adds ~47s CPU (pool=20) — likely needs grader GPU.

---

## 2  Repository layout

```
├── main.py              # Entry point: run(queries), build_offline_index()
├── chunk.py             # Passage chunking                    ← Ron
├── embed.py             # MiniLM encode wrapper (index only)  ← Ron
├── index.py             # Build + load FAISS/numpy index      ← Ron
├── lexical.py           # BM25 build + load + tokenize        ← Ron (build), Yehoraz (query)
├── retrieve.py          # Query-time search + fusion + PRF    ← Yehoraz
├── diagnostics.py       # Shared eval harness (set-aware)     ← shared (both use)
├── eval.py              # NDCG@10 evaluation (READ-ONLY)
├── utils.py             # Shared constants & helpers          ← shared
├── scripts/
│   ├── build_index.py   # Offline build driver                ← Ron
│   ├── eval_public.py   # Public self-test (canonical score)
│   ├── diagnose.py      # Diagnostic harness CLI              ← shared
│   └── sweep_rerank_ab.py  # A/B: baseline vs CE rerank     ← Yehoraz (exploration)
├── artifacts/           # Committed index files (Ron builds, never Yehoraz)
│   ├── index_vectors.npy   # float32 (n_chunks, 384)
│   ├── index_meta.json     # page_ids, chunk_ids, build params
│   ├── index.faiss         # FAISS IndexFlatIP
│   ├── bm25_tf.npz         # CSR term-frequency per chunk (E2)
│   ├── bm25_vocab.json     # token → IDF
│   ├── bm25_meta.json      # BM25 corpus stats
│   └── chunk_texts.npy     # ★ REQUIRED for E6 rerank — NOT YET BUILT (§4.3)
├── data/
│   ├── public_queries.json   # 50 labelled queries (tracked)
│   └── Wikipedia Entries/    # Raw corpus (gitignored — Ron's VM only)
├── results/             # Diagnostic JSON outputs (gitignored)
├── requirements.txt
└── WORKFLOW.md          # ← this file
```

---

## 3  Ownership & responsibilities

### 3.1  Ron — indexing / corpus side

**Files owned:** `chunk.py`, `embed.py`, `index.py`, `scripts/build_index.py`

**Responsibilities:**
- All offline index builds run on Ron's VM (only machine with corpus + GPU).
- Commit `artifacts/` to `main` after every accepted improvement.
- **Ron is the sole committer of artifact binaries** — prevents divergent blobs.
- Verify every merge passes `eval_public.py` on a fresh clone (no rebuild).

**Experiments (priority order):**

| ID | Experiment | Files touched | Status |
|----|-----------|---------------|--------|
| E1 | Chunking sweep (2×2 + `title_150`). See decision log §8. | `chunk.py`, `utils.py` | **done** — `title_150` locked |
| E2 | Lexical index — BM25 artifacts. | `lexical.py`, `index.py` | **done** — production `artifacts/` (Jun 6) |
| **E6** | **Chunk text artifact for cross-encoder reranking.** Save passage strings at build time. **Ron is the blocker.** See §4.3 and §8.4. | `index.py`, `scripts/build_index.py` | **TODO — highest Ron priority** |
| E5 | Per-page title embedding artifact (title-vector fusion). | `index.py` (Ron) + `retrieve.py` (Yehoraz) | deferred until after E6 or in parallel |

**E1 solo arms (fully in Ron's scope — change chunk text/size, rebuild, measure with `diagnostics.py`):**

| Arm | Change | Hypothesis |
|-----|--------|-----------|
| A | No title prefix (`body` only) | Baseline: does the title actually help or hurt the chunk embedding? |
| B | Title prefix `f"{title}. {body}"` (current) | Status quo; entity anchoring across coref-heavy passages. |
| D | Smaller `CHUNK_WORDS` (100/120) x title on/off | Eliminates truncation + concentrates the gold sentence; title matters more when body is short. Preview token cost with `scripts/audit_tokens.py --chunk-words N` (no GPU). |

Front placement of the title is kept deliberately: truncation cuts the tail, so a prefix survives the 256-token cap while a suffix would not.

### 3.2  Yehoraz — query / ranking side

**Files owned:** `retrieve.py`, `utils.py` (query-time constants), `rerank.py` (when E6 lands)

**Responsibilities:**
- All query-time ranking logic (retrieval, aggregation, fusion, PRF, reranking).
- Keep `diagnostics.py` / `scripts/diagnose.py` in sync with `retrieve.py` so sanity checks pass.
- Keep query-phase latency within grading budget (Ron validates absolute timing on VM).
- Never rebuild or commit `artifacts/` — treat them as read-only inputs.

**Experiments (priority order):**

| ID | Experiment | Files touched | Status |
|----|-----------|---------------|--------|
| E3 | Page-scope mean-all aggregation; `TOP_CHUNKS` 500. | `retrieve.py`, `utils.py`, `diagnostics.py` | **done** — 0.2476 |
| E4 | BM25 + dense RRF fusion. | `retrieve.py`, `utils.py`, `diagnostics.py` | **done** — 0.2993 |
| PRF | Rocchio page-level query expansion (query-side, while E5 blocked). | `retrieve.py`, `utils.py`, `diagnostics.py` | **done** — **0.3113 (current)** |
| RRF-K | Shared/asymmetric K tuning. | analysis only | **done** — K=60 validated, no change |
| **E6** | Cross-encoder rerank (Option A) on RRF shortlist. | `rerank.py`, `retrieve.py`, `diagnostics.py` | **A/B done; blocked on §4.3** |
| E5 | Title-vector fusion — Ron artifact + blend at query time. | `retrieve.py` (Yehoraz) | blocked on Ron |
| follow-up | BM25 candidate generation (union BM25 top-pages with dense pool). | `retrieve.py` | exploratory — surfaced by RRF-K analysis |

**Current `utils.py` production constants (on `yehoraz_develop`, not merged to `main`):**
```python
TOP_CHUNKS = 500
AGG_SCOPE = "page"
PAGE_POOL_K = 0          # mean of ALL page chunks
FUSION = "rrf"
RRF_K = 60
BM25_PAGE_AGG = "max"
BM25_SCOPE = "window"
PRF = True
PRF_ALPHA = 0.9
PRF_TOPN = 10
PRF_PAGE_REPR = "mean"
# E6 (not yet enabled):
# RERANK = False
# RERANK_POOL = 20
# RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### 3.3  Shared — both on Day 1

- **Eval harness (built): `diagnostics.py` + `scripts/diagnose.py`.** Single internal evaluation tool for BOTH teammates — set-aware NDCG@10 (matches `eval.py`), recall@{10,50,100}, MRR, per-relevant-page ranks, chunk-level diagnostics, per-bucket (by n_relevant), 5-fold CV, data-quality checks, **sanity check** (harness top-10 == `retrieve.search_batch`), and **query-phase timing**. CLI flags mirror `utils.py`: `--scope`, `--pool-k`, `--top-chunks`, `--fusion`, `--prf`/`--no-prf`. Run `python scripts/diagnose.py --tag <name>`; compare with `--compare A.json B.json`. Results in `results/` (gitignored).
- **Evaluation discipline:** use 5-fold CV mean ± std (not a single number) — 50 queries are noisy. Use **split-half held-out tests** before adopting new fusion/rerank knobs (see RRF-K and E6 A/B lessons in §8). Isolate *which side* can move a metric: gold-chunk rank low → Ron (chunk/embedding); gold-chunk high but page rank low → Yehoraz (aggregation/fusion/rerank).
- **Artifact contracts:** E2 lexical (§4.2, done), E6 chunk text (§4.3, **pending**), E5 title-vector (TBD).
- **Additional LLM rule:** pretrained models beyond MiniLM are allowed **only for reranking** (E6 cross-encoder). MiniLM remains the sole indexing/first-stage retrieval encoder.

---

## 4  Artifact contract (interface between Ron & Yehoraz)

### 4.1  Existing artifacts (dense retrieval)

| File | Shape / format | Producer | Consumer |
|------|---------------|----------|----------|
| `index_vectors.npy` | `float32 (n_chunks, 384)` L2-normalized | `index.build_index()` | `index.load_index()` → `retrieve.py` |
| `index_meta.json` | `{"page_ids": [...], "chunk_ids": [...], "model": str, ...}` | `index.build_index()` | `index.load_index()` → `retrieve.py` |
| `index.faiss` | FAISS `IndexFlatIP` over chunk vectors | `index.build_index()` | `index.load_index()` → `retrieve.py` |

### 4.2  Lexical / BM25 artifacts (E2 → E4)

> **Status:** **built and verified** (2026-06-06). Ron's E2 scope complete; Yehoraz unblocked for E4.
> Chunk config locked: `CHUNK_WORDS=150`, `CHUNK_OVERLAP=33`, `PREFIX_TITLE=True`.
> **Dense-only NDCG unchanged by BM25 files** until E4 fusion is wired in `retrieve.py`.

| File | Format | Contents |
|------|--------|----------|
| `bm25_vocab.json` | `{"token": idf_float, ...}` | Precomputed IDF per in-vocab token |
| `bm25_tf.npz` | CSR arrays + `vocab` | `data`, `indices`, `indptr` (scipy CSR layout), `vocab` (object array: `vocab[col]` = token) |
| `bm25_meta.json` | JSON object | Corpus stats + BM25 hyperparameters (see schema below) |

**Definitions:**
- **Document unit = one chunk.** CSR row `i` aligns 1:1 with `index_meta.json` `page_ids[i]` / `chunk_ids[i]`.
- **`n_docs`** = number of chunks (not pages).
- **`avg_dl`** = mean token count per chunk (build-time tokenizer).
- **IDF:** `log((N - df + 0.5) / (df + 0.5) + 1)` where `N = n_docs`, `df` = chunks containing the term.
- **Vocab pruning:** terms with `df < min_df` (default 2) are dropped; `min_df` stored in meta.

**Tokenization (Ron and Yehoraz must match):**
- Import `tokenize` from `lexical.py`: `re.findall(r"[a-z0-9]+", text.lower())`.
- No stemming or stopwords in v1. Meta field `tokenizer`: `"regex_[a-z0-9]+_lower"`.

**`bm25_meta.json` schema:**
```json
{
  "n_docs": 521322,
  "vocab_size": 123456,
  "avg_dl": 201.0,
  "k1": 1.5,
  "b": 0.75,
  "min_df": 2,
  "tokenizer": "regex_[a-z0-9]+_lower",
  "chunk_words": 150,
  "chunk_overlap": 33,
  "prefix_title": true
}
```

**Okapi BM25 term score** (reference in `lexical.bm25_score_row`):
```
score(q, d) = sum over distinct t in q: IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avg_dl))
```
where `tf` = term freq in chunk, `dl` = chunk length (token count). Each query term counts once (classic Okapi; no query-TF multiplier).

**E4 integration pattern (Yehoraz — implemented 2026-06-07, query time):**

1. **Dense retrieve:** FAISS → top `TOP_CHUNKS`(=500) chunks → distinct **candidate pages**.
2. **Dense page score (E3):** each candidate scored by **mean cosine of ALL its chunks** (page-scope, `PAGE_POOL_K=0`).
3. **BM25 page score:** `BM25_PAGE_AGG="max"` over each candidate's chunks **inside the dense window only** (`BM25_SCOPE="window"`).
4. **Fuse at page level:** Reciprocal Rank Fusion (`RRF_K=60`) of dense and BM25 **page rankings** — not chunk-level fusion.
5. **PRF (optional, on):** two-pass dense query expansion before step 1–2; BM25 still uses original query terms.

```python
from lexical import load_bm25, tokenize, bm25_score_row
bm25 = load_bm25()
q_terms = tokenize(query)  # original query; not PRF-expanded
# BM25 per chunk row, aggregated to page via max over in-window chunks
# Dense + BM25 page rankings fused via RRF in retrieve._rrf_fuse()
```

**Load helpers:**
- `lexical.load_bm25(artifacts_dir)` → `Bm25Index` dataclass
- `index.load_bm25_index(artifacts_dir)` — thin wrapper

### 4.2.1  Page-level embeddings (E5)

> **Status:** build script ready (`scripts/build_page_index.py`). **Chunk-config independent** — one copy in `artifacts/` shared across all variant dirs.
> **Build on VM only** (needs full corpus). No second FAISS index; query-time lookup by `page_id`.

| File | Format | Contents |
|------|--------|----------|
| `page_vectors.npy` | `float32 (n_pages, 384)` L2-normalized | MiniLM embeddings |
| `page_meta.json` | JSON | `page_ids` (sorted), `recipe`, `model`, `dim`, `num_pages` |

**Embed text recipe** (`page_index.page_embed_text`): `title . first_sentence . last_sentence` (last omitted if same as first). Built by `page_index.build_page_index()` / `python scripts/build_page_index.py`.

**Query-time (Yehoraz — after E4 or with fusion):**
```python
from index import load_page_index
pages = load_page_index(artifacts_dir)  # or artifacts_dir=Path(...)
page_score = pages.score(query_vector, page_id)  # dot product, vectors normalized
```

Optional neighbor + page fusion (see session notes): `s* = a*s(chunk) + b*s(prev) + c*s(next) + d*page_score`, then max-pool to pages.

> **Important:** If this format changes, Ron rebuilds on the VM, commits, and notifies Yehoraz to `git pull`. Batch format changes to minimize round-trips.

### 4.3  Chunk text artifact (E6 → cross-encoder reranking)

> **Status:** **NOT BUILT.** This is the **#1 blocker** for enabling reranking. Yehoraz A/B test (2026-06-07) used a BM25-token proxy; results are indicative only (+0.017 NDCG) but not shippable.

#### Why the existing index is not enough

The current artifacts support **bi-encoder** retrieval (MiniLM): query and chunks are embedded into vectors; query time compares numbers. **No passage text is needed** for FAISS, dense page scoring, BM25, PRF, or RRF.

A **cross-encoder reranker** scores `(query, passage)` pairs by running both strings through a transformer jointly. It cannot consume:
- `index_vectors.npy` (384-dim floats — not readable text)
- `bm25_tf.npz` (token counts without word order — bag-of-words, not passages)

The chunk **text** already exists at build time (`c.text for c in chunks` in `index.build_index()`) but is **discarded** after embedding. E6 only requires **persisting** those same strings.

#### Artifact spec

| File | Format | Producer | Consumer |
|------|--------|----------|----------|
| `chunk_texts.npy` | `numpy.ndarray` dtype `object`, shape `(n_chunks,)`, `chunk_texts[i]` is `str` | `index.build_index()` | `index.load_chunk_texts()` → `retrieve.py` / `rerank.py` |

**Alignment contract (must hold):**
```
row i of chunk_texts.npy
  == row i of index_vectors.npy
  == row i of bm25_tf CSR
  == page_ids[i] / chunk_ids[i] in index_meta.json
```

**Text content:** exact string passed to `embed_texts()` at build time (title-prefixed passage per `chunk.py` / `PREFIX_TITLE=True`). Do not re-chunk or re-tokenize differently.

**Size estimate:** ~521k chunks × ~150 words × ~6 chars ≈ **400–600 MB** uncompressed `.npy`; consider `numpy.savez_compressed` if git/LFS is tight. No re-embedding required — additive artifact only.

#### Ron implementation checklist (E6)

1. **In `index.build_index()`** — after `texts = [c.text for c in chunks]`:
   ```python
   CHUNK_TEXTS_NAME = "chunk_texts.npy"
   np.save(out_dir / CHUNK_TEXTS_NAME, np.asarray(texts, dtype=object))
   ```
2. **In `index.py`** — add loader:
   ```python
   def load_chunk_texts(artifacts_dir=None) -> np.ndarray:
       path = (artifacts_dir or ARTIFACTS_DIR) / "chunk_texts.npy"
       if not path.exists():
           raise FileNotFoundError(f"Missing {path.name} — rebuild with build_index.py")
       return np.load(path, allow_pickle=True)
   ```
3. **Rebuild on VM** from existing `title_150` chunk config (same `CHUNK_WORDS/OVERLAP/PREFIX_TITLE` as current `artifacts/`). Dense vectors and FAISS **unchanged** if chunking params unchanged — only add the new file. If unsure, full rebuild is safest.
4. **Verify alignment:** `len(chunk_texts) == meta["num_vectors"] == bm25.n_docs`.
5. **Commit** `chunk_texts.npy` to git (LFS if needed) on `main` / notify Yehoraz to `git pull`.
6. **Optional but recommended:** on VM with GPU, run `python scripts/sweep_rerank_ab.py` and confirm total query path < 60s.

**What Ron does NOT need to do:**
- Change `embed.py` or the MiniLM model
- Re-embed the corpus (unless chunking params change)
- Touch `retrieve.py`

#### Yehoraz integration pattern (after Ron ships artifact)

Query-time pipeline (current + E6):

1. Stages 1–3 unchanged: PRF → FAISS → page-scope dense + BM25/RRF → fused page list.
2. **Rerank (new):** take top `RERANK_POOL` fused pages (sweep showed 20 best); for each page, score `(query, best_dense_chunk_text)` with cross-encoder; final order = CE score (Option A).
3. Return top 10.

**Passage per page:** best-matching chunk **in the dense window** (highest cosine vs PRF-expanded query) — not full page text, not BM25 proxy.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (start); optional `BAAI/bge-reranker-base` if GPU budget allows. Loaded only in `rerank.py` — never used for indexing.

**Merge criteria (Yehoraz, before enabling in production):**
- `python scripts/sweep_rerank_ab.py` with real `chunk_texts.npy`: k-fold gain ≥ +0.005 **and** split-half stable (both halves improve or within noise).
- `diagnose.py` sanity PASSED with `--rerank`.
- `eval_public.py` `query_phase_time` < 60s on grading hardware.

#### E6 A/B results so far (proxy text — do not merge)

| Variant | NDCG@10 | 5-fold ± | Notes |
|---------|---------|----------|-------|
| A — baseline (no rerank) | 0.3113 | ±0.083 | matches live pipeline |
| B — CE rerank pool=20 | 0.3284 | ±0.085 | +0.017 full-set |
| B — CE rerank pool=30/40/50 | 0.3266–0.3277 | ±0.087 | diminishing returns |

**Split-half (pool=20):** half-A +0.068, half-B −0.034 → **unstable; rerun required with real text.**

**Latency (CPU, CE only):** pool=20 ≈ 47s; full pipeline ≈ 67–72s → **over 60s budget on CPU.** Grader GPU timing TBD.

> **Important:** If this format changes, Ron rebuilds on the VM, commits, and notifies Yehoraz to `git pull`.

---

## 5  Environment setup for Yehoraz (no VM needed)

```bash
git clone <repo-url>
cd ProjectA-SectionB
python -m venv .venv && .venv/Scripts/activate   # Windows; use source .venv/bin/activate on Linux
pip install -r requirements.txt
# MiniLM downloads on first run (~80 MB). CE rerank model downloads on first sweep_rerank_ab.py run.

# Canonical score (should match current yehoraz_develop stack)
python scripts/eval_public.py          # expect mean_ndcg@10 ≈ 0.3113

# Full diagnostics + sanity + timing
python scripts/diagnose.py --tag prf_page_mean

# E6 A/B (after Ron ships chunk_texts.npy)
python scripts/sweep_rerank_ab.py
```

**What you need from git (all tracked):**
- All `.py` files
- `artifacts/` — `index_vectors.npy`, `index.faiss`, `index_meta.json`, BM25 files (Ron commits; never rebuild locally)
- `artifacts/chunk_texts.npy` — **pending from Ron (§4.3)**
- `data/public_queries.json`

**What you do NOT need:**
- `data/Wikipedia Entries/` (gitignored, Ron's VM only)
- GPU for scoring (CPU scores match GPU); GPU may be required for E6 within 60s timing budget

---

## 6  Git workflow

### Branches

| Branch | Purpose | Rule |
|--------|---------|------|
| `main` | Always-green graded branch | Must pass `eval_public.py` on fresh clone |
| `ron_develop` | Ron's working branch | Merge to `main` via PR with score report |
| `yehoraz_develop` | Yehoraz's working branch | Merge to `main` via PR with score report |
| feature branches | Per-experiment (`ron/sentence-chunking`, `yehoraz/bm25-fusion`) | Short-lived |

### Merge rules

1. Every PR description includes **before/after holdout NDCG@10**.
2. No merge if holdout score regresses vs current `main`.
3. After merge to `main`, **Ron promotes the winning index** into `artifacts/` only (single LFS set). The three `artifacts_variants/` dirs stay on `ron_develop` for experimentation — not required on `main`.
4. Both run `eval_public.py` after pulling `main` to confirm.

### Git LFS on `ron_develop`

| Location | Contents | When |
|----------|----------|------|
| `artifacts/` | Production default (currently `title_150`) | `main` + `ron_develop` |
| `artifacts_variants/{title_150,notitle_150,notitle_180}/` | Full six-file index per E1 arm | `ron_develop` only |

Ron is the sole committer of LFS blobs. Yehoraz: `git lfs pull` after every pull that touches artifacts.

---

## 7  Timeline (7 days)

### Day 1 — Foundation (both, pair session)
- [v] Lock baseline NDCG@10 number
- [v] Build shared eval harness: per-query scores, 35/15 holdout split, timing, results log
- [ ] Agree on E2 lexical artifact format (Section 4.2 above)
- [ ] Yehoraz: set up local env, confirm `eval_public.py` runs

### Day 2 — First experiments (parallel)
- [x] **Ron → E1:** chunking parameter sweep → `title_150` locked
- [x] **Yehoraz → E3:** page-scope mean-all aggregation (0.2476)

### Day 3 — Lexical handoff
- [x] **Ron → E2:** BM25 artifacts on VM
- [x] **Yehoraz → E4:** BM25 + dense RRF fusion (0.2993)

### Day 4 — Tune fusion + PRF
- [x] **Yehoraz → E4/PRF:** RRF K validated; PRF query expansion (0.3113)
- [x] **Yehoraz → E6 A/B:** CE rerank tested (proxy text; +0.017, not merged)

### Day 5 — Integration + rerank unblock
- [ ] Merge Yehoraz stack (E3+E4+PRF) to `main` if team agrees
- [ ] **Ron → E6:** build `chunk_texts.npy` (§8.4) — **critical path**
- [ ] **Yehoraz:** rerun A/B with real text; integrate rerank if stable + fast enough
- [ ] Ron: verify query-phase timing on VM (GPU)

### Day 6 — Hardening
- [ ] Fresh-clone reproducibility test
- [ ] Edge cases: empty pages, queries returning < 10 results
- [ ] Final tuning on holdout only
- [ ] **Code freeze**

### Day 7 — Packaging & submission
- [ ] Finalize README (artifact paths, design decisions)
- [ ] Record video (Ron: indexing/chunking; Yehoraz: ranking/fusion)
- [ ] Submission dry-run
- [ ] Buffer hours for surprises

---

## 8  Decision log

Record every experiment result here so both teammates (and agents) have context.

| Date | Exp | Branch | NDCG@10 | Delta vs baseline | Merged? | Notes |
|------|-----|--------|---------|-------------------|---------|-------|
| 2026-06-06 | baseline (full corpus) | `ron_develop` | 0.1295 | — | — | 5-fold mean (std 0.075) over 50 public queries, 27,074 pages / 437,237 chunks (CHUNK_WORDS=180, overlap=40, TOP_CHUNKS=200). recall@10/50/100 = 0.18/0.39/0.51. Per `diagnostics.py`: gold chunk rank approx equals gold page rank -> max-pool aggregation is near-lossless, so the bottleneck is chunk/embedding quality (Ron side), not aggregation. Union-oracle ceiling approx 0.19 (13 duplicate query strings carry conflicting labels). Per-bucket NDCG: n_rel=1 -> 0.19, n_rel=2 -> 0.00, n_rel=3 -> 0.17, n_rel=4 -> 0.03. |
| 2026-06-06 | token-truncation audit (E1 prep) | `ron_develop` | — | — | n/a | `scripts/audit_tokens.py` on full corpus: chunk token lengths median 240, mean 238.8, p90 272, p95 285, max 1124 vs MiniLM cap 256. 98,523 chunks (22.5%) exceed 256 and are silently truncated at encode time (median 13 tokens lost). Motivates E1: smaller CHUNK_WORDS (~100-120) to eliminate truncation and concentrate the gold sentence. |
| 2026-06-06 | E1 `notitle_180` (arm A) | `ron_develop` | 0.1115 | −0.0180 | no | 180w/40, no title. 437k chunks. k-fold 0.1115 ± 0.043. recall@10=0.185. Gold-chunk median rank 240 (better than baseline 268) but NDCG worse — title helps page-level ranking after max-pool. |
| 2026-06-06 | E1 `title_120` (arm D, title) | `ron_develop` | 0.1159 | −0.0136 | no | 120w/30, title on. 674k chunks. k-fold 0.1159 ± 0.095. recall@10=0.157. Worst gold-chunk ranks (median 448). Smaller windows + title = fragmentation + title noise. |
| 2026-06-06 | E1 `notitle_120` (arm D, no title) | `ron_develop` | 0.1322 | +0.0027 | no | 120w/30, no title. 674k chunks. k-fold 0.1322 ± 0.079. recall@10=0.205, MRR=0.134. 8 query wins / 8 losses vs baseline (34 ties). Superseded by `title_150`. |
| 2026-06-06 | E1 `title_150` token audit | `ron_develop` | — | — | n/a | 150w/33, title on (preview only). Full corpus: median 201 tokens, 2.1% >256 (vs 22.5% at 180w). ~521k chunks expected. Middle ground on truncation without +54% chunk count of 120w. |
| 2026-06-06 | E1 `title_150` (follow-up) | `ron_develop` | 0.1332 | +0.0037 | **yes (locked)** | 150w/33, title on. 521,322 chunks. k-fold 0.1332 ± 0.078. recall@10=0.195, MRR=0.128. query_phase ~1.83s. **Best E1 arm — chunk config locked for E2 rebuild.** |
| 2026-06-07 | E1 `notitle_150` | `ron_develop` | 0.1290 | −0.0005 | no | 150w/33, no title. 521,322 chunks. k-fold 0.1290 ± 0.072. recall@10=0.195. Title prefix still wins at 150w (+0.0042 vs this arm). |
| 2026-06-06 | E2 production rebuild | `ron_develop` | 0.1332 | +0.0037 | **yes** | `artifacts/`: title_150 dense (764M vectors + 764M faiss) + BM25 (`bm25_tf.npz` 393M, `bm25_vocab.json` 9.6M, vocab=319,990, avg_dl=152.4, min_df=2). `eval_public.py` NDCG=0.1332, query_phase=3.0s. `diagnose --tag production_e2`: sanity PASSED, query_phase=1.9s OK. **No score lift from BM25 until E4** — artifacts ready for Yehoraz. Future rebuild tip: copy dense from `artifacts_sweep/title_150/` + BM25-only to skip re-embed. |
| 2026-06-07 | E3a window mean-of-top-K (superseded) | `yehoraz_develop` | 0.1612 | +0.0280 | no (superseded by E3b) | First E3 step: page score = mean of its top-2 chunk scores **within the retrieved window** + `TOP_CHUNKS` 200→500. 5-fold 0.1612 ± 0.085. Every window mean-K beat max-pool; `sum`-of-top-N strictly worse (rewards long pages). Swept with `scripts/sweep_e3.py`. Kept only as the stepping stone to E3b (page scope). |
| 2026-06-07 | **E3b page-scope mean-all** | `yehoraz_develop` | **0.2476** | **+0.1144** | no (on `yehoraz_develop`) | **Two-stage rerank** (`AGG_SCOPE="page"`, `PAGE_POOL_K=0`): FAISS top-`TOP_CHUNKS`(=500) selects CANDIDATE pages, then each candidate is rescored by the **mean cosine of ALL its chunks** vs the query (not just windowed ones). 5-fold **0.2476 ± 0.107** vs baseline 0.1332 ± 0.067. NDCG curve rose monotonically with K and plateaued once K covered the page (mean100=mean1000), i.e. parameter-free page-mean. Broad gains: recall@10 0.19→0.36, recall@50 0.37→0.64, queries-with-hit 13→23, every n_rel bucket up. `eval_public.py`=0.2476, `diagnose --tag page_meanall` sanity **PASSED**, query_phase 8.6-13s CPU (OK). Touches `retrieve.py` (`_rank_pages_page_scope`) + `utils.{AGG_SCOPE,PAGE_POOL_K,TOP_CHUNKS}`. **Shared harness updated:** `diagnostics.py`/`diagnose.py` are now aggregation-aware (`--scope`/`--pool-k`/`--top-chunks`, default from utils) so they mirror `retrieve.py` again. Headroom: union-oracle ceiling 0.357 (capped by 13 duplicate-label queries). |
| 2026-06-07 | **E4 BM25 + dense RRF fusion** | `yehoraz_develop` | **0.2993** | **+0.1661** | no (on `yehoraz_develop`) | **Reciprocal Rank Fusion** of the E3 dense page ranking with a BM25 page ranking (`FUSION="rrf"`, `RRF_K=60`). BM25 page score = **max** over the page's chunk BM25 scores (best-matching passage; `BM25_PAGE_AGG="max"`), computed only over the page's chunks **inside the dense window** (`BM25_SCOPE="window"`) — `page` scope (all chunks) was only +0.0095 (within ±0.095 noise) but ~15-20× more BM25 work and risked the 60s budget. Swept fusion (RRF vs weighted-sum) × agg (max/sum/mean) × scope (window/page) × params with `scripts/sweep_e4.py`: RRF+max won and is **insensitive to k (10-100 all ≈0.309)**; `sum` collapses (long-page bias); pure-dense (α=1) reproduces 0.2476 (harness sanity). 5-fold **0.2993 ± 0.095** vs E3 0.2476. recall@10 0.36→0.43, queries-with-hit 23→28, n_rel=1 bucket ndcg 0.44. `eval_public.py`=0.2993 (query 19.5s), `diagnose --tag e4_rrf` sanity **PASSED**, query_phase 24.9s CPU incl. 393M BM25 load (OK <60s). Touches `retrieve.py` (`_rrf_fuse`/`_page_bm25_scores`/`_collect_candidates`) + `utils.{FUSION,RRF_K,BM25_PAGE_AGG,BM25_SCOPE}`. **Shared harness updated:** `diagnostics.py`/`diagnose.py` are fusion-aware (`--fusion`, BM25-mirroring `aggregate_to_pages`). Headroom: union-oracle ceiling 0.418. |
| 2026-06-07 | **PRF query expansion (Rocchio, page-level)** | `yehoraz_develop` | **0.3113** | **+0.1781** | no (on `yehoraz_develop`) | **Two-pass pseudo-relevance feedback** on the dense query (`PRF=True`, `PRF_ALPHA=0.9`, `PRF_TOPN=10`, `PRF_PAGE_REPR="mean"`): first pass picks the top-10 pseudo-relevant **pages**, each represented by its **mean chunk vector**; their centroid expands the query `q' = norm(0.9·q + 0.1·centroid)`; second pass ranks with `q'`. BM25/RRF unchanged (original query terms). Done while E5 blocked on Ron — entirely query-side. Swept level (chunk/page) × repr (mean/best) × α × N with `scripts/sweep_prf.py`: **page-level decisively beat chunk-level** (every chunk config ≤ no-PRF, down to 0.218 — redundant chunks of one page drift the centroid); **light expansion (α=0.9) won**; α=1.0 reproduces 0.2993 (sanity). Picked the **conservative** config (mean/N=10), not the literal top (best/N=20=0.3129), to avoid overfit; top region (α=0.9, N=10-20, both reprs) stable at 0.310-0.313. 5-fold **0.3113 ± 0.083** (tighter than E4's ±0.095). PRF's second pass lifts **recall too** (recall@100 0.638→0.675, queries-with-hit 28→31) — attacks the candidate-set ceiling that reorder-only fusion can't. `eval_public.py`=0.3113 (query 15.6s), `diagnose --tag prf_page_mean` sanity **PASSED**, query_phase 23.1s (OK). Touches `retrieve.py` (`_prf_expand_query` + two-pass in `search_batch`) + `utils.{PRF,PRF_ALPHA,PRF_TOPN,PRF_PAGE_REPR}`. **Shared harness updated:** `diagnostics.py`/`diagnose.py` are PRF-aware (`--prf`/`--no-prf`, importing `_prf_expand_query` to mirror production). Headroom: union-oracle ceiling 0.427. |
| 2026-06-07 | RRF K-tuning analysis (no change) | `yehoraz_develop` | 0.3113 | 0 | n/a (validated current) | Investigated whether `RRF_K=60` is optimal and whether an **asymmetric** RRF (different K per ranker) is justified. On the shared candidate set: fusion (0.310) beats both singles — semantic-alone 0.2554 (±0.107), **BM25-alone 0.2774 (±0.044)**; BM25 is the more reliable single ranker (MRR 0.325 vs 0.291; head-to-head 20 vs 13). Shared-K is a **smooth flat plateau** k=10→1000 (0.3102-0.3107) → k=60 confirmed. Fine asymmetric grid (k_d,k_b ∈ 40-70) showed a *jagged* surface with apparent BM25-favored peaks (e.g. (70,55)=0.3142), but a **split-half held-out test debunked it**: cell tuned on half A → held-out B = 0.3023 vs symmetric 0.3300 (−0.028); best cell differs per half ((70,40) vs (70,55)). **Asymmetric K overfits the public 50 → rejected; kept symmetric K=60.** Real lever surfaced instead: BM25 candidate generation (union with dense candidates) to lift recall ceiling. Analysis only (`sweep_rrf_k.py`, since removed); no code/score change. |
| 2026-06-07 | **E6 CE rerank A/B (not merged)** | `yehoraz_develop` | 0.3284 (B) | +0.017 vs 0.3113 | no (blocked) | **Cross-encoder rerank** (Option A: CE-only on shortlist) A/B via `scripts/sweep_rerank_ab.py`. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. **Passage text: BM25 token proxy** (no `chunk_texts.npy` yet) — architecture test only. A=0.3113, B(pool=20)=0.3284 (+0.017), B(pool=30–50)≈0.327. Split-half unstable: half-A +0.068, half-B −0.034. CE stage alone ~47s CPU (pool=20); estimated full pipeline ~67–72s → over 60s on CPU. **Not merged.** Blocked on Ron §4.3 (`chunk_texts.npy`) + GPU timing + stable rerun. Preserves all upstream optimizations (PRF/E3/E4/RRF) — CE only reorders top-M fused pages. |

### 8.1  E1 2×2 synthesis & Ron next direction (2026-06-06, updated 2026-06-07)

**Factorial results (title × size):**

| | 180w / ovlp 40 | 150w / ovlp 33 | 120w / ovlp 30 |
|---|---|---|---|
| **title ON** | 0.1295 baseline | **0.1332** | 0.1159 |
| **title OFF** | 0.1115 | 0.1290 | 0.1322 |

**Key findings:**
- **Strong interaction:** title helps at 180w (+0.018) but hurts at 120w (−0.016). No universal “title on” or “smaller is better.”
- **Truncation hypothesis mostly rejected:** 120w nearly eliminates truncation but still underperforms; bottleneck is chunk *matching*, not tail clipping (consistent with gold-chunk rank ≈ page rank).
- **Chunking alone has a low ceiling** (~±0.02 NDCG on 50 public queries). Multi-relevant buckets (n_rel≥2) stay weak across all arms.
- **E4 (BM25 + dense fusion)** remains the highest-expected-impact track per §3.2; E2 unblocks it.

**Agreed Ron priority (updated 2026-06-07):**
1. **E1 + E2 complete.** Production `artifacts/`: title_150 dense + BM25.
2. **Yehoraz E3 + E4 + PRF complete** on `yehoraz_develop` (0.3113) — not yet merged to `main`.
3. **Ron → E6 (NOW):** build and commit `chunk_texts.npy` per §4.3 — **unblocks reranking**.
4. **Ron → E5 (optional, parallel or after E6):** per-page title embedding artifact.
5. Sentence-aware splitting: still deferred.

### 8.2  Yehoraz query-side progress summary (2026-06-07)

**Score progression on public queries (branch `yehoraz_develop`):**

```
0.1332  baseline (dense max-pool, TOP_CHUNKS=200)
  ↓ E3b page-scope mean-all + TOP_CHUNKS=500
0.2476
  ↓ E4 BM25 + dense RRF (k=60, BM25-max, window scope)
0.2993
  ↓ PRF Rocchio (page/mean, N=10, α=0.9)
0.3113  ← current production config
  ↓ E6 CE rerank (A/B only, proxy text — not merged)
0.3284  (indicative; blocked)
```

**Files changed (Yehoraz side, uncommitted):**
- `retrieve.py` — E3 page-scope, E4 RRF fusion, PRF two-pass
- `utils.py` — all constants listed in §3.2
- `diagnostics.py` + `scripts/diagnose.py` — aggregation/fusion/PRF-aware; sanity check mirrors `retrieve.py`
- `scripts/sweep_rerank_ab.py` — E6 A/B harness (keep until real text rerun)

**Diagnostic artifacts in `results/` (gitignored):**
- `diag_page_meanall.json` (E3, 0.2476)
- `diag_e4_rrf.json` (E4, 0.2993)
- `diag_prf_page_mean.json` (PRF, 0.3113)

**Not merged to `main` yet** — awaiting team decision / Ron E6 artifact.

### 8.3  Ron E2 handoff (complete)

**Artifacts (VM, ready on `ron_develop`):**
- **Default:** `artifacts/` — `title_150` (521,322 chunks, 150w/33/title) + BM25
- **Variants for A/B:** `artifacts_variants/{title_150,notitle_150,notitle_180}/` — same six-file layout (§4.3). `title_180` scores: `results/diag_baseline.json`. Pass `artifacts_dir` to `search_batch` / `load_bm25` / `diagnose --artifacts-dir`.

**Missing for E6:** `chunk_texts.npy` — see §8.4.

### 8.4  Ron E6 handoff checklist — enable cross-encoder reranking

> **This is Ron's immediate next task** to unblock the largest remaining query-side gain.

#### What Ron needs to know

1. **Yehoraz cannot build this artifact** — no access to `data/Wikipedia Entries/` (gitignored, VM only).
2. **No re-embedding needed** — save the same `c.text` strings already produced during `build_index()`. MiniLM vectors stay as-is.
3. **Alignment is critical** — `chunk_texts[i]` must match row `i` of vectors, BM25 CSR, and `index_meta.json` page_ids. A single row mismatch breaks rerank scoring.
4. **Why BM25 is not a substitute** — BM25 stores token frequencies, not ordered passage text. Yehoraz's A/B used a BM25-token proxy; split-half was unstable. Real text required for production.
5. **Size** — expect ~400–600 MB for `chunk_texts.npy`; use git LFS. Optional: `np.savez_compressed` for smaller blob.
6. **Latency** — reranking adds query time. Yehoraz measured ~47s CE-only on CPU (pool=20, 50 queries). Ron should verify **total** `query_phase_time < 60s` on **grading GPU** after Yehoraz integrates (or flag if too slow).

#### Ron step-by-step

| Step | Action | Verify |
|------|--------|--------|
| 1 | Add `CHUNK_TEXTS_NAME = "chunk_texts.npy"` constant to `index.py` | — |
| 2 | In `build_index()`, after chunking: `np.save(out_dir / CHUNK_TEXTS_NAME, np.asarray([c.text for c in chunks], dtype=object))` | File exists |
| 3 | Add `load_chunk_texts(artifacts_dir)` to `index.py` (see §4.3) | `len(texts) == len(page_ids)` |
| 4 | Rebuild on VM (`python scripts/build_index.py`) — full rebuild safest; or add-only pass if chunking unchanged | `eval_public.py` NDCG unchanged (0.1332 dense-only; Yehoraz stack is query-side) |
| 5 | Commit `chunk_texts.npy` (+ code) to git; notify Yehoraz | `git pull` on Yehoraz machine |
| 6 | (Optional) Run `python scripts/sweep_rerank_ab.py` on VM GPU; share timing | `query_phase` < 60s |

#### After Ron ships — Yehoraz step-by-step

| Step | Action | Verify |
|------|--------|--------|
| 1 | `git pull` — confirm `artifacts/chunk_texts.npy` present | `load_chunk_texts()` works |
| 2 | Rerun `python scripts/sweep_rerank_ab.py` (real text, not proxy) | A=0.3113; B improves; split-half stable |
| 3 | If pass: implement `rerank.py`, wire into `retrieve.py`, mirror in `diagnostics.py` | `diagnose.py` sanity PASSED |
| 4 | `eval_public.py` + `diagnose.py` on GPU hardware | NDCG gain + time < 60s |
| 5 | Record in §8 decision log; PR to `main` | — |

#### What stays unchanged when E6 lands

- MiniLM embedding model and `index_vectors.npy` / `index.faiss`
- Chunking params (`CHUNK_WORDS=150`, `CHUNK_OVERLAP=33`, `PREFIX_TITLE=True`)
- BM25 artifacts (unless full rebuild)
- All Yehoraz query logic upstream of rerank: PRF, page-scope mean-all, BM25/RRF fusion, `RRF_K=60`

---

## 9  Agent instructions

> This section is for AI coding agents that Yehoraz (or Ron) may use during development.

### If you are Yehoraz's agent:

1. **Your scope:** `retrieve.py`, `rerank.py` (when E6 lands), `utils.py` (query constants). Keep `diagnostics.py` / `scripts/diagnose.py` in sync with `retrieve.py`.
2. **Do not** modify `eval.py` (read-only per assignment rules).
3. **Do not** modify `chunk.py`, `embed.py`, or `index.py` — those are Ron's.
4. **Do not** rebuild or overwrite anything in `artifacts/` — treat as read-only. **Exception:** consume `chunk_texts.npy` once Ron commits it.
5. **Available data:** `artifacts/` (dense + BM25; `chunk_texts.npy` pending) and `data/public_queries.json`. Raw corpus not available.
6. **Test changes:** `python scripts/eval_public.py` (canonical score) **and** `python scripts/diagnose.py --tag <name>` (sanity + timing). Sanity must PASS.
7. **Current production config (§3.2):** E3 page-scope mean-all + E4 RRF + PRF → **0.3113**. Do not regress without documenting in §8.
8. **Next priority — E6 rerank (blocked):**
   - Wait for Ron's `chunk_texts.npy` (§4.3, §8.4).
   - Rerun `python scripts/sweep_rerank_ab.py` with real text.
   - If stable + fast: implement Option A CE rerank (`RERANK_POOL≈20`, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - Additional pretrained models **only for reranking** — never replace MiniLM for indexing.
9. **Exploratory (lower priority):** BM25 candidate generation (union BM25 top-pages with dense pool).
10. **Latency:** 60s query-phase budget. CE rerank was ~47s CPU alone — profile on GPU before merging.
11. **Always** record before/after NDCG@10 in §8 decision log.

### If you are Ron's agent:

1. **Your scope:** `chunk.py`, `embed.py`, `index.py`, `lexical.py`, `scripts/build_index.py`, artifact generation.
2. **Do not** modify `eval.py` or `retrieve.py`.
3. **After any index change**, rebuild with `python scripts/build_index.py`, test `python scripts/eval_public.py`.
4. **Priority (updated 2026-06-07):**
   - **E1 + E2:** done. Chunk config locked: `title_150`.
   - **E6 (NOW — highest priority):** persist `chunk_texts.npy` at build time + `load_chunk_texts()`. See §4.3 and §8.4 step-by-step. **This unblocks Yehoraz reranking.**
   - **E5 (optional):** per-page title embedding artifact for title-vector fusion.
   - Sentence-aware splitting: still deferred.
5. **E6 implementation notes:**
   - Save `np.asarray([c.text for c in chunks], dtype=object)` — same strings sent to `embed_texts()`.
   - Row alignment with vectors/BM25/meta is mandatory.
   - No MiniLM model change; no mandatory re-embed if chunking params unchanged.
   - Commit via git LFS (~400–600 MB).
   - Notify Yehoraz after `git push`.
6. **Always** record before/after NDCG@10 for every change in §8.
7. **Commit artifacts** to `main` only after confirming dense-only score does not regress.
