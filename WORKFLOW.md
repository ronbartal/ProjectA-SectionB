# Section B — Workflow & Task Division

> **Team:** Ron (indexing / corpus side) · Yehoraz (query / ranking side)
>
> **Goal:** Maximize mean NDCG@10 on 50 hidden queries within a 1-week sprint.
>
> **Last updated:** 2026-06-04

---

## 1  Project overview

A semantic retrieval pipeline over ~9 600 Wikipedia pages.
The grader calls `main.run(queries)` once with all evaluation queries.
Only the first 10 page_ids per query are scored (NDCG@10, binary relevance).

### Pipeline stages

```
[OFFLINE — not timed, Ron's VM]          [QUERY TIME — timed, grader GPU]
corpus JSON → chunk → embed → FAISS+np   queries → embed → FAISS search → aggregate → page_ids
              ~~~~~~~~~~~~~~~~~~~~~~       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              Ron owns this side            Yehoraz owns this side
```

### Key constraints

- Embedding model is fixed: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- Allowed deps: `numpy`, `sentence-transformers`, `faiss-cpu` (see `requirements.txt`).
- Staff do **not** rebuild the index — committed `artifacts/` are graded as-is.
- `eval.py` is **read-only** (do not modify).

---

## 2  Repository layout

```
├── main.py              # Entry point: run(queries), build_offline_index()
├── chunk.py             # Passage chunking               ← Ron
├── embed.py             # MiniLM encode wrapper           ← Ron
├── index.py             # Build + load FAISS/numpy index  ← Ron
├── retrieve.py          # Query-time search + aggregation ← Yehoraz
├── eval.py              # NDCG@10 evaluation (READ-ONLY)
├── utils.py             # Shared constants & helpers       ← shared
├── scripts/
│   ├── build_index.py   # Offline build driver
│   └── eval_public.py   # Public self-test
├── artifacts/           # Committed index files (Ron builds, never Yehoraz)
│   ├── index_vectors.npy
│   ├── index_meta.json
│   └── index.faiss
├── data/
│   ├── public_queries.json   # 50 labelled queries (tracked)
│   └── Wikipedia Entries/    # Raw corpus (gitignored — Ron's VM only)
├── requirements.txt
└── WORKFLOW.md               # ← this file
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

| ID | Experiment | Files touched | Expected impact |
|----|-----------|---------------|-----------------|
| E1 | Chunking sweep — sentence-aware splits, smaller windows (80-120 words), higher overlap | `chunk.py`, `utils.py` | Medium |
| E2 | Lexical index — persist per-chunk term frequencies, IDF stats, and vocabulary as new artifacts so Yehoraz can implement BM25 at query time | `index.py`, new artifact files | **High** (enables E4) |

### 3.2  Yehoraz — query / ranking side

**Files owned:** `retrieve.py`

**Responsibilities:**
- All query-time ranking logic.
- Keep query-phase latency within grading budget (Ron validates absolute timing on VM).
- Never rebuild or commit `artifacts/` — treat them as read-only inputs.

**Experiments (priority order):**

| ID | Experiment | Files touched | Expected impact |
|----|-----------|---------------|-----------------|
| E3 | Aggregation sweep — try sum-of-top-N chunk scores per page instead of max-pool; tune `TOP_CHUNKS` | `retrieve.py`, `utils.py` | Medium |
| E4 | BM25 + dense fusion — weighted combination or RRF using lexical artifacts from E2 | `retrieve.py`, possibly `utils.py` | **High** (biggest expected score jump) |

### 3.3  Shared — both on Day 1

- **Eval harness:** per-query NDCG@10, holdout split (35 tune / 15 holdout from the 50 public queries), timing, results log.
- **Artifact contract:** agree on the format of any new lexical artifacts (E2) before parallel work begins.

---

## 4  Artifact contract (interface between Ron & Yehoraz)

### 4.1  Existing artifacts (dense retrieval)

| File | Shape / format | Producer | Consumer |
|------|---------------|----------|----------|
| `index_vectors.npy` | `float32 (n_chunks, 384)` L2-normalized | `index.build_index()` | `index.load_index()` → `retrieve.py` |
| `index_meta.json` | `{"page_ids": [...], "chunk_ids": [...], "model": str, ...}` | `index.build_index()` | `index.load_index()` → `retrieve.py` |
| `index.faiss` | FAISS `IndexFlatIP` over chunk vectors | `index.build_index()` | `index.load_index()` → `retrieve.py` |

### 4.2  New artifacts needed for E4 (lexical / BM25)

> **Status:** to be built by Ron (E2), consumed by Yehoraz (E4).
> Agree on this schema on Day 1 before parallel work begins.

Proposed files (Ron will create under `artifacts/`):

| File | Format | Contents |
|------|--------|----------|
| `bm25_vocab.json` | `{"token": idf_float, ...}` | Vocabulary with precomputed IDF values |
| `bm25_tf.npz` | numpy `.npz` (CSR arrays: data, indices, indptr) | Per-chunk term-frequency matrix (n_chunks × vocab_size) |
| `bm25_meta.json` | `{"avg_dl": float, "n_docs": int, "vocab_size": int}` | Corpus-level BM25 statistics |

**Yehoraz:** at query time, tokenize the query, look up IDF from vocab, compute BM25 scores against chunks using the TF matrix + `avg_dl`, then fuse with dense FAISS scores.

> **Important:** If this format changes during the sprint, Ron rebuilds on the VM, commits, and notifies Yehoraz to `git pull`. Batch format changes to minimize round-trips.

---

## 5  Environment setup for Yehoraz (no VM needed)

```bash
git clone <repo-url>
cd ProjectA_SectionB
pip install -r requirements.txt
# MiniLM downloads automatically on first run (~80 MB, CPU is fine)

# Verify baseline
python scripts/eval_public.py
```

**What you need from git (all tracked):**
- All `.py` files
- `artifacts/` (Ron commits these — never rebuild locally)
- `data/public_queries.json`

**What you do NOT need:**
- `data/Wikipedia Entries/` (gitignored, only on Ron's VM)
- GPU (CPU works for query-time eval; times will be slower but scores are identical)

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
3. After merge to `main`, **Ron rebuilds artifacts on VM** (if indexing changed) and commits them.
4. Both run `eval_public.py` after pulling `main` to confirm.

---

## 7  Timeline (7 days)

### Day 1 — Foundation (both, pair session)
- [ ] Lock baseline NDCG@10 number
- [ ] Build shared eval harness: per-query scores, 35/15 holdout split, timing, results log
- [ ] Agree on E2 lexical artifact format (Section 4.2 above)
- [ ] Yehoraz: set up local env, confirm `eval_public.py` runs

### Day 2 — First experiments (parallel)
- [ ] **Ron → E1:** chunking parameter sweep (window size, overlap, sentence-aware splits)
- [ ] **Yehoraz → E3:** aggregation sweep (max-pool vs sum-of-top-N, `TOP_CHUNKS` tuning)
- [ ] Merge winners to `main`

### Day 3 — Lexical handoff
- [ ] **Ron → E2:** build lexical index artifacts, commit to `main`
- [ ] **Yehoraz → E4:** scaffold BM25 fusion in `retrieve.py` against new artifacts

### Day 4 — Tune fusion (parallel)
- [ ] **Yehoraz → E4:** tune fusion weights on holdout (expected biggest jump)
- [ ] **Ron:** re-tune chunking if fusion changes what "good chunks" means

### Day 5 — Integration
- [ ] Merge best chunking + best fusion + best aggregation into `main`
- [ ] Confirm holdout ≥ each individual best
- [ ] Ron: final artifact rebuild on VM, commit to `main`
- [ ] Ron: verify query-phase timing on VM

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

| Date | Exp | Branch | Holdout NDCG@10 | Delta vs baseline | Merged? | Notes |
|------|-----|--------|-----------------|-------------------|---------|-------|
| 2026-06-04 | baseline | `main` | _TBD_ | — | yes | Initial starter code |

---

## 9  Agent instructions

> This section is for AI coding agents that Yehoraz (or Ron) may use during development.

### If you are Yehoraz's agent:

1. **Your scope:** `retrieve.py` and query-time logic only. You may read any file but should only edit `retrieve.py` (and `utils.py` for shared constants like `TOP_CHUNKS`).
2. **Do not** modify `eval.py` (read-only per assignment rules).
3. **Do not** modify `chunk.py`, `embed.py`, or `index.py` — those are Ron's.
4. **Do not** rebuild or overwrite anything in `artifacts/` — treat as read-only.
5. **Available data:** `artifacts/` (dense index) and `data/public_queries.json`. The raw corpus (`data/Wikipedia Entries/`) is not available to you.
6. **Test your changes** by running `python scripts/eval_public.py` and reporting the `mean_ndcg@10` score.
7. **Priority experiments** (in order):
   - **E3:** In `retrieve.py`, change `_rank_pages_from_chunks` to try sum-of-top-N chunk scores instead of max-pool. Sweep N ∈ {1, 2, 3, 5}. Also try tuning `TOP_CHUNKS` in `utils.py` (try 100, 200, 300, 500).
   - **E4:** Once `bm25_*.json`/`.npz` artifacts exist in `artifacts/`, implement BM25 scoring at query time and fuse with dense FAISS scores using a weighted sum. Tune the weight (e.g. 0.3 BM25 + 0.7 dense).
8. **Always** record before/after NDCG@10 for every change.
9. **Latency matters:** the query phase is timed. Avoid O(n²) loops over the full corpus at query time. Vectorized numpy operations are preferred.

### If you are Ron's agent:

1. **Your scope:** `chunk.py`, `embed.py`, `index.py`, `scripts/build_index.py`, and artifact generation.
2. **Do not** modify `eval.py` or `retrieve.py`.
3. **After any index change**, rebuild artifacts by running `python scripts/build_index.py`, then test with `python scripts/eval_public.py`.
4. **Priority experiments:**
   - **E1:** Sweep `CHUNK_WORDS` (80, 100, 120, 150, 180) and `CHUNK_OVERLAP` (20, 40, 60) in `utils.py`. Try sentence-aware splitting in `chunk.py`.
   - **E2:** Add lexical artifact generation to `index.py` — compute per-chunk TF, corpus IDF, avg document length. Save as `artifacts/bm25_vocab.json`, `artifacts/bm25_tf.npz`, `artifacts/bm25_meta.json` per the contract in Section 4.2.
5. **Always** record before/after NDCG@10 for every change.
6. **Commit artifacts** to `main` only after confirming the score does not regress.
