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
| E1 | Chunking sweep (2×2 factorial + `title_150` follow-up). **Status: 2×2 complete; `title_150` GPU run in flight.** See decision log §8. | `chunk.py`, `utils.py` | Medium (diminishing returns observed) |
| E2 | Lexical index — BM25 artifacts in `lexical.py`, built by `index.build_index()`. **Status: complete** — production `artifacts/` has dense title_150 + BM25 (Jun 6). | `lexical.py`, `index.py` | **High** (enables E4) |

**E1 solo arms (fully in Ron's scope — change chunk text/size, rebuild, measure with `diagnostics.py`):**

| Arm | Change | Hypothesis |
|-----|--------|-----------|
| A | No title prefix (`body` only) | Baseline: does the title actually help or hurt the chunk embedding? |
| B | Title prefix `f"{title}. {body}"` (current) | Status quo; entity anchoring across coref-heavy passages. |
| D | Smaller `CHUNK_WORDS` (100/120) x title on/off | Eliminates truncation + concentrates the gold sentence; title matters more when body is short. Preview token cost with `scripts/audit_tokens.py --chunk-words N` (no GPU). |

Front placement of the title is kept deliberately: truncation cuts the tail, so a prefix survives the 256-token cap while a suffix would not.

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
| E5 | Title-vector fusion (cross-cutting) — Ron builds a per-page title embedding artifact (offline, like E2); Yehoraz blends its score with the chunk score at query time. Keeps the chunk embedding "pure", frees token budget, and adds entity signal. Same shape as E4 fusion. | `index.py` (Ron: artifact) + `retrieve.py` (Yehoraz: blend) | Medium-High |

### 3.3  Shared — both on Day 1

- **Eval harness (built): `diagnostics.py` + `scripts/diagnose.py`.** This is the single internal evaluation tool for BOTH teammates — set-aware NDCG@10 (matches `eval.py`), recall@{10,50,100}, MRR, per-relevant-page ranks, chunk-level diagnostics (gold-chunk rank, recall within `TOP_CHUNKS`), per-bucket (by n_relevant), 5-fold CV, and data-quality checks. Run `python scripts/diagnose.py --tag <name>`; compare runs with `--compare A.json B.json`. Results land in `results/` (gitignored).
- **Evaluation discipline:** use the 5-fold CV mean +/- std (not a single split) to judge changes — 50 queries are noisy. When analyzing a result, isolate *which side* can move it: gold-chunk rank low -> chunk/embedding (Ron); gold-chunk rank high but page rank low -> aggregation/fusion (Yehoraz).
- **Artifact contract:** agree on the format of any new artifacts (E2 lexical, E5 title-vector) before parallel work begins.

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

**E4 integration pattern (Yehoraz — query time, fits 60s budget):**

1. **Dense retrieve (existing):** FAISS → top `TOP_CHUNKS` chunk indices + cosine scores.
2. **BM25 rescore (new):** For those indices only, score each row via CSR slice:
   ```python
   from lexical import load_bm25, tokenize, bm25_score_row
   bm25 = load_bm25()  # or index.load_bm25_index()
   q_terms = tokenize(query)
   for row in chunk_indices:
       lex = bm25_score_row(
           bm25.data, bm25.indices, bm25.indptr, row,
           q_terms, bm25.idf, bm25.vocab, bm25.avg_dl, bm25.k1, bm25.b
       )
   ```
3. **Fuse per chunk:** e.g. `alpha * dense_score + (1-alpha) * lex`, or RRF.
4. **Aggregate to pages:** existing max-pool in `retrieve.py`.

**Load helpers:**
- `lexical.load_bm25(artifacts_dir)` → `Bm25Index` dataclass
- `index.load_bm25_index(artifacts_dir)` — thin wrapper

> **Important:** If this format changes, Ron rebuilds on the VM, commits, and notifies Yehoraz to `git pull`. Batch format changes to minimize round-trips.

---

## 5  Environment setup for Yehoraz (no VM needed)

```bash
git clone <repo-url>
cd ProjectA-SectionB
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
- [v] Lock baseline NDCG@10 number
- [v] Build shared eval harness: per-query scores, 35/15 holdout split, timing, results log
- [ ] Agree on E2 lexical artifact format (Section 4.2 above)
- [ ] Yehoraz: set up local env, confirm `eval_public.py` runs

### Day 2 — First experiments (parallel)
- [v] **Ron → E1:** chunking parameter sweep (window size, overlap, sentence-aware splits)
- [ ] **Yehoraz → E3:** aggregation sweep (max-pool vs sum-of-top-N, `TOP_CHUNKS` tuning)
- [ ] Merge winners to `main`

### Day 3 — Lexical handoff
- [x] **Ron → E2:** build lexical index artifacts on VM (`artifacts/` title_150 + BM25); code in `lexical.py` + `index.py`
- [ ] **Ron:** commit code + `artifacts/` to git so Yehoraz can `git pull`
- [ ] **Yehoraz → E4:** scaffold BM25 fusion in `retrieve.py` against new artifacts (see §4.2)

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

| Date | Exp | Branch | NDCG@10 | Delta vs baseline | Merged? | Notes |
|------|-----|--------|---------|-------------------|---------|-------|
| 2026-06-06 | baseline (full corpus) | `ron_develop` | 0.1295 | — | — | 5-fold mean (std 0.075) over 50 public queries, 27,074 pages / 437,237 chunks (CHUNK_WORDS=180, overlap=40, TOP_CHUNKS=200). recall@10/50/100 = 0.18/0.39/0.51. Per `diagnostics.py`: gold chunk rank approx equals gold page rank -> max-pool aggregation is near-lossless, so the bottleneck is chunk/embedding quality (Ron side), not aggregation. Union-oracle ceiling approx 0.19 (13 duplicate query strings carry conflicting labels). Per-bucket NDCG: n_rel=1 -> 0.19, n_rel=2 -> 0.00, n_rel=3 -> 0.17, n_rel=4 -> 0.03. |
| 2026-06-06 | token-truncation audit (E1 prep) | `ron_develop` | — | — | n/a | `scripts/audit_tokens.py` on full corpus: chunk token lengths median 240, mean 238.8, p90 272, p95 285, max 1124 vs MiniLM cap 256. 98,523 chunks (22.5%) exceed 256 and are silently truncated at encode time (median 13 tokens lost). Motivates E1: smaller CHUNK_WORDS (~100-120) to eliminate truncation and concentrate the gold sentence. |
| 2026-06-06 | E1 `notitle_180` (arm A) | `ron_develop` | 0.1115 | −0.0180 | no | 180w/40, no title. 437k chunks. k-fold 0.1115 ± 0.043. recall@10=0.185. Gold-chunk median rank 240 (better than baseline 268) but NDCG worse — title helps page-level ranking after max-pool. |
| 2026-06-06 | E1 `title_120` (arm D, title) | `ron_develop` | 0.1159 | −0.0136 | no | 120w/30, title on. 674k chunks. k-fold 0.1159 ± 0.095. recall@10=0.157. Worst gold-chunk ranks (median 448). Smaller windows + title = fragmentation + title noise. |
| 2026-06-06 | E1 `notitle_120` (arm D, no title) | `ron_develop` | 0.1322 | +0.0027 | no | 120w/30, no title. 674k chunks. k-fold 0.1322 ± 0.079. recall@10=0.205, MRR=0.134. 8 query wins / 8 losses vs baseline (34 ties). Superseded by `title_150`. |
| 2026-06-06 | E1 `title_150` token audit | `ron_develop` | — | — | n/a | 150w/33, title on (preview only). Full corpus: median 201 tokens, 2.1% >256 (vs 22.5% at 180w). ~521k chunks expected. Middle ground on truncation without +54% chunk count of 120w. |
| 2026-06-06 | E1 `title_150` (follow-up) | `ron_develop` | 0.1332 | +0.0037 | **yes (locked)** | 150w/33, title on. 521,322 chunks. k-fold 0.1332 ± 0.078. recall@10=0.195, MRR=0.128. query_phase ~1.83s. **Best E1 arm — chunk config locked for E2 rebuild.** |
| 2026-06-06 | E2 production rebuild | `ron_develop` | 0.1332 | +0.0037 | **yes** | `artifacts/`: title_150 dense (764M vectors + 764M faiss) + BM25 (`bm25_tf.npz` 393M, `bm25_vocab.json` 9.6M, vocab=319,990, avg_dl=152.4, min_df=2). `eval_public.py` NDCG=0.1332, query_phase=3.0s. `diagnose --tag production_e2`: sanity PASSED, query_phase=1.9s OK. **No score lift from BM25 until E4** — artifacts ready for Yehoraz. Future rebuild tip: copy dense from `artifacts_sweep/title_150/` + BM25-only to skip re-embed. |

### 8.1  E1 2×2 synthesis & Ron next direction (2026-06-06)

**2×2 results (title × size):**

| | 180w / ovlp 40 | 120w / ovlp 30 |
|---|---|---|
| **title ON** | 0.1295 baseline | 0.1159 |
| **title OFF** | 0.1115 | **0.1322** |

**Key findings:**
- **Strong interaction:** title helps at 180w (+0.018) but hurts at 120w (−0.016). No universal “title on” or “smaller is better.”
- **Truncation hypothesis mostly rejected:** 120w nearly eliminates truncation but still underperforms; bottleneck is chunk *matching*, not tail clipping (consistent with gold-chunk rank ≈ page rank).
- **Chunking alone has a low ceiling** (~±0.02 NDCG on 50 public queries). Multi-relevant buckets (n_rel≥2) stay weak across all arms.
- **E4 (BM25 + dense fusion)** remains the highest-expected-impact track per §3.2; E2 unblocks it.

**Agreed Ron priority (updated after E2):**
1. **E1 + E2 complete.** Production `artifacts/` on VM: title_150 dense + BM25 per §4.2.
2. **Yehoraz → E4:** BM25 + dense fusion in `retrieve.py` (Ron does not touch `retrieve.py`).
3. **Ron next (optional):** commit code + artifacts to git for Yehoraz `git pull`; sentence-aware splitting deferred until after E4 results.

### 8.2  Yehoraz E4 handoff checklist (Ron E2 complete)

**Artifacts in `artifacts/` (VM, ready):**
- Dense: `index_vectors.npy`, `index.faiss`, `index_meta.json` (521,322 chunks, 150w/33/title)
- Lexical: `bm25_tf.npz`, `bm25_vocab.json`, `bm25_meta.json`

**Code to import:**
- `from lexical import tokenize, load_bm25, bm25_score_row`
- Integration pattern: §4.2 (FAISS top-K → BM25 rescore rows → fuse → max-pool)

**Verified dense baseline (no fusion yet):** NDCG@10 = **0.1332**, query_phase **~2s**, within 60s budget.

**Notify Yehoraz:** `git pull` after Ron commits artifacts (~1.9GB total on VM).

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
   - **E4:** BM25 artifacts are in `artifacts/` (`bm25_vocab.json`, `bm25_tf.npz`, `bm25_meta.json`). Import `tokenize`, `load_bm25`, `bm25_score_row` from `lexical.py` (or `index.load_bm25_index()`). Rescore top-K FAISS hits only — see §4.2. Fuse with dense scores (weighted sum or RRF), then max-pool to pages.
8. **Always** record before/after NDCG@10 for every change.
9. **Latency matters:** the query phase is timed. Avoid O(n²) loops over the full corpus at query time. Vectorized numpy operations are preferred.

### If you are Ron's agent:

1. **Your scope:** `chunk.py`, `embed.py`, `index.py`, `scripts/build_index.py`, and artifact generation.
2. **Do not** modify `eval.py` or `retrieve.py`.
3. **After any index change**, rebuild artifacts by running `python scripts/build_index.py`, then test with `python scripts/eval_public.py`.
4. **Priority experiments (updated 2026-06-06):**
   - **E1:** 2×2 complete; `title_150` follow-up in flight. Lock chunk config after it lands — **no further size sweeps** unless `title_150` is inconclusive.
   - **E2 (done):** `lexical.py` + hook in `index.build_index()`. VM production rebuild verified (`diag_production_e2.json`). Commit code + `artifacts/` for Yehoraz.
   - **Sentence-aware splitting:** deferred until after E2 handoff or final rebuild (see §8.1).
5. **Always** record before/after NDCG@10 for every change.
6. **Commit artifacts** to `main` only after confirming the score does not regress.
