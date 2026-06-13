# FedInsight — Plan to a Fully Functional, Useful RAG

This plan takes FedInsight from a working end-to-end skeleton to a genuinely
useful retrieval system. The pipeline already runs today
(scrape → parse → chunk → embed → store → **retrieve** → generate), and a
ChromaDB store with ~4,000 FOMC-minutes chunks ships in the repo. The work
below is about **retrieval quality, coverage, and trust**, not building the
basics from scratch.

Legend: ✅ done · ⬜ todo. Each item lists the files it touches and how to
verify it.

---

## Current state (verified)

- ✅ FOMC Minutes scraper — `src/scraper/fomc_minutes_scraper.py`
- ✅ Column-aware PDF parser (PyMuPDF) — `src/parser/pdf_parser.py`
- ✅ Chunker — `src/chunker/text_chunker.py`
- ✅ Embedder (sentence-transformers, auto CUDA) — `src/embeddings/embedder.py`
- ✅ Vector stores: ChromaDB (default) + pgvector — `src/vectorstore/`
- ✅ Retriever: embed query → top-k cosine, supports `metadata_filter` — `src/rag/retriever.py`
- ✅ RAG pipeline with source citations, OpenAI/Ollama/offline fallback — `src/rag/`
- ✅ Streamlit UI + Docker deploy to HF Spaces — `app.py`, `Dockerfile`
- ⬜ No automated tests, no retrieval evaluation, no CI

---

## Phase 1 — Trust the retrieval (highest priority)

You can't improve what you can't measure, and a RAG demo is only useful if its
answers are right.

### 1.1 ⬜ Golden eval set
- Create `eval/qa_set.jsonl`: ~25–40 hand-written questions about known FOMC
  content, each with the expected meeting date(s) and a short reference answer
  or quote.
- **Verify:** file loads; every referenced date exists in the ingested corpus.

### 1.2 ⬜ Retrieval metrics harness
- Add `scripts/eval_retrieval.py`: for each question, run `Retriever.search`
  and compute **hit@k** and **MRR** against the expected date(s)/chunk(s).
- Print a summary table; fail loudly if hit@5 drops below a threshold.
- **Verify:** `python scripts/eval_retrieval.py` prints metrics on the shipped
  Chroma store.

### 1.3 ⬜ Test suite + lint baseline
- Add `tests/` with unit tests for: chunker boundaries, retriever empty-query
  handling, `build_context`/citation numbering, vector-store factory selection,
  LLM provider auto-resolution (mock `requests`).
- Add `pyproject.toml` with ruff config so `python -m ruff check src scripts app.py`
  has a stable rule set.
- **Verify:** `python -m pytest -q` green; `ruff check` clean.

---

## Phase 2 — Make retrieval actually good

Plain cosine similarity over a single small embedding model is the weakest link
for keyword/date-heavy Fed text.

### 2.1 ⬜ Hybrid search (dense + sparse)
- Add BM25/keyword retrieval and fuse with vector hits (Reciprocal Rank Fusion).
- New module `src/rag/hybrid.py`; wire an option into `Retriever`.
- **Verify:** hit@5 on the eval set improves vs. dense-only.

### 2.2 ⬜ Cross-encoder reranking
- Rerank the top-N candidates with a cross-encoder
  (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`); return reranked top-k.
- Make it optional (CPU cost) and configurable via `.env`.
- **Verify:** MRR improves on the eval set; latency noted in the UI.

### 2.3 ⬜ Better chunking for FOMC structure
- Respect section/paragraph boundaries; add overlap; carry section headers into
  chunk metadata so citations are more precise.
- **Verify:** re-ingest a sample, confirm chunks don't split mid-sentence and
  eval metrics hold or improve.

### 2.4 ⬜ Upgrade embedding model (optional)
- Evaluate `bge-small`/`bge-base`/`e5` vs. current `all-MiniLM-L6-v2` on the
  eval set; switch only if it clearly wins. Keep dimension wired through the
  store factory.
- **Verify:** eval metrics; note model + dim in README.

---

## Phase 3 — Make it useful in the UI

### 3.1 ⬜ Metadata filters in the UI
- Surface date-range / year filters in the Streamlit sidebar that pass through
  to `Retriever.search(metadata_filter=...)` (already supported end-to-end).
- **Verify:** filtering to a single year only returns chunks from that year.

### 3.2 ⬜ Click-through sources
- In the sources expander, link each `[n]` to the original PDF/page and show the
  meeting date + section header prominently.
- **Verify:** every cited `[n]` resolves to a real source entry.

### 3.3 ⬜ Answer-faithfulness guardrail
- When the LLM cites `[n]` not in the retrieved set, or returns no citation,
  flag it in the UI rather than presenting it as grounded.
- **Verify:** a deliberately off-topic question shows the "not enough
  information" path, not a hallucinated answer.

---

## Phase 4 — Coverage and freshness

### 4.1 ⬜ More document types
- Beige Book, FOMC speeches, SEP scrapers under `src/scraper/`, each writing the
  same `data/raw` + metadata layout the ingest pipeline expects.
- **Verify:** ingest runs across mixed types; `document_type` filter works.

### 4.2 ⬜ Incremental, idempotent ingest
- Dedup by content hash / `(document_type, meeting_date, chunk_idx)` so re-runs
  don't duplicate vectors; support "ingest only new since last run".
- **Verify:** running ingest twice leaves the vector count unchanged.

### 4.3 ⬜ Docker Compose for local pgvector
- `docker-compose.yml` bringing up Postgres + pgvector for the production
  backend path, documented in the README.
- **Verify:** `VECTOR_BACKEND=pgvector` ingest + query works against the
  compose stack.

---

## Suggested order

1. **Phase 1** in full — without eval + tests, every later change is a guess.
2. **2.1 → 2.2** (hybrid + rerank) — biggest retrieval-quality wins.
3. **3.1 → 3.3** — turns better retrieval into a better product.
4. **Phase 4** as coverage demand grows.

## Verification commands

- Lint: `python -m ruff check src scripts app.py`
- Tests: `python -m pytest -q`
- Retrieval eval: `python scripts/eval_retrieval.py`
