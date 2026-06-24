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

## Phase 3 — Make the retrieval screen & results more helpful

Today each source shows only `[n] date · similarity` + raw text. The goal: every
result should let the user **verify the claim at the source in one click** and
**see why it matched**. The chunk metadata already carried end-to-end
(`pdf_url`, `meeting_date`, `release_date`, `document_type`, `chunk_index`,
`total_chunks`) makes most of this cheap.

### 3.1 ⬜ Open-the-document links
- Render each source's `pdf_url` as a clickable "📄 Open PDF" link. The Fed
  serves these at `federalreserve.gov/monetarypolicy/files/...`, already in
  metadata — no new data needed.
- Where the parser can capture it, store a `page` number per chunk and deep-link
  to it (`...pdf#page=N`) so the user lands near the quote.
- **Verify:** clicking a source opens the correct FOMC minutes PDF (and page,
  once page numbers are stored).

### 3.2 ⬜ Document location & context
- Show a clear provenance line per result: document type · meeting date ·
  "chunk N of M" so the user knows *where* in the document it came from.
- Capture the nearest section heading during parsing/chunking (e.g.
  "Developments in Financial Markets") into chunk metadata and display it.
- **Verify:** each result shows a human-readable location; section headings are
  present for newly ingested docs.

### 3.3 ⬜ Better snippets (highlight + expand)
- Show a trimmed, query-relevant snippet by default with **highlighted query
  terms**; add an "expand" toggle for the full chunk and a "± neighbors" toggle
  that pulls the adjacent `chunk_index` rows for surrounding context.
- Add a one-click "Copy quote with citation" (text + date + URL).
- **Verify:** highlights match query terms; expand/neighbors return the right
  adjacent chunks.

### 3.4 ⬜ Result controls & transparency
- Sidebar metadata filters (date-range / year / document type) passed to
  `Retriever.search(metadata_filter=...)` — already supported end-to-end.
- Show the similarity score as a small bar/badge and let the user sort by score
  or date; surface "no strong matches" when top score is below a threshold.
- **Verify:** filtering to one year returns only that year; low-confidence
  queries are visibly flagged.

### 3.5 ⬜ Answer-faithfulness guardrail
- When the LLM cites `[n]` not in the retrieved set, or returns no citation,
  flag it in the UI rather than presenting it as grounded.
- **Verify:** a deliberately off-topic question shows the "not enough
  information" path, not a hallucinated answer.

**Suggested first slice (high value, low effort):** 3.1 open-PDF links + 3.2
provenance line + 3.3 highlighted snippet — all drivable from existing metadata,
no re-ingest required.

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

### 4.3 ✅ Docker Compose for local pgvector
- `docker-compose.yml` brings up `pgvector/pgvector:pg16` with a healthcheck and
  a named volume; pgvector is now the project default backend.
- `scripts/migrate_chroma_to_pgvector.py` loads the shipped Chroma vectors into
  Postgres (no re-scrape). The HF Space stays on Chroma via `VECTOR_BACKEND` in
  the Dockerfile.
- **Done:** `docker compose up -d` → migrate → `python scripts/query.py ...`
  returns hits from pgvector.

---

## Phase 5 — Harden the code

The skeleton favored clarity over robustness. Before this is something to trust
with real users (or a public Space), close these gaps. Roughly ordered by
risk-reduction per unit effort.

### 5.1 ⬜ Security: SQL, secrets, and inputs
- **Parameterize identifiers / validate table name.** `PgVectorStore` interpolates
  `self.table_name` into every SQL string (`src/vectorstore/pgvector_store.py`).
  It's not user-supplied today, but allow-list it against `^[A-Za-z_][A-Za-z0-9_]*$`
  so it can never become an injection vector. Values are already parameterized — keep it that way.
- **Never log secrets.** Ensure `DATABASE_URL` / `OPENAI_API_KEY` are never logged
  (the connection log currently prints table/dim only — keep it so).
- **Scraper/URL safety.** In the scraper, validate that download URLs stay on
  `federalreserve.gov`, set timeouts, and cap response sizes.
- **Verify:** add tests for a rejected bad table name; grep logs for secret leakage.

### 5.2 ⬜ Robust LLM/HTTP calls
- Add retries with backoff + explicit timeouts for OpenAI/Ollama calls
  (`src/rag/llm.py` already sets timeouts; add bounded retries on 5xx/timeouts).
- Cap context size sent to the LLM (token/char budget) so large `k` can't blow
  past model limits or cost.
- **Verify:** unit tests with a mocked `requests` simulating timeout → retry → fail-soft
  to the extractive fallback.

### 5.3 ⬜ Connection & resource lifecycle
- `PgVectorStore` opens a single connection with `autocommit=True`; add reconnect-on-
  drop and ensure `close()` is always called (the Streamlit `@st.cache_resource`
  pipeline holds one for the process — document/guard that).
- Consider a small connection pool if concurrency grows.
- **Verify:** killing/restarting the DB container mid-session recovers without a crash.

### 5.4 ⬜ Input validation & error surfaces
- Validate `k` (1..N), empty/oversized queries, and metadata-filter shapes at the
  `Retriever`/UI boundary with clear messages.
- Replace bare `except Exception` swallowing where it hides real bugs; log with
  context and fail toward the graceful path only for known, expected failures.
- **Verify:** tests for empty query, huge `k`, malformed filter.

### 5.5 ⬜ Tooling & CI gate
- Add `pyproject.toml` with ruff config (ignore `E402` in `scripts/` for the
  `sys.path` bootstrap), pytest config, and `bandit` for security linting.
- GitHub Actions: run ruff + pytest + bandit on push/PR.
- Pin/lock dependencies (hashes or a lockfile) for reproducible, audited installs.
- **Verify:** CI is green and blocks merges on lint/test/security failure.

### 5.6 ⬜ Index & data correctness
- IVFFlat with `lists=100` on ~4k rows returns *approximate* neighbors and needs
  `ANALYZE` + a sensible `ivfflat.probes`; for this corpus size consider HNSW or
  raising probes. (Symptom seen: a 2023 query surfacing 2019 chunks.)
- Add `ANALYZE` after bulk loads and document tuning knobs.
- **Verify:** recall@k on the Phase 1 eval set before/after tuning.

---

## Suggested order

1. **Phase 1** in full — without eval + tests, every later change is a guess.
2. **5.1 + 5.5** — close the security gaps and stand up a CI gate early.
3. **2.1 → 2.2** (hybrid + rerank) and **5.6** — biggest retrieval-quality wins.
4. **Phase 3** (3.1→3.3 first) — turns better retrieval into a better product.
5. **Phase 4 / remaining Phase 5** as coverage and load grow.

## Verification commands

- Lint: `python -m ruff check src scripts app.py`
- Tests: `python -m pytest -q`
- Retrieval eval: `python scripts/eval_retrieval.py`
- Start the vector DB: `docker compose up -d`
- Migrate vectors into pgvector: `python scripts/migrate_chroma_to_pgvector.py`
