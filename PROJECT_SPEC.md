# FedInsight — Project Specification & Architecture

## 1. Project Goal

A RAG (Retrieval-Augmented Generation) system for working with U.S. Federal
Reserve communications. It:

- Collects Fed documents (FOMC Minutes today; Beige Book, speeches, SEP planned)
- Turns them into a searchable vector knowledge base
- Answers natural-language questions with accurate quotes and source attribution
- Runs fully locally (no mandatory API keys), or switches to a hosted LLM

**Target audience**: quants, economists, traders, students, and monetary-policy
researchers.

## 2. High-level Architecture

```
Raw PDFs (federalreserve.gov)
        ↓  src/scraper/        requests + BeautifulSoup
   Scraper Layer
        ↓  src/parser/         PyMuPDF (column-aware), pdfplumber fallback
   Parser + Cleaner
        ↓  src/chunker/        RecursiveCharacterTextSplitter
   Chunker
        ↓  src/embeddings/     sentence-transformers (CUDA / Apple MPS / CPU)
   Embedder
        ↓  src/vectorstore/    pgvector (default) | ChromaDB
   Vector Store
        ↓  src/rag/retriever   cosine top-k + metadata filter
   Retriever
        ↓  src/rag/rag_pipeline grounded prompt → LLM, cited sources
   RAG Pipeline
        ↓  app.py              Streamlit chat UI
   UI
```

### Components & status

| Layer             | Technology                          | Status | Location              |
|-------------------|-------------------------------------|--------|-----------------------|
| Scraper           | requests + BeautifulSoup            | ✅ FOMC Minutes | `src/scraper/`   |
| Parser            | PyMuPDF (+ pdfplumber fallback)     | ✅     | `src/parser/`         |
| Chunker           | langchain-text-splitters            | ✅     | `src/chunker/`        |
| Embeddings        | sentence-transformers `all-MiniLM-L6-v2` | ✅ | `src/embeddings/`     |
| Vector DB         | pgvector (default) / ChromaDB       | ✅     | `src/vectorstore/`    |
| Retriever         | cosine top-k + metadata filter      | ✅     | `src/rag/retriever.py`|
| RAG pipeline      | OpenAI / Ollama / offline fallback  | ✅     | `src/rag/`            |
| UI                | Streamlit                           | ✅     | `app.py`              |
| Deployment        | Docker → Hugging Face Spaces        | ✅     | `Dockerfile`          |
| Evaluation        | retrieval metrics + RAGAS (optional)| ⬜ planned | `eval/` (planned) |
| Test suite        | pytest                              | ⬜ planned | `tests/` (planned)|

## 3. Data

### Sources (priority order)

1. **FOMC Minutes** — implemented (8 regular meetings/year + extraordinary)
2. **Beige Book** — planned
3. **FOMC Speeches** — planned
4. **Summary of Economic Projections (SEP)** — planned
5. **Monetary Policy Report**, **Financial Stability Report** — later

### Storage layout

- `data/raw/fomc_minutes/YYYY-MM-DD_fomc_minutes.pdf` — downloaded PDFs
- `data/processed/…` — scraper metadata + logs
- Vectors: PostgreSQL/pgvector table `fed_chunks` (default), or on-disk Chroma
  at `data/chroma/` (used by the shipped/Hugging Face build)

**Chunk metadata** (carried end-to-end into search results): `meeting_date`,
`release_date`, `document_type`, `pdf_url`, `chunk_index`, `total_chunks`.

> **Note:** the repo ships a prebuilt Chroma store (~4,075 FOMC chunks) but no
> source PDFs. `scripts/migrate_chroma_to_pgvector.py` copies those vectors into
> pgvector, so you get the full knowledge base without re-scraping.

## 4. Vector backends

Two interchangeable backends implement one interface (`src/vectorstore/base.py`),
selected by `VECTOR_BACKEND`:

- **pgvector (default)** — PostgreSQL + pgvector. Strong metadata filtering
  (JSONB `@>`), SQL power, production-ready. Start locally with
  `docker compose up -d` (see `docker-compose.yml`).
- **ChromaDB** — server-less, on-disk. Zero setup; used by the Hugging Face
  Space, which pins `VECTOR_BACKEND=chroma` in the `Dockerfile`.

## 5. Technical decisions

- **Embeddings**: `all-MiniLM-L6-v2` (384-dim, fast). The embedder auto-selects
  the fastest device — **CUDA → Apple MPS → CPU** — and normalizes vectors for
  cosine similarity. Upgrade path: `bge-*` / `e5-*` (re-ingest required).
- **LLM**: provider is `auto` — OpenAI if `OPENAI_API_KEY` is set, else a local
  Ollama server if reachable, else `none` (returns retrieved excerpts, fully
  offline). See `src/rag/llm.py`.
- **Chunking**: 800-char chunks, 150-char overlap (tunable via ingest flags).

## 6. Development Roadmap

- **Phase 0 — Initialization** ✅ structure, scraper, requirements, docs
- **Phase 1 — End-to-end skeleton** ✅ parse → chunk → embed → store → retrieve → generate, both backends, Streamlit UI, Docker deploy
- **Phase 2 — Retrieval quality** ⬜ eval harness, hybrid (dense+BM25) search, reranking, better chunking *(see `plan.md`)*
- **Phase 3 — UX & trust** ⬜ source links/snippets, metadata filters in UI, faithfulness guardrails *(see `plan.md`)*
- **Phase 4 — Coverage & hardening** ⬜ more document types, incremental ingest, test suite, input/SQL hardening *(see `plan.md`)*

## 7. Project principles

- **Skeleton first**: a working end-to-end flow before polishing parts (done).
- **Observable**: good logging and progress bars (loguru + tqdm).
- **Resumable**: scrapers and pipelines support resuming.
- **Interchangeable backends**: nothing above the vector store knows which DB is in use.
- **Clean & educational**: readable, well-documented code; English only.

---

**Current status**: Phase 1 complete (end-to-end RAG on pgvector by default).
Next: Phase 2 retrieval quality. See [plan.md](plan.md) for the detailed,
prioritized next steps.
