# FedInsight — Project Specification & Architecture

## 1. Project Goal

Build a production-grade RAG system for working with Federal Reserve documents.

The system should be able to:
- Collect documents (FOMC Minutes, Beige Book, Speeches, SEP, etc.)
- Turn them into a searchable knowledge base
- Answer questions with accurate quotes and sources
- Work fully locally (or easily switchable to API)

**Тarget audience**: quants, economists, traders, students, and monetary policy researchers.

## 2. High-level Architecture

```
Raw Documents (PDF/HTML)
        ↓
   Scraper Layer
        ↓
   Parser + Cleaner
        ↓
   Chunker (semantic / recursive)
        ↓
   Embedder (sentence-transformers)
        ↓
   Vector Store (pgvector)
        ↓
   Retriever (similarity + metadata filter)
        ↓
   RAG Pipeline (context + query → LLM)
        ↓
   Streamlit UI / API
```

### Components

| Layer                | Technology                     | Status     | Location                  |
|----------------------|--------------------------------|------------|---------------------------|
| Scraper              | requests + BeautifulSoup       | In progress| src/scraper/              |
| Parser               | pypdf / pdfplumber             | Planned    | src/parser/               |
| Chunker              | LangChain or custom            | Planned    | src/chunker/              |
| Embeddings           | sentence-transformers          | Planned    | src/embeddings/           |
| Vector DB            | pgvector (PostgreSQL)          | Planned    | src/vectorstore/          |
| RAG Orchestration    | LangChain / custom             | Planned    | src/rag/                  |
| UI                   | Streamlit                      | Planned    | src/ui/                   |
| Evaluation           | custom + RAGAS (optional)      | Planned    | src/eval/                 |

## 3. Data

### Main sources (priority)

1. **FOMC Minutes** — most important (8 times per year + extraordinary meetings)
2. **Beige Book** — regional economic conditions (8 times per year)
3. **FOMC Speeches** — speeches by FOMC members
4. **Summary of Economic Projections (SEP)**
5. **Monetary Policy Report** (semiannual)
6. **Financial Stability Report**

### Storage format

- `data/raw/fomc_minutes/YYYY-MM-DD_fomc_minutes.pdf`
- `data/processed/fomc_minutes/YYYY-MM-DD.json` (text + metadata)
- PostgreSQL database with pgvector extension for embeddings

**Minimum metadata**:
- `meeting_date`
- `release_date`
- `document_type`
- `url`
- `title`
- `participants` (if extractable)

## 4. Development Roadmap

### Phase 0 — Initialization (current)
- [x] Project structure
- [x] README + PROJECT_SPEC (English)
- [x] requirements.txt
- [x] FOMC Minutes scraper (improved version)

### Phase 1 — Skeleton (priority right now)
Build a **working end-to-end pipeline** as fast as possible:
- PDF parser
- Simple chunker
- Embeddings
- pgvector storage + retrieval
- Basic RAG query (CLI or simple script)
- Then we improve quality and add features

### Phase 2 — Core improvements
- Better PDF text extraction and cleaning
- Semantic chunking
- Metadata filtering in pgvector
- Proper prompt engineering with source attribution

### Phase 3 — Full features
- Beige Book + Speeches scrapers
- Streamlit interface
- Hybrid search / reranking
- Evaluation framework

### Phase 4 — Production touches (later)
- Caching
- Incremental updates
- Docker compose for easy local Postgres + pgvector
- Advanced RAG techniques (query rewriting, multi-step reasoning, etc.)

## 5. Technical decisions

### Why pgvector?
- More production-ready than Chroma for many use cases
- Excellent metadata filtering + SQL power
- You can run it locally with Docker or use managed services (Supabase, Neon, etc.)
- Good performance and ecosystem

### Embeddings
We start with `sentence-transformers/all-MiniLM-L6-v2` (fast, 384 dim).  
Later we can upgrade to better models (`bge-m3`, `e5-large`, etc.).

### LLM
- Development: Grok, Claude 3.5, or GPT-4o
- Production/local: Llama-3.1, Qwen2, Mistral, etc. via Ollama or vLLM

## 6. Project principles

- **Skeleton first**: Get a working end-to-end flow before polishing individual parts
- **Observable**: Good logging and progress bars
- **Resumable**: Scrapers and pipelines should support resuming
- **Clean & educational**: Code should be readable and well-documented
- **English only**: All code, comments, docs, and messages are in English

---

**Сurrent status**: Phase 0 → Phase 1 (building the PDF parser and moving toward end-to-end skeleton)

Ready when you are. Just say the word.