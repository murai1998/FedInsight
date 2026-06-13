# FedInsight 📈

> **AI-powered semantic search and RAG over Federal Reserve communications**  
> FOMC Minutes • Beige Book • Speeches • SEP • and more

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FedInsight is a complete RAG (Retrieval-Augmented Generation) project that lets you "talk" to Federal Reserve documents. Ask questions in natural language and get accurate quotes + synthesized answers with sources.

## Why this is useful

- Analysts and traders want to quickly find what the Fed said about inflation, interest rates, or the labor market in specific periods.
- Researchers study the evolution of monetary policy rhetoric over time.
- It's just interesting: "What did the Fed say about the 2023 banking crisis?"

## Current capabilities

- ✅ Full FOMC Minutes scraper (with metadata, resume support, logging)
- ✅ PDF parsing → clean text + metadata (PyMuPDF, column-aware)
- ✅ Chunking + embeddings (sentence-transformers, GPU-accelerated)
- ✅ Vector database: ChromaDB (local default) **or** pgvector (PostgreSQL)
- ✅ RAG pipeline (retrieval + generation with source attribution)
- ✅ Streamlit chat interface
- ✅ Fully local operation (no mandatory API keys)
- 🔄 Beige Book scraper (planned)

## Quick start

```bash
git clone https://github.com/amorosov2006/FedInsight.git
cd FedInsight

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows (PowerShell)

pip install -r requirements.txt
```

> **NVIDIA GPU (optional but recommended):** the default `torch` wheel is CPU-only.
> For GPU acceleration install the matching CUDA build, e.g. for an RTX 50-series
> (Blackwell) card:
>
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cu128 torch
> ```
>
> The embedder auto-detects CUDA and uses it when available.

## Usage — the full pipeline, step by step

Run everything from the project root **with the virtualenv activated** (so the
correct Python/torch is used). On Windows you can also prefix commands with
`.\.venv\Scripts\python.exe` instead of activating.

Optional configuration lives in `.env` (copy from `.env.example`):

```bash
cp .env.example .env   # then edit as needed (vector backend, LLM provider, keys)
```

### 1. Scrape FOMC minutes (downloads PDFs + metadata)

```bash
python src/scraper/fomc_minutes_scraper.py            # full historical scrape
python src/scraper/fomc_minutes_scraper.py --years 2022 2023 2024
```

### 2. Ingest: parse → chunk → embed → store in the vector DB

```bash
python scripts/ingest_sample.py --clear               # ingest everything (fresh)
python scripts/ingest_sample.py --limit 5             # quick test on 5 docs
python scripts/ingest_sample.py --backend chroma      # force a backend
```

Default backend is **ChromaDB** (local, no server). To use **pgvector**, set
`VECTOR_BACKEND=pgvector` in `.env` and point it at a running Postgres.

### 3a. Simple semantic search (retrieval only, no LLM)

```bash
python scripts/query.py "What did the Fed say about inflation in 2023?"
python scripts/query.py "labor market" --k 10
```

### 3b. Ask a question (full RAG: retrieval + generated answer + sources)

```bash
python scripts/ask.py "How did the Committee view inflation risks in 2022?"
python scripts/ask.py "What was decided about the federal funds rate?" --k 8
python scripts/ask.py "Balance sheet plans" --provider ollama
```

The LLM provider is set by `LLM_PROVIDER` (default `auto`): OpenAI if
`OPENAI_API_KEY` is set, else a local Ollama server if reachable, else `none`
(prints the most relevant excerpts — works fully offline, no keys required).

### 4. Chat UI (Streamlit)

```bash
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501). Pick the backend,
LLM provider, and number of retrieved chunks in the sidebar.

## How to get the project

### Recommended: Clone from GitHub

```bash
git clone https://github.com/amorosov2006/FedInsight.git
cd FedInsight
```

**Advantages:**
- Always up-to-date version
- Easy to work locally
- You can commit your own changes
- I can push updates directly to this repository

### Alternative: Download from Grok Sandbox

If you want to grab the files directly from this chat:

1. In the Grok interface, find the **Download / Export / Build** button for the project
2. Select the `FedInsight/` folder
3. Download the archive and unzip it

This works but is less convenient than Git.

## Current workflow

- I develop in the sandbox and periodically push changes to your GitHub repository
- You just `git pull` on your side
- You can also commit and push yourself anytime

## Project structure

```
FedInsight/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── .env.example             # copy to .env to configure
├── app.py                   # Streamlit chat UI  (step 4)
├── src/
│   ├── scraper/
│   │   └── fomc_minutes_scraper.py
│   ├── parser/
│   │   └── pdf_parser.py     # PyMuPDF-based, column-aware
│   ├── chunker/
│   │   └── text_chunker.py
│   ├── embeddings/
│   │   └── embedder.py       # sentence-transformers, auto CUDA
│   ├── vectorstore/
│   │   ├── base.py           # shared interface
│   │   ├── chroma_store.py   # default local backend
│   │   ├── pgvector_store.py # PostgreSQL backend
│   │   └── __init__.py       # get_vector_store() factory
│   └── rag/
│       ├── retriever.py      # query → top-k chunks
│       ├── llm.py            # openai / ollama / none
│       └── rag_pipeline.py   # retrieve + generate + sources
├── scripts/
│   ├── ingest_sample.py      # parse→chunk→embed→store  (step 2)
│   ├── query.py              # semantic search          (step 3a)
│   └── ask.py                # full RAG Q&A             (step 3b)
├── data/
│   ├── raw/                  # downloaded PDFs
│   ├── processed/            # metadata + logs
│   └── chroma/               # local Chroma DB (auto-created)
└── notebooks/
```

## Tech stack

- **Scraping**: requests + BeautifulSoup
- **PDF parsing**: pypdf / pdfplumber
- **Embeddings**: sentence-transformers
- **Vector DB**: pgvector (PostgreSQL)
- **Orchestration**: LangChain (or clean custom code for learning)
- **UI**: Streamlit
- **Logging**: loguru

## Development philosophy

We follow a **working skeleton first** approach:
1. Build a minimal but complete end-to-end pipeline (scrape → parse → chunk → embed → store → retrieve → generate)
2. Then iteratively improve quality, features, and robustness.

## How to continue development

Just write messages like:
- "Continue FedInsight"
- "Improve the scraper"
- "Build the PDF parser"
- "Add Beige Book scraper"
- "Create the pgvector store"
- "Build Streamlit interface"

I will work inside the `FedInsight/` folder, keep everything in English, and push updates to your repository.

---

Made with love for monetary policy and clean code ❤️

*Learning project: building a full RAG + Vector DB system from scratch.*