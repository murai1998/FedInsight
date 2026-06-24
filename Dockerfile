# FedInsight — Streamlit RAG app on Hugging Face Spaces (Docker SDK).
# HF removed the managed Streamlit SDK, so we run Streamlit ourselves in Docker.
FROM python:3.11-slim

# Build tools for any packages without manylinux wheels (e.g. hnswlib via chromadb).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The Space ships a prebuilt Chroma store and has no Postgres, so pin the
# backend to chroma here (the project default is pgvector for local use).
ENV VECTOR_BACKEND=chroma \
    # Caches that must be writable at runtime. HF model downloads go to HF_HOME.
    HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1

# Install deps first for better layer caching. requirements.txt pins CPU torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the prebuilt Chroma vector store (shipped via Git LFS).
COPY . .

# Make app dir (incl. data/chroma and caches) writable for the runtime user.
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache /app/data

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
