"""
Vector store layer: persist and query embeddings.

Two interchangeable backends implement the same interface (see base.py):
    - "chroma"   : local, server-less ChromaDB (default, great for testing)
    - "pgvector" : PostgreSQL + pgvector (production, needs a running server)

Select the backend with the VECTOR_BACKEND environment variable, or override
per call via get_vector_store(backend=...).

    from src.vectorstore import get_vector_store
    store = get_vector_store()              # uses VECTOR_BACKEND (default "chroma")
    store = get_vector_store("pgvector")    # explicit override
"""

import os
from typing import Any

from .base import BaseVectorStore

# Global backend selector. Change the env var (or DEFAULT_BACKEND) to switch.
DEFAULT_BACKEND = "chroma"
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", DEFAULT_BACKEND).strip().lower()


def get_vector_store(backend: str | None = None, **kwargs: Any) -> BaseVectorStore:
    """
    Factory that returns the configured vector-store backend.

    Args:
        backend: "chroma" or "pgvector". If None, uses the VECTOR_BACKEND global.
        **kwargs: forwarded to the backend constructor (e.g. collection_name,
                  embedding_dim, persist_dir, connection_string).
    """
    chosen = (backend or VECTOR_BACKEND).strip().lower()

    if chosen in ("chroma", "chromadb"):
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(**kwargs)

    if chosen in ("pgvector", "postgres", "postgresql"):
        from .pgvector_store import PgVectorStore

        return PgVectorStore(**kwargs)

    raise ValueError(
        f"Unknown vector store backend: {chosen!r}. "
        f"Expected one of: 'chroma', 'pgvector'."
    )


__all__ = ["BaseVectorStore", "get_vector_store", "VECTOR_BACKEND", "DEFAULT_BACKEND"]
