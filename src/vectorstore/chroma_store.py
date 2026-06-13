#!/usr/bin/env python3
"""
FedInsight - ChromaDB Vector Store

A local, server-less vector store backend used for development and testing.
ChromaDB persists to disk, so no Docker or external database is required.

This implements the same interface as PgVectorStore (see base.py), so the two
backends are interchangeable via the factory in this package.

Environment variables:
    CHROMA_PERSIST_DIR   directory for the on-disk database (default: data/chroma)
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Sequence

import numpy as np
from loguru import logger

from .base import BaseVectorStore

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "fed_chunks"
# Scalar types ChromaDB accepts in metadata
_SCALAR_TYPES = (str, int, float, bool)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB only accepts str/int/float/bool metadata values and rejects None.
    Drop None values and stringify anything non-scalar so ingestion never fails.
    """
    clean: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class ChromaVectorStore(BaseVectorStore):
    """Store and query document chunk embeddings in a local ChromaDB."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        **_ignored: Any,
    ):
        # embedding_dim is accepted for interface parity but Chroma infers it.
        import chromadb

        self.collection_name = collection_name
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR
        )
        self.embedding_dim = embedding_dim

        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(
            f"Opening ChromaDB | dir='{self.persist_dir}' "
            f"| collection='{self.collection_name}'"
        )

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.success(f"ChromaDB store ready ({self.count()} existing chunks)")

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
    ) -> int:
        if len(chunks) == 0:
            logger.warning("add_documents called with no chunks")
            return 0

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )

        ids = [uuid.uuid4().hex for _ in chunks]
        documents = [chunk.get("content", "") for chunk in chunks]
        metadatas = [_sanitize_metadata(chunk.get("metadata", {})) for chunk in chunks]
        embeddings_list = embeddings.tolist()

        # ChromaDB enforces a hard cap on items per add() call. Split into
        # batches that stay safely under that limit.
        batch_size = self._max_batch_size()
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            if total > batch_size:
                logger.info(f"  added {end}/{total} chunks")

        logger.success(
            f"Inserted {total} chunks into Chroma collection "
            f"'{self.collection_name}'"
        )
        return total

    def _max_batch_size(self) -> int:
        """Largest number of items allowed in a single add() call."""
        # Newer Chroma exposes this on the client; fall back conservatively.
        for getter in ("get_max_batch_size",):
            fn = getattr(self.client, getter, None)
            if callable(fn):
                try:
                    return max(1, int(fn()))
                except Exception:
                    pass
        value = getattr(self.client, "max_batch_size", None)
        if isinstance(value, int) and value > 0:
            return value
        return 5000

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=metadata_filter or None,
        )

        # Chroma returns lists-of-lists (one per query); we only sent one query.
        ids = res.get("ids", [[]])[0]
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        results = []
        for i in range(len(ids)):
            distance = float(distances[i])
            results.append(
                {
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i] or {},
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine similarity
                }
            )

        logger.info(f"similarity_search returned {len(results)} results (k={k})")
        return results

    # ------------------------------------------------------------------ #
    # Management
    # ------------------------------------------------------------------ #
    def clear_collection(self) -> None:
        logger.warning(f"Clearing Chroma collection '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    # Quick local smoke test (no server needed)
    store = ChromaVectorStore(collection_name="smoke_test")
    store.clear_collection()

    sample_chunks = [
        {"content": "The Fed raised interest rates.", "metadata": {"year": 2023}},
        {"content": "Unemployment remained low.", "metadata": {"year": 2024}},
    ]
    sample_embeddings = np.random.rand(2, 8).astype(np.float32)

    store.add_documents(sample_chunks, sample_embeddings)
    print(f"Count after insert: {store.count()}")

    hits = store.similarity_search(sample_embeddings[0], k=2)
    for h in hits:
        print(h["content"], "| score=", round(h["score"], 3))

    store.clear_collection()
    store.close()
