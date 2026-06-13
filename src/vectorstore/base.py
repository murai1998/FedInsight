#!/usr/bin/env python3
"""
FedInsight - Vector Store Interface

Defines the common contract every vector-store backend must implement so that
the rest of the pipeline (ingestion, retrieval, RAG) does not care whether the
embeddings live in ChromaDB (local/testing) or pgvector (PostgreSQL/production).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence

import numpy as np


class BaseVectorStore(ABC):
    """Abstract interface shared by all vector-store backends."""

    @abstractmethod
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
    ) -> int:
        """
        Insert chunks and their embeddings.

        Args:
            chunks: list of dicts with keys 'content' and 'metadata'
            embeddings: array-like of shape (len(chunks), embedding_dim)

        Returns:
            number of rows inserted
        """

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the k most similar chunks by cosine distance.

        Each result is a dict with keys:
            {id, content, metadata, distance, score}
        where score = 1 - cosine_distance.
        """

    @abstractmethod
    def clear_collection(self) -> None:
        """Remove all stored chunks (keeps the backend usable)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""

    def close(self) -> None:
        """Release any resources. Override if the backend needs cleanup."""

    def __enter__(self) -> "BaseVectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
