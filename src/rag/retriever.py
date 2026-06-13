#!/usr/bin/env python3
"""
FedInsight - Retriever

Ties together the embedder and the vector store: turns a natural-language query
into an embedding and returns the most relevant document chunks.

This is the shared building block used by both the simple query script and the
full RAG pipeline.
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from src.embeddings import get_default_embedder, Embedder
from src.vectorstore import get_vector_store, BaseVectorStore


class Retriever:
    """Embed a query and fetch the top-k similar chunks from the vector store."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[BaseVectorStore] = None,
        backend: Optional[str] = None,
    ):
        self.embedder = embedder or get_default_embedder()
        self.store = store or get_vector_store(
            backend=backend, embedding_dim=self.embedder.dimension
        )

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the k most relevant chunks for a query.

        Returns a list of dicts: {id, content, metadata, distance, score}
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to Retriever.search")
            return []

        query_embedding = self.embedder.embed_single(query)
        results = self.store.similarity_search(
            query_embedding, k=k, metadata_filter=metadata_filter
        )
        return results
