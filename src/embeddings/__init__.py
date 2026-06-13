"""Embeddings layer: turn text chunks into vectors."""

from .embedder import Embedder, get_default_embedder

__all__ = ["Embedder", "get_default_embedder"]
