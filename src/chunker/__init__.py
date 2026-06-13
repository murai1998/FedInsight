"""Chunker layer: split documents into embeddable chunks."""

from .text_chunker import chunk_text, chunk_documents

__all__ = ["chunk_text", "chunk_documents"]
