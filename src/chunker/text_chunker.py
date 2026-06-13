#!/usr/bin/env python3
"""
FedInsight - Text Chunker

Splits long documents into smaller chunks suitable for embedding and retrieval.

For the initial skeleton we use LangChain's RecursiveCharacterTextSplitter.
It is reliable, respects sentence boundaries reasonably well, and is widely used.

Later we can replace it with a more advanced semantic chunker if needed.
"""

from typing import List, Dict, Any
from loguru import logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    metadata: Dict[str, Any] | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Split a document into chunks while preserving metadata.

    Each returned dict contains:
        - content: chunk text
        - metadata: original metadata + chunk-specific info (chunk_index, etc.)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to chunker")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    result = []
    base_metadata = metadata.copy() if metadata else {}

    for i, chunk in enumerate(chunks):
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_index": i,
            "chunk_size": len(chunk),
            "total_chunks": len(chunks),
        })

        result.append({
            "content": chunk,
            "metadata": chunk_metadata
        })

    logger.info(f"Created {len(result)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return result


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk multiple parsed documents at once.

    Each document should have at least 'text' and optionally 'metadata'.
    """
    all_chunks = []

    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})

        chunks = chunk_text(text, meta, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.success(f"Chunked {len(documents)} documents → {len(all_chunks)} total chunks")
    return all_chunks


if __name__ == "__main__":
    # Quick test
    sample_text = """
    The Federal Open Market Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent.

    The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.

    In assessing the appropriate stance of monetary policy, the Committee will continue to monitor the implications of incoming information.
    """ * 5

    chunks = chunk_text(sample_text, metadata={"meeting_date": "2024-01-31"}, chunk_size=300, chunk_overlap=50)

    print(f"Created {len(chunks)} chunks")
    for i, c in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i} ---")
        print(c["content"][:200])
        print(c["metadata"])