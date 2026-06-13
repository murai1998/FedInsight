#!/usr/bin/env python3
"""
FedInsight - Sample Ingestion Script (Skeleton

This script creates a working end-to-end pipeline:
    PDF files + metadata  →  Parser  →  Chunker  →  Embedder  →  pgvector

Usage examples:
    # Ingest up to 5 most recent documents
    python scripts/ingest_sample.py --limit 5

    # Ingest everything (can be slow on first run)
    python scripts/ingest_sample.py

    # Clear existing data and re-ingest
    python scripts/ingest_sample.py --limit 3 --clear
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

# Make the project root importable when running this script directly
# (e.g. `python scripts/ingest_sample.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from src.parser.pdf_parser import parse_pdf
from src.chunker.text_chunker import chunk_documents
from src.embeddings.embedder import get_default_embedder
from src.vectorstore import get_vector_store, VECTOR_BACKEND


def load_scraper_metadata(metadata_path: Path = Path("data/processed/fomc_minutes_metadata.json")) -> List[Dict]:
    """Load metadata produced by the FOMC scraper."""
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.info("Please run the scraper first: python src/scraper/fomc_minutes_scraper.py")
        return []

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Loaded {len(metadata)} records from scraper metadata")
    return metadata


def find_pdf_files(raw_dir: Path = Path("data/raw/fomc_minutes")) -> List[Path]:
    """Find all PDF files in the raw directory."""
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return []

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {raw_dir}")
    return pdf_files


def main():
    parser = argparse.ArgumentParser(description="FedInsight Sample Ingestion Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents to process (useful for testing)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing data in pgvector before ingesting")
    parser.add_argument("--chunk-size", type=int, default=800,
                        help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=150,
                        help="Chunk overlap in characters")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["chroma", "pgvector"],
                        help="Vector store backend (default: VECTOR_BACKEND env var)")
    args = parser.parse_args()

    logger.info("=== FedInsight Ingestion Pipeline (Skeleton) ===")

    # 1. Load metadata from scraper
    metadata_list = load_scraper_metadata()
    if not metadata_list:
        return

    # 2. Find PDF files
    pdf_files = find_pdf_files()
    if not pdf_files:
        return

    # Apply limit if specified
    if args.limit:
        pdf_files = pdf_files[-args.limit:]  # take the most recent ones
        logger.info(f"Limited to last {args.limit} documents")

    # 3. Initialize components
    embedder = get_default_embedder()
    backend = args.backend or VECTOR_BACKEND
    logger.info(f"Using vector store backend: {backend}")
    vector_store = get_vector_store(
        backend=args.backend, embedding_dim=embedder.dimension
    )

    if args.clear:
        vector_store.clear_collection()

    # 4. Process documents
    parsed_docs = []
    all_chunks = []

    for pdf_path in pdf_files:
        # Try to find matching metadata
        meeting_date = pdf_path.stem.split("_")[0]  # e.g. 2024-01-31 from filename
        meta = next((m for m in metadata_list if m.get("meeting_date") == meeting_date), {})

        try:
            # Parse
            parsed = parse_pdf(pdf_path, metadata=meta)
            parsed_docs.append(parsed)

            # Chunk
            chunks = chunk_documents(
                [parsed],
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    if not all_chunks:
        logger.warning("No chunks were created. Exiting.")
        return

    logger.info(f"Total chunks ready for embedding: {len(all_chunks)}")

    # 5. Generate embeddings
    texts = [chunk["content"] for chunk in all_chunks]
    embeddings = embedder.embed_texts(texts)

    # 6. Store in the vector backend
    vector_store.add_documents(all_chunks, embeddings)

    logger.success("=== Ingestion completed successfully ===")
    logger.info(
        f"Processed {len(parsed_docs)} documents → {len(all_chunks)} chunks "
        f"stored in '{backend}' (total now: {vector_store.count()})"
    )


if __name__ == "__main__":
    main()