#!/usr/bin/env python3
"""
FedInsight - Migrate vectors from ChromaDB into pgvector.

The repo ships a prebuilt Chroma store (data/chroma) but no source PDFs, so this
copies the existing chunks (text + metadata + embeddings) straight into
PostgreSQL/pgvector. No re-scraping or re-embedding required.

Usage:
    docker compose up -d                              # start Postgres first
    python scripts/migrate_chroma_to_pgvector.py            # migrate everything
    python scripts/migrate_chroma_to_pgvector.py --clear    # wipe pg table first
    python scripts/migrate_chroma_to_pgvector.py --batch-size 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.chroma_store import ChromaVectorStore, DEFAULT_COLLECTION  # noqa: E402
from src.vectorstore.pgvector_store import PgVectorStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate Chroma vectors into pgvector")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="Chroma collection name (default: fed_chunks)")
    ap.add_argument("--batch-size", type=int, default=1000,
                    help="Rows per insert batch")
    ap.add_argument("--clear", action="store_true",
                    help="TRUNCATE the pgvector table before migrating")
    args = ap.parse_args()

    src = ChromaVectorStore(collection_name=args.collection)
    total = src.count()
    if total == 0:
        logger.error(f"Chroma collection '{args.collection}' is empty; nothing to migrate.")
        return
    logger.info(f"Chroma collection '{args.collection}' has {total} chunks")

    # Infer embedding dimension from one row so the pg schema matches.
    probe = src.collection.get(limit=1, include=["embeddings"])
    dim = len(probe["embeddings"][0])
    logger.info(f"Embedding dimension: {dim}")

    dst = PgVectorStore(embedding_dim=dim)
    if args.clear:
        dst.clear_collection()

    migrated = 0
    for offset in range(0, total, args.batch_size):
        batch = src.collection.get(
            limit=args.batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        documents = batch["documents"]
        metadatas = batch["metadatas"]
        embeddings = np.asarray(batch["embeddings"], dtype=np.float32)

        chunks = [
            {"content": documents[i], "metadata": metadatas[i] or {}}
            for i in range(len(documents))
        ]
        dst.add_documents(chunks, embeddings)
        migrated += len(chunks)
        logger.info(f"  migrated {migrated}/{total}")

    final = dst.count()
    dst.close()
    logger.success(f"Migration complete: {migrated} chunks copied; pgvector now holds {final}")


if __name__ == "__main__":
    main()
