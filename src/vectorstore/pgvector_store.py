#!/usr/bin/env python3
"""
FedInsight - pgvector Vector Store

A thin, dependable wrapper around PostgreSQL + pgvector for storing and
retrieving document chunk embeddings.

Design goals for the skeleton:
- Simple, explicit SQL (easy to read and debug)
- Connection details via environment variables (.env friendly)
- Automatic schema bootstrap (extension + table + index)
- Cosine similarity search with optional metadata filtering

Environment variables (any one approach works):
    DATABASE_URL=postgresql://user:pass@localhost:5432/fedinsight
  or the individual parts:
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import os
import json
from typing import List, Dict, Any, Optional, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from loguru import logger

from .base import BaseVectorStore

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


# Default matches sentence-transformers/all-MiniLM-L6-v2
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_TABLE = "fed_chunks"


def _build_connection_string() -> str:
    """Resolve a libpq connection string from the environment."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "fedinsight")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


class PgVectorStore(BaseVectorStore):
    """Store and query document chunk embeddings in PostgreSQL/pgvector."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = DEFAULT_TABLE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_setup: bool = True,
    ):
        self.connection_string = connection_string or _build_connection_string()
        self.table_name = table_name
        self.embedding_dim = embedding_dim

        logger.info(
            f"Connecting to pgvector | table='{self.table_name}' "
            f"| dim={self.embedding_dim}"
        )
        self.conn = psycopg.connect(self.connection_string, autocommit=True)

        # Ensure the vector extension exists before registering the adapter
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

        if auto_setup:
            self._ensure_schema()

        logger.success("pgvector store ready")

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #
    def _ensure_schema(self) -> None:
        """Create the chunks table and similarity index if they don't exist."""
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id          BIGSERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                embedding   VECTOR({self.embedding_dim}) NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        # IVFFlat index for cosine distance. Safe to attempt repeatedly.
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
            ON {self.table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )

        # GIN index for fast metadata filtering
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx
            ON {self.table_name}
            USING gin (metadata)
            """
        )

    def clear_collection(self) -> None:
        """Remove all rows from the table (keeps the schema)."""
        logger.warning(f"Clearing all rows from '{self.table_name}'")
        self.conn.execute(f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY")

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
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
        if len(chunks) == 0:
            logger.warning("add_documents called with no chunks")
            return 0

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != expected {self.embedding_dim}"
            )

        rows = [
            (
                chunk.get("content", ""),
                embeddings[i],
                json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
            )
            for i, chunk in enumerate(chunks)
        ]

        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table_name} (content, embedding, metadata)
                VALUES (%s, %s, %s)
                """,
                rows,
            )

        logger.success(f"Inserted {len(rows)} chunks into '{self.table_name}'")
        return len(rows)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the k most similar chunks by cosine distance.

        Args:
            query_embedding: 1D embedding of the query
            k: number of results
            metadata_filter: optional exact-match filter on metadata keys,
                             e.g. {"document_type": "fomc_minutes"}

        Returns:
            list of dicts: {id, content, metadata, distance, score}
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        where_clause = ""
        params: List[Any] = [query_embedding]
        if metadata_filter:
            where_clause = "WHERE metadata @> %s"
            params.append(json.dumps(metadata_filter, ensure_ascii=False))

        params_with_k = params + [k]

        sql = f"""
            SELECT id, content, metadata, embedding <=> %s AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance ASC
            LIMIT %s
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, params_with_k)
            records = cur.fetchall()

        results = []
        for row in records:
            row_id, content, metadata, distance = row
            results.append(
                {
                    "id": row_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": float(distance),
                    "score": 1.0 - float(distance),  # cosine similarity
                }
            )

        logger.info(f"similarity_search returned {len(results)} results (k={k})")
        return results

    def count(self) -> int:
        """Return the number of stored chunks."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]

    def close(self) -> None:
        if getattr(self, "conn", None) is not None:
            self.conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    # Quick smoke test (requires a running Postgres with pgvector)
    store = PgVectorStore()
    print(f"Current chunk count: {store.count()}")
    store.close()
