#!/usr/bin/env python3
"""
FedInsight - Simple Query Script (retrieval only)

Embeds your question and prints the most relevant FOMC chunks from the vector
store. No LLM involved - this is pure semantic search, useful for sanity checks.

Usage:
    python scripts/query.py "What did the Fed say about inflation in 2023?"
    python scripts/query.py "labor market" --k 10
    python scripts/query.py "rate decision" --backend chroma
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="FedInsight semantic search")
    parser.add_argument("query", type=str, help="Your search query (in quotes)")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["chroma", "pgvector"],
                        help="Vector store backend (default: VECTOR_BACKEND env var)")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text instead of a truncated preview")
    args = parser.parse_args()

    retriever = Retriever(backend=args.backend)
    results = retriever.search(args.query, k=args.k)

    if not results:
        logger.warning("No results. Have you run the ingestion step?")
        return

    print(f"\nTop {len(results)} results for: {args.query!r}\n" + "=" * 70)
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        date = meta.get("meeting_date", "unknown date")
        score = r.get("score", 0.0)
        content = r.get("content", "").strip()
        if not args.full and len(content) > 400:
            content = content[:400].rstrip() + "..."
        print(f"\n[{i}] score={score:.3f} | {date}")
        print("-" * 70)
        print(content)
    print()


if __name__ == "__main__":
    main()
