#!/usr/bin/env python3
"""
FedInsight - Ask (full RAG from the command line)

Retrieves relevant FOMC context and generates a grounded answer with sources.

The LLM provider is chosen by the LLM_PROVIDER env var (default "auto"):
    OpenAI if OPENAI_API_KEY is set, else a local Ollama server if reachable,
    else "none" (prints the most relevant excerpts - works fully offline).

Usage:
    python scripts/ask.py "How did the Fed view inflation risks in 2022?"
    python scripts/ask.py "What was decided about the federal funds rate?" --k 8
    python scripts/ask.py "Labor market outlook" --provider ollama
    python scripts/ask.py "Balance sheet plans" --provider openai
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="FedInsight RAG question answering")
    parser.add_argument("query", type=str, help="Your question (in quotes)")
    parser.add_argument("--k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["chroma", "pgvector"],
                        help="Vector store backend (default: VECTOR_BACKEND env var)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["auto", "openai", "ollama", "none"],
                        help="LLM provider (default: LLM_PROVIDER env var or 'auto')")
    args = parser.parse_args()

    pipeline = RAGPipeline(backend=args.backend, llm_provider=args.provider)
    result = pipeline.answer(args.query, k=args.k)

    print("\n" + "=" * 70)
    print(f"Q: {result['query']}")
    print(f"(LLM provider: {result['provider']})")
    print("=" * 70)
    print(f"\n{result['answer']}\n")

    if result["sources"]:
        print("-" * 70)
        print("Sources:")
        for i, s in enumerate(result["sources"], start=1):
            meta = s.get("metadata", {})
            date = meta.get("meeting_date", "unknown date")
            score = s.get("score", 0.0)
            print(f"  [{i}] {date} (score={score:.3f})")
    print()


if __name__ == "__main__":
    main()
