#!/usr/bin/env python3
"""
FedInsight - RAG Pipeline

Orchestrates the full Retrieval-Augmented Generation flow:

    query -> retrieve top-k chunks -> build grounded prompt -> LLM -> answer

The answer is grounded in the retrieved FOMC context and cites sources as [1],
[2], ... matching the returned `sources` list. If no LLM is configured, the
pipeline degrades gracefully and returns the most relevant excerpts.
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from .retriever import Retriever
from .llm import get_llm, BaseLLM


SYSTEM_PROMPT = (
    "You are FedInsight, an assistant specialized in U.S. Federal Reserve "
    "communications (FOMC minutes and related documents). Answer the user's "
    "question using ONLY the provided context excerpts. Be precise and concise. "
    "Cite the excerpts you rely on using their bracketed numbers, e.g. [1], [2]. "
    "When relevant, mention the meeting date. If the answer is not contained in "
    "the context, say you don't have enough information rather than guessing."
)


def _format_source_label(metadata: Dict[str, Any]) -> str:
    """Human-readable label for a chunk, e.g. 'FOMC minutes 2020-12-16'."""
    doc_type = (metadata or {}).get("document_type", "document").replace("_", " ")
    date = (metadata or {}).get("meeting_date", "unknown date")
    return f"{doc_type} {date}".strip()


def build_context(hits: List[Dict[str, Any]]) -> str:
    """Build the numbered context block fed to the LLM."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        label = _format_source_label(hit.get("metadata", {}))
        content = hit.get("content", "").strip()
        blocks.append(f"[{i}] ({label})\n{content}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, context: str) -> str:
    return (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite excerpts as [n] and mention meeting dates where useful):"
    )


def _extractive_fallback(hits: List[Dict[str, Any]]) -> str:
    """Answer used when no LLM is available: surface the top excerpts."""
    if not hits:
        return "No relevant excerpts were found in the knowledge base."
    lines = [
        "No language model is configured (set OPENAI_API_KEY or run Ollama). "
        "Here are the most relevant excerpts:\n"
    ]
    for i, hit in enumerate(hits, start=1):
        label = _format_source_label(hit.get("metadata", {}))
        snippet = hit.get("content", "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "..."
        lines.append(f"[{i}] ({label}) {snippet}")
    return "\n\n".join(lines)


class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[BaseLLM] = None,
        backend: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.retriever = retriever or Retriever(backend=backend)
        self.llm = llm or get_llm(llm_provider)
        logger.info(f"RAGPipeline ready (LLM provider: {self.llm.name})")

    def answer(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG flow and return:
            {query, answer, provider, sources}
        where `sources` is the list of retrieved chunks.
        """
        hits = self.retriever.search(query, k=k, metadata_filter=metadata_filter)

        if not hits:
            return {
                "query": query,
                "answer": "I couldn't find anything relevant in the knowledge base. "
                "Have you run the ingestion step?",
                "provider": self.llm.name,
                "sources": [],
            }

        if not self.llm.available:
            answer = _extractive_fallback(hits)
        else:
            context = build_context(hits)
            user_prompt = build_user_prompt(query, context)
            try:
                answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.error(f"LLM generation failed ({self.llm.name}): {e}")
                answer = (
                    f"LLM generation failed ({self.llm.name}): {e}\n\n"
                    + _extractive_fallback(hits)
                )

        return {
            "query": query,
            "answer": answer,
            "provider": self.llm.name,
            "sources": hits,
        }
