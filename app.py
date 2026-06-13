#!/usr/bin/env python3
"""
FedInsight - Streamlit Chat UI

A simple chat interface over the FOMC knowledge base. Ask questions in natural
language; answers are grounded in retrieved minutes with expandable sources.

Run it (from the project root, with the venv active):
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import RAGPipeline
from src.vectorstore import VECTOR_BACKEND


st.set_page_config(page_title="FedInsight", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading model and vector store...")
def load_pipeline(backend: str, provider: str) -> RAGPipeline:
    """Build the pipeline once and reuse it across reruns."""
    return RAGPipeline(backend=backend, llm_provider=provider)


def main():
    st.title("FedInsight 📈")
    st.caption("Ask questions about Federal Reserve FOMC minutes. Answers are grounded in retrieved documents.")

    with st.sidebar:
        st.header("Settings")
        backend = st.selectbox(
            "Vector store backend",
            options=["chroma", "pgvector"],
            index=0 if VECTOR_BACKEND != "pgvector" else 1,
        )
        provider = st.selectbox(
            "LLM provider",
            options=["auto", "openai", "ollama", "none"],
            index=0,
            help="auto: OpenAI if key set, else Ollama if running, else excerpts-only.",
        )
        top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=15, value=5)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown(
            "**Setup steps**\n\n"
            "1. `python src/scraper/fomc_minutes_scraper.py`\n"
            "2. `python scripts/ingest_sample.py --clear`\n"
            "3. `streamlit run app.py`"
        )

    try:
        pipeline = load_pipeline(backend, provider)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()

    st.info(f"Backend: **{backend}** · LLM provider: **{pipeline.llm.name}**", icon="⚙️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    prompt = st.chat_input("Ask about inflation, rate decisions, the labor market...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the minutes and generating an answer..."):
            result = pipeline.answer(prompt, k=top_k)
        st.markdown(result["answer"])
        _render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )


def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            meta = s.get("metadata", {})
            date = meta.get("meeting_date", "unknown date")
            score = s.get("score", 0.0)
            content = s.get("content", "").strip()
            st.markdown(f"**[{i}] {date}** · similarity {score:.3f}")
            st.write(content)
            st.divider()


if __name__ == "__main__":
    main()
