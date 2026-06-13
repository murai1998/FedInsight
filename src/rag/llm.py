#!/usr/bin/env python3
"""
FedInsight - LLM providers

A tiny, dependency-light abstraction over chat LLMs used to generate answers
from retrieved context. Three providers are supported:

    - "openai" : OpenAI-compatible Chat Completions API (needs OPENAI_API_KEY)
    - "ollama" : a local Ollama server (no API key, fully local)
    - "none"   : no generation; the pipeline returns retrieved excerpts instead

The default provider is "auto":
    OpenAI if OPENAI_API_KEY is set, else Ollama if a local server is reachable,
    else "none" (so the project always works out of the box, even offline).

All HTTP calls use `requests` to avoid extra dependencies.
"""

import os
from typing import Optional

import requests
from loguru import logger

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class BaseLLM:
    name = "base"
    available = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    name = "openai"
    available = True

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class OllamaLLM(BaseLLM):
    name = "ollama"
    available = True

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.temperature = temperature
        self.timeout = timeout

    @staticmethod
    def is_reachable(host: Optional[str] = None, timeout: int = 2) -> bool:
        host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        try:
            r = requests.get(f"{host}/api/tags", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()


class NoLLM(BaseLLM):
    """Fallback used when no LLM is configured. Generation is handled by the
    pipeline (it returns retrieved excerpts), so generate() is never called."""

    name = "none"
    available = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "No language model is configured, so I can't synthesize an answer. "
            "See the retrieved excerpts below."
        )


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Resolve and construct an LLM provider.

    Args:
        provider: "auto" (default), "openai", "ollama", or "none".
    """
    chosen = (provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()

    if chosen == "auto":
        if os.getenv("OPENAI_API_KEY"):
            chosen = "openai"
        elif OllamaLLM.is_reachable():
            chosen = "ollama"
        else:
            chosen = "none"
        logger.info(f"LLM provider auto-resolved to: {chosen}")

    if chosen == "openai":
        return OpenAILLM()
    if chosen == "ollama":
        return OllamaLLM()
    if chosen == "none":
        return NoLLM()

    raise ValueError(
        f"Unknown LLM provider: {chosen!r}. Expected 'auto', 'openai', 'ollama', or 'none'."
    )
