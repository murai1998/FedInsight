#!/usr/bin/env python3
"""
FedInsight - Embedder

Generates vector embeddings for text chunks using sentence-transformers.

This module is intentionally simple for the skeleton phase.
We can later add:
- Batching
- Different models
- Caching
- Normalization options
"""

from typing import List
import numpy as np
import torch
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error("sentence-transformers is required. Install it with: pip install sentence-transformers")
    raise e


# Default model - fast and decent quality for starting
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# GPU is much faster for embedding. Larger batches utilize the GPU better.
DEFAULT_BATCH_SIZE_GPU = 256
DEFAULT_BATCH_SIZE_CPU = 32


def _mps_available() -> bool:
    """True on Apple Silicon Macs with a working Metal (MPS) backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend) and backend.is_available()


def resolve_device(device: str = "auto") -> str:
    """
    Resolve the compute device.

    Args:
        device: "auto" (CUDA, then Apple MPS, then CPU), "cuda", "mps", or "cpu"

    Returns:
        "cuda", "mps", or "cpu"
    """
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():  # Apple Silicon GPU (MacBook Pro M-series)
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if device == "mps" and not _mps_available():
        logger.warning("MPS requested but not available; falling back to CPU")
        return "cpu"
    return device


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        """
        Initialize the embedding model.

        Args:
            model_name: Hugging Face model name or path
            device: "auto" (default, uses CUDA when available), "cuda", or "cpu"
        """
        self.device = resolve_device(device)
        self.batch_size = (
            DEFAULT_BATCH_SIZE_GPU
            if self.device in ("cuda", "mps")
            else DEFAULT_BATCH_SIZE_CPU
        )

        if self.device == "cuda":
            logger.info(
                f"Loading embedding model: {model_name} on cuda "
                f"({torch.cuda.get_device_name(0)})"
            )
        elif self.device == "mps":
            logger.info(f"Loading embedding model: {model_name} on mps (Apple Silicon GPU)")
        else:
            logger.info(f"Loading embedding model: {model_name} on cpu")

        self.model = SentenceTransformer(model_name, device=self.device)

        # get_sentence_embedding_dimension was renamed; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

        logger.success(
            f"Model loaded on {self.device}. Embedding dimension: {self.dimension}"
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts on {self.device}...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # good for cosine similarity
        )
        logger.success(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (returns 1D array)."""
        return self.embed_texts([text])[0]


# Global instance for convenience during skeleton development
_default_embedder = None

def get_default_embedder() -> Embedder:
    """Get or create a default embedder instance (singleton pattern for simplicity)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


if __name__ == "__main__":
    embedder = Embedder()

    texts = [
        "The Federal Reserve raised interest rates to combat inflation.",
        "The labor market remains strong with low unemployment.",
        "Committee members discussed risks to the economic outlook."
    ]

    embeddings = embedder.embed_texts(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 8 values): {embeddings[0][:8]}")