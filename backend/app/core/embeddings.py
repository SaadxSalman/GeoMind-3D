"""
Multi-modal spatial embeddings — 1536-dim dense vectors.

Two interchangeable backends:

* ``LocalEmbedder``  — offline, deterministic feature-hashing embedder.
  Tokens, word bigrams and character trigrams are hashed (BLAKE2b, stable
  across processes) into a 1536-dim signed vector with sqrt-inverse-frequency
  weighting, then L2-normalised. Cosine similarity behaves like a lexical/soft
  semantic overlap metric — no network, no model weights, fully reproducible.

* ``OpenAIEmbedder`` — calls the OpenAI embeddings endpoint
  (`text-embedding-3-small`, 1536 dims) when ``OPENAI_API_KEY`` is configured,
  falling back to the local embedder when ``EMBEDDING_FALLBACK=true``.

Both expose ``embed(texts) -> np.ndarray (N, 1536) float32``.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords are down-weighted (not removed) so long report prose still matches.
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "by",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "with",
    "as", "from", "into", "than", "then", "them", "their", "there", "which",
    "we", "our", "can", "may", "also", "such", "has", "have", "had",
}


def _stable_hash(data: str) -> int:
    """Process-stable 64-bit hash (unlike Python's randomized ``hash``)."""
    digest = hashlib.blake2b(data.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _signed_index(data: str, dim: int) -> tuple[int, float]:
    h = _stable_hash(data)
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0


class BaseEmbedder:
    """Interface every embedding backend implements."""

    dim: int = 1536
    name: str = "base"

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (N, dim) float32
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:  # (dim,)
        return self.embed([text])[0]


class LocalEmbedder(BaseEmbedder):
    """Deterministic hashing embedder — works completely offline."""

    def __init__(self, dim: int | None = None):
        self.dim = int(dim or settings.embedding_dim)
        self.name = "local-hash-v1"

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        tokens = _TOKEN_RE.findall(text.lower())
        n = len(tokens)
        for i, tok in enumerate(tokens):
            freq_w = 2.0 / math.sqrt(1.0 + math.log1p(i + 1))
            weight = 0.35 if tok in _STOPWORDS else 1.0
            yield f"w:{tok}", freq_w * weight
            if i + 1 < n:
                yield f"b:{tok}_{tokens[i + 1]}", 0.7 * weight
            padded = f"^{tok}$"  # character trigrams: sub-word robustness
            for j in range(len(padded) - 2):
                yield f"c:{padded[j:j + 3]}", 0.18 * weight

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, w in self._features(text):
                idx, sign = _signed_index(feat, self.dim)
                out[row, idx] += sign * w
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings with graceful local fallback."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 dim: int = 1536, fallback: Optional[BaseEmbedder] = None,
                 batch: int = 64, timeout: float = 30.0):
        import httpx  # lazy import: tests inject a MockTransport via monkeypatch

        self.dim = dim
        self.name = f"openai:{model}"
        self._model = model
        self._batch = batch
        self._fallback = fallback
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        try:
            for i in range(0, len(texts), self._batch):
                part = list(texts[i:i + self._batch])
                resp = self._client.post(
                    "/embeddings",
                    json={"model": self._model, "input": part, "dimensions": self.dim},
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                chunks.append(np.asarray([d["embedding"] for d in data], dtype=np.float32))
            vecs = np.vstack(chunks) if chunks else np.zeros((0, self.dim), np.float32)
            return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
        except Exception:
            if self._fallback is not None:
                return self._fallback.embed(texts)
            raise


def get_embedder() -> BaseEmbedder:
    """Factory honouring EMBEDDING_PROVIDER / OPENAI_API_KEY from ``.env``."""
    provider = settings.embedding_provider
    local = LocalEmbedder()
    if provider == "openai" and settings.openai_api_key and "REPLACE" not in settings.openai_api_key:
        import os
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
            dim=settings.embedding_dim,
            fallback=local if settings.embedding_fallback else None,
            batch=int(os.environ.get("OPENAI_EMBEDDING_BATCH", "64")),
        )
    return local
