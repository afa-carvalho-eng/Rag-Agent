"""Text embedders: Amazon Bedrock for production, a hashing embedder for offline use."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Protocol

_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it of on or our that the "
    "their there this to was we what when where which who why will with you your".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashEmbedder:
    """Deterministic bag-of-words embedder with no model and no network.

    It only captures word overlap, not meaning. It exists so the pipeline,
    the tests and the retrieval eval run offline; use BedrockEmbedder for
    real retrieval quality.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalise(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class BedrockEmbedder:
    """Amazon Titan Text Embeddings V2 through the Bedrock runtime API."""

    def __init__(self, client: Any, model_id: str, dim: int = 1024) -> None:
        self._client = client
        self._model_id = model_id
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            response = self._client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}),
            )
            payload = json.loads(response["body"].read())
            vector = payload["embedding"]
            if len(vector) != self.dim:
                raise ValueError(
                    f"{self._model_id} returned {len(vector)} dimensions, expected {self.dim}"
                )
            vectors.append(vector)
        return vectors
