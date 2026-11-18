"""Vector stores: a JSON-backed local store and OpenSearch Serverless."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[SearchHit]: ...

    def delete_source(self, source: str) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class LocalVectorStore:
    """Brute-force cosine search over vectors kept in memory.

    With a path, the store is loaded from and saved to a JSON file, which is
    enough for local development and for the retrieval eval. It is not meant
    for more than a few thousand chunks.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        if self._path and self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**item["chunk"]) for item in data]
            self._vectors = [item["vector"] for item in data]

    def __len__(self) -> int:
        return len(self._chunks)

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"chunk": asdict(chunk), "vector": vector}
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        self._path.write_text(json.dumps(data), encoding="utf-8")

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)
        self._save()

    def delete_source(self, source: str) -> None:
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._save()

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        hits = [
            SearchHit(chunk, _cosine(vector, stored))
            for chunk, stored in zip(self._chunks, self._vectors, strict=True)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]


class OpenSearchVectorStore:
    """k-NN index in an OpenSearch Serverless vector collection.

    Vectors are expected to be unit length, so L2 distance ranks them the
    same way cosine similarity does.
    """

    def __init__(self, client: Any, index: str, dim: int) -> None:
        self._client = client
        self._index = index
        self._dim = dim

    def ensure_index(self) -> None:
        if self._client.indices.exists(index=self._index):
            return
        self._client.indices.create(
            index=self._index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self._dim,
                            "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"},
                        },
                        "text": {"type": "text"},
                        "source": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                    }
                },
            },
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        # Serverless vector collections assign document IDs themselves, so
        # documents are indexed without one.
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._client.index(index=self._index, body={**asdict(chunk), "embedding": vector})

    def delete_source(self, source: str) -> None:
        """Remove every chunk of a document, so re-ingesting does not duplicate it."""
        response = self._client.search(
            index=self._index,
            body={"size": 1000, "_source": False, "query": {"term": {"source": source}}},
        )
        for hit in response["hits"]["hits"]:
            self._client.delete(index=self._index, id=hit["_id"])

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        response = self._client.search(
            index=self._index,
            body={
                "size": k,
                "_source": ["text", "source", "chunk_index"],
                "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
            },
        )
        return [
            SearchHit(
                Chunk(
                    text=hit["_source"]["text"],
                    source=hit["_source"]["source"],
                    chunk_index=hit["_source"]["chunk_index"],
                ),
                float(hit["_score"]),
            )
            for hit in response["hits"]["hits"]
        ]
