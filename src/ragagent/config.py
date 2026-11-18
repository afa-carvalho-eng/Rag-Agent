"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

LOCAL = "local"
AWS = "aws"


@dataclass(frozen=True)
class Settings:
    backend: str = LOCAL
    aws_region: str = "eu-west-1"
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024
    opensearch_endpoint: str = ""
    opensearch_index: str = "documents"
    history_table: str = ""
    data_dir: str = ".data"
    top_k: int = 4
    max_history_messages: int = 20

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ) if env is None else env
        backend = env.get("RAG_BACKEND", LOCAL).lower()
        if backend not in (LOCAL, AWS):
            raise ValueError(f"RAG_BACKEND must be 'local' or 'aws', got {backend!r}")
        # Bedrock model IDs carry an "anthropic." prefix; the direct API does not.
        default_model = "anthropic.claude-opus-5-5" if backend == AWS else "claude-opus-5-5"
        settings = cls(
            backend=backend,
            aws_region=env.get("AWS_REGION", cls.aws_region),
            model=env.get("RAG_MODEL", default_model),
            effort=env.get("RAG_EFFORT", cls.effort),
            embedding_model=env.get("RAG_EMBEDDING_MODEL", cls.embedding_model),
            embedding_dim=int(env.get("RAG_EMBEDDING_DIM", cls.embedding_dim)),
            opensearch_endpoint=env.get("OPENSEARCH_ENDPOINT", ""),
            opensearch_index=env.get("OPENSEARCH_INDEX", cls.opensearch_index),
            history_table=env.get("HISTORY_TABLE", ""),
            data_dir=env.get("RAG_DATA_DIR", cls.data_dir),
            top_k=int(env.get("RAG_TOP_K", cls.top_k)),
            max_history_messages=int(env.get("RAG_MAX_HISTORY", cls.max_history_messages)),
        )
        if backend == AWS:
            missing = [
                name
                for name, value in (
                    ("OPENSEARCH_ENDPOINT", settings.opensearch_endpoint),
                    ("HISTORY_TABLE", settings.history_table),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"RAG_BACKEND=aws requires: {', '.join(missing)}")
        return settings
