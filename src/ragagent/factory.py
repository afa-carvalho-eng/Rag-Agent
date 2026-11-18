"""Build the concrete components for the configured backend (local or AWS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent import RagAgent
from .config import AWS, Settings
from .embeddings import BedrockEmbedder, Embedder, HashEmbedder
from .history import DynamoHistory, History, InMemoryHistory
from .vectorstore import LocalVectorStore, OpenSearchVectorStore, VectorStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.backend == AWS:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        return BedrockEmbedder(client, settings.embedding_model, settings.embedding_dim)
    return HashEmbedder()


def build_store(settings: Settings) -> VectorStore:
    if settings.backend == AWS:
        import boto3
        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

        credentials = boto3.Session().get_credentials()
        host = settings.opensearch_endpoint.removeprefix("https://").rstrip("/")
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=AWSV4SignerAuth(credentials, settings.aws_region, "aoss"),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        store = OpenSearchVectorStore(client, settings.opensearch_index, settings.embedding_dim)
        store.ensure_index()
        return store
    return LocalVectorStore(Path(settings.data_dir) / "vectors.json")


def build_history(settings: Settings) -> History:
    if settings.backend == AWS:
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
        return DynamoHistory(dynamodb.Table(settings.history_table))
    return InMemoryHistory()


def build_client(settings: Settings) -> Any:
    import anthropic

    if settings.backend == AWS:
        return anthropic.AnthropicBedrockMantle(aws_region=settings.aws_region)
    # Credentials come from the environment (ANTHROPIC_API_KEY or an
    # `ant auth login` profile); none are read or stored by this project.
    return anthropic.Anthropic()


def build_agent(settings: Settings) -> RagAgent:
    return RagAgent(
        build_client(settings),
        settings.model,
        build_embedder(settings),
        build_store(settings),
        build_history(settings),
        top_k=settings.top_k,
        effort=settings.effort,
        max_history_messages=settings.max_history_messages,
        # Server-side refusal fallbacks exist on the Claude API only.
        server_fallbacks=settings.backend != AWS,
    )
