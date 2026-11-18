"""History, ingestion, configuration, the HTTP API and the retrieval eval."""

import json
from pathlib import Path

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from evals.retrieval_eval import run_eval
from ragagent.agent import AgentAnswer
from ragagent.api import app, get_agent
from ragagent.config import Settings
from ragagent.embeddings import HashEmbedder
from ragagent.history import HISTORY_TTL_SECONDS, DynamoHistory, InMemoryHistory
from ragagent.ingest import UnsupportedDocument, extract_text, ingest_document, process_sqs_event
from ragagent.vectorstore import LocalVectorStore
from tests.fakes import FakeDynamoTable, FakeS3

ROOT = Path(__file__).resolve().parent.parent


# --- history ---------------------------------------------------------------


def test_dynamo_history_returns_the_latest_messages_oldest_first():
    ticks = iter(range(1_000_000_000, 10_000_000_000, 1_000_000_000))
    history = DynamoHistory(FakeDynamoTable(), clock_ns=lambda: next(ticks))
    for role, content in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]:
        history.append("s1", role, content)
    history.append("s2", "user", "other session")

    assert history.load("s1", limit=2) == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_dynamo_history_never_starts_with_an_assistant_message():
    ticks = iter(range(1, 100))
    history = DynamoHistory(FakeDynamoTable(), clock_ns=lambda: next(ticks))
    for role, content in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]:
        history.append("s1", role, content)
    assert [m["content"] for m in history.load("s1", limit=3)] == ["q2", "a2"]


def test_dynamo_history_items_expire():
    table = FakeDynamoTable()
    DynamoHistory(table, clock_ns=lambda: 5_000_000_000).append("s1", "user", "hello")
    assert table.items[0]["expires_at"] == 5 + HISTORY_TTL_SECONDS


def test_in_memory_history_respects_the_limit():
    history = InMemoryHistory()
    for index in range(6):
        history.append("s1", "user" if index % 2 == 0 else "assistant", str(index))
    assert [m["content"] for m in history.load("s1", limit=4)] == ["2", "3", "4", "5"]
    assert history.load("s1", limit=0) == []


# --- ingestion -------------------------------------------------------------


def s3_message(message_id, bucket, key):
    s3_record = {"bucket": {"name": bucket}, "object": {"key": key}}
    body = {"Records": [{"eventSource": "aws:s3", "s3": s3_record}]}
    return {"messageId": message_id, "body": json.dumps(body)}


def test_reingesting_a_document_replaces_it():
    store, embedder = LocalVectorStore(), HashEmbedder()
    assert ingest_document("a.md", "Old content about vacation.", embedder, store) == 1
    assert ingest_document("a.md", "New content about travel.", embedder, store) == 1
    assert len(store) == 1
    assert "travel" in store.search(embedder.embed(["travel"])[0], 1)[0].chunk.text


def test_failed_embedding_keeps_the_previous_version():
    class FailingEmbedder:
        dim = 4

        def embed(self, texts):
            raise RuntimeError("embedding service down")

    store = LocalVectorStore()
    ingest_document("a.md", "Original.", HashEmbedder(), store)
    with pytest.raises(RuntimeError):
        ingest_document("a.md", "Replacement.", FailingEmbedder(), store)
    assert len(store) == 1


def test_extract_text_rejects_unknown_types():
    assert extract_text("notes.MD", b"hello") == "hello"
    with pytest.raises(UnsupportedDocument):
        extract_text("image.png", b"\x89PNG")


def test_sqs_batch_ingests_objects_and_reports_only_failures():
    s3 = FakeS3({("docs", "guides/my file.md"): b"Canary releases get five percent of traffic."})
    store = LocalVectorStore()
    event = {
        "Records": [
            s3_message("m1", "docs", "guides/my+file.md"),  # S3 URL-encodes keys
            s3_message("m2", "docs", "missing.md"),
            s3_message("m3", "docs", "photo.png"),
            {"messageId": "m4", "body": json.dumps({"Event": "s3:TestEvent"})},
        ]
    }
    result = process_sqs_event(event, s3, HashEmbedder(), store)

    assert result == {"batchItemFailures": [{"itemIdentifier": "m2"}]}
    assert len(store) == 1
    top_hit = store.search(HashEmbedder().embed(["canary"])[0], 1)[0]
    assert top_hit.chunk.source == "guides/my file.md"


# --- configuration ---------------------------------------------------------


def test_settings_default_to_the_local_backend():
    settings = Settings.from_env({})
    assert settings.backend == "local"
    assert settings.model == "claude-opus-5-5"


def test_aws_backend_uses_the_bedrock_model_id_and_needs_its_endpoints():
    with pytest.raises(ValueError, match="OPENSEARCH_ENDPOINT, HISTORY_TABLE"):
        Settings.from_env({"RAG_BACKEND": "aws"})
    settings = Settings.from_env(
        {
            "RAG_BACKEND": "aws",
            "OPENSEARCH_ENDPOINT": "https://x.aoss.amazonaws.com",
            "HISTORY_TABLE": "t",
        }
    )
    assert settings.model == "anthropic.claude-opus-5-5"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="RAG_BACKEND"):
        Settings.from_env({"RAG_BACKEND": "azure"})


# --- HTTP API --------------------------------------------------------------


class StubAgent:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def answer(self, session_id, question):
        self.calls.append((session_id, question))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_the_answer(client):
    agent = StubAgent(AgentAnswer("25 days.", "end_turn", ["vacation.md"], 1))
    app.dependency_overrides[get_agent] = lambda: agent
    reply = client.post("/chat", json={"session_id": "user-1", "message": "Vacation?"})
    assert reply.status_code == 200
    assert reply.json() == {
        "answer": "25 days.",
        "stop_reason": "end_turn",
        "retrieved_sources": ["vacation.md"],
        "tool_calls": 1,
    }
    assert agent.calls == [("user-1", "Vacation?")]


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "", "message": "hi"},
        {"session_id": "has spaces", "message": "hi"},
        {"session_id": "ok", "message": ""},
        {"session_id": "ok", "message": "x" * 8001},
        {"message": "hi"},
    ],
)
def test_chat_validates_input(client, payload):
    app.dependency_overrides[get_agent] = lambda: StubAgent(AgentAnswer("", "end_turn"))
    assert client.post("/chat", json=payload).status_code == 422


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_status_error(anthropic.RateLimitError, 429), 429),
        (_status_error(anthropic.InternalServerError, 500), 502),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            503,
        ),
    ],
)
def test_chat_maps_model_errors_to_http_status(client, error, expected):
    app.dependency_overrides[get_agent] = lambda: StubAgent(error)
    reply = client.post("/chat", json={"session_id": "s", "message": "hi"})
    assert reply.status_code == expected


# --- retrieval eval --------------------------------------------------------


def test_retrieval_eval_on_the_sample_documents():
    result = run_eval(
        HashEmbedder(), ROOT / "sample_docs", ROOT / "evals" / "retrieval_dataset.jsonl", k=4
    )
    assert result.total == 15
    # With five one-chunk documents, hit@4 is nearly guaranteed; hit@1 is the
    # number that says something.
    assert result.hit_at_1 >= 0.7
