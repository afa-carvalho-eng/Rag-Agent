"""Test doubles for the model client and the AWS clients."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(query: Any, block_id: str = "toolu_1", name: str = "search_documents"):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input={"query": query})


def response(stop_reason: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class _Messages:
    def __init__(self, script: list[Any], calls: list[dict[str, Any]]) -> None:
        self._script = script
        self._calls = calls

    def create(self, **kwargs: Any) -> Any:
        # Snapshot the messages: the agent keeps appending to the same list.
        self._calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeAnthropic:
    """Replays scripted responses and records every request it receives."""

    def __init__(self, *script: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.beta_calls: list[dict[str, Any]] = []
        remaining = list(script)
        self.messages = _Messages(remaining, self.calls)
        self.beta = SimpleNamespace(messages=_Messages(remaining, self.beta_calls))


class FakeBedrockRuntime:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.requests: list[dict[str, Any]] = []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append({**kwargs, "body": json.loads(kwargs["body"])})
        vector = [1.0] + [0.0] * (self.dim - 1)
        return {"body": io.BytesIO(json.dumps({"embedding": vector}).encode())}


class FakeDynamoTable:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []

    def put_item(self, Item: dict[str, Any]) -> None:  # noqa: N803 - boto3 argument name
        self.items.append(Item)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        session_id = kwargs["ExpressionAttributeValues"][":sid"]
        items = sorted(
            (item for item in self.items if item["session_id"] == session_id),
            key=lambda item: item["ts"],
            reverse=not kwargs.get("ScanIndexForward", True),
        )
        return {"Items": items[: kwargs["Limit"]]}


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


class FakeOpenSearch:
    """Just enough of the opensearch-py client for OpenSearchVectorStore."""

    def __init__(self) -> None:
        self.created: dict[str, Any] = {}
        self.docs: dict[str, dict[str, Any]] = {}
        self.searches: list[dict[str, Any]] = []
        self.indices = SimpleNamespace(exists=self._exists, create=self._create)

    def _exists(self, index: str) -> bool:
        return index in self.created

    def _create(self, index: str, body: dict[str, Any]) -> None:
        self.created[index] = body

    def index(self, index: str, body: dict[str, Any]) -> None:
        self.docs[f"doc-{len(self.docs)}"] = body

    def delete(self, index: str, id: str) -> None:  # noqa: A002 - client argument name
        del self.docs[id]

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searches.append(body)
        query = body["query"]
        if "term" in query:
            matches = [
                (doc_id, doc)
                for doc_id, doc in self.docs.items()
                if doc["source"] == query["term"]["source"]
            ]
        else:
            matches = list(self.docs.items())[: body["size"]]
        return {
            "hits": {
                "hits": [{"_id": doc_id, "_score": 0.5, "_source": doc} for doc_id, doc in matches]
            }
        }
