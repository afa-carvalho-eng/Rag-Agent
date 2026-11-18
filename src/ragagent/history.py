"""Conversation history: in memory for local use, DynamoDB in production."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Protocol

Message = dict[str, str]

HISTORY_TTL_SECONDS = 30 * 24 * 3600


class History(Protocol):
    def load(self, session_id: str, limit: int) -> list[Message]: ...

    def append(self, session_id: str, role: str, content: str) -> None: ...


def _starting_with_user(messages: list[Message]) -> list[Message]:
    """Drop leading assistant messages: the Messages API needs a user turn first."""
    for index, message in enumerate(messages):
        if message["role"] == "user":
            return messages[index:]
    return []


class InMemoryHistory:
    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)

    def load(self, session_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        return _starting_with_user(list(self._sessions[session_id][-limit:]))

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})


class DynamoHistory:
    """One item per message: partition key `session_id`, sort key `ts` (ns).

    `table` is a boto3 DynamoDB Table resource. Items carry an `expires_at`
    attribute so the table's TTL setting removes old conversations.
    """

    def __init__(self, table: Any, clock_ns: Any = time.time_ns) -> None:
        self._table = table
        self._clock_ns = clock_ns

    def load(self, session_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        response = self._table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": session_id},
            ScanIndexForward=False,
            Limit=limit,
            ConsistentRead=True,
        )
        newest_first = response.get("Items", [])
        messages = [
            {"role": item["role"], "content": item["content"]} for item in reversed(newest_first)
        ]
        return _starting_with_user(messages)

    def append(self, session_id: str, role: str, content: str) -> None:
        now_ns = self._clock_ns()
        self._table.put_item(
            Item={
                "session_id": session_id,
                "ts": now_ns,
                "role": role,
                "content": content,
                "expires_at": now_ns // 1_000_000_000 + HISTORY_TTL_SECONDS,
            }
        )
