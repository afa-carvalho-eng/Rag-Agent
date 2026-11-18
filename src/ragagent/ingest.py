"""Document ingestion: extract text, chunk, embed and index.

`handler` is the AWS Lambda entry point. It receives SQS messages that wrap
S3 "object created" events.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote_plus

from .chunking import chunk_text
from .embeddings import Embedder
from .vectorstore import Chunk, VectorStore

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


class UnsupportedDocument(ValueError):
    pass


def extract_text(name: str, data: bytes) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise UnsupportedDocument(f"unsupported file type: {name}")


def ingest_document(
    source: str,
    text: str,
    embedder: Embedder,
    store: VectorStore,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> int:
    """Index one document, replacing any earlier version. Returns the chunk count."""
    pieces = chunk_text(text, max_chars=max_chars, overlap=overlap)
    # Embed before touching the index, so a failed embedding call leaves the
    # previous version of the document searchable.
    vectors = embedder.embed(pieces) if pieces else []
    store.delete_source(source)
    if not pieces:
        logger.warning("no text extracted from %s", source)
        return 0
    chunks = [Chunk(text=piece, source=source, chunk_index=i) for i, piece in enumerate(pieces)]
    store.add(chunks, vectors)
    return len(chunks)


def _s3_objects(sqs_body: str) -> list[tuple[str, str]]:
    """(bucket, key) pairs from an S3 event notification. Test events have none."""
    event = json.loads(sqs_body)
    return [
        (record["s3"]["bucket"]["name"], unquote_plus(record["s3"]["object"]["key"]))
        for record in event.get("Records", [])
        if record.get("eventSource") == "aws:s3"
    ]


def process_sqs_event(
    event: dict[str, Any], s3: Any, embedder: Embedder, store: VectorStore
) -> dict[str, Any]:
    """Ingest every object named in the batch.

    Returns the partial-batch response SQS expects: only the messages listed
    in `batchItemFailures` are retried, and after the queue's receive limit
    they move to the dead-letter queue.
    """
    failures = []
    for record in event.get("Records", []):
        try:
            for bucket, key in _s3_objects(record["body"]):
                if PurePosixPath(key).suffix.lower() not in SUPPORTED_SUFFIXES:
                    # Retrying cannot help, so the message is acknowledged
                    # without downloading the object.
                    logger.warning("skipping unsupported object s3://%s/%s", bucket, key)
                    continue
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                count = ingest_document(key, extract_text(key, body), embedder, store)
                logger.info("ingested s3://%s/%s as %d chunks", bucket, key, count)
        except Exception:
            logger.exception("failed to process message %s", record.get("messageId"))
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}


_dependencies: tuple[Any, Embedder, VectorStore] | None = None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _dependencies
    if _dependencies is None:
        import boto3

        from .config import Settings
        from .factory import build_embedder, build_store

        logging.getLogger().setLevel(logging.INFO)
        settings = Settings.from_env()
        _dependencies = (
            boto3.client("s3", region_name=settings.aws_region),
            build_embedder(settings),
            build_store(settings),
        )
    s3, embedder, store = _dependencies
    return process_sqs_event(event, s3, embedder, store)
