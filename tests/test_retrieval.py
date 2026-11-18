import math

import pytest

from ragagent.chunking import chunk_text
from ragagent.embeddings import BedrockEmbedder, HashEmbedder
from ragagent.vectorstore import Chunk, LocalVectorStore, OpenSearchVectorStore
from tests.fakes import FakeBedrockRuntime, FakeOpenSearch


def test_short_text_is_one_chunk():
    assert chunk_text("One paragraph.\n\nAnother one.") == ["One paragraph.\n\nAnother one."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("  \n\n  ") == []


def test_chunks_respect_the_size_limit_and_keep_every_word():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(12)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=500, overlap=60)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    joined = " ".join(chunks)
    assert all(f"Paragraph {i} " in joined for i in range(12))


def test_a_paragraph_longer_than_the_limit_is_split_on_spaces():
    chunks = chunk_text("alpha " * 300, max_chars=200, overlap=0)
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(set(chunk.split()) == {"alpha"} for chunk in chunks)


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Sentence number {i} has some words in it." for i in range(40))
    chunks = chunk_text(text, max_chars=300, overlap=80)
    first_words_of_second = chunks[1].split("\n\n")[0]
    assert first_words_of_second in chunks[0]


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_are_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap=overlap)


def test_hash_embedder_is_deterministic_and_unit_length():
    embedder = HashEmbedder(dim=64)
    first, second = embedder.embed(["vacation days per year", "vacation days per year"])
    assert first == second
    assert len(first) == 64
    assert math.isclose(sum(v * v for v in first), 1.0)


def test_hash_embedder_ranks_shared_words_higher():
    embedder = HashEmbedder()
    query, related, unrelated = embedder.embed(
        ["vacation days", "employees receive vacation days", "canary release traffic"]
    )

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_bedrock_embedder_sends_the_titan_request_shape():
    runtime = FakeBedrockRuntime(dim=8)
    vectors = BedrockEmbedder(runtime, "amazon.titan-embed-text-v2:0", dim=8).embed(["hello"])
    assert vectors == [[1.0, 0, 0, 0, 0, 0, 0, 0]]
    request = runtime.requests[0]
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert request["body"] == {"inputText": "hello", "dimensions": 8, "normalize": True}


def test_bedrock_embedder_rejects_an_unexpected_dimension():
    with pytest.raises(ValueError, match="expected 16"):
        BedrockEmbedder(FakeBedrockRuntime(dim=8), "model", dim=16).embed(["hello"])


def _chunk(source, index=0, text="text"):
    return Chunk(text=text, source=source, chunk_index=index)


def test_local_store_returns_the_nearest_vectors_first():
    store = LocalVectorStore()
    store.add([_chunk("a.md"), _chunk("b.md"), _chunk("c.md")], [[1, 0], [0.6, 0.8], [0, 1]])
    hits = store.search([1, 0], k=2)
    assert [hit.chunk.source for hit in hits] == ["a.md", "b.md"]
    assert hits[0].score == pytest.approx(1.0)


def test_local_store_persists_and_deletes_by_source(tmp_path):
    path = tmp_path / "nested" / "vectors.json"
    store = LocalVectorStore(path)
    store.add([_chunk("a.md"), _chunk("a.md", 1), _chunk("b.md")], [[1, 0], [1, 0], [0, 1]])
    store.delete_source("a.md")

    reloaded = LocalVectorStore(path)
    assert len(reloaded) == 1
    assert reloaded.search([0, 1], k=5)[0].chunk == _chunk("b.md")


def test_local_store_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        LocalVectorStore().add([_chunk("a.md")], [])


def test_opensearch_store_creates_a_knn_index_once():
    client = FakeOpenSearch()
    store = OpenSearchVectorStore(client, "documents", dim=8)
    store.ensure_index()
    store.ensure_index()
    mapping = client.created["documents"]["mappings"]["properties"]["embedding"]
    assert client.created["documents"]["settings"] == {"index": {"knn": True}}
    assert mapping["type"] == "knn_vector"
    assert mapping["dimension"] == 8


def test_opensearch_store_indexes_searches_and_deletes():
    client = FakeOpenSearch()
    store = OpenSearchVectorStore(client, "documents", dim=2)
    store.add([_chunk("a.md", 0, "alpha"), _chunk("b.md", 0, "beta")], [[1, 0], [0, 1]])

    hits = store.search([1, 0], k=1)
    assert hits[0].chunk == _chunk("a.md", 0, "alpha")
    assert client.searches[-1]["query"] == {"knn": {"embedding": {"vector": [1, 0], "k": 1}}}

    store.delete_source("a.md")
    assert [doc["source"] for doc in client.docs.values()] == ["b.md"]
