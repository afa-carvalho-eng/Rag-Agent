import pytest

from ragagent.agent import REFUSAL_TEXT, SEARCH_TOOL, STEP_LIMIT_TEXT, RagAgent
from ragagent.embeddings import HashEmbedder
from ragagent.history import InMemoryHistory
from ragagent.ingest import ingest_document
from ragagent.vectorstore import LocalVectorStore
from tests.fakes import FakeAnthropic, response, text_block, tool_use_block


@pytest.fixture
def store():
    store = LocalVectorStore()
    embedder = HashEmbedder()
    ingest_document("vacation.md", "Employees receive 25 vacation days per year.", embedder, store)
    ingest_document("deploy.md", "Canary releases get five percent of traffic.", embedder, store)
    return store


def make_agent(client, store, history=None, **kwargs):
    return RagAgent(
        client, "test-model", HashEmbedder(), store, history or InMemoryHistory(), **kwargs
    )


def test_answers_from_retrieved_passages(store):
    client = FakeAnthropic(
        response("tool_use", tool_use_block("vacation days per year")),
        response("end_turn", text_block("You get 25 days [vacation.md].")),
    )
    answer = make_agent(client, store, top_k=1).answer("s1", "How much vacation do I get?")

    assert answer.text == "You get 25 days [vacation.md]."
    assert answer.stop_reason == "end_turn"
    assert answer.sources == ["vacation.md"]
    assert answer.tool_calls == 1

    first, second = client.calls
    assert first["model"] == "test-model"
    assert first["tools"] == [SEARCH_TOOL]
    assert first["thinking"] == {"type": "adaptive"}
    assert first["messages"] == [{"role": "user", "content": "How much vacation do I get?"}]
    # The follow-up request carries the assistant turn and one tool_result.
    tool_result = second["messages"][-1]["content"][0]
    assert second["messages"][-2]["role"] == "assistant"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert "25 vacation days" in tool_result["content"]
    assert "is_error" not in tool_result


def test_parallel_tool_calls_return_in_one_user_message(store):
    client = FakeAnthropic(
        response(
            "tool_use",
            tool_use_block("vacation days", "toolu_a"),
            tool_use_block("canary traffic", "toolu_b"),
        ),
        response("end_turn", text_block("Done.")),
    )
    answer = make_agent(client, store, top_k=1).answer("s1", "Two things?")

    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert answer.tool_calls == 2
    assert answer.sources == ["vacation.md", "deploy.md"]


def test_history_is_saved_and_replayed(store):
    history = InMemoryHistory()
    client = FakeAnthropic(
        response("end_turn", text_block("First answer.")),
        response("end_turn", text_block("Second answer.")),
    )
    agent = make_agent(client, store, history)
    agent.answer("s1", "First question")
    agent.answer("s1", "Second question")

    assert client.calls[1]["messages"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question"},
    ]
    assert history.load("other-session", 10) == []


@pytest.mark.parametrize("bad_query", ["", "   ", None, 42])
def test_invalid_tool_input_is_reported_to_the_model(store, bad_query):
    client = FakeAnthropic(
        response("tool_use", tool_use_block(bad_query)),
        response("end_turn", text_block("Sorry.")),
    )
    make_agent(client, store).answer("s1", "Question")
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_unknown_tool_is_reported_to_the_model(store):
    client = FakeAnthropic(
        response("tool_use", tool_use_block("x", name="delete_everything")),
        response("end_turn", text_block("Sorry.")),
    )
    make_agent(client, store).answer("s1", "Question")
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "delete_everything" in tool_result["content"]


def test_search_failure_becomes_an_error_result(store):
    class BrokenStore:
        def search(self, vector, k):
            raise RuntimeError("index offline")

    client = FakeAnthropic(
        response("tool_use", tool_use_block("vacation")),
        response("end_turn", text_block("Search is down.")),
    )
    answer = make_agent(client, BrokenStore()).answer("s1", "Question")
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "index offline" not in tool_result["content"]
    assert answer.text == "Search is down."


def test_refusal_is_not_saved_to_history(store):
    history = InMemoryHistory()
    client = FakeAnthropic(response("refusal"))
    answer = make_agent(client, store, history).answer("s1", "Question")
    assert answer.text == REFUSAL_TEXT
    assert answer.stop_reason == "refusal"
    assert history.load("s1", 10) == []


def test_loop_stops_at_the_step_limit(store):
    client = FakeAnthropic(*[response("tool_use", tool_use_block("again")) for _ in range(3)])
    answer = make_agent(client, store, max_steps=3).answer("s1", "Question")
    assert answer.text == STEP_LIMIT_TEXT
    assert answer.stop_reason == "step_limit"
    assert len(client.calls) == 3


def test_truncated_answer_is_marked(store):
    client = FakeAnthropic(response("max_tokens", text_block("Partial")))
    answer = make_agent(client, store).answer("s1", "Question")
    assert answer.text.startswith("Partial")
    assert "cut off" in answer.text


def test_paused_turn_is_resumed(store):
    client = FakeAnthropic(
        response("pause_turn", text_block("Working")),
        response("end_turn", text_block("Finished.")),
    )
    answer = make_agent(client, store).answer("s1", "Question")
    assert answer.text == "Finished."
    assert client.calls[1]["messages"][-1]["role"] == "assistant"


def test_server_fallbacks_use_the_beta_endpoint(store):
    client = FakeAnthropic(response("end_turn", text_block("Hi.")))
    make_agent(client, store, server_fallbacks=True).answer("s1", "Question")
    assert client.calls == []
    assert client.beta_calls[0]["fallbacks"] == "default"
    assert client.beta_calls[0]["betas"] == ["server-side-fallback-2026-07-01"]
