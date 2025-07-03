"""
Unit‑tests for the :pyclass:`knowai.core.KnowAIAgent`.

The real LangGraph workflow is replaced with a lightweight stub so that the
tests can run without external dependencies, I/O, or environment variables.
"""

from __future__ import annotations

import anyio
from typing import Any, Dict, List

import pytest

# --------------------------------------------------------------------------- #
# Dummy LangGraph stubs
# --------------------------------------------------------------------------- #


class _DummyGraph:
    """Mimic the object returned by ``graph_app.get_graph()``."""

    @staticmethod
    def draw_mermaid() -> str:  # pragma: no cover
        return "graph TD; A-->B;"


class _DummyGraphApp:
    """Minimal async interface compatible with KnowAIAgent expectations."""

    def __init__(self) -> None:
        self._graph = _DummyGraph()

    # Signature mirrors the real method used by KnowAIAgent
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        """Echo the user's question to ``generation`` and add dummy data."""
        await anyio.sleep(0)  # Yield control to satisfy the event‑loop
        return {
            "generation": f"Echo: {state.get('question')}",
            "documents_by_file": {"file1": []},
            "raw_documents_for_synthesis": "dummy raw text",
        }

    def get_graph(self) -> _DummyGraph:  # noqa: D401
        """Return a graph object exposing ``draw_mermaid``."""
        return self._graph


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def patch_graph_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically replace ``knowai.core.create_graph_app`` with the stub.

    The patch is applied for *all* tests in this module.
    """

    from knowai import core as _core_module

    monkeypatch.setattr(_core_module, "create_graph_app", lambda: _DummyGraphApp())


@pytest.fixture()
def agent() -> "knowai.core.KnowAIAgent":  # type: ignore[name-defined]
    """
    Instantiate :pyclass:`KnowAIAgent` pointing at a dummy vector‑store path.

    The constructor will receive the stubbed LangGraph thanks to the
    ``patch_graph_app`` autouse fixture.
    """
    from knowai.core import KnowAIAgent

    return KnowAIAgent(vectorstore_path="tests/fixtures/vectorstore")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_initial_session_state(agent: "knowai.core.KnowAIAgent") -> None:  # type: ignore[name-defined]
    """Verify that constructor populates all mandatory ``session_state`` keys."""
    required_keys = {
        "embeddings",
        "vectorstore_path",
        "vectorstore",
        "llm_large",
        "retriever",
        "allowed_files",
        "question",
        "documents_by_file",
        "n_alternatives",
        "k_per_query",
        "generation",
        "conversation_history",
        "raw_documents_for_synthesis",
        "k_chunks_retriever",
        "combine_threshold",
    }

    assert required_keys.issubset(agent.session_state.keys())


@pytest.mark.anyio
async def test_process_turn_updates_state(agent: "knowai.core.KnowAIAgent") -> None:  # type: ignore[name-defined]
    """
    Ensure ``process_turn`` returns the expected structure and updates history.
    """
    result = await agent.process_turn(
        user_question="What is RAG?",
        selected_files=["file1"],
    )

    # Top‑level return keys
    expected_keys = {
        "generation",
        "documents_by_file",
        "raw_documents_for_synthesis",
    }
    assert expected_keys == set(result.keys())

    # Values come from the dummy graph
    assert result["generation"] == "Echo: What is RAG?"

    # Conversation history should contain exactly one turn
    history: List[Dict[str, str]] = agent.session_state["conversation_history"]  # type: ignore[assignment]
    assert len(history) == 1
    assert history[0]["user_question"] == "What is RAG?"
    assert history[0]["assistant_response"] == "Echo: What is RAG?"


@pytest.mark.anyio
async def test_conversation_history_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Confirm that history is trimmed to ``max_conversation_turns`` entries.
    """
    from knowai.core import KnowAIAgent

    # Create agent with a small history buffer for easier testing
    agent = KnowAIAgent(
        vectorstore_path="tests/fixtures/vectorstore",
        max_conversation_turns=3,
    )

    # Simulate four conversational turns
    for i in range(4):
        await agent.process_turn(
            user_question=f"Question {i}",
            selected_files=["file1"],
        )

    history = agent.session_state["conversation_history"]  # type: ignore[assignment]
    assert len(history) == 3
    # Oldest turn should be "Question 1"
    assert history[0]["user_question"] == "Question 1"
    