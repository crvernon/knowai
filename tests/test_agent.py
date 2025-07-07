"""
Unit‑tests for core node functions in *knowai.agent*.

All external services (Azure OpenAI, FAISS, etc.) are stubbed so that the
tests run offline and deterministically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import anyio
import pytest

# --------------------------------------------------------------------------- #
# Dummy stubs to replace heavyweight LangChain components
# --------------------------------------------------------------------------- #
class _DummyEmbeddings:
    """Mimic AzureOpenAIEmbeddings with async / sync embed helpers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self.args = args
        self.kwargs = kwargs

    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:  # noqa: D401
        """Return one fake 3‑dim vector per text."""
        return [[float(i), float(len(t)), 0.0] for i, t in enumerate(texts)]


class _DummyRetriever:  # noqa: D401
    """Minimal retriever placeholder."""
    def __init__(self, docs: Dict[str, List["Document"]]) -> None:
        self._docs = docs


class _DummyVectorStore:  # noqa: D401
    """Stub of a FAISS vector store exposing required methods."""

    def __init__(self, docs_by_file: Dict[str, List["Document"]]) -> None:
        from types import SimpleNamespace

        self.index = SimpleNamespace(ntotal=sum(len(v) for v in docs_by_file.values()))
        self._docs_by_file = docs_by_file

    # Sync API used inside `instantiate_retriever`
    def as_retriever(self, *, search_kwargs: Dict[str, Any]) -> _DummyRetriever:  # noqa: D401
        return _DummyRetriever(self._docs_by_file)

    # Async similarity search used inside `_async_retrieve_docs_with_embeddings_for_file`
    async def asimilarity_search_by_vector(
        self,
        *,
        embedding: List[float],
        k: int,
        filter: Dict[str, str],
    ) -> List["Document"]:  # noqa: D401
        await anyio.sleep(0)
        fname = filter.get("file")  # type: ignore[arg-type]
        return self._docs_by_file.get(fname, [])[:k]


class _DummyLLM:
    """Stand‑in for AzureChatOpenAI returning canned output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    async def ainvoke(self, prompt_dict: Dict[str, str]) -> str:  # noqa: D401
        await anyio.sleep(0)
        return "Synthesized answer"


class _DummyChain:
    """Used to stub `MultiQueryRetriever.from_llm(...).llm_chain`."""

    output_key = "result"

    async def ainvoke(self, inputs: Dict[str, str]) -> str:  # noqa: D401
        await anyio.sleep(0)
        # Return two alternative queries separated by newlines
        return "alt query 1\nalt query 2"


# --------------------------------------------------------------------------- #
# Pytest fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_heavy_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace Azure/FAISS/MQR classes inside *knowai.agent* with dummy stubs.
    """
    import knowai.agent as agent_mod

    monkeypatch.setattr(agent_mod, "AzureOpenAIEmbeddings", _DummyEmbeddings)
    monkeypatch.setattr(agent_mod, "AzureChatOpenAI", _DummyLLM)

    # Patch FAISS.load_local -> returns _DummyVectorStore seeded with no docs
    def _fake_load_local(folder_path: str, embeddings, allow_dangerous_deserialization: bool):  # noqa: D401
        return _DummyVectorStore(docs_by_file={})

    monkeypatch.setattr(agent_mod.FAISS, "load_local", _fake_load_local)

    # Patch MultiQueryRetriever.from_llm to return an object with attribute llm_chain
    import types

    fake_mqr_cls = types.SimpleNamespace()
    fake_mqr_cls.llm_chain = _DummyChain()

    def _fake_from_llm(*args: Any, **kwargs: Any):  # noqa: D401
        return fake_mqr_cls

    monkeypatch.setattr(agent_mod.MultiQueryRetriever, "from_llm", staticmethod(_fake_from_llm))


@pytest.fixture()
def base_state(tmp_path: Path) -> Dict[str, Any]:
    """Return a minimal GraphState‑like dict with defaults."""
    return {
        "embeddings": None,
        "vectorstore_path": str(tmp_path),  # will exist
        "vectorstore": None,
        "llm_large": None,
        "retriever": None,
        "allowed_files": None,
        "question": None,
        "documents_by_file": None,
        "n_alternatives": 2,
        "k_per_query": 5,
        "generation": None,
        "conversation_history": [],
        "raw_documents_for_synthesis": None,
        "k_chunks_retriever": 3,
    }


# --------------------------------------------------------------------------- #
# Tests for individual helper functions / nodes
# --------------------------------------------------------------------------- #
def test_instantiate_embeddings(base_state: Dict[str, Any]) -> None:
    """`instantiate_embeddings` should attach `_DummyEmbeddings`."""
    import knowai.agent as agent_mod

    new_state = agent_mod.instantiate_embeddings(base_state)
    assert isinstance(new_state["embeddings"], _DummyEmbeddings)


def test_instantiate_llm_large(base_state: Dict[str, Any]) -> None:
    """`instantiate_llm_large` should attach `_DummyLLM`."""
    import knowai.agent as agent_mod

    new_state = agent_mod.instantiate_llm_large(base_state)
    assert isinstance(new_state["llm_large"], _DummyLLM)


def test_load_faiss_vectorstore(tmp_path: Path, base_state: Dict[str, Any]) -> None:
    """Should call patched `FAISS.load_local` and attach `_DummyVectorStore`."""
    import knowai.agent as agent_mod

    # Ensure the directory path exists
    vec_path = tmp_path / "faiss_store"
    vec_path.mkdir()
    base_state["vectorstore_path"] = str(vec_path)
    base_state = agent_mod.instantiate_embeddings(base_state)

    new_state = agent_mod.load_faiss_vectorstore(base_state)
    assert isinstance(new_state["vectorstore"], _DummyVectorStore)
    # ntotal is zero because _fake_load_local seeded empty
    assert new_state["vectorstore"].index.ntotal == 0  # type: ignore[attr-defined]


def test_instantiate_retriever(base_state: Dict[str, Any]) -> None:
    """Should create a `_DummyRetriever` from `_DummyVectorStore`."""
    import knowai.agent as agent_mod

    # Inject a ready vectorstore
    base_state["vectorstore"] = _DummyVectorStore({})
    new_state = agent_mod.instantiate_retriever(base_state)
    assert isinstance(new_state["retriever"], _DummyRetriever)


def test_format_conversation_history() -> None:
    """Check pretty‑printing of conversation history."""
    import knowai.agent as agent_mod

    hist = [
        {"user_question": "Hi", "assistant_response": "Hello"},
        {"user_question": "How are you?", "assistant_response": "Fine"},
    ]
    formatted = agent_mod._format_conversation_history(hist)
    expected = "User: Hi\nAssistant: Hello\n\nUser: How are you?\nAssistant: Fine"
    assert formatted == expected

    # Empty history returns placeholder
    assert (
        agent_mod._format_conversation_history([]) == "No previous conversation history."
    )