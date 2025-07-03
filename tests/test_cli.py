import os
import tempfile
import pytest
from fastapi.testclient import TestClient

import knowai.cli as cli_module

@pytest.fixture(autouse=True)
def patch_agent(monkeypatch):
    """
    Monkey-patch KnowAIAgent to a dummy implementation for testing.
    """
    class DummyAgent:
        def __init__(self, vectorstore_path, combine_threshold, max_conversation_turns):
            # Store init parameters for verification if needed
            self.vectorstore_path = vectorstore_path

        async def process_turn(
            self,
            user_question,
            selected_files=None,
            n_alternatives_override=None,
            k_per_query_override=None
        ):
            return {
                "generation": f"echo: {user_question}",
                "documents_by_file": {},
                "raw_documents_for_synthesis": None
            }

    # Patch the KnowAIAgent class and clear any existing sessions
    monkeypatch.setattr(cli_module, "KnowAIAgent", DummyAgent)
    cli_module._sessions.clear()
    yield

# Create a TestClient for the FastAPI app
client = TestClient(cli_module.app)

def test_initialize_with_local_path(tmp_path):
    """
    Test that /initialize accepts a local directory path and returns a session_id.
    """
    vector_dir = tmp_path / "vec"
    vector_dir.mkdir()
    response = client.post(
        "/initialize",
        json={"vectorstore_s3_uri": str(vector_dir)}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] in cli_module._sessions

def test_ask_after_initialize(tmp_path):
    """
    Test that /ask returns the dummy agent's echo response after a valid initialize.
    """
    # Initialize session first
    vector_dir = tmp_path / "vec"
    vector_dir.mkdir()
    init_resp = client.post(
        "/initialize",
        json={"vectorstore_s3_uri": str(vector_dir)}
    )
    session_id = init_resp.json()["session_id"]

    # Now ask a question
    payload = {
        "session_id": session_id,
        "question": "Hello, world!",
        "selected_files": ["file1.txt"],
    }
    ask_resp = client.post("/ask", json=payload)
    assert ask_resp.status_code == 200
    result = ask_resp.json()
    assert result["generation"] == "echo: Hello, world!"

def test_ask_unknown_session():
    """
    Test that /ask returns 404 for an unknown session_id.
    """
    response = client.post(
        "/ask",
        json={"session_id": "nonexistent", "question": "Hi"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown session_id"