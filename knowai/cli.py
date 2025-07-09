"""
knowai CLI – launches a FastAPI micro‑service that exposes the KnowAIAgent
over HTTP so other containers (e.g., Svelte front‑end) can converse with it.
Run via:  `docker run … knowai`  (Dockerfile entrypoint already points here)
"""
import asyncio
import os
import uuid
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from typing import List, Optional, Dict, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .core import KnowAIAgent, get_workflow_mermaid_diagram
from .errors import RateLimitError, TokenLimitError


try:
    _KNOWAI_VERSION = _pkg_version("knowai")
except PackageNotFoundError:
    _KNOWAI_VERSION = "0.0.0"

app = FastAPI(title="knowai‑service", version=_KNOWAI_VERSION)

# --------------------------------------------------------------------------- #
# Session management (very simple in‑memory cache; swap for Redis if needed)
# --------------------------------------------------------------------------- #
_sessions: Dict[str, KnowAIAgent] = {}


class InitPayload(BaseModel):
    """
    Payload to initialize a KnowAIAgent session.

    Attributes:
        vectorstore_s3_uri (str): URI of the vectorstore, either an S3 URI
            (e.g., "s3://bucket/path") or a local filesystem path.
        max_conversation_turns (Optional[int]): Maximum number of past
            conversation turns to retain. Defaults to None for the agent's
            default.

    """
    vectorstore_s3_uri: str
    max_conversation_turns: Optional[int] = None



class AskPayload(BaseModel):
    """
    Payload for a conversational turn with the KnowAIAgent.

    Attributes:
        session_id (str): Unique identifier for the conversation session.
        question (str): The user's current question to process.
        selected_files (Optional[List[str]]): List of file paths to include in
            retrieval. Defaults to None.
        n_alternatives_override (Optional[int]): Override for the number of
            answer alternatives. Defaults to None.
        k_per_query_override (Optional[int]): Override for the number of chunks
            retrieved per query. Defaults to None.
    """
    session_id: str
    question: str
    selected_files: Optional[List[str]] = None
    n_alternatives_override: Optional[int] = None
    k_per_query_override: Optional[int] = None


class AskStreamPayload(BaseModel):
    """
    Payload for a streaming conversational turn with the KnowAIAgent.

    Attributes:
        session_id (str): Unique identifier for the conversation session.
        question (str): The user's current question to process.
        selected_files (Optional[List[str]]): List of file paths to include in
            retrieval. Defaults to None.
        n_alternatives_override (Optional[int]): Override for the number of
            answer alternatives. Defaults to None.
        k_per_query_override (Optional[int]): Override for the number of chunks
            retrieved per query. Defaults to None.
    """
    session_id: str
    question: str
    selected_files: Optional[List[str]] = None
    n_alternatives_override: Optional[int] = None
    k_per_query_override: Optional[int] = None


def _download_vectorstore(s3_uri: str, dst_dir: str = "/tmp/vectorstore") -> str:
    """
    Download the FAISS vectorstore from S3 if not already present locally.

    Args:
        s3_uri (str): S3 URI of the vectorstore prefix (e.g., "s3://bucket/path").
        dst_dir (str): Destination directory for download. Defaults to "/tmp/vectorstore".

    Returns:
        str: Path to the local vectorstore directory.
    """
    import boto3
    from pathlib import Path

    dst = Path(dst_dir)
    if dst.exists() and any(dst.iterdir()):
        return str(dst)

    bucket, key_prefix = s3_uri.replace("s3://", "").split("/", 1)
    s3 = boto3.resource("s3")
    bucket_obj = s3.Bucket(bucket)

    for obj in bucket_obj.objects.filter(Prefix=key_prefix):
        target = dst / obj.key[len(key_prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        bucket_obj.download_file(obj.key, str(target))

    return str(dst)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/initialize")
async def initialize(payload: InitPayload):
    """
    Initialize a KnowAIAgent session and download or locate the vectorstore.

    Args:
        payload (InitPayload): Initialization parameters, including the
            vectorstore URI (S3 or local) and maximum conversation turns.

    Returns:
        dict: A JSON-serializable dict containing:
            session_id (str): Unique identifier for the created session.
    """
    # Determine vectorstore path: download if on S3, else use local path
    uri = payload.vectorstore_s3_uri
    if uri.startswith("s3://"):
        vec_path = _download_vectorstore(uri)
    else:
        vec_path = uri
    agent = KnowAIAgent(
        vectorstore_path=vec_path,
        max_conversation_turns=payload.max_conversation_turns or 20,

    )
    session_id = str(uuid.uuid4())
    _sessions[session_id] = agent
    return {"session_id": session_id}


@app.post("/ask")
async def ask(payload: AskPayload):
    """
    Process a conversational turn by invoking the KnowAIAgent.

    Args:
        payload (AskPayload): Payload containing session ID, question, and
            optional parameters for file selection and overrides.

    Returns:
        dict: JSON-serializable result from the agent, including generation
            and retrieved documents.
    """
    agent = _sessions.get(payload.session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    result = await agent.process_turn(
        user_question=payload.question,
        selected_files=payload.selected_files,
        n_alternatives_override=payload.n_alternatives_override,
        k_per_query_override=payload.k_per_query_override,
    )
    return result


@app.post("/ask-stream")
async def ask_stream(payload: AskStreamPayload):
    """
    Process a conversational turn with streaming response.

    This endpoint streams the LLM response in real-time as it's being generated,
    providing a more responsive user experience.

    Args:
        payload (AskStreamPayload): Payload containing session ID, question, and
            optional parameters for file selection and overrides.

    Returns:
        StreamingResponse: Server-sent events stream containing the generated response.
    """
    agent = _sessions.get(payload.session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    # Create a queue to collect tokens from the streaming callback
    token_queue = asyncio.Queue()

    def stream_callback(token: str):
        """Callback to queue tokens for streaming."""
        token_queue.put_nowait(token)

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        # Start the processing in the background
        processing_task = asyncio.create_task(
            agent.process_turn(
                user_question=payload.question,
                selected_files=payload.selected_files,
                n_alternatives_override=payload.n_alternatives_override,
                k_per_query_override=payload.k_per_query_override,
                streaming_callback=stream_callback
            )
        )

        # Stream tokens as they arrive
        while True:
            try:
                # Wait for next token with timeout
                token = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                yield f"data: {token}\n\n"
            except asyncio.TimeoutError:
                # Check if processing is complete
                if processing_task.done():
                    break
                else:
                    # Send keepalive
                    yield "data: \n\n"

        # Wait for processing to complete and handle exceptions
        try:
            await processing_task
        except RateLimitError as e:
            # Rate limit error - send error message and end stream
            yield f"data: {str(e)}\n\n"
        except TokenLimitError as e:
            # Token limit error - send error message and end stream
            yield f"data: {str(e)}\n\n"

        # Send end marker
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@app.get("/workflow-diagram")
async def get_workflow_diagram():
    """
    Get a Mermaid diagram representation of the KnowAI workflow.

    Returns:
        dict: A JSON-serializable dict containing:
            mermaid_diagram (str): Mermaid diagram string.
    """
    diagram = get_workflow_mermaid_diagram()
    return {"mermaid_diagram": diagram}


def _main():
    """Entry point for the CLI application."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    _main()
