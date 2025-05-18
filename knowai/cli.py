"""
knowai CLI – launches a FastAPI micro‑service that exposes the KnowAIAgent
over HTTP so other containers (e.g., Svelte front‑end) can converse with it.
Run via:  `docker run … knowai`  (Dockerfile entrypoint already points here)
"""
import asyncio
import os
import uuid
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core import KnowAIAgent


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
    """Payload to start a fresh KnowAIAgent session."""
    vectorstore_s3_uri: str
    combine_threshold: Optional[int] = None
    max_conversation_turns: Optional[int] = None


class AskPayload(BaseModel):
    """Payload for each conversational turn."""
    session_id: str
    question: str
    selected_files: Optional[List[str]] = None
    bypass_individual_gen: bool = False
    n_alternatives_override: Optional[int] = None
    k_per_query_override: Optional[int] = None


def _download_vectorstore(s3_uri: str, dst_dir: str = "/tmp/vectorstore") -> str:
    """Download the FAISS vector‑store from S3 the first time the service starts."""
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
    """Create a new KnowAIAgent, download the vectorstore, return a session_id."""
    vec_path = _download_vectorstore(payload.vectorstore_s3_uri)
    agent = KnowAIAgent(
        vectorstore_path=vec_path,
        combine_threshold=payload.combine_threshold or 50,
        max_conversation_turns=payload.max_conversation_turns or 20,
    )
    session_id = str(uuid.uuid4())
    _sessions[session_id] = agent
    return {"session_id": session_id}


@app.post("/ask")
async def ask(payload: AskPayload):
    """Run a single RAG turn and return JSON results."""
    agent = _sessions.get(payload.session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    result = await agent.process_turn(
        user_question=payload.question,
        selected_files=payload.selected_files,
        bypass_individual_gen=payload.bypass_individual_gen,
        n_alternatives_override=payload.n_alternatives_override,
        k_per_query_override=payload.k_per_query_override,
    )
    return result


# --------------------------------------------------------------------------- #
# Main (only executed when container starts)
# --------------------------------------------------------------------------- #
def _main():
    import uvicorn

    uvicorn.run("knowai.cli:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    _main()
