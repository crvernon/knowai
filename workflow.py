import base64
import os
import time
import asyncio
import fitz  # PyMuPDF
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS # Using community import
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field # For function calling output schema
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence, Dict, Optional
from collections import defaultdict
import tiktoken

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Import AIMessage


load_dotenv() # Load environment variables from .env file

k_chunks_retriever = 25


# Fetch Azure credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # Default model
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large") # Default embedding model
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # API version


DEBUG_CAPTURE = [] # For capturing debug timing or messages



def load_faiss_vectorstore(path: str, embeddings) -> Optional[FAISS]:
    """
    Load a FAISS vectorstore that was previously saved to disk.

    Parameters
    ----------
    path : str
        Directory containing the serialized FAISS index (e.g., ``index.faiss``) and
        accompanying metadata (``index.pkl``).
    embeddings :
        The embeddings object that was originally used to build the index
        (e.g., ``AzureOpenAIEmbeddings``, ``OpenAIEmbeddings``, etc.). The same
        dimensionality must be used when loading.

    Returns
    -------
    FAISS | None
        The loaded FAISS vectorstore, or ``None`` if loading fails.

    Notes
    -----
    * ``allow_dangerous_deserialization=True`` is required because FAISS stores
      pickle metadata alongside the binary index. Only load indices from trusted
      locations.
    * Any exceptions during loading are caught and logged; the function returns
      ``None`` so callers can handle the failure gracefully.
    """
    if not os.path.exists(path):
        logging.error("FAISS vectorstore path does not exist: %s", path)
        return None

    if not os.path.isdir(path):
        logging.error("Provided FAISS vectorstore path is not a directory: %s", path)
        return None

    try:
        logging.info("Loading FAISS vectorstore from '%s' ...", path)
        vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logging.info("FAISS vectorstore loaded with %d embeddings.", vectorstore.index.ntotal)
        return vectorstore
    except Exception:
        logging.exception("Failed to load FAISS vectorstore from '%s'", path)
        return None


