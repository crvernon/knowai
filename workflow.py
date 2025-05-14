import base64
import os
import time
import asyncio
import fitz  # PyMuPDF
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
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

logger = logging.getLogger(__name__)



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


def list_vectorstore_files(vectorstore) -> List[str]:
    """
    Return a sorted list of unique filenames stored in the FAISS vectorstore.

    The filenames are extracted from the ``metadata`` dictionary of each
    document under the key ``"file"``. If the vectorstore is ``None`` or no
    filenames are present, an empty list is returned.

    Parameters
    ----------
    vectorstore : FAISS | None
        The FAISS vectorstore instance from which to extract filenames.

    Returns
    -------
    List[str]
        Alphabetically sorted list of unique filenames found in the metadata.
    """
    if vectorstore is None:
        logging.error("Cannot list files: vectorstore is None")
        return []

    files = set()
    # Access the underlying docstore dictionary
    try:
        for _, doc in vectorstore.docstore._dict.items():
            filename = doc.metadata.get("file")
            if filename:
                files.add(filename)
    except Exception:
        logging.exception("Failed to access docstore when listing files")
        return []

    file_list = sorted(files)
    logging.info("Files in vectorstore: %s", file_list)
    return file_list


def pdf_to_chunks(
    file_path: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
) -> List[Document]:
    """
    Load a single PDF file and split its pages into overlapping text chunks.

    Each chunk is returned as a ``Document`` whose ``metadata`` dictionary
    contains:
        {"file": "<pdf-filename>", "page": <page-number>}

    Parameters
    ----------
    file_path : str
        Path to the PDF file.
    chunk_size : int, default 1500
        Maximum characters per chunk.
    chunk_overlap : int, default 300
        Number of characters overlapped between adjacent chunks.

    Returns
    -------
    List[Document]
        ``Document`` objects ready for vectorstore ingestion.
    """
    if not os.path.exists(file_path):
        logging.error("PDF file does not exist: %s", file_path)
        return []

    filename = os.path.basename(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: List[Document] = []
    try:
        pdf = fitz.open(file_path)
    except Exception:
        logging.exception("Failed to open PDF: %s", file_path)
        return []

    for page_num in range(len(pdf)):
        try:
            text = pdf.load_page(page_num).get_text()
        except Exception:
            logging.exception("Failed to read page %d of %s", page_num + 1, filename)
            continue

        if not text:
            continue

        for piece in splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={"file": filename, "page": page_num + 1},
                )
            )

    pdf.close()
    logging.info("Extracted %d chunks from %s", len(chunks), filename)
    return chunks


def add_new_pdfs_to_vectorstore(
    directory_path: str,
    vectorstore: FAISS,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
) -> int:
    """
    Scan a directory for PDF files that are not yet present in the given
    vectorstore and add their text chunks to the store.

    A PDF is considered *already present* if its filename appears in the
    ``"file"`` metadata of any existing document in the vectorstore.

    Parameters
    ----------
    directory_path : str
        Directory containing one or more ``.pdf`` files.
    vectorstore : FAISS
        An *already instantiated* FAISS vectorstore to update.
    chunk_size : int, default 1500
        Maximum characters per chunk when splitting text.
    chunk_overlap : int, default 300
        Number of characters overlapped between consecutive chunks.

    Returns
    -------
    int
        Total number of **new chunks** added to the vectorstore.
    """
    if vectorstore is None:
        logging.error("Vectorstore is None; cannot add new PDFs.")
        return 0

    if not os.path.isdir(directory_path):
        logging.error("Provided directory does not exist: %s", directory_path)
        return 0

    # Collect filenames already indexed
    existing_files = set()
    for _, doc in vectorstore.docstore._dict.items():
        filename = doc.metadata.get("file")
        if filename:
            existing_files.add(filename)

    new_docs: List[Document] = []
    pdf_files = sorted(
        f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")
    )

    for filename in pdf_files:
        if filename in existing_files:
            logging.debug("Skipping %s (already indexed).", filename)
            continue

        file_path = os.path.join(directory_path, filename)
        docs = pdf_to_chunks(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        new_docs.extend(docs)

    if not new_docs:
        logging.info("No new PDFs found in %s.", directory_path)
        return 0

    vectorstore.add_documents(new_docs)
    logging.info(
        "Added %d chunks from %d new PDF(s) to vectorstore.",
        len(new_docs),
        len({d.metadata['file'] for d in new_docs}),
    )
    return len(new_docs)
