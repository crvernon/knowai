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
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers import VectorStoreRetriever
from dotenv import load_dotenv
from typing import (
    List, 
    TypedDict, 
    Annotated, 
    Sequence, 
    Dict, 
    Optional,
    Union 
)

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


class GraphState(TypedDict):

    embeddings: Union[None, AzureOpenAIEmbeddings]
    vectorstore_path: Union[None, str]
    vectorstore: Union[None, FAISS]
    question: str
    documents: List[Document]
    individual_answers: Dict[str, str]


workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("load_vectorstore", load_faiss_vectorstore)


def instantiate_embeddings(state: GraphState):

    # Retrieve current embeddings instantiation state
    embeddings_state = state.get("embeddings", None) 

    if embeddings_deployment:
        return state
    
    else:
        return {
            **state, 
            "embeddings": AzureOpenAIEmbeddings(
                azure_deployment=embeddings_deployment,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                openai_api_version=openai_api_version
            )
        }


def load_faiss_vectorstore(state: GraphState) -> Optional[FAISS]:
    """
    Load a FAISS vectorstore that was previously saved to disk.

    Parameters
    ----------
    vectorstore_path : str
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
    # Retrieve state variables
    vectorstore_path = state.get("vectorstore_path", None)
    vectorstore = state.get("vectorstore", None)
    embeddings = state.get("embeddings", None)

    if vectorstore:
        return state
    
    else:
        if not os.path.exists(vectorstore_path):
            logging.error("FAISS vectorstore path does not exist: %s", vectorstore_path)
            return None

        if not os.path.isdir(vectorstore_path):
            logging.error("Provided FAISS vectorstore path is not a directory: %s", vectorstore_path)
            return None
        
        if embeddings is None:
            logging.error("Embeddings not yet instantiated.  Review workflow to ensure this node is available to the graph.")
            return None 
        
        try:
            logging.info("Loading FAISS vectorstore from '%s' ...", vectorstore_path)
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("FAISS vectorstore loaded with %d embeddings.", vectorstore.index.ntotal)
            return vectorstore
        except Exception:
            logging.exception("Failed to load FAISS vectorstore from '%s'", vectorstore_path)
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


def instantiate_retriever(
    vectorstore: FAISS,
    allowed_files: Optional[List[str]] = None,
    top_n_similar: int = 25,
):
    """
    Instantiate a retriever from an existing FAISS vectorstore.

    Parameters
    ----------
    vectorstore : FAISS
        The FAISS vectorstore that already contains your document embeddings.
    allowed_files : list[str] | None, optional
        Filenames to restrict retrieval to. When ``None`` or an empty list,
        the retriever searches across **all** chunks in the store.
    top_n_similar : int, default 25
        Number of most‑similar chunks to return.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever configured with the specified constraints.

    Raises
    ------
    ValueError
        If *vectorstore* is ``None``.
    """
    if vectorstore is None:
        raise ValueError("`vectorstore` must be a valid FAISS instance.")

    # Base kwargs
    search_kwargs = {"k": top_n_similar}

    # Add metadata filter only when filenames are supplied
    if allowed_files:
        search_kwargs["filter"] = {"file": {"$in": allowed_files}}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def extract_using_multiquery(
    question: str,
    llm: "BaseLanguageModel",
    retriever: "VectorStoreRetriever",
    n_alternatives: int = 4,
    k_per_query: int = 25,
) -> List["Document"]:
    """
    Generate alternative phrasings of *question* (via ``MultiQueryRetriever``'s components)
    **and** return the unique chunks retrieved from the provided *retriever*.

    The function workflow is:

    1. Use the ``llm_chain`` from ``MultiQueryRetriever`` to generate a text containing
       multiple query formulations.
    2. Parse this text (typically newline-separated queries) into a list of queries.
    3. For each generated query (plus the original), run
       ``retriever.get_relevant_documents`` (with *k_per_query*).
    4. Merge all returned `Document`s, deduplicate them
       (file‑name + page‑number + content), and return the unique list.

    Parameters
    ----------
    question : str
        The original user question.
    llm : BaseLanguageModel
        The LLM used by ``MultiQueryRetriever`` to craft alternative queries.
    retriever : VectorStoreRetriever
        A retriever tied to the target vectorstore.
    n_alternatives : int, default 4
        Number of alternative queries to generate.
    k_per_query : int, default 25
        Top‑*k* chunks to retrieve per query.

    Returns
    -------
    List[Document]
        Unique chunks drawn from the vectorstore across **all** generated
        queries (including the original question).

    Notes
    -----
    * The function silently falls back to returning chunks for just the
      original question if alternative query generation fails.
    * Deduplication key: ``(file, page, page_content)``.
    """

    # Initialize MultiQueryRetriever to access its llm_chain
    # Note: Ensure the MultiQueryRetriever import path is correct for your Langchain version.
    mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    alt_queries: List[str] = []
    try:
        # Directly use the llm_chain from MultiQueryRetriever to generate queries.
        # The input to the llm_chain is typically a dict with the key "question".
        chain_input = {"question": question}
        # Invoke the chain; it will use default callback handling.
        llm_response = mqr.llm_chain.invoke(chain_input)

        raw_queries_text = ""
        # Process the llm_response to get a single string of newline-separated queries.
        if isinstance(llm_response, dict):
            # Default output key for LLMChain is 'text'.
            # MultiQueryRetriever's llm_chain is an LLMChain.
            raw_queries_text = llm_response.get(mqr.llm_chain.output_key, "")
            logging.debug(f"LLMChain response was a dict. Raw text: '{raw_queries_text[:100]}...'")
        elif isinstance(llm_response, list):
            logging.debug(
                f"LLMChain response was a list. Attempting to process as list of query strings. Content: {llm_response}"
            )
            # If the LLM directly returns a list of query strings
            if all(isinstance(item, str) for item in llm_response):
                raw_queries_text = "\n".join(llm_response) # Join them into a single string
                logging.debug(f"Joined list of strings into: '{raw_queries_text[:100]}...'")
            else:
                logging.warning(
                    "LLMChain response was a list, but not all items are strings. Cannot process."
                )
        elif isinstance(llm_response, str):
            logging.debug(
                f"LLMChain response was a string directly. Processing as raw text. Response: {llm_response[:100]}..."
            )
            raw_queries_text = llm_response
        else:
            logging.warning(
                f"Unexpected response type from llm_chain.invoke: {type(llm_response)}. Expected dict, list, or str."
            )

        if raw_queries_text:
            # The default prompt for MultiQueryRetriever asks for newline-separated queries.
            # Split the raw text by newlines and filter out any empty strings.
            alt_queries = [q.strip() for q in raw_queries_text.split("\n") if q.strip()]
            logging.debug(f"Parsed queries from raw text: {alt_queries}")
        else:
            logging.warning("No raw query text obtained or processed from LLM chain response for multi-query.")
            alt_queries = []

    except Exception:
        logging.exception(
            "Failed to generate or parse alternative queries using mqr.llm_chain.invoke"
        )
        alt_queries = [] # Ensure alt_queries is defined and empty on failure

    # Clean & truncate to desired count (duplicates are already handled by list(dict.fromkeys(...)))
    # Remove empty strings (already done by strip() and check in list comprehension)
    # and duplicates, then truncate.
    alt_queries = list(dict.fromkeys(alt_queries)) # Remove duplicates while preserving order
    alt_queries = alt_queries[:n_alternatives]

    # Always include the original question for retrieval
    query_list = [question] + alt_queries

    logging.info(f"Running retrieval for {len(query_list)} queries: {query_list}")

    # Retrieve docs for each query
    retrieved_docs: List["Document"] = []
    for q_text in query_list:
        try:
            # Attempt to use 'k' parameter
            docs_for_query = retriever.get_relevant_documents(q_text, k=k_per_query)
        except TypeError as e:
            # Check if TypeError is due to unexpected 'k' argument
            if 'unexpected keyword argument \'k\'' in str(e).lower() or \
               'got an unexpected keyword argument \'k\'' in str(e).lower():
                logging.debug(f"Retriever for query '{q_text[:50]}...' does not support 'k' arg, retrieving all and slicing.")
                docs_for_query = retriever.get_relevant_documents(q_text)
                docs_for_query = docs_for_query[:k_per_query] # Manual slicing
            else:
                # Different TypeError, log and skip this query's docs
                logging.exception(f"TypeError during retrieval for query: '{q_text[:50]}...'")
                continue
        except Exception:
            logging.exception(f"Retrieval failed for query: '{q_text[:50]}...'")
            continue # Skip this query's docs if any other exception occurs
        retrieved_docs.extend(docs_for_query)

    # Deduplicate documents
    # Using a dictionary to store unique documents based on a key
    unique_docs_map: Dict[tuple, "Document"] = {}
    for doc in retrieved_docs:
        # Create a unique key for each document.
        # Ensure page_content is stripped for consistent keying.
        # Handle cases where metadata might be missing 'file' or 'page'.
        doc_file = doc.metadata.get("file") if hasattr(doc, 'metadata') and doc.metadata else None
        doc_page = doc.metadata.get("page") if hasattr(doc, 'metadata') and doc.metadata else None
        key = (
            doc_file,
            doc_page,
            doc.page_content.strip() if hasattr(doc, 'page_content') else "",
        )
        if key not in unique_docs_map:
            unique_docs_map[key] = doc

    final_unique_docs = list(unique_docs_map.values())
    logging.info(f"Retrieved {len(retrieved_docs)} chunks -> {len(final_unique_docs)} unique after dedup.")

    return final_unique_docs


# generate a function to 
