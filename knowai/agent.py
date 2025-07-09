# knowai/agent.py
"""
Contains the LangGraph agent definition, including GraphState, node functions,
and graph compilation logic.
"""
import asyncio
import logging
import os
import time
from typing import (
    List, 
    TypedDict, 
    Dict, 
    Optional, 
    Callable, 
    Tuple, 
    Any
)

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as Graph

from .errors import (
    handle_llm_error
)

from .utils import (
    estimate_tokens,
    estimate_synthesis_tokens,
)

from .prompts import (
    get_synthesis_prompt_template,
    get_consolidation_prompt_template,
    get_batch_combination_prompt_template,
    get_hierarchical_consolidation_prompt_template,
    get_progress_message
)

# global progress callback
GLOBAL_PROGRESS_CB = None

# instantiate logger
logger = logging.getLogger(__name__)

# Token management constants
GPT4_1_CONTEXT_WINDOW = 1_000_000  # GPT-4.1 context window
TOKEN_SAFETY_MARGIN = 0.1  # 10% safety margin
MAX_COMPLETION_TOKENS = 32768  # Maximum tokens for completion
MAX_CONCURRENT_LLM_CALLS = 10  # Limit concurrent LLM calls to 10
MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION = 10  # Maximum number of files before using hierarchical consolidation
HIERARCHICAL_CONSOLIDATION_BATCH_SIZE = 10  # Batch size for hierarchical consolidation
MAX_TOKENS_PER_HIERARCHICAL_BATCH = 25_000  # Maximum tokens per hierarchical consolidation batch

# Limit for per-file responses to avoid huge, slow outputs
INDIVIDUAL_FILE_MAX_TOKENS = 4096  # tokens (≈ four-times characters)


def create_content_batches(
    documents: List[Document],
    max_tokens_per_batch: int,
    question: str,
    conversation_history: str,
    files_no_info: str,
    files_errors: str
) -> List[Tuple[List[Document], int]]:
    """
    Create batches of documents that fit within token limits using tiktoken.
    
    Parameters
    ----------
    documents : List[Document]
        List of documents to batch.
    max_tokens_per_batch : int
        Maximum tokens allowed per batch.
    question : str
        The user's question.
    conversation_history : str
        Formatted conversation history.
    files_no_info : str
        List of files with no information.
    files_errors : str
        List of files with errors.
        
    Returns
    -------
    List[Tuple[List[Document], int]]
        List of (documents, estimated_tokens) tuples for each batch.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Calculate overhead tokens (question, history, etc.)
    overhead_tokens = estimate_synthesis_tokens(
        question=question,
        content="",  # Empty content to get just overhead
        conversation_history=conversation_history,
        files_no_info=files_no_info,
        files_errors=files_errors
    )
    
    # Ensure overhead doesn't exceed the batch limit
    if overhead_tokens >= max_tokens_per_batch:
        logging.warning(
            f"Overhead tokens ({overhead_tokens:,}) exceed batch limit ({max_tokens_per_batch:,}). "
            f"Using minimum batch size."
        )
        available_tokens = 1000  # Minimum available tokens
    else:
        available_tokens = max_tokens_per_batch - overhead_tokens
    
    for doc in documents:
        # Format document content
        fname = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted_content = f"--- File: {fname} | Page: {page} ---\n{doc.page_content}"
        
        # Estimate tokens for this document
        doc_tokens = estimate_tokens(formatted_content)
        
        # Check if this single document exceeds the available tokens
        if doc_tokens > available_tokens:
            logging.warning(
                f"Document {fname} exceeds token limit ({doc_tokens:,} > {available_tokens:,}). "
                f"Truncating content."
            )
            
            # Truncate the document content to fit within limits
            # Estimate how many characters we can keep (rough estimate: 4 chars per token)
            max_chars = int(available_tokens * 4)
            truncated_content = doc.page_content[:max_chars] + "\n\n[Content truncated due to token limits]"
            
            # Create a new document with truncated content
            truncated_doc = Document(
                page_content=truncated_content,
                metadata=doc.metadata
            )
            
            # If we have a current batch, save it and start fresh
            if current_batch:
                batches.append((current_batch, current_tokens + overhead_tokens))
                current_batch = []
                current_tokens = 0
            
            # Add the truncated document as its own batch
            truncated_formatted = f"--- File: {fname} | Page: {page} ---\n{truncated_content}"
            truncated_tokens = estimate_tokens(truncated_formatted)
            batches.append(([truncated_doc], truncated_tokens + overhead_tokens))
            
        elif current_tokens + doc_tokens > available_tokens and current_batch:
            # Save current batch and start a new one
            batches.append((current_batch, current_tokens + overhead_tokens))
            current_batch = [doc]
            current_tokens = doc_tokens
        else:
            # Add to current batch
            current_batch.append(doc)
            current_tokens += doc_tokens
    
    # Add the last batch if it has content
    if current_batch:
        batches.append((current_batch, current_tokens + overhead_tokens))
    
    return batches


def create_individual_file_response_batches(
    individual_file_responses: Dict[str, str],
    allowed_files: List[str],
    max_tokens_per_batch: int,
    question: str,
    conversation_history: str
) -> List[Tuple[List[str], int]]:
    """
    Create batches of individual file responses that fit within token limits using tiktoken.
    
    This function groups files based on their response token count rather than file count,
    ensuring that each batch stays within the specified token limit.
    
    Parameters
    ----------
    individual_file_responses : Dict[str, str]
        Mapping of filenames to their individual LLM responses.
    allowed_files : List[str]
        List of files in the order they should be processed.
    max_tokens_per_batch : int
        Maximum tokens allowed per batch.
    question : str
        The user's question.
    conversation_history : str
        Formatted conversation history.
        
    Returns
    -------
    List[Tuple[List[str], int]]
        List of (file_list, estimated_tokens) tuples for each batch.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Calculate overhead tokens (question, history, etc.)
    overhead_tokens = estimate_synthesis_tokens(
        question=question,
        content="",  # Empty content to get just overhead
        conversation_history=conversation_history,
        files_no_info="",  # Empty for individual processing
        files_errors=""    # Empty for individual processing
    )
    
    # Ensure overhead doesn't exceed the batch limit
    if overhead_tokens >= max_tokens_per_batch:
        logging.warning(
            f"Overhead tokens ({overhead_tokens:,}) exceed batch limit ({max_tokens_per_batch:,}). "
            f"Using minimum batch size."
        )
        available_tokens = 1000  # Minimum available tokens
    else:
        available_tokens = max_tokens_per_batch - overhead_tokens
    
    for filename in allowed_files:
        if filename not in individual_file_responses:
            # Skip files without responses
            continue
            
        response = individual_file_responses[filename]
        
        # Format the response for batching (same format as used in hierarchical consolidation)
        formatted_response = f"--- File: {filename} ---\n{response}"
        
        # Estimate tokens for this formatted response
        response_tokens = estimate_tokens(formatted_response)
        
        # Check if this single response exceeds the available tokens
        if response_tokens > available_tokens:
            logging.warning(
                f"File response {filename} exceeds token limit ({response_tokens:,} > {available_tokens:,}). "
                f"Creating separate batch for this file."
            )
            
            # If we have a current batch, save it and start fresh
            if current_batch:
                batches.append((current_batch, current_tokens + overhead_tokens))
                current_batch = []
                current_tokens = 0
            
            # Add the large response as its own batch
            batches.append(([filename], response_tokens + overhead_tokens))
            
        elif current_tokens + response_tokens > available_tokens and current_batch:
            # Save current batch and start a new one
            batches.append((current_batch, current_tokens + overhead_tokens))
            current_batch = [filename]
            current_tokens = response_tokens
        else:
            # Add to current batch
            current_batch.append(filename)
            current_tokens += response_tokens
    
    # Add the last batch if it has content
    if current_batch:
        batches.append((current_batch, current_tokens + overhead_tokens))
    
    return batches


class GraphState(TypedDict):
    """
    Typed dictionary representing the mutable state that flows through the
    LangGraph agent.

    Attributes
    ----------
    embeddings : Optional[LangchainEmbeddings]
        Embeddings model instance. ``None`` until instantiated.
    vectorstore_path : str
        Path to the FAISS vector‑store directory on disk.
    vectorstore : Optional[FAISS]
        Loaded FAISS vector store. ``None`` until loaded.
    llm_large : Optional[AzureChatOpenAI]
        Large language model used for query generation and synthesis.
    llm_small : Optional[AzureChatOpenAI]
        Small language model used for query generation.
    retriever : Optional[VectorStoreRetriever]
        Retriever built from the FAISS vector store.
    allowed_files : Optional[List[str]]
        Filenames selected by the user for the current question.
    question : Optional[str]
        The user's current question.
    documents_by_file : Optional[Dict[str, List[Document]]]
        Mapping of filenames to the list of retrieved document chunks.
    n_alternatives : Optional[int]
        Number of alternative queries to generate per question.
    k_per_query : Optional[int]
        Chunks to retrieve per alternative query.
    generation : Optional[str]
        Final synthesized answer.
    conversation_history : Optional[List[Dict[str, str]]]
        List of previous conversation turns.
    raw_documents_for_synthesis : Optional[str]
        Raw document text formatted for the synthesizer.
    combined_documents : Optional[List[Document]]
        Combined list of all retrieved documents.
    detailed_response_desired : Optional[bool]
        Whether to use detailed (large) or simple (small) LLM.
    k_chunks_retriever : int
        Total chunks to retrieve for the base retriever.
    k_chunks_retriever_all_docs : int
        Total documents to fetch internally for filtering.
    generated_queries : Optional[List[str]]
        List of generated alternative queries.
    query_embeddings : Optional[List[List[float]]]
        Embeddings for generated queries.
    streaming_callback : Optional[Callable[[str], None]]
        Callback function for streaming tokens.
    max_tokens_per_batch : int
        Maximum tokens allowed per synthesis batch.
    batch_results : Optional[List[str]]
        Results from processing multiple batches.
    process_files_individually : bool
        Whether to process each file individually and then consolidate responses.
    individual_file_responses : Optional[Dict[str, str]]
        Mapping of filenames to their individual LLM responses.
    hierarchical_consolidation_results : Optional[List[str]]
        Results from hierarchical consolidation of individual file responses in token-based batches.
    show_detailed_individual_responses : bool
        Whether to include individual document responses in the final output.
    detailed_responses_for_ui : Optional[str]
        Detailed individual responses formatted for UI display.
    rate_limit_error_occurred: bool
    """
    embeddings: Optional[LangchainEmbeddings]
    vectorstore_path: str
    vectorstore: Optional[FAISS]
    llm_large: Optional[AzureChatOpenAI]
    llm_small: Optional[AzureChatOpenAI]
    retriever: Optional[VectorStoreRetriever]
    allowed_files: Optional[List[str]]
    question: Optional[str]
    documents_by_file: Optional[Dict[str, List[Document]]]
    n_alternatives: Optional[int]
    k_per_query: Optional[int]
    generation: Optional[str]
    conversation_history: Optional[List[Dict[str, str]]]
    raw_documents_for_synthesis: Optional[str]
    combined_documents: Optional[List[Document]]
    detailed_response_desired: Optional[bool]
    k_chunks_retriever: int
    k_chunks_retriever_all_docs: int
    generated_queries: Optional[List[str]]
    query_embeddings: Optional[List[List[float]]]
    streaming_callback: Optional[Callable[[str], None]]
    max_tokens_per_batch: int
    batch_results: Optional[List[str]]
    process_files_individually: bool
    individual_file_responses: Optional[Dict[str, str]]
    hierarchical_consolidation_results: Optional[List[str]]
    show_detailed_individual_responses: bool
    detailed_responses_for_ui: Optional[str]
    rate_limit_error_occurred: bool


def _log_node_start(node_name: str) -> float:
    """
    Log the start of a node execution and return the start time.

    Parameters
    ----------
    node_name : str
        Name of the node being executed.

    Returns
    -------
    float
        Start time for performance measurement.
    """
    start_time = time.perf_counter()
    logging.info(f"--- Starting Node: {node_name} ---")
    return start_time


def _log_node_end(node_name: str, start_time: float) -> None:
    """
    Log the end of a node execution with duration.

    Parameters
    ----------
    node_name : str
        Name of the node that finished.
    start_time : float
        Start time from _log_node_start.
    """
    duration = time.perf_counter() - start_time
    logging.info(f"--- Node: {node_name} finished in {duration:.4f} seconds ---")


def _update_progress_callback(
    state: GraphState,
    node_name: str,
    stage: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update progress callback if available in state.

    Parameters
    ----------
    state : GraphState
        Current state containing progress callback.
    node_name : str
        Name of the current node.
    stage : str
        Current processing stage.
    metadata : Optional[Dict[str, Any]]
        Additional metadata for progress updates (e.g., counters).
    """
    # Try to get progress callback from state first, then fall back to global
    progress_cb = state.get("__progress_cb__")
    if not progress_cb:
        global GLOBAL_PROGRESS_CB
        progress_cb = GLOBAL_PROGRESS_CB
    
    if progress_cb:
        # Get the base progress message
        base_message = get_progress_message(stage, node_name)
        
        # If we have metadata with counters, use a custom message
        if metadata and "completed" in metadata and "total" in metadata:
            completed = metadata["completed"]
            total = metadata["total"]
            if "current_batch" in metadata:
                # Batch processing
                current_batch = metadata["current_batch"]
                message = f"Summarizing document batch {current_batch} of {total}..."
            elif completed == 0:
                # Individual file processing - starting
                message = f"Starting to process {total} files asynchronously in batches of {MAX_CONCURRENT_LLM_CALLS}..."
            elif completed == total:
                # Individual file processing - completed
                message = f"Completed processing {completed} files successfully"
            else:
                # Individual file processing - in progress
                message = f"Processed {completed} of {total} files..."
        else:
            message = base_message
        
        try:
            progress_cb(
                message,
                "info",
                {"node": node_name, "stage": stage, **(metadata or {})}
            )
        except Exception as e:
            pass
    else:
        pass


def instantiate_embeddings(state: GraphState) -> GraphState:
    """
    Instantiate and attach an Azure OpenAI embeddings model to the graph
    state.

    The function checks whether an embeddings model already exists in
    ``state``. If absent, it creates a new
    :class:`langchain_openai.AzureOpenAIEmbeddings` instance using the Azure
    configuration provided by module‑level environment variables. Any
    exception during instantiation is logged and the ``embeddings`` field is
    set to ``None``.

    Parameters
    ----------
    state : GraphState
        Current state dictionary flowing through the LangGraph agent.

    Returns
    -------
    GraphState
        Updated state containing the embeddings model (or ``None`` on
        failure).
    """
    start_time = _log_node_start("instantiate_embeddings_node")
    _update_progress_callback(state, "instantiate_embeddings_node", "initialization")

    if not state.get("embeddings"):
        logging.info("Instantiating embeddings model")
        try:
            # Log configuration for debugging
            deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_4p1_VERSION")
                        
            new_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=deployment,
                azure_endpoint=endpoint,
                api_key=api_key,
                openai_api_version=api_version
            )
            
            logging.info("[instantiate_embeddings_node] Embeddings model created successfully")
            state = {**state, "embeddings": new_embeddings}
            
        except Exception as e:
            logging.error(f"[instantiate_embeddings_node] Failed to instantiate embeddings model: {e}")
            logging.error(f"[instantiate_embeddings_node] Error type: {type(e).__name__}")
            logging.error(f"[instantiate_embeddings_node] Error details: {e}")
            
            # Log additional context for debugging
            if hasattr(e, 'response'):
                logging.error(f"[instantiate_embeddings_node] API Response: {e.response}")
            if hasattr(e, 'status_code'):
                logging.error(f"[instantiate_embeddings_node] Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                logging.error(f"[instantiate_embeddings_node] Response Body: {e.body}")
            
            state = {**state, "embeddings": None}
    else:
        logging.info("Using pre-instantiated embeddings model")

    _log_node_end("instantiate_embeddings_node", start_time)
    return state


def instantiate_llm_large(state: GraphState) -> GraphState:
    """
    Instantiate and attach a large Azure OpenAI chat model to the graph
    state for query generation.

    The function first checks whether an LLM instance already exists in
    ``state``. If it does not, a new
    :class:`langchain_openai.AzureChatOpenAI` model is created using the
    deployment, endpoint, API key, and version specified by the
    module‑level Azure configuration variables. On any exception, the error
    is logged and the ``llm_large`` field is set to ``None``.

    Parameters
    ----------
    state : GraphState
        Current state dictionary flowing through the LangGraph agent.

    Returns
    -------
    GraphState
        Updated state containing the large LLM instance (or ``None`` if
        instantiation failed).
    """
    start_time = _log_node_start("instantiate_llm_large_node")
    _update_progress_callback(state, "instantiate_llm_large_node", "initialization")

    if not state.get("llm_large"):
        try:
            logging.info("[instantiate_llm_large_node] Creating large LLM instance")
            
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            api_version = os.getenv("AZURE_OPENAI_API_4p1_VERSION")

            new_llm = AzureChatOpenAI(
                temperature=0.1,
                api_version=api_version,
                azure_deployment=deployment,
                max_tokens=MAX_COMPLETION_TOKENS,
                request_timeout=300,  # 5 minute timeout
            )
            
            logging.info(f"[instantiate_llm_large_node] Large LLM instance created successfully with max_tokens={MAX_COMPLETION_TOKENS:,}")
            state = {**state, "llm_large": new_llm}
            
        except Exception as e:
            logging.error(f"[instantiate_llm_large_node] Failed to instantiate large LLM model: {e}")
            logging.error(f"[instantiate_llm_large_node] Error type: {type(e).__name__}")
            logging.error(f"[instantiate_llm_large_node] Error details: {e}")
            
            # Log additional context for debugging
            if hasattr(e, 'response'):
                logging.error(f"[instantiate_llm_large_node] API Response: {e.response}")
            if hasattr(e, 'status_code'):
                logging.error(f"[instantiate_llm_large_node] Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                logging.error(f"[instantiate_llm_large_node] Response Body: {e.body}")
            
            state = {**state, "llm_large": None}
    else:
        logging.info("Using pre-instantiated large LLM model (for query generation)")

    _log_node_end("instantiate_llm_large_node", start_time)
    return state


def instantiate_llm_small(state: GraphState) -> GraphState:
    """
    Instantiate and attach a small Azure OpenAI chat model to the graph
    state for query generation.

    The function first checks whether an LLM instance already exists in
    ``state``. If it does not, a new
    :class:`langchain_openai.AzureChatOpenAI` model is created using the
    deployment, endpoint, API key, and version specified by the
    module‑level Azure configuration variables. On any exception, the error
    is logged and the ``llm_small`` field is set to ``None``.

    Parameters
    ----------
    state : GraphState
        Current state dictionary flowing through the LangGraph agent.

    Returns
    -------
    GraphState
        Updated state containing the small LLM instance (or ``None`` if
        instantiation failed).
    """
    start_time = _log_node_start("instantiate_llm_small_node")
    _update_progress_callback(state, "instantiate_llm_small_node", "initialization")

    if not state.get("llm_small"):
        try:
            logging.info("[instantiate_llm_small_node] Creating small LLM instance")
            
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NANO")
            api_version = os.getenv("AZURE_OPENAI_API_4p1_VERSION")

            new_llm = AzureChatOpenAI(
                temperature=0.1,
                api_version=api_version,
                azure_deployment=deployment,
                max_tokens=MAX_COMPLETION_TOKENS,
                request_timeout=300,  # 5 minute timeout
            )
            
            logging.info(f"[instantiate_llm_small_node] Small LLM instance created successfully with max_tokens={MAX_COMPLETION_TOKENS:,}")
            state = {**state, "llm_small": new_llm}
            
        except Exception as e:
            logging.error(f"[instantiate_llm_small_node] Failed to instantiate small LLM model: {e}")
            logging.error(f"[instantiate_llm_small_node] Error type: {type(e).__name__}")
            logging.error(f"[instantiate_llm_small_node] Error details: {e}")
            
            # Log additional context for debugging
            if hasattr(e, 'response'):
                logging.error(f"[instantiate_llm_small_node] API Response: {e.response}")
            if hasattr(e, 'status_code'):
                logging.error(f"[instantiate_llm_small_node] Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                logging.error(f"[instantiate_llm_small_node] Response Body: {e.body}")
            
            state = {**state, "llm_small": None}
    else:
        logging.info("Using pre-instantiated small LLM model (for query generation)")

    _log_node_end("instantiate_llm_small_node", start_time)
    return state


def load_faiss_vectorstore(state: GraphState) -> GraphState:
    """
    Load a local FAISS vector store from the path stored in ``state`` and
    attach it to the graph state.

    The function validates that a vector‑store path exists, an embeddings
    model has been instantiated, and the target directory is present on
    disk. If any check fails or loading raises an exception, the
    ``vectorstore`` field in the returned state is set to ``None`` and the
    error is logged. When loading succeeds, the resulting
    :class:`langchain_community.vectorstores.FAISS` instance is saved back
    into the state under the ``vectorstore`` key.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state flowing through the LangGraph agent.

    Returns
    -------
    GraphState
        Updated state whose ``vectorstore`` key holds the loaded FAISS
        instance, or ``None`` if loading failed.
    """
    start_time = _log_node_start("load_vectorstore_node")
    _update_progress_callback(state, "load_vectorstore_node", "initialization")

    current_vectorstore_path = state.get("vectorstore_path")
    embeddings = state.get("embeddings")

    if "vectorstore" not in state:
        state["vectorstore"] = None

    if state.get("vectorstore"):
        logging.info("Vectorstore already exists in state.")
    elif not current_vectorstore_path:
        logging.error("Vectorstore path not provided in state.")
        state["vectorstore"] = None
    elif not embeddings:
        logging.error("Embeddings not instantiated.")
        state["vectorstore"] = None
    elif not os.path.exists(current_vectorstore_path) or not os.path.isdir(current_vectorstore_path):
        logging.error(
            f"FAISS vectorstore path does not exist or is not a directory: "
            f"{current_vectorstore_path}"
        )
        state["vectorstore"] = None
    else:
        try:
            logging.info(f"Loading FAISS vectorstore from '{current_vectorstore_path}' ...")
            loaded_vectorstore = FAISS.load_local(
                folder_path=current_vectorstore_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info(
                f"FAISS vectorstore loaded with {loaded_vectorstore.index.ntotal} "
                f"embeddings."
            )
            state = {**state, "vectorstore": loaded_vectorstore}
        except Exception as e:
            logging.exception(f"Failed to load FAISS vectorstore: {e}")
            state["vectorstore"] = None

    _log_node_end("load_vectorstore_node", start_time)
    return state


def instantiate_retriever(state: GraphState) -> GraphState:
    """
    Instantiate and attach a base retriever built from the loaded FAISS
    vector store.

    The function checks that a FAISS vector store is present in ``state``.
    If available, it constructs a
    :class:`langchain_core.vectorstores.VectorStoreRetriever` using the
    ``k`` value stored in ``state['k_chunks_retriever']`` (falling back to
    the module‑level default). On success the new retriever is written back
    to ``state`` under the ``retriever`` key. If the vector store is
    missing or instantiation fails, the key is set to ``None`` and the error
    is logged.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state flowing through the LangGraph agent.

    Returns
    -------
    GraphState
        Updated state whose ``retriever`` key holds the instantiated
        :class:`langchain_core.vectorstores.VectorStoreRetriever`, or
        ``None`` if creation was unsuccessful.
    """
    start_time = _log_node_start("instantiate_retriever_node")
    _update_progress_callback(state, "instantiate_retriever_node", "initialization")

    if "retriever" not in state:
        state["retriever"] = None

    vectorstore = state.get("vectorstore")
    k_retriever = state.get("k_chunks_retriever")
    k_retriever_all_docs = state.get("k_chunks_retriever_all_docs")

    if vectorstore is None:
        logging.error("Vectorstore not loaded.")
        state["retriever"] = None
    else:
        if k_retriever is None:
            logging.error("k_chunks_retriever not set.")
            state["retriever"] = None
        elif k_retriever_all_docs is None:
            logging.error("k_chunks_retriever_all_docs not set.")
            state["retriever"] = None
        else:
            search_kwargs = {"k": k_retriever, "fetch_k": k_retriever_all_docs}

            try:
                base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
                logging.info(f"Base retriever instantiated with default k={k_retriever}.")
                state = {**state, "retriever": base_retriever}
            except Exception as e:
                logging.exception(f"Failed to instantiate base retriever: {e}")
                state["retriever"] = None

    _log_node_end("instantiate_retriever_node", start_time)
    return state


async def _async_retrieve_docs_with_embeddings_for_file(
    vectorstore: FAISS,
    file_name: str,
    query_embeddings_list: List[List[float]],
    query_list_texts: List[str],
    k_per_query: int,
    k_retriever_all_docs: int
) -> tuple[str, Optional[List[Document]]]:
    """
    Retrieve document chunks for a single file using pre‑computed query
    embeddings and return a unique list of results.

    Parameters
    ----------
    vectorstore : FAISS
        Loaded FAISS vector store containing all indexed document chunks.
    file_name : str
        Name of the file (as stored in document metadata) whose passages
        should be retrieved.
    query_embeddings_list : List[List[float]]
        Pre‑computed embedding vectors corresponding to each query variant.
    query_list_texts : List[str]
        Textual form of the queries (parallel to
        ``query_embeddings_list``). Used only for logging.
    k_per_query : int
        Number of document chunks to retrieve per query embedding.
    k_retriever_all_docs : int
        Number of documents to fetch internally for filtering.

    Returns
    -------
    tuple[str, Optional[List[Document]]]
        Two‑element tuple ``(file_name, docs)`` where ``docs`` is a list of
        unique :class:`langchain_core.documents.Document` instances on
        success, or ``None`` if retrieval fails.
    """
    retrieved_docs: List[Document] = []
    try:
        for i, query_embedding in enumerate(query_embeddings_list):
            docs_for_embedding = await vectorstore.asimilarity_search_by_vector(
                embedding=query_embedding,
                k=k_per_query,
                fetch_k=k_retriever_all_docs,
                filter={"file_name": file_name}
            )
            retrieved_docs.extend(docs_for_embedding)

        unique_docs_map: Dict[tuple, Document] = {}
        for doc in retrieved_docs:
            key = (
                doc.metadata.get("file_name"),
                doc.metadata.get("page"),
                doc.page_content.strip() if hasattr(doc, 'page_content') else ""
            )
            if key not in unique_docs_map:
                unique_docs_map[key] = doc

        final_unique_docs = list(unique_docs_map.values())

        logging.info(
            f"[{file_name}] Retrieved {len(retrieved_docs)} raw -> "
            f"{len(final_unique_docs)} unique docs."
        )
        return file_name, final_unique_docs

    except Exception as e_retrieve:
        logging.exception(
            f"[{file_name}] Error during similarity search by vector: {e_retrieve}"
        )
        return file_name, None


async def generate_multi_queries_node(state: GraphState) -> GraphState:
    """
    Generate alternative queries for the user's question using MultiQueryRetriever.

    This node uses the MultiQueryRetriever to generate alternative phrasings
    of the user's question to improve document retrieval. The generated queries
    are stored in the state for use by downstream nodes.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state. Expected to contain the keys
        ``question``, ``llm_small``, ``retriever``, and ``n_alternatives``.

    Returns
    -------
    GraphState
        Updated state containing ``generated_queries`` and ``query_embeddings``.
    """
    start_time = _log_node_start("generate_multi_queries_node")
    _update_progress_callback(state, "generate_multi_queries_node", "query_generation")

    question = state.get("question")
    llm_small = state.get("llm_small")
    base_retriever = state.get("retriever")
    n_alternatives = state.get("n_alternatives", 4)
    embeddings_model = state.get("embeddings")

    # Initialize with original question
    query_list: List[str] = [question] if question else []
    query_embeddings_list: List[List[float]] = []

    if not question:
        logging.info(
            "[generate_multi_queries_node] No question provided. "
            "Skipping query generation."
        )
        return {**state, "generated_queries": query_list, "query_embeddings": query_embeddings_list}

    if not all([llm_small, base_retriever, embeddings_model]):
        logging.error(
            "[generate_multi_queries_node] Missing components for query generation. "
            "Using original question only."
        )
        return {**state, "generated_queries": query_list, "query_embeddings": query_embeddings_list}

    try:
        logging.info("[generate_multi_queries_node] Generating alternative queries...")

        mqr_llm_chain = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=llm_small
        ).llm_chain

        llm_response = await mqr_llm_chain.ainvoke({"question": question})
        raw_queries_text = ""

        if isinstance(llm_response, dict):
            raw_queries_text = str(llm_response.get(mqr_llm_chain.output_key, ""))
        elif isinstance(llm_response, str):
            raw_queries_text = llm_response
        elif isinstance(llm_response, list):
            raw_queries_text = "\n".join(
                str(item).strip() for item in llm_response 
                if isinstance(item, str) and str(item).strip()
            )
        else:
            raw_queries_text = str(llm_response)

        alt_queries = [q.strip() for q in raw_queries_text.split("\n") if q.strip()]
        query_list.extend(list(dict.fromkeys(alt_queries))[:n_alternatives])

        logging.info(
            f"[generate_multi_queries_node] Generated {len(query_list)} "
            f"total unique queries."
        )

    except Exception as e_query_gen:
        logging.error(f"[generate_multi_queries_node] Failed to generate alt queries: {e_query_gen}")
        logging.error(f"[generate_multi_queries_node] Error type: {type(e_query_gen).__name__}")
        logging.error(f"[generate_multi_queries_node] Error details: {e_query_gen}")
        
        # Log additional context for debugging
        if hasattr(e_query_gen, 'response'):
            logging.error(f"[generate_multi_queries_node] API Response: {e_query_gen.response}")
        if hasattr(e_query_gen, 'status_code'):
            logging.error(f"[generate_multi_queries_node] Status Code: {e_query_gen.status_code}")
        if hasattr(e_query_gen, 'body'):
            logging.error(f"[generate_multi_queries_node] Response Body: {e_query_gen.body}")
        
        # Use the new error handling function
        error_msg = handle_llm_error(e_query_gen, None, "generate_multi_queries_node")
        logging.warning(f"[generate_multi_queries_node] {error_msg} - Using original question only.")
        # In all cases, we fall back to the original question

    # Generate embeddings for all queries
    try:
        logging.info(f"[generate_multi_queries_node] Embedding {len(query_list)} queries...")
        query_embeddings_list = await embeddings_model.aembed_documents(query_list)
    except Exception as e_embed:
        logging.exception(f"[generate_multi_queries_node] Failed to embed queries: {e_embed}")
        query_embeddings_list = []

    if not query_embeddings_list or len(query_embeddings_list) != len(query_list):
        logging.error(
            "[generate_multi_queries_node] Query embedding failed/mismatched. "
            "Using empty embeddings."
        )
        query_embeddings_list = []

    _log_node_end("generate_multi_queries_node", start_time)

    return {
        **state,
        "generated_queries": query_list,
        "query_embeddings": query_embeddings_list
    }


async def extract_documents_parallel_node(state: GraphState) -> GraphState:
    """
    Extract relevant document chunks in parallel for each user‑selected file.

    The node performs the following steps:

    1. Uses pre-generated queries and embeddings from the multi-query generation node.
    2. For every file in ``state['allowed_files']`` retrieve the top
       ``k_per_query`` chunks per query embedding from the FAISS vector
       store with an asynchronous similarity search.
    3. Deduplicate retrieved chunks per file.
    4. Store the resulting mapping in ``state['documents_by_file']``.

    If any required component (question, allowed files, vector store,
    retriever, generated queries, or query embeddings) is missing, the function returns early
    with an empty ``documents_by_file`` dictionary.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state. Expected to contain the keys
        ``question``, ``vectorstore``, ``allowed_files``,
        ``generated_queries``, and ``query_embeddings``.

    Returns
    -------
    GraphState
        Updated state where ``documents_by_file`` maps each allowed filename
        to a list of retrieved :class:`langchain_core.documents.Document`
        instances (or an empty list on failure).
    """
    start_time = _log_node_start("extract_documents_node")
    _update_progress_callback(state, "extract_documents_node", "document_retrieval")

    question = state.get("question")
    base_retriever = state.get("retriever")
    vectorstore = state.get("vectorstore")
    allowed_files = state.get("allowed_files")
    k_per_query = state.get("k_chunks_retriever")
    k_retriever_all_docs = state.get("k_chunks_retriever_all_docs")
    query_list = state.get("generated_queries")
    query_embeddings_list = state.get("query_embeddings")
    current_documents_by_file: Dict[str, List[Document]] = {}

    if not question:
        logging.info("[extract_documents_node] No question. Skipping extraction.")
        return {**state, "documents_by_file": current_documents_by_file}
    if not allowed_files:
        logging.info("[extract_documents_node] No files selected. Skipping extraction.")
        return {**state, "documents_by_file": current_documents_by_file}
    if not all([base_retriever, vectorstore, query_list, query_embeddings_list]):
        logging.error("[extract_documents_node] Missing components for extraction. Halting.")
        return {**state, "documents_by_file": current_documents_by_file}
    if len(query_list) != len(query_embeddings_list):
        logging.error(
            "[extract_documents_node] Mismatch between number of queries and embeddings. "
            "Halting."
        )
        return {**state, "documents_by_file": current_documents_by_file}

    tasks = [
        _async_retrieve_docs_with_embeddings_for_file(
            vectorstore,
            f_name,
            query_embeddings_list,
            query_list,
            k_per_query,
            k_retriever_all_docs
        ) for f_name in allowed_files
    ]

    if tasks:
        results = await asyncio.gather(*tasks)
        for f_name, docs in results:
            current_documents_by_file[f_name] = docs if docs else []
        # Build a flattened list of all docs across files
        combined_docs_list: List[Document] = []
        for docs in current_documents_by_file.values():
            if docs:
                combined_docs_list.extend(docs)
    else:
        combined_docs_list: List[Document] = []

    _log_node_end("extract_documents_node", start_time)
    return {
        **state,
        "documents_by_file": current_documents_by_file,
        "combined_documents": combined_docs_list
    }


def format_raw_documents_for_synthesis_node(state: GraphState) -> GraphState:
    """
    Format retrieved document chunks into a single raw‑text block for
    downstream answer synthesis.

    The node iterates over the `state['allowed_files']` list and, for each
    file, concatenates the page‑level text stored in
    `state['documents_by_file']` into a structured plain‑text section:

    ```
    --- Start of Context from File: <filename> ---

    Page X:
    <page content>

    ---
    ```

    The assembled text for *all* files is saved under the
    ``raw_documents_for_synthesis`` key so that the synthesis LLM can
    answer the user's question.

    If no documents were retrieved for the selected files, or if no files
    were selected, the function writes an explanatory placeholder string
    instead.

    Parameters
    ----------
    state : GraphState
        Current mutable LangGraph state. Expected keys include
        ``documents_by_file`` and ``allowed_files``.

    Returns
    -------
    GraphState
        Updated state with ``raw_documents_for_synthesis`` containing the
        formatted context text or a descriptive placeholder.
    """
    start_time = _log_node_start("format_raw_documents_for_synthesis_node")
    _update_progress_callback(
        state, "format_raw_documents_for_synthesis_node", "document_preparation"
    )

    documents_by_file = state.get("documents_by_file")
    allowed_files = state.get("allowed_files") if state.get("allowed_files") is not None else []
    formatted_raw_docs = ""

    if documents_by_file:
        for filename in allowed_files:
            docs_list = documents_by_file.get(filename)
            if docs_list:
                formatted_raw_docs += f"--- Start of Context from File: {filename} ---\n\n"
                for doc in docs_list:
                    page = doc.metadata.get('page', 'N/A')
                    formatted_raw_docs += f"Page {page}:\n{doc.page_content}\n\n---\n\n"
                formatted_raw_docs += f"--- End of Context from File: {filename} ---\n\n"
            else:
                formatted_raw_docs += (
                    f"--- No Content Extracted for File: {filename} "
                    f"(no matching document chunks found) ---\n\n"
                )

    if not formatted_raw_docs and allowed_files:
        formatted_raw_docs = "No documents were retrieved for the selected files and question."
    elif not allowed_files:
        formatted_raw_docs = "No files were selected for processing."

    _log_node_end("format_raw_documents_for_synthesis_node", start_time)
    return {**state, "raw_documents_for_synthesis": formatted_raw_docs.strip()}


async def process_batches_node(state: GraphState) -> GraphState:
    """
    Process documents in batches to avoid exceeding token limits or route to individual file processing.
    
    This node checks the `process_files_individually` flag to determine the processing approach:
    1. If `process_files_individually` is True: Skip batch processing and route to individual file processing
    2. If `process_files_individually` is False: Check if estimated tokens exceed limits and process in batches if needed
    
    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing documents and token limits.
        
    Returns
    -------
    GraphState
        Updated state with batch results or routing to individual processing.
    """
    start_time = _log_node_start("process_batches_node")
    _update_progress_callback(state, "process_batches_node", "synthesis")
    
    question = state.get("question")
    allowed_files = state.get("allowed_files")
    conversation_history = state.get("conversation_history")
    combined_docs_list = state.get("combined_documents", [])
    process_files_individually = state.get("process_files_individually", False)
    max_tokens_per_batch = state.get("max_tokens_per_batch", int(GPT4_1_CONTEXT_WINDOW * (1 - TOKEN_SAFETY_MARGIN)))
    
    if not question or not combined_docs_list:
        # No processing needed
        return state
    
    if process_files_individually:
        # Route to individual file processing - skip batch processing
        _update_progress_callback(state, "process_batches_node", "routing_to_individual_processing")
        return state
    
    # Traditional batch processing logic
    conversation_history_str = _format_conversation_history(conversation_history)
    
    # Track files with no docs
    docs_by_file = state.get("documents_by_file", {})
    no_info_list = []
    if allowed_files:
        for af in allowed_files:
            if not docs_by_file.get(af):
                no_info_list.append(f"`{af}` (no chunks extracted)")
    error_list = ["Error tracking for raw path not detailed here."]
    
    # Estimate total tokens for all documents
    all_content = "\n\n".join([
        f"--- File: {doc.metadata.get('file_name', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')} ---\n{doc.page_content}"
        for doc in combined_docs_list
    ])
    
    total_estimated_tokens = estimate_synthesis_tokens(
        question=question,
        content=all_content,
        conversation_history=conversation_history_str,
        files_no_info=", ".join(no_info_list) if no_info_list else "None",
        files_errors=", ".join(error_list) if error_list else "None"
    )
    
    logging.info(f"[process_batches_node] Total estimated tokens: {total_estimated_tokens:,}")
    logging.info(f"[process_batches_node] Max tokens per batch: {max_tokens_per_batch:,}")
    logging.info(f"[process_batches_node] Using accurate token counting with tiktoken")
    
    if total_estimated_tokens <= max_tokens_per_batch:
        # Can process all documents in one batch
        logging.info("[process_batches_node] Processing all documents in single batch")
        return state
    else:
        # Need to process in batches
        logging.info(f"[process_batches_node] Processing {len(combined_docs_list)} documents in batches")
        _update_progress_callback(state, "process_batches_node", "processing_in_batches")
        
        # Create batches
        batches = create_content_batches(
            documents=combined_docs_list,
            max_tokens_per_batch=max_tokens_per_batch,
            question=question,
            conversation_history=conversation_history_str,
            files_no_info=", ".join(no_info_list) if no_info_list else "None",
            files_errors=", ".join(error_list) if error_list else "None"
        )
        
        logging.info(f"[process_batches_node] Created {len(batches)} batches")
        
        # Process each batch
        detailed_flag = state.get("detailed_response_desired", True)
        # llm_instance = state.get("llm_large") if detailed_flag else state.get("llm_small")
        llm_instance = state.get("llm_large")
        combo_prompt = get_synthesis_prompt_template()
        streaming_callback = state.get("streaming_callback")
        
        batch_results = []
        
        for i, (batch_docs, batch_tokens) in enumerate(batches):
            logging.info(f"[process_batches_node] Processing batch {i+1}/{len(batches)} ({batch_tokens:,} tokens)")
            _update_progress_callback(
                state, 
                "process_batches_node", 
                f"processing_batch_{i+1}_{len(batches)}",
                {"completed": i, "total": len(batches), "current_batch": i+1}
            )
            
            # Format batch content
            batch_content = "\n\n".join([
                f"--- File: {doc.metadata.get('file_name', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')} ---\n{doc.page_content}"
                for doc in batch_docs
            ])
            
            # Process batch
            batch_result = await _stream_final_generation(
                question=question,
                content_llm=batch_content,
                llm_instance=llm_instance,
                combo_prompt=combo_prompt,
                conversation_history_str=conversation_history_str,
                no_info_list=no_info_list,
                error_list=error_list,
                streaming_callback=streaming_callback
            )
            
            batch_results.append(batch_result)
            
            # Add batch separator for streaming
            if streaming_callback and i < len(batches) - 1:
                streaming_callback("\n\n--- Processing next batch ---\n\n")
        
        # Store batch results for final combination
        state = {**state, "batch_results": batch_results}
        
        _log_node_end("process_batches_node", start_time)
        return state


def _format_conversation_history(
    history: Optional[List[Dict[str, str]]]
) -> str:
    """
    Format the prior conversation turns into a readable multi‑line string.

    Each turn is rendered as two lines—one for the user question and one
    for the assistant response—separated by a blank line between turns. If
    *history* is ``None`` or empty, a placeholder message is returned
    instead.

    Parameters
    ----------
    history : Optional[List[Dict[str, str]]]
        Conversation history where each element is a dictionary containing
        the keys ``'user_question'`` and ``'assistant_response'``.

    Returns
    -------
    str
        Formatted conversation history or a message indicating that no
        previous history is available.
    """
    if not history:
        return "No previous conversation history."

    return "\n\n".join(
        [
            (
                f"User: {t.get('user_question', 'N/A')}\n"
                f"Assistant: {t.get('assistant_response', 'N/A')}"
            )
            for t in history
        ]
    )


async def _stream_final_generation(
    question: str,
    content_llm: str,
    llm_instance: BaseLanguageModel,
    combo_prompt: PromptTemplate,
    conversation_history_str: str,
    no_info_list: List[str],
    error_list: List[str],
    streaming_callback: Optional[Callable[[str], None]],
    additional_variables: Optional[Dict[str, Any]] = None,
    max_tokens_override: Optional[int] = None
) -> str:
    """
    Stream the final generation using the LLM with a callback for real-time updates.

    Parameters
    ----------
    question : str
        The user's question.
    content_llm : str
        The content to synthesize.
    llm_instance : BaseLanguageModel
        The LLM instance to use for generation.
    combo_prompt : PromptTemplate
        The prompt template for synthesis.
    conversation_history_str : str
        Formatted conversation history.
    no_info_list : List[str]
        List of files with no relevant information.
    error_list : List[str]
        List of files with errors.
    streaming_callback : Optional[Callable[[str], None]]
        Callback function to stream tokens as they're generated.

    Returns
    -------
    str
        The complete generated response.
    """
    try:
        logging.info("[_stream_final_generation] Creating streaming chain")
        
        # Log LLM configuration for debugging
        if hasattr(llm_instance, 'max_tokens'):
            logging.info(f"[_stream_final_generation] LLM max_tokens: {llm_instance.max_tokens:,}")
        if hasattr(llm_instance, 'azure_deployment'):
            logging.info(f"[_stream_final_generation] LLM deployment: {llm_instance.azure_deployment}")
        
        # If a max_tokens_override was provided, bind it to the LLM so the
        # OpenAI request enforces that tighter cap (this avoids gigantic,
        # runaway completions).
        if max_tokens_override is not None:
            try:
                llm_runnable = llm_instance.bind(max_tokens=max_tokens_override)
            except Exception:
                # Fallback: if .bind is unsupported, just use the original instance.
                llm_runnable = llm_instance
        else:
            llm_runnable = llm_instance

        # Create the streaming chain
        chain = combo_prompt | llm_runnable | StrOutputParser()

        # Prepare the input
        input_data = {
            "question": question,
            "formatted_answers_or_raw_docs": content_llm,
            "files_no_info": ", ".join(no_info_list) if no_info_list else "None",
            "files_errors": ", ".join(error_list) if error_list else "None",
            "conversation_history": conversation_history_str
        }
        
        # Add additional variables if provided
        if additional_variables:
            input_data.update(additional_variables)
        
        logging.info(f"[_stream_final_generation] Input data prepared - question length: {len(question)}, content length: {len(content_llm)}")

        if streaming_callback:
            # Use streaming if callback is provided
            logging.info("[_stream_final_generation] Using streaming mode")
            full_response = ""
            try:
                async for chunk in chain.astream(input_data):
                    if chunk:
                        full_response += chunk
                        streaming_callback(chunk)
                
                # Check if response exceeds MAX_COMPLETION_TOKENS
                estimated_tokens = len(full_response) // 4
                if estimated_tokens > MAX_COMPLETION_TOKENS:
                    logging.error(f"[_stream_final_generation] Response estimated at {estimated_tokens:,} tokens, exceeds MAX_COMPLETION_TOKENS ({MAX_COMPLETION_TOKENS:,})")
                    logging.error(f"[_stream_final_generation] Response length: {len(full_response):,} characters")
                    
                    # Truncate to a reasonable length
                    max_chars = MAX_COMPLETION_TOKENS * 4
                    full_response = full_response[:max_chars] + "\n\n[Response truncated due to token limit]"
                    logging.warning(f"[_stream_final_generation] Truncated response to {len(full_response):,} characters")
                
                logging.info(f"[_stream_final_generation] Streaming completed - response length: {len(full_response)}")
                return full_response
            except Exception as stream_error:
                logging.error(f"[_stream_final_generation] Streaming error: {stream_error}")
                logging.error(f"[_stream_final_generation] Streaming error type: {type(stream_error).__name__}")
                if hasattr(stream_error, 'response'):
                    logging.error(f"[_stream_final_generation] API Response: {stream_error.response}")
                if hasattr(stream_error, 'status_code'):
                    logging.error(f"[_stream_final_generation] Status Code: {stream_error.status_code}")
                raise
        else:
            # Use regular invocation if no streaming callback
            logging.info("[_stream_final_generation] Using regular invocation mode")
            try:
                result = await chain.ainvoke(input_data)
                
                # Check if response exceeds MAX_COMPLETION_TOKENS
                estimated_tokens = len(result) // 4
                if estimated_tokens > MAX_COMPLETION_TOKENS:
                    logging.error(f"[_stream_final_generation] Response estimated at {estimated_tokens:,} tokens, exceeds MAX_COMPLETION_TOKENS ({MAX_COMPLETION_TOKENS:,})")
                    logging.error(f"[_stream_final_generation] Response length: {len(result):,} characters")
                    
                    # Truncate to a reasonable length
                    max_chars = MAX_COMPLETION_TOKENS * 4
                    result = result[:max_chars] + "\n\n[Response truncated due to token limit]"
                    logging.warning(f"[_stream_final_generation] Truncated response to {len(result):,} characters")
                
                logging.info(f"[_stream_final_generation] Regular invocation completed - response length: {len(result)}")
                return result
            except Exception as invoke_error:
                logging.error(f"[_stream_final_generation] Invocation error: {invoke_error}")
                logging.error(f"[_stream_final_generation] Invocation error type: {type(invoke_error).__name__}")
                if hasattr(invoke_error, 'response'):
                    logging.error(f"[_stream_final_generation] API Response: {invoke_error.response}")
                if hasattr(invoke_error, 'status_code'):
                    logging.error(f"[_stream_final_generation] Status Code: {invoke_error.status_code}")
                raise

    except Exception as e:
        logging.error(f"[_stream_final_generation] Critical error in generation: {e}")
        logging.error(f"[_stream_final_generation] Error type: {type(e).__name__}")
        logging.error(f"[_stream_final_generation] Error details: {e}")
        
        # Log additional context for debugging
        if hasattr(e, 'response'):
            logging.error(f"[_stream_final_generation] API Response: {e.response}")
        if hasattr(e, 'status_code'):
            logging.error(f"[_stream_final_generation] Status Code: {e.status_code}")
        if hasattr(e, 'body'):
            logging.error(f"[_stream_final_generation] Response Body: {e.body}")
        
        # Use the new error handling function
        return handle_llm_error(e, streaming_callback, "_stream_final_generation")



async def hierarchical_consolidation_node(state: GraphState) -> GraphState:
    """
    Consolidate individual file responses in token-based batches to ensure all information is preserved.
    
    This node processes individual file responses in hierarchical batches based on token count
    rather than file count. Each batch is limited to MAX_TOKENS_PER_HIERARCHICAL_BATCH tokens
    to ensure we stay within context window limits.
    
    For example, if there are 4 files with responses of 10k, 100k, 290k, and 150k tokens:
    - Batch 1: Files 1, 2, 3 (400k tokens total)
    - Batch 2: File 4 (150k tokens)
    
    This ensures that no information is lost when processing many documents, as each batch
    summary preserves all important details from its constituent files while staying within
    token limits.
    
    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing individual file responses.
        
    Returns
    -------
    GraphState
        Updated state with hierarchical consolidation results.
    """
    start_time = _log_node_start("hierarchical_consolidation_node")
    _update_progress_callback(state, "hierarchical_consolidation_node", "hierarchical_consolidation")
    
    question = state.get("question")
    allowed_files = state.get("allowed_files")
    individual_file_responses = state.get("individual_file_responses", {})
    conversation_history = state.get("conversation_history")
    process_files_individually = state.get("process_files_individually", False)
    
    # Only run if we're in individual file processing mode and have responses
    if not process_files_individually or not individual_file_responses or not allowed_files:
        logging.info("[hierarchical_consolidation_node] Skipping - not in individual file processing mode or no responses")
        logging.info(f"[hierarchical_consolidation_node] process_files_individually: {process_files_individually}")
        logging.info(f"[hierarchical_consolidation_node] individual_file_responses: {len(individual_file_responses) if individual_file_responses else 0}")
        logging.info(f"[hierarchical_consolidation_node] allowed_files: {len(allowed_files) if allowed_files else 0}")
        return {**state, "hierarchical_consolidation_results": None}
    
    # Check if we have enough files to warrant hierarchical consolidation (more than n)
    logging.info(f"[hierarchical_consolidation_node] Checking file count: {len(allowed_files)} vs threshold: {MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION}")
    if len(allowed_files) <= MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION:
        logging.info(f"[hierarchical_consolidation_node] Only {len(allowed_files)} files - no hierarchical consolidation needed")
        return {**state, "hierarchical_consolidation_results": None}
    
    logging.info(f"[hierarchical_consolidation_node] Starting token-based hierarchical consolidation for {len(allowed_files)} files")
    
    detailed_flag = state.get("detailed_response_desired", True)
    # llm_instance = state.get("llm_small") or state.get("llm_large")
    llm_instance = state.get("llm_large")
    streaming_callback = None
    conversation_history_str = _format_conversation_history(conversation_history)
    
    if not llm_instance:
        logging.error("[hierarchical_consolidation_node] LLM instance not available")
        return {**state, "hierarchical_consolidation_results": None}
    
    # Get the hierarchical consolidation prompt
    hierarchical_prompt = get_hierarchical_consolidation_prompt_template()
    
    # Create token-based batches
    batches = create_individual_file_response_batches(
        individual_file_responses=individual_file_responses,
        allowed_files=allowed_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question=question,
        conversation_history=conversation_history_str
    )
    
    total_batches = len(batches)
    logging.info(f"[hierarchical_consolidation_node] Created {total_batches} token-based batches")
    
    # Log batch details for debugging
    for i, (batch_files, batch_tokens) in enumerate(batches):
        logging.info(f"[hierarchical_consolidation_node] Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")
        for filename in batch_files:
            response_length = len(individual_file_responses.get(filename, ""))
            logging.info(f"[hierarchical_consolidation_node]   - {filename}: {response_length:,} chars")
    
    # Process each batch
    batch_results = []
    
    for batch_num, (batch_files, batch_tokens) in enumerate(batches):
        logging.info(f"[hierarchical_consolidation_node] Processing batch {batch_num + 1}/{total_batches} ({batch_tokens:,} tokens, {len(batch_files)} files)")
        # Update progress callback with batch information
        if total_batches == 1:
            stage = "hierarchical_consolidation"
        elif total_batches == 2:
            stage = f"processing_batch_{batch_num + 1}_2"
        elif total_batches == 3:
            stage = f"processing_batch_{batch_num + 1}_3"
        else:
            stage = "hierarchical_consolidation"
        
        _update_progress_callback(
            state, 
            "hierarchical_consolidation_node", 
            stage,
            {"completed": batch_num, "total": total_batches, "current_batch": batch_num + 1}
        )
        
        # Format responses for this batch
        batch_responses = []
        for filename in batch_files:
            if filename in individual_file_responses:
                response = individual_file_responses[filename]
                batch_responses.append(f"--- File: {filename} ---\n{response}")
            else:
                batch_responses.append(f"--- File: {filename} ---\nNo response generated for this file.")
        
        batch_content = "\n\n".join(batch_responses)
        
        # Generate consolidated summary for this batch
        batch_start_time = time.perf_counter()
        try:
            logging.info(f"[hierarchical_consolidation_node] Starting LLM call for batch {batch_num + 1}")
            batch_summary = await _stream_final_generation(
                question=question,
                content_llm=batch_content,
                llm_instance=llm_instance,
                combo_prompt=hierarchical_prompt,
                conversation_history_str=conversation_history_str,
                no_info_list=[],  # Already handled in individual processing
                error_list=[],    # Already handled in individual processing
                streaming_callback=streaming_callback,
                additional_variables={"batch_number": batch_num + 1}
            )
            
            batch_end_time = time.perf_counter()
            batch_duration = batch_end_time - batch_start_time
            # Check if response exceeds MAX_COMPLETION_TOKENS (rough estimate: 4 chars per token)
            estimated_tokens = len(batch_summary) // 4
            if estimated_tokens > MAX_COMPLETION_TOKENS:
                logging.error(f"[hierarchical_consolidation_node] Batch {batch_num + 1} summary estimated at {estimated_tokens:,} tokens, exceeds MAX_COMPLETION_TOKENS ({MAX_COMPLETION_TOKENS:,})")
                logging.error(f"[hierarchical_consolidation_node] Summary length: {len(batch_summary):,} characters")
                
                # Truncate to a reasonable length (roughly MAX_COMPLETION_TOKENS * 4 chars)
                max_chars = MAX_COMPLETION_TOKENS * 4
                batch_summary = batch_summary[:max_chars] + "\n\n[Summary truncated due to token limit]"
                logging.warning(f"[hierarchical_consolidation_node] Truncated summary to {len(batch_summary):,} characters")
            
            # Additional safety check for extremely long responses
            max_hierarchical_length = 100000  # characters (much higher than before)
            if len(batch_summary) > max_hierarchical_length:
                logging.error(f"[hierarchical_consolidation_node] Batch {batch_num + 1} summary was {len(batch_summary):,} chars, truncating to {max_hierarchical_length:,}")
                batch_summary = batch_summary[:max_hierarchical_length] + "\n\n[Summary truncated due to length limits]"
            
            batch_results.append(batch_summary)
            logging.info(f"[hierarchical_consolidation_node] Completed batch {batch_num + 1} ({len(batch_summary)} chars) in {batch_duration:.2f} seconds")
            
        except Exception as e:
            logging.error(f"[hierarchical_consolidation_node] Error consolidating batch {batch_num + 1}: {e}")
            logging.error(f"[hierarchical_consolidation_node] Error type: {type(e).__name__}")
            logging.error(f"[hierarchical_consolidation_node] Error details: {e}")
            
            # Log additional context for debugging
            if hasattr(e, 'response'):
                logging.error(f"[hierarchical_consolidation_node] API Response: {e.response}")
            if hasattr(e, 'status_code'):
                logging.error(f"[hierarchical_consolidation_node] Status Code: {e.status_code}")
            if hasattr(e, 'body'):
                logging.error(f"[hierarchical_consolidation_node] Response Body: {e.body}")
            
            # Use the new error handling function
            error_msg = handle_llm_error(e, None, f"hierarchical_consolidation_node_batch_{batch_num + 1}")
            batch_results.append(error_msg)
    
    logging.info(f"[hierarchical_consolidation_node] Completed token-based hierarchical consolidation - {len(batch_results)} batch summaries created")
    _update_progress_callback(
        state, 
        "hierarchical_consolidation_node", 
        "hierarchical_consolidation",
        {"completed": total_batches, "total": total_batches}
    )
    
    _log_node_end("hierarchical_consolidation_node", start_time)
    return {**state, "hierarchical_consolidation_results": batch_results}


async def combine_answers_node(state: GraphState) -> GraphState:
    """
    Synthesize a final answer for the user by combining raw document text, batch results, or individual file responses.

    The function handles three processing modes:
    1. Individual file processing: Combines responses from individual file processing
    2. Multi-batch processing: Combines results from batch processing when documents exceed token limits
    3. Single-batch processing: Traditional approach when documents fit within token limits
    
    Content‑policy violations are propagated using the global :data:`CONTENT_POLICY_MESSAGE`.

    Error and "no‑info" conditions are tracked per file and injected back
    into the final prompt so that the LLM can acknowledge gaps or issues.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing (among others) the keys
        ``question``, ``allowed_files``, ``raw_documents_for_synthesis``,
        ``batch_results``, ``individual_file_responses``, ``process_files_individually``,
        and ``conversation_history``.

    Returns
    -------
    GraphState
        Updated state where ``generation`` holds the synthesized answer,
        a content‑policy message, or an error explanation.
    """
    start_time = _log_node_start("combine_answers_node")
    _update_progress_callback(state, "combine_answers_node", "synthesis")

    question = state.get("question")
    allowed_files = state.get("allowed_files")
    conversation_history = state.get("conversation_history")
    batch_results = state.get("batch_results")
    individual_file_responses = state.get("individual_file_responses", {})
    process_files_individually = state.get("process_files_individually", False)
    output_generation: Optional[str] = "Error during synthesis."
    state_to_return = {**state}

    if not allowed_files:
        output_generation = "Please select files to analyze."
    elif not question:
        output_generation = (
            f"Files selected: {', '.join(allowed_files) if allowed_files else 'any'}. "
            f"Ask a question."
        )
    else:
        conversation_history_str = _format_conversation_history(conversation_history)

        if process_files_individually and individual_file_responses:
            # Check if hierarchical consolidation was performed
            hierarchical_results = state.get("hierarchical_consolidation_results")
            
            if hierarchical_results and len(hierarchical_results) > 1:
                # Use hierarchical consolidation results - combine the batch summaries
                logging.info(f"[combine_answers_node] Combining {len(hierarchical_results)} hierarchical consolidation results")
                _update_progress_callback(state, "combine_answers_node", "combining_hierarchical_results")
                
                detailed_flag = state.get("detailed_response_desired", True)
                # llm_instance = state.get("llm_small") or state.get("llm_large")
                llm_instance = state.get("llm_large")
                streaming_callback = state.get("streaming_callback")
                
                # Get batch combination prompt for combining hierarchical results
                batch_combination_prompt = get_batch_combination_prompt_template()
                
                # Format hierarchical results
                formatted_batches = []
                for i, result in enumerate(hierarchical_results):
                    formatted_batches.append(f"--- Hierarchical Batch {i+1} ---\n{result}")
                combined_content = "\n\n".join(formatted_batches)
                
                # Generate final combined response from hierarchical results
                output_generation = await _stream_final_generation(
                    question=question,
                    content_llm=combined_content,
                    llm_instance=llm_instance,
                    combo_prompt=batch_combination_prompt,
                    conversation_history_str=conversation_history_str,
                    no_info_list=[],  # Already handled in hierarchical processing
                    error_list=[],    # Already handled in hierarchical processing
                    streaming_callback=streaming_callback
                )
                
            elif hierarchical_results and len(hierarchical_results) == 1:
                # Only one hierarchical batch - use it directly
                logging.info("[combine_answers_node] Using single hierarchical consolidation result")
                output_generation = hierarchical_results[0]
                
            else:
                # No hierarchical consolidation - use traditional approach for ≤10 files
                logging.info(f"[combine_answers_node] Consolidating {len(individual_file_responses)} individual file responses (traditional approach)")
                _update_progress_callback(state, "combine_answers_node", "consolidating_individual_responses")
                
                detailed_flag = state.get("detailed_response_desired", True)
                # llm_instance = state.get("llm_small") or state.get("llm_large")
                llm_instance = state.get("llm_large")
                streaming_callback = state.get("streaming_callback")
                
                # Get consolidation prompt for individual file responses
                consolidation_prompt = get_consolidation_prompt_template()
                
                # Format individual file responses
                formatted_responses = []
                for filename in allowed_files:
                    if filename in individual_file_responses:
                        response = individual_file_responses[filename]
                        formatted_responses.append(f"--- File: {filename} ---\n{response}")
                    else:
                        formatted_responses.append(f"--- File: {filename} ---\nNo response generated for this file.")
                
                combined_content = "\n\n".join(formatted_responses)
                
                # Generate final consolidated response
                output_generation = await _stream_final_generation(
                    question=question,
                    content_llm=combined_content,
                    llm_instance=llm_instance,
                    combo_prompt=consolidation_prompt,
                    conversation_history_str=conversation_history_str,
                    no_info_list=[],  # Already handled in individual processing
                    error_list=[],    # Already handled in individual processing
                    streaming_callback=streaming_callback
                )
            
        elif batch_results and len(batch_results) > 1:
            # Multiple batches were processed, combine the results
            logging.info(f"[combine_answers_node] Combining {len(batch_results)} batch results")
            _update_progress_callback(state, "combine_answers_node", "combining_batch_results")
            
            detailed_flag = state.get("detailed_response_desired", True)
            # llm_instance = state.get("llm_large") if detailed_flag else state.get("llm_small")
            llm_instance = state.get("llm_large")

            streaming_callback = state.get("streaming_callback")
            
            # Get batch combination prompt for combining batch results
            batch_combination_prompt = get_batch_combination_prompt_template()
            
            # Format batch results
            formatted_batches = []
            for i, result in enumerate(batch_results):
                formatted_batches.append(f"--- Batch {i+1} ---\n{result}")
            combined_content = "\n\n".join(formatted_batches)
            
            # Generate final combined response
            output_generation = await _stream_final_generation(
                question=question,
                content_llm=combined_content,
                llm_instance=llm_instance,
                combo_prompt=batch_combination_prompt,
                conversation_history_str=conversation_history_str,
                no_info_list=[],  # Already handled in batches
                error_list=[],    # Already handled in batches
                streaming_callback=streaming_callback
            )
            
        elif batch_results and len(batch_results) == 1:
            # Single batch was processed, use the result directly
            logging.info("[combine_answers_node] Using single batch result")
            output_generation = batch_results[0]
            
        else:
            # No batch processing occurred, use traditional single-batch approach
            logging.info("[combine_answers_node] Using traditional single-batch processing")
            _update_progress_callback(state, "combine_answers_node", "traditional_synthesis")
            
            detailed_flag = state.get("detailed_response_desired", True)
            # llm_instance = state.get("llm_small") or state.get("llm_large")
            llm_instance = state.get("llm_large")
            combo_prompt = get_synthesis_prompt_template()
            streaming_callback = state.get("streaming_callback")

            no_info_list: List[str] = []
            error_list: List[str] = []
            content_llm: str = ""

            combined_docs_list = state.get("combined_documents") or []

            if combined_docs_list:
                temp_lines = []
                for doc in combined_docs_list:
                    fname = doc.metadata.get("file_name", "unknown")
                    page = doc.metadata.get("page", "N/A")
                    temp_lines.append(
                        f"--- File: {fname} | Page: {page} ---\n{doc.page_content}"
                    )
                content_llm = "\n\n".join(temp_lines)
            else:
                content_llm = state.get("raw_documents_for_synthesis", "No raw documents.")

            # Track files with no docs
            docs_by_file = state.get("documents_by_file", {})
            if allowed_files:
                for af in allowed_files:
                    if not docs_by_file.get(af):
                        no_info_list.append(f"`{af}` (no chunks extracted)")
            error_list.append("Error tracking for raw path not detailed here.")

            if content_llm:
                output_generation = await _stream_final_generation(
                    question=question,
                    content_llm=content_llm,
                    llm_instance=llm_instance,
                    combo_prompt=combo_prompt,
                    conversation_history_str=conversation_history_str,
                    no_info_list=no_info_list,
                    error_list=error_list,
                    streaming_callback=streaming_callback
                )

    # Store individual responses separately for UI display
    if state.get("show_detailed_individual_responses", False):
        # Create the detailed responses section for separate display
        detailed_section = ""
        if individual_file_responses and process_files_individually:
            # Add each individual response
            for filename in allowed_files:
                if filename in individual_file_responses:
                    response = individual_file_responses[filename]
                    detailed_section += f"## {filename}\n{response}\n\n"
                else:
                    detailed_section += f"## {filename} \nNo response generated for this file.\n\n"
        else:
            # No individual responses available - show message
            detailed_section += "Individual file responses are not available in the current processing mode.\n\n"
            detailed_section += "To see individual responses:\n"
            detailed_section += "1. Ensure 'Process files individually' is enabled\n"
            detailed_section += "2. Make sure documents are available for processing\n"
            detailed_section += "3. Check that the vectorstore contains indexed documents\n\n"
        
        # Store detailed responses in state for UI to access
        state_to_return["detailed_responses_for_ui"] = detailed_section
    
    # If a streaming callback exists but we generated the answer without streaming,
    # send the full answer once so the chat UI receives the response.
    final_cb = state.get("streaming_callback")
    if final_cb:
        try:
            final_cb(output_generation)
        except Exception:
            # Swallow any UI-side errors so they do not break the agent.
            pass
    
    state_to_return["generation"] = output_generation
    _log_node_end("combine_answers_node", start_time)
    return state_to_return


async def process_individual_files_node(state: GraphState) -> GraphState:
    """
    Process each file individually to generate separate responses for each file.
    
    This node takes the documents retrieved for each file and generates an
    individual LLM response for each file asynchronously in parallel, with
    a maximum of 10 concurrent LLM calls to prevent overwhelming the service.
    The responses are stored in `individual_file_responses` for later consolidation.
    
    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing documents and processing parameters.
        
    Returns
    -------
    GraphState
        Updated state with individual file responses.
    """
    start_time = _log_node_start("process_individual_files_node")
    _update_progress_callback(state, "process_individual_files_node", "individual_processing")
    
    question = state.get("question")
    allowed_files = state.get("allowed_files")
    documents_by_file = state.get("documents_by_file", {})
    conversation_history = state.get("conversation_history")
    detailed_flag = state.get("detailed_response_desired", True)
    # llm_instance = state.get("llm_small") or state.get("llm_large")
    llm_instance = state.get("llm_large")
    
    if not question or not allowed_files or not documents_by_file:
        logging.info("[process_individual_files_node] Missing required data. Skipping individual processing.")
        return {**state, "individual_file_responses": {}}
    
    if not llm_instance:
        logging.error("[process_individual_files_node] LLM instance not available. Cannot process files.")
        return {**state, "individual_file_responses": {}}
    
    conversation_history_str = _format_conversation_history(conversation_history)
    combo_prompt = get_synthesis_prompt_template()
    
    # Limit concurrent LLM calls to 10
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    total_files = len(allowed_files)
    logging.info(f"[process_individual_files_node] Processing {total_files} files asynchronously (max {MAX_CONCURRENT_LLM_CALLS} concurrent)")
    
    # Progress tracking
    completed_files = 0
    progress_lock = asyncio.Lock()
    
    async def update_progress():
        """Update progress callback with current completion status."""
        nonlocal completed_files
        async with progress_lock:
            completed_files += 1
            _update_progress_callback(
                state, 
                "process_individual_files_node", 
                "individual_processing",
                {"completed": completed_files, "total": total_files}
            )
    
    async def process_single_file(filename: str) -> tuple[str, str]:
        """
        Process a single file asynchronously with concurrency control.
        
        Parameters
        ----------
        filename : str
            Name of the file to process.
            
        Returns
        -------
        tuple[str, str]
            Tuple of (filename, response).
        """
        try:
            async with semaphore:  # Limit concurrent access
                logging.info(f"[process_individual_files_node] Starting async processing for {filename}")
                
                docs_list = documents_by_file.get(filename, [])
                
                if not docs_list:
                    logging.info(f"[process_individual_files_node] No documents found for {filename}")
                    await update_progress()
                    return filename, f"No relevant information found in {filename}."
                
                # Format documents for this file
                file_content_lines = []
                for doc in docs_list:
                    page = doc.metadata.get('page', 'N/A')
                    file_content_lines.append(f"Page {page}:\n{doc.page_content}")
                
                file_content = "\n\n---\n\n".join(file_content_lines)
                
                # Track files with no info and errors (empty for individual processing)
                no_info_list = []
                error_list = []
                
                try:
                    logging.info(f"[process_individual_files_node] Making LLM call for {filename}")
                    
                    # Generate response for this individual file
                    file_response = await _stream_final_generation(
                        question=question,
                        content_llm=file_content,
                        llm_instance=llm_instance,
                        combo_prompt=combo_prompt,
                        conversation_history_str=conversation_history_str,
                        no_info_list=no_info_list,
                        error_list=error_list,
                        streaming_callback=None,  # Don't stream individual file responses
                        max_tokens_override=INDIVIDUAL_FILE_MAX_TOKENS
                    )
                    
                    # Check if response exceeds MAX_COMPLETION_TOKENS (rough estimate: 4 chars per token)
                    estimated_tokens = len(file_response) // 4
                    if estimated_tokens > MAX_COMPLETION_TOKENS:
                        logging.error(f"[process_individual_files_node] Response for {filename} estimated at {estimated_tokens:,} tokens, exceeds MAX_COMPLETION_TOKENS ({MAX_COMPLETION_TOKENS:,})")
                        logging.error(f"[process_individual_files_node] Response length: {len(file_response):,} characters")
                        
                        # Truncate to a reasonable length (roughly MAX_COMPLETION_TOKENS * 4 chars)
                        max_chars = MAX_COMPLETION_TOKENS * 4
                        file_response = file_response[:max_chars] + "\n\n[Response truncated due to token limit]"
                        logging.warning(f"[process_individual_files_node] Truncated response to {len(file_response):,} characters")
                    
                    # Additional safety check for extremely long responses
                    max_response_length = 50000  # characters (much higher than before)
                    if len(file_response) > max_response_length:
                        logging.error(f"[process_individual_files_node] Response for {filename} was {len(file_response):,} chars, truncating to {max_response_length:,}")
                        file_response = file_response[:max_response_length] + "\n\n[Response truncated due to length limits]"
                    
                    logging.info(f"[process_individual_files_node] Completed async processing for {filename} ({len(file_response)} chars)")
                    await update_progress()
                    return filename, file_response
                    
                except Exception as e:
                    # Log detailed error information
                    logging.error(f"[process_individual_files_node] Error processing {filename}: {e}")
                    logging.error(f"[process_individual_files_node] Error type: {type(e).__name__}")
                    logging.error(f"[process_individual_files_node] Error details: {e}")
                    
                    # Log additional context for debugging
                    if hasattr(e, 'response'):
                        logging.error(f"[process_individual_files_node] API Response: {e.response}")
                    if hasattr(e, 'status_code'):
                        logging.error(f"[process_individual_files_node] Status Code: {e.status_code}")
                    if hasattr(e, 'body'):
                        logging.error(f"[process_individual_files_node] Response Body: {e.body}")
                    
                    # Use the new error handling function
                    error_msg = handle_llm_error(e, None, f"process_individual_files_node_{filename}")
                    
                    await update_progress()
                    return filename, error_msg
                    
        except Exception as outer_e:
            # Catch any errors in the semaphore or outer processing
            error_msg = f"Critical error processing {filename}: {str(outer_e)}"
            logging.error(f"[process_individual_files_node] {error_msg}")
            logging.error(f"[process_individual_files_node] Outer error type: {type(outer_e).__name__}")
            logging.error(f"[process_individual_files_node] Outer error details: {outer_e}")
            await update_progress()
            return filename, error_msg
    
    try:
        # Initial progress update
        _update_progress_callback(
            state, 
            "process_individual_files_node", 
            "individual_processing",
            {"completed": 0, "total": total_files}
        )
        
        # Process all files asynchronously with concurrency control
        logging.info(f"[process_individual_files_node] Creating {total_files} async tasks")
        tasks = [process_single_file(filename) for filename in allowed_files]
        
        logging.info(f"[process_individual_files_node] Starting asyncio.gather with {len(tasks)} tasks")
        
        # Wait for all files to complete processing
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logging.info(f"[process_individual_files_node] asyncio.gather completed with {len(results)} results")
        
    except Exception as gather_error:
        logging.error(f"[process_individual_files_node] Error in asyncio.gather: {gather_error}")
        logging.error(f"[process_individual_files_node] Gather error type: {type(gather_error).__name__}")
        logging.error(f"[process_individual_files_node] Gather error details: {gather_error}")
        
        # Return empty results on critical error
        results = []
    
    # Convert results to dictionary
    individual_responses: Dict[str, str] = {}
    successful_files = 0
    failed_files = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed_files += 1
            logging.error(f"[process_individual_files_node] Task {i} failed with exception: {result}")
            logging.error(f"[process_individual_files_node] Task {i} exception type: {type(result).__name__}")
            continue
        
        if isinstance(result, tuple) and len(result) == 2:
            filename, response = result
            individual_responses[filename] = response
            successful_files += 1
        else:
            failed_files += 1
            logging.error(f"[process_individual_files_node] Task {i} returned invalid result: {result}")
    
    logging.info(f"[process_individual_files_node] Processing summary: {successful_files} successful, {failed_files} failed")
    logging.info(f"[process_individual_files_node] Completed async processing for {len(individual_responses)} files")
    
    # Final progress update
    _update_progress_callback(
        state, 
        "process_individual_files_node", 
        "individual_processing",
        {"completed": len(individual_responses), "total": total_files}
    )
    
    _log_node_end("process_individual_files_node", start_time)
    return {**state, "individual_file_responses": individual_responses}


def create_graph_app() -> Graph:
    """
    Build and compile the LangGraph workflow for the KnowAI agent.

    The workflow wires together the individual LangGraph nodes that
    perform each stage of the question‑answer pipeline:

    1. Instantiate embeddings, LLM, vector store, and retriever.
    2. Generate multi-queries for the user's question.
    3. Extract document chunks relevant to the user's question.
    4. Format raw documents for synthesis.
    5. Route to either batch processing or individual file processing based on configuration.
    6. Combine raw text into a final synthesized response.

    Returns
    -------
    Graph
        A compiled, ready‑to‑run LangGraph representing the complete agent
        workflow.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("instantiate_embeddings_node", instantiate_embeddings)
    workflow.add_node("instantiate_llm_large_node", instantiate_llm_large)
    workflow.add_node("instantiate_llm_small_node", instantiate_llm_small)
    workflow.add_node("load_vectorstore_node", load_faiss_vectorstore)
    workflow.add_node("instantiate_retriever_node", instantiate_retriever)
    workflow.add_node("generate_multi_queries_node", generate_multi_queries_node)
    workflow.add_node("extract_documents_node", extract_documents_parallel_node)
    workflow.add_node("format_raw_documents_node", format_raw_documents_for_synthesis_node)
    workflow.add_node("process_batches_node", process_batches_node)
    workflow.add_node("process_individual_files_node", process_individual_files_node)
    workflow.add_node("hierarchical_consolidation_node", hierarchical_consolidation_node)
    workflow.add_node("combine_answers_node", combine_answers_node)

    workflow.set_entry_point("instantiate_embeddings_node")
    workflow.add_edge("instantiate_embeddings_node", "instantiate_llm_large_node")
    workflow.add_edge("instantiate_llm_large_node", "instantiate_llm_small_node")
    workflow.add_edge("instantiate_llm_small_node", "load_vectorstore_node")
    workflow.add_edge("load_vectorstore_node", "instantiate_retriever_node")
    workflow.add_edge("instantiate_retriever_node", "generate_multi_queries_node")
    workflow.add_edge("generate_multi_queries_node", "extract_documents_node")
    workflow.add_edge("extract_documents_node", "format_raw_documents_node")
    workflow.add_edge("format_raw_documents_node", "process_batches_node")
    
    # Conditional routing based on process_files_individually flag
    def route_to_processing(state: GraphState) -> str:
        """Route to individual file processing or combine answers based on configuration."""
        if state.get("process_files_individually", False):
            return "process_individual_files_node"
        else:
            return "combine_answers_node"
    
    workflow.add_conditional_edges(
        "process_batches_node",
        route_to_processing,
        {
            "process_individual_files_node": "process_individual_files_node",
            "combine_answers_node": "combine_answers_node"
        }
    )
    
    # Add hierarchical consolidation step after individual file processing
    workflow.add_edge("process_individual_files_node", "hierarchical_consolidation_node")
    workflow.add_edge("hierarchical_consolidation_node", "combine_answers_node")
    workflow.add_edge("combine_answers_node", END)

    return workflow.compile()

