# knowai/agent.py
"""
Contains the LangGraph agent definition, including GraphState, node functions,
and graph compilation logic.
"""
import asyncio
import logging
import os
import time
from typing import List, TypedDict, Dict, Optional, Callable

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as Graph

from .prompts import (
    get_synthesis_prompt_template,
    CONTENT_POLICY_MESSAGE,
    get_progress_message
)


logger = logging.getLogger(__name__)


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
    k_chunks_retriever : int
        Total chunks to retrieve for the base retriever.
    combine_threshold : int
        Maximum number of individual answers that may be combined in a
        single batch before hierarchical combining is used.
    detailed_response_desired : Optional[bool]
        Whether to use detailed (large) or simple (small) LLM.
    generated_queries : Optional[List[str]]
        List of generated alternative queries.
    query_embeddings : Optional[List[List[float]]]
        Embeddings for generated queries.
    streaming_callback : Optional[Callable[[str], None]]
        Callback function for streaming tokens.
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
    combine_threshold: int
    generated_queries: Optional[List[str]]
    query_embeddings: Optional[List[List[float]]]
    streaming_callback: Optional[Callable[[str], None]]


def _is_content_policy_error(e: Exception) -> bool:
    """
    Determine whether an exception message indicates an AI content‑policy
    violation.

    Parameters
    ----------
    e : Exception
        Exception raised by the LLM provider.

    Returns
    -------
    bool
        ``True`` if the exception message contains any keyword that signals
        a policy‑related block; otherwise ``False``.
    """
    error_message = str(e).lower()
    keywords = [
        "content filter",
        "content management policy",
        "responsible ai",
        "safety policy",
        "prompt blocked"  # Common for Azure
    ]
    return any(keyword in error_message for keyword in keywords)


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
    stage: str
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
    """
    progress_cb = state.get("__progress_cb__")
    if progress_cb:
        progress_cb(
            get_progress_message(stage, node_name),
            "info",
            {"node": node_name, "stage": stage}
        )


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
            new_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_4p1_VERSION")
            )
            state = {**state, "embeddings": new_embeddings}
        except Exception as e:
            logging.error(f"Failed to instantiate embeddings model: {e}")
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
            new_llm = AzureChatOpenAI(
                temperature=0.1,
                api_version=os.getenv("AZURE_OPENAI_API_4p1_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            )
            state = {**state, "llm_large": new_llm}
        except Exception as e:
            logging.error(f"Failed to instantiate large LLM model: {e}")
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
            new_llm = AzureChatOpenAI(
                temperature=0.1,
                api_version=os.getenv("AZURE_OPENAI_API_4p1_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NANO"),
            )
            state = {**state, "llm_small": new_llm}
        except Exception as e:
            logging.error(f"Failed to instantiate small LLM model: {e}")
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
            retriever=base_retriever, llm=llm_small
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
        if _is_content_policy_error(e_query_gen):
            logging.warning(
                "[generate_multi_queries_node] Content policy violation during "
                f"query generation. Using original question only. Error: {e_query_gen}"
            )
        else:
            logging.exception(
                f"[generate_multi_queries_node] Failed to generate alt queries: {e_query_gen}"
            )
        # In both cases, we fall back to the original question

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
    streaming_callback: Optional[Callable[[str], None]]
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
        # Create the streaming chain
        chain = combo_prompt | llm_instance | StrOutputParser()

        # Prepare the input
        input_data = {
            "question": question,
            "formatted_answers_or_raw_docs": content_llm,
            "files_no_info": ", ".join(no_info_list) if no_info_list else "None",
            "files_errors": ", ".join(error_list) if error_list else "None",
            "conversation_history": conversation_history_str
        }

        if streaming_callback:
            # Use streaming if callback is provided
            full_response = ""
            async for chunk in chain.astream(input_data):
                if chunk:
                    full_response += chunk
                    streaming_callback(chunk)
            return full_response
        else:
            # Use regular invocation if no streaming callback
            return await chain.ainvoke(input_data)

    except Exception as e:
        if _is_content_policy_error(e):
            logging.warning(f"Content policy violation during streaming generation: {e}")
            return CONTENT_POLICY_MESSAGE
        else:
            logging.exception(f"Error during streaming generation: {e}")
            return f"Generation error: {e}. Content: {content_llm[:200]}..."


async def combine_answers_node(state: GraphState) -> GraphState:
    """
    Synthesize a final answer for the user by combining raw document text.

    The function performs hierarchical combination when the number of
    documents exceeds ``state['combine_threshold']`` by chunking the inputs
    and calling synthesis asynchronously, followed by a final synthesis step.
    Content‑policy violations are propagated using the global
    :data:`CONTENT_POLICY_MESSAGE`.

    Error and "no‑info" conditions are tracked per file and injected back
    into the final prompt so that the LLM can acknowledge gaps or issues.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing (among others) the keys
        ``question``, ``allowed_files``, ``raw_documents_for_synthesis``,
        ``combine_threshold``, and ``conversation_history``.

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
    raw_docs_for_synthesis = state.get("raw_documents_for_synthesis")
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

        detailed_flag = state.get("detailed_response_desired", True)
        llm_instance = state.get("llm_large") if detailed_flag else state.get("llm_small")

        # Use centralized prompt template for raw documents
        combo_prompt = get_synthesis_prompt_template()

        no_info_list: List[str] = []
        error_list: List[str] = []
        content_llm: str = ""

        logging.info("[combine_answers_node] Combining raw documents.")
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
            content_llm = raw_docs_for_synthesis if raw_docs_for_synthesis else "No raw documents."

        # Track files with no docs
        docs_by_file = state.get("documents_by_file", {})
        if allowed_files:
            for af in allowed_files:
                if not docs_by_file.get(af):
                    no_info_list.append(f"`{af}` (no chunks extracted)")
        error_list.append("Error tracking for raw path not detailed here.")

        if content_llm and (output_generation == "Error during synthesis." or True):
            # Get streaming callback from state
            streaming_callback = state.get("streaming_callback")

            # Use streaming generation
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

    state_to_return["generation"] = output_generation
    _log_node_end("combine_answers_node", start_time)
    return state_to_return


def create_graph_app() -> Graph:
    """
    Build and compile the LangGraph workflow for the KnowAI agent.

    The workflow wires together the individual LangGraph nodes that
    perform each stage of the question‑answer pipeline:

    1. Instantiate embeddings, LLM, vector store, and retriever.
    2. Generate multi-queries for the user's question.
    3. Extract document chunks relevant to the user's question.
    4. Format raw documents for synthesis.
    5. Combine raw text into a final synthesized response.

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
    workflow.add_node("combine_answers_node", combine_answers_node)

    workflow.set_entry_point("instantiate_embeddings_node")
    workflow.add_edge("instantiate_embeddings_node", "instantiate_llm_large_node")
    workflow.add_edge("instantiate_llm_large_node", "instantiate_llm_small_node")
    workflow.add_edge("instantiate_llm_small_node", "load_vectorstore_node")
    workflow.add_edge("load_vectorstore_node", "instantiate_retriever_node")
    workflow.add_edge("instantiate_retriever_node", "generate_multi_queries_node")
    workflow.add_edge("generate_multi_queries_node", "extract_documents_node")
    workflow.add_edge("extract_documents_node", "format_raw_documents_node")
    workflow.add_edge("format_raw_documents_node", "combine_answers_node")
    workflow.add_edge("combine_answers_node", END)

    return workflow.compile()
