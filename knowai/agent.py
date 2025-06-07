# knowai/agent.py
"""
Contains the LangGraph agent definition, including GraphState, node functions,
and graph compilation logic.
"""
import asyncio
import logging
import os
import time 
from typing import List, TypedDict, Dict, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever 
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langgraph.graph import StateGraph, END, Graph


# Content Policy Error Handling
CONTENT_POLICY_MESSAGE = "Due to content management policy issues with the AI provider, we are not able to provide a response to this topic. Please rephrase your question and try again."


# Fetch Azure credentials from environment variables (loaded by core.py)
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

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
        The user’s current question.
    documents_by_file : Optional[Dict[str, List[Document]]]
        Mapping of filenames to the list of retrieved document chunks.
    individual_answers : Optional[Dict[str, str]]
        Answers generated for each file individually.
    n_alternatives : Optional[int]
        Number of alternative queries to generate per question.
    k_per_query : Optional[int]
        Chunks to retrieve per alternative query.
    generation : Optional[str]
        Final synthesized answer.
    conversation_history : Optional[List[Dict[str, str]]]
        List of previous conversation turns.
    bypass_individual_generation : Optional[bool]
        Whether to skip individual‑file answer generation.
    raw_documents_for_synthesis : Optional[str]
        Raw document text formatted for the synthesizer.
    k_chunks_retriever : int
        Total chunks to retrieve for the base retriever.
    combine_threshold : int
        Maximum number of individual answers that may be combined in a
        single batch before hierarchical combining is used.
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
    individual_answers: Optional[Dict[str, str]] 
    n_alternatives: Optional[int] 
    k_per_query: Optional[int]
    generation: Optional[str] 
    conversation_history: Optional[List[Dict[str, str]]]
    bypass_individual_generation: Optional[bool]
    raw_documents_for_synthesis: Optional[str]
    combined_documents: Optional[List[Document]]
    detailed_response_desired: Optional[bool]
    k_chunks_retriever: int
    k_chunks_retriever_all_docs: int
    combine_threshold: int
    max_tokens_individual_answer: int 

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
        "prompt blocked" # Common for Azure
    ]
    return any(keyword in error_message for keyword in keywords)


def instantiate_embeddings(state: GraphState) -> GraphState:
    """
    Instantiate and attach an Azure OpenAI embeddings model to the graph
    state.

    The function checks whether an embeddings model already exists in
    ``state``. If absent, it creates a new
    :class:`langchain_openai.AzureOpenAIEmbeddings` instance using the Azure
    configuration provided by module‑level environment variables.  Any
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_embeddings_node"
    logging.info(f"--- Starting Node: {node_name} ---")
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
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_llm_large_node"
    logging.info(f"--- Starting Node: {node_name} (for query generation) ---")
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
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_llm_small_node"
    logging.info(f"--- Starting Node: {node_name} (for query generation) ---")
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
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return state


def load_faiss_vectorstore(state: GraphState) -> GraphState:
    """
    Load a local FAISS vector store from the path stored in ``state`` and
    attach it to the graph state.

    The function validates that a vector‑store path exists, an embeddings
    model has been instantiated, and the target directory is present on
    disk. If any check fails or loading raises an exception, the
    ``vectorstore`` field in the returned state is set to ``None`` and the
    error is logged.  When loading succeeds, the resulting
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
    t_node_start = time.perf_counter()
    node_name = "load_vectorstore_node"
    logging.info(f"--- Starting Node: {node_name} ---")
    current_vectorstore_path = state.get("vectorstore_path") 
    embeddings = state.get("embeddings")
    
    if "vectorstore" not in state: state["vectorstore"] = None 

    if state.get("vectorstore"): 
        logging.info("Vectorstore already exists in state.")
    elif not current_vectorstore_path: 
        logging.error("Vectorstore path not provided in state."); state["vectorstore"] = None
    elif not embeddings: 
        logging.error("Embeddings not instantiated."); state["vectorstore"] = None
    elif not os.path.exists(current_vectorstore_path) or not os.path.isdir(current_vectorstore_path):
        logging.error(f"FAISS vectorstore path does not exist or is not a directory: {current_vectorstore_path}"); state["vectorstore"] = None
    else:
        try:
            logging.info(f"Loading FAISS vectorstore from '{current_vectorstore_path}' ...")
            loaded_vectorstore = FAISS.load_local(
                folder_path=current_vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True
            )
            logging.info(f"FAISS vectorstore loaded with {loaded_vectorstore.index.ntotal} embeddings.")
            state = {**state, "vectorstore": loaded_vectorstore}
        except Exception as e:
            logging.exception(f"Failed to load FAISS vectorstore: {e}"); state["vectorstore"] = None
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return state


def instantiate_retriever(state: GraphState) -> GraphState:
    """
    Instantiate and attach a base retriever built from the loaded FAISS
    vector store.

    The function checks that a FAISS vector store is present in ``state``.
    If available, it constructs a
    :class:`langchain_core.vectorstores.VectorStoreRetriever` using the
    ``k`` value stored in ``state['k_chunks_retriever']`` (falling back to
    the module‑level default).  On success the new retriever is written back
    to ``state`` under the ``retriever`` key.  If the vector store is
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_retriever_node"
    logging.info(f"--- Starting Node: {node_name} ---")
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
            
        search_kwargs = {"k": k_retriever, "fetch_k": k_retriever_all_docs}

        try:
            base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            logging.info(f"Base retriever instantiated with default k={k_retriever}.")
            state = {**state, "retriever": base_retriever}
        except Exception as e:
            logging.exception(f"Failed to instantiate base retriever: {e}"); state["retriever"] = None
            
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
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

    Returns
    -------
    tuple[str, Optional[List[Document]]]
        Two‑element tuple ``(file_name, docs)`` where ``docs`` is a list of
        unique :class:`langchain_core.documents.Document` instances on
        success, or ``None`` if retrieval fails.
    """
    # logging.info(f"Retrieving for file: {file_name} using {len(query_embeddings_list)} pre-computed query embeddings.")
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
            key = (doc.metadata.get("file_name"), doc.metadata.get("page"), doc.page_content.strip() if hasattr(doc, 'page_content') else "")
            if key not in unique_docs_map: 
                unique_docs_map[key] = doc

        final_unique_docs = list(unique_docs_map.values())

        logging.info(f"[{file_name}] Retrieved {len(retrieved_docs)} raw -> {len(final_unique_docs)} unique docs.")
        return file_name, final_unique_docs
    
    except Exception as e_retrieve:
        logging.exception(f"[{file_name}] Error during similarity search by vector: {e_retrieve}")
        return file_name, None


async def extract_documents_parallel_node(state: GraphState) -> GraphState:
    """
    Extract relevant document chunks in parallel for each user‑selected file.

    The node performs the following steps:

    1. Generate alternative queries for the user’s question with a large
       language model (via :pyclass:`langchain.retrievers.MultiQueryRetriever`)
       up to ``n_alternatives`` variants.
    2. Embed each query using the embeddings model stored in ``state``.
    3. For every file in ``state['allowed_files']`` retrieve the top
       ``k_per_query`` chunks per query embedding from the FAISS vector
       store with an asynchronous similarity search.
    4. Deduplicate retrieved chunks per file.
    5. Store the resulting mapping in ``state['documents_by_file']``.

    If any required component (question, allowed files, vector store,
    embeddings, retriever, or LLM) is missing, the function returns early
    with an empty ``documents_by_file`` dictionary.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state. Expected to contain the keys
        ``question``, 
        ``llm_large``, 
        ``llm_small``, 
        ``retriever``, 
        ``vectorstore``,
        ``embeddings``, 
        ``allowed_files``.

    Returns
    -------
    GraphState
        Updated state where ``documents_by_file`` maps each allowed filename
        to a list of retrieved :class:`langchain_core.documents.Document`
        instances (or an empty list on failure).
    """
    t_node_start = time.perf_counter()
    node_name = "extract_documents_node"
    logging.info(f"--- Starting Node: {node_name} (Async) ---")
    question, llm_large, llm_small, base_retriever, vectorstore, embeddings_model, allowed_files = (
        state.get("question"), 
        state.get("llm_large"), 
        state.get("llm_small"),
        state.get("retriever"),
        state.get("vectorstore"), 
        state.get("embeddings"), 
        state.get("allowed_files")
    )

    n_alternatives = state.get("n_alternatives", 4)
    k_per_query = state.get("k_chunks_retriever")
    k_retriever_all_docs = state.get("k_chunks_retriever_all_docs")
    current_documents_by_file: Dict[str, List[Document]] = {}

    if not question: 
        logging.info(f"[{node_name}] No question. Skipping extraction.")
        return {**state, "documents_by_file": current_documents_by_file}
    
    if not allowed_files: 
        logging.info(f"[{node_name}] No files selected. Skipping extraction.")
        return {**state, "documents_by_file": current_documents_by_file}
    
    if not all([llm_large, llm_small, base_retriever, vectorstore, embeddings_model]):
        logging.error(f"[{node_name}] Missing components for extraction. Halting.")
        return {**state, "documents_by_file": current_documents_by_file}

    query_list: List[str] = [question]
    try:
        logging.info(f"[{node_name}] Generating alternative queries...")

        mqr_llm_chain = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm_small).llm_chain

        llm_response = await mqr_llm_chain.ainvoke({"question": question})
        raw_queries_text = ""
        if isinstance(llm_response, dict): 
            raw_queries_text = str(llm_response.get(mqr_llm_chain.output_key, ""))

        elif isinstance(llm_response, str): 
            raw_queries_text = llm_response

        elif isinstance(llm_response, list): 
            raw_queries_text = "\n".join(str(item).strip() for item in llm_response if isinstance(item, str) and str(item).strip())

        else: 
            raw_queries_text = str(llm_response)
        
        alt_queries = [q.strip() for q in raw_queries_text.split("\n") if q.strip()]
        query_list.extend(list(dict.fromkeys(alt_queries))[:n_alternatives])

        logging.info(f"[{node_name}] Generated {len(query_list)} total unique queries.")

    except Exception as e_query_gen:
        if _is_content_policy_error(e_query_gen):
            logging.warning(f"[{node_name}] Content policy violation during query generation. Using original question only. Error: {e_query_gen}")
            # query_list is already initialized with the original question
        else:
            logging.exception(f"[{node_name}] Failed to generate alt queries: {e_query_gen}")
        # In both cases (policy or other error), we fall back to the original question if query_list isn't populated beyond it.

    query_embeddings_list: List[List[float]] = []
    try:
        logging.info(f"[{node_name}] Embedding {len(query_list)} queries...")
        query_embeddings_list = await embeddings_model.aembed_documents(query_list) 
    except Exception as e_embed: 
        logging.exception(f"[{node_name}] Failed to embed queries: {e_embed}")
        return {**state, "documents_by_file": current_documents_by_file}
    
    if not query_embeddings_list or len(query_embeddings_list) != len(query_list):
        logging.error(f"[{node_name}] Query embedding failed/mismatched.")
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

    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return {
        **state,
        "documents_by_file": current_documents_by_file,
        "combined_documents": combined_docs_list
    }


async def generate_individual_answers_node(state: GraphState) -> GraphState:
    """
    Generate detailed answers for each user‑selected file in parallel.

    The node iterates over the filenames in ``state['allowed_files']`` and
    produces an answer per file using only the document chunks previously
    extracted for that file (``state['documents_by_file']``).  For every
    file it constructs a prompt that:

    * Presents the file‑specific context chunks.
    * Asks the large language model (LLM) to answer the user’s question
      **solely** from that context.
    * Requires inline citations in the form
      ``"quoted text…" (filename, Page X)``.

    Answers are generated asynchronously to improve latency.  If a file has
    no retrieved chunks, a default “no relevant information” message is
    stored.  Content‑policy violations or other LLM errors are caught and
    logged; the corresponding answer is set to a predefined policy message
    or a generic error explanation.

    The resulting mapping is written back to
    ``state['individual_answers']``.  The existing synthesized answer in
    ``state['generation']`` is preserved for downstream nodes.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state.  Expected keys include
        ``question``, ``allowed_files``, and ``documents_by_file``.

    Returns
    -------
    GraphState
        Updated state where ``individual_answers`` maps each allowed
        filename to its generated answer (or a placeholder string).
    """
    t_node_start = time.perf_counter()
    node_name = "generate_answers_node"
    logging.info(f"--- Starting Node: {node_name} (Async) ---")
    
    question = state.get("question")
    documents_by_file = state.get("documents_by_file")
    _allowed_files = state.get("allowed_files")
    initial_files_for_answers = _allowed_files if _allowed_files is not None else []

    current_individual_answers: Dict[str, str] = {
        filename: f"No relevant information found in '{filename}' for this question." for filename in initial_files_for_answers
    }
    current_generation = state.get("generation")
    state_to_return = {**state, "individual_answers": current_individual_answers, "generation": current_generation}

    if not question: logging.info(f"[{node_name}] No question. Skipping.")
    elif not documents_by_file: logging.info(f"[{node_name}] No 'documents_by_file'. Skipping.")
    else:
        prompt_text = """You are an expert assistant. Answer the user's question based ONLY on the provided context from a SINGLE FILE.
Context from File '{filename}' (Chunks from Pages X, Y, Z...):
{context}
Question: {question}
Detailed Answer (with citations like "quote..." ({filename}, Page X)):"""
        prompt_template = PromptTemplate(template=prompt_text, input_variables=["context", "question", "filename"])

        # llm = state.get("llm_large")
        llm = state.get("llm_small")
        # llm = AzureChatOpenAI(
        #     temperature=0.1, 
        #     api_version=os.getenv("AZURE_OPENAI_API_4p1_VERSION"),
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        #     max_tokens=state.get("max_tokens_individual_answer")
        # )
        chain = prompt_template | llm | StrOutputParser()
        
        async def _gen_ans(fname: str, fdocs: List[Document], q: str) -> tuple[str, str]:
            if not fdocs: return fname, f"No relevant documents found in '{fname}' to answer the question."
            ctx = "\n\n".join([f"--- Context from Page {d.metadata.get('page', 'N/A')} (File: {fname}) ---\n{d.page_content}" for d in fdocs])
            try: 
                return fname, await chain.ainvoke({"context": ctx, "question": q, "filename": fname})
            except Exception as e:
                if _is_content_policy_error(e):
                    logging.warning(f"Content policy violation for file {fname}: {e}")
                    return fname, CONTENT_POLICY_MESSAGE
                logging.exception(f"Error generating answer for file {fname}: {e}")
                return fname, f"An error occurred while generating the answer for file '{fname}': {str(e)}" # Generic error
        
        tasks = []
        for fname_allowed in initial_files_for_answers:
            if fname_allowed in documents_by_file and documents_by_file[fname_allowed]: # type: ignore
                tasks.append(_gen_ans(fname_allowed, documents_by_file[fname_allowed], question)) # type: ignore
            else:
                logging.info(f"File '{fname_allowed}' no docs or not in documents_by_file. Default message retained.")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, tuple) and len(res) == 2: current_individual_answers[res[0]] = res[1]
                elif isinstance(res, Exception): logging.error(f"Task error in answer gen: {res}")
                else: logging.error(f"Unexpected task result in answer gen: {res}")
        else:
            logging.info(f"[{node_name}] No tasks for answer generation.")
        state_to_return["individual_answers"] = current_individual_answers
    
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return state_to_return


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
    answer the user’s question when individual‑file generation is
    bypassed.

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
    t_node_start = time.perf_counter()
    node_name = "format_raw_documents_for_synthesis_node"
    logging.info(f"--- Starting Node: {node_name} ---")
    
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
                formatted_raw_docs += f"--- No Content Extracted for File: {filename} ---\n\n"
    if not formatted_raw_docs and allowed_files: formatted_raw_docs = "No documents were retrieved for the selected files and question."
    elif not allowed_files: formatted_raw_docs = "No files were selected for processing."

    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
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


async def _async_combine_answer_chunk(
    question: str, 
    answer_chunk_input: Union[Dict[str, str], str], 
    llm_combiner: BaseLanguageModel,
    combination_prompt_template: PromptTemplate, 
    chunk_name: str, 
    conversation_history_str: str,
    is_raw_chunk: bool
) -> str:
    """
    Combine a single group of answers or raw‑text chunks into an
    intermediate synthesized result using an LLM.

    The coroutine formats *answer_chunk_input* according to
    *combination_prompt_template* and invokes *llm_combiner* to produce a
    unified chunk of text. It supports two modes:

    * **Processed‑answers mode** (`is_raw_chunk=False`) – *answer_chunk_input*
      is a mapping from filename to the answer previously generated for
      that file.
    * **Raw‑chunk mode** (`is_raw_chunk=True`) – *answer_chunk_input* is the
      raw text extracted from documents, already concatenated for prompt
      injection.

    If any sub‑answer equals the global ``CONTENT_POLICY_MESSAGE`` the
    function short‑circuits and returns that message unchanged. Exceptions
    raised during LLM invocation are caught; policy violations return
    ``CONTENT_POLICY_MESSAGE`` while other errors yield a descriptive
    string.

    Parameters
    ----------
    question : str
        The user’s current question.
    answer_chunk_input : Union[Dict[str, str], str]
        Answers dictionary (processed mode) or raw text block (raw‑chunk
        mode).
    llm_combiner : BaseLanguageModel
        Language model used to perform the combination.
    combination_prompt_template : PromptTemplate
        Prompt template that wraps the chunk content before LLM invocation.
    chunk_name : str
        Human‑readable identifier for logging.
    conversation_history_str : str
        Pre‑formatted conversation history passed to the prompt.
    is_raw_chunk : bool
        ``True`` if *answer_chunk_input* is raw text, ``False`` if it is a
        processed answers mapping.

    Returns
    -------
    str
        Synthesized answer chunk, the policy‑violation message, or an error
        description.
    """
    logging.info(f"Combining answer chunk: {chunk_name} (is_raw_chunk: {is_raw_chunk}).")

    formatted_chunk_content_for_prompt: str
    if is_raw_chunk and isinstance(answer_chunk_input, str):
        formatted_chunk_content_for_prompt = answer_chunk_input
    elif not is_raw_chunk and isinstance(answer_chunk_input, dict):
        temp_content = ""
        for filename, answer in answer_chunk_input.items():
            if answer == CONTENT_POLICY_MESSAGE: # If a chunk is already a policy message, pass it as is
                return CONTENT_POLICY_MESSAGE 
            temp_content += f"--- Answer based on file: {filename} ---\n{answer}\n\n"

        formatted_chunk_content_for_prompt = temp_content.strip()

        if not formatted_chunk_content_for_prompt: # All answers in chunk were policy messages
             return CONTENT_POLICY_MESSAGE 
    else: return f"Error: Invalid input for combining chunk {chunk_name}."

    chain = combination_prompt_template | llm_combiner | StrOutputParser()

    try:
        no_info_placeholder = "Not applicable for this intermediate chunk."
        errors_placeholder = "Not applicable for this intermediate chunk."
        combined_text = await chain.ainvoke({
            "question": question, 
            "formatted_answers_or_raw_docs": formatted_chunk_content_for_prompt,
            "files_no_info": no_info_placeholder,
            "files_errors": errors_placeholder,
            "conversation_history": conversation_history_str 
        })
        return combined_text
    except Exception as e:
        if _is_content_policy_error(e):
            logging.warning(f"Content policy violation during chunk combination {chunk_name}: {e}")
            return CONTENT_POLICY_MESSAGE
        logging.exception(f"Error combining answer chunk {chunk_name}: {e}")
        return f"Error combining chunk {chunk_name}. Content:\n" + formatted_chunk_content_for_prompt


async def combine_answers_node(state: GraphState) -> GraphState:
    """
    Synthesize a final answer for the user by combining individual file‑
    based answers or raw document text.

    The node supports two workflows controlled by
    ``state['bypass_individual_generation']``:

    * **Standard mode** (`bypass_individual_generation=False`) – Combine the
      pre‑processed answers stored in ``state['individual_answers']``.
    * **Bypass mode** (`bypass_individual_generation=True`) – Skip individual
      answers and instead combine the raw document text assembled in
      ``state['raw_documents_for_synthesis']``.

    The function performs hierarchical combination when the number of
    answers exceeds ``state['combine_threshold']`` by chunking the inputs
    and calling :pyfunc:`_async_combine_answer_chunk` asynchronously,
    followed by a final synthesis step. Content‑policy violations are
    propagated using the global :data:`CONTENT_POLICY_MESSAGE`.

    Error and “no‑info” conditions are tracked per file and injected back
    into the final prompt so that the LLM can acknowledge gaps or issues.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state containing (among others) the keys
        ``question``, ``allowed_files``, ``individual_answers``,
        ``raw_documents_for_synthesis``, ``combine_threshold``, and
        ``conversation_history``.

    Returns
    -------
    GraphState
        Updated state where ``generation`` holds the synthesized answer,
        a content‑policy message, or an error explanation.
    """
    t_node_start = time.perf_counter()
    node_name = "combine_answers_node"
    logging.info(f"--- Starting Node: {node_name} (Async) ---")

    question, individual_answers, allowed_files, conversation_history, bypass_flag, raw_docs_for_synthesis = (
        state.get("question"), 
        state.get("individual_answers"), 
        state.get("allowed_files"),
        state.get("conversation_history"), 
        state.get("bypass_individual_generation", False),
        state.get("raw_documents_for_synthesis")
    )
    combine_thresh = state.get("combine_threshold")
    output_generation: Optional[str] = "Error during synthesis."
    state_to_return = {**state}

    if not allowed_files: output_generation = "Please select files to analyze."
    elif not question: output_generation = f"Files selected: {', '.join(allowed_files) if allowed_files else 'any'}. Ask a question."
    else:
        conversation_history_str = _format_conversation_history(conversation_history)

        detailed_flag = state.get("detailed_response_desired", True)    
        llm_instance = state.get("llm_large") if detailed_flag else state.get("llm_small")

        # llm_instance = AzureChatOpenAI(
        #     temperature=0.0, 
        #     api_version=os.getenv("AZURE_OPENAI_API_4p1_VERSION"),
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NANO"),
        # )
        
        prompt_processed = """You are an expert synthesis assistant. Combine PRE-PROCESSED answers.
Conversation History: {conversation_history}
User's CURRENT Question: {question}
Individual PRE-PROCESSED Answers: {formatted_answers_or_raw_docs}
Files with No Relevant Info: {files_no_info}
Files with Errors: {files_errors}
Instructions: Synthesize, preserve details & citations (e.g., "quote..." (file.pdf, Page X)). Attribute. Structure. Handle contradictions. Acknowledge files with no info/errors.
Synthesized Answer:"""

        prompt_raw = """You are an expert AI assistant. Answer CURRENT question based ONLY on RAW text chunks.
Conversation History: {conversation_history}
User's CURRENT Question: {question}
RAW Text Chunks: {formatted_answers_or_raw_docs}
Files with No Relevant Info (no chunks extracted): {files_no_info}
Files with Errors (extraction errors): {files_errors}
Instructions: Read raw text. Answer ONLY from raw text. Quote with citations (e.g., "quote..." (file.pdf, Page X)). If info not found, state it. Structure logically.
Synthesized Answer from RAW Docs:"""
        
        active_prompt_text = prompt_raw if bypass_flag else prompt_processed
        combo_prompt = PromptTemplate(
            template=active_prompt_text, 
            input_variables=[
                "question", 
                "formatted_answers_or_raw_docs", 
                "files_no_info", 
                "files_errors", 
                "conversation_history"
            ]
        )
        
        no_info_list: List[str] = []
        error_list: List[str] = []
        content_llm: str = ""

        if bypass_flag:
            logging.info(f"[{node_name}] Combining raw documents.")
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
                        no_info_list.append(f"`{af}`")
            error_list.append("Error tracking for raw path not detailed here.")
        else: 
            if not individual_answers: output_generation = "No individual answers to combine."
            else:
                ans_to_combine: Dict[str, str] = {}
                for fname, ans in individual_answers.items():
                    if ans == CONTENT_POLICY_MESSAGE: error_list.append(f"`{fname}` (Content Policy)") # Specifically track policy issues
                    elif "An error occurred" in ans: error_list.append(f"`{fname}`")
                    elif "No relevant information found" in ans or "No relevant documents were found" in ans: no_info_list.append(f"`{fname}`")
                    else: ans_to_combine[fname] = ans
                
                if not ans_to_combine: # All answers were errors or no_info or policy
                    if any(CONTENT_POLICY_MESSAGE in ans for ans in individual_answers.values()):
                         output_generation = CONTENT_POLICY_MESSAGE
                    else:
                        msg_parts = [f"I couldn't find specific information to answer: '{question}'."]
                        if no_info_list: msg_parts.append(f"No info in: {', '.join(no_info_list)}.")
                        if error_list: msg_parts.append(f"Issues with: {', '.join(error_list)}.") # Modified to be more generic
                        output_generation = "\n".join(msg_parts)
                else:
                    if len(ans_to_combine) <= combine_thresh:
                        content_llm = "\n\n".join([f"--- Answer from file: {fn} ---\n{an}" for fn, an in ans_to_combine.items()])
                    else: 
                        items = list(ans_to_combine.items())
                        tasks_s1 = [
                            _async_combine_answer_chunk(question, dict(items[i:i+combine_thresh]), llm_instance, combo_prompt, f"ProcChunk{i//combine_thresh+1}", conversation_history_str, False)
                            for i in range(0, len(items), combine_thresh)
                        ]
                        interm_res = await asyncio.gather(*tasks_s1, return_exceptions=True)
                        
                        # Check if all intermediate results are content policy messages
                        if all(res == CONTENT_POLICY_MESSAGE for res in interm_res if isinstance(res, str)):
                            output_generation = CONTENT_POLICY_MESSAGE
                            content_llm = "" # Prevent further processing
                        else:
                            valid_texts = [r for r in interm_res if isinstance(r, str) and r != CONTENT_POLICY_MESSAGE and "Error combining chunk" not in r]
                            error_chunks = [r for r in interm_res if not (isinstance(r, str) and r != CONTENT_POLICY_MESSAGE and "Error combining chunk" not in r)]
                            policy_chunks = [r for r in interm_res if isinstance(r, str) and r == CONTENT_POLICY_MESSAGE]

                            if not valid_texts and policy_chunks: # Only policy violations or errors
                                output_generation = CONTENT_POLICY_MESSAGE
                                content_llm = ""
                            elif not valid_texts and error_chunks:
                                output_generation = "Failed to combine intermediate answer chunks due to errors."
                                content_llm = ""
                            else:
                                content_llm = "\n\n".join([f"--- Synthesized Batch {i+1} ---\n{t}" for i, t in enumerate(valid_texts)])
                                if policy_chunks: error_list.append(f"{len(policy_chunks)} intermediate chunk(s) hit content policy.")
                                if error_chunks: error_list.append(f"{len(error_chunks)} intermediate chunk(s) had errors.")
        
        if content_llm and (output_generation == "Error during synthesis." or (ans_to_combine if not bypass_flag else True)):
            try:
                final_chain = combo_prompt | llm_instance | StrOutputParser()
                output_generation = await final_chain.ainvoke({
                    "question": question, "formatted_answers_or_raw_docs": content_llm,
                    "files_no_info": ", ".join(no_info_list) if no_info_list else "None",
                    "files_errors": ", ".join(error_list) if error_list else "None", # error_list now includes policy issues
                    "conversation_history": conversation_history_str
                })
            except Exception as e:
                if _is_content_policy_error(e):
                    logging.warning(f"Content policy violation during final combination: {e}")
                    output_generation = CONTENT_POLICY_MESSAGE
                else:
                    logging.exception(f"[{node_name}] Final combination LLM error: {e}")
                    output_generation = f"Final synthesis error: {e}. Content: {content_llm[:200]}..."
        
    state_to_return["generation"] = output_generation
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return state_to_return


def decide_processing_path_after_extraction(state: GraphState) -> str:
    """
    Decide which processing path should follow document extraction.

    The choice depends on three pieces of information stored in *state*:

    * ``question`` – the user’s current question.
    * ``allowed_files`` – the list of filenames selected by the user.
    * ``bypass_individual_generation`` – ``True`` when individual‑file
      answer generation should be skipped.

    Routing logic
    -------------
    * If either *question* or *allowed_files* is missing, route to
      ``"to_generate_individual_answers"`` so the UI can prompt the user.
    * If *bypass_individual_generation* is ``True``, route to
      ``"to_format_raw_for_synthesis"`` to format raw text for a single
      synthesis pass.
    * Otherwise, route to ``"to_generate_individual_answers"`` to create
      answers for each file separately.

    Parameters
    ----------
    state : GraphState
        Current mutable graph state produced by
        :pyfunc:`extract_documents_parallel_node`.

    Returns
    -------
    str
        Workflow edge label that indicates the next node to execute.
    """
    node_name = "decide_processing_path_after_extraction"
    bypass = state.get("bypass_individual_generation", False)
    question = state.get("question")
    allowed_files = state.get("allowed_files")

    if not question or not allowed_files:
        logging.info(f"[{node_name}] No question/files. Default to standard generation for messaging.")
        return "to_generate_individual_answers"
    if bypass:
        logging.info(f"[{node_name}] Bypass TRUE. Routing to format raw documents.")
        return "to_format_raw_for_synthesis"
    else:
        logging.info(f"[{node_name}] Bypass FALSE. Routing to generate individual answers.")
        return "to_generate_individual_answers"


def create_graph_app() -> Graph:
    """
    Build and compile the LangGraph workflow for the KnowAI agent.

    The workflow wires together the individual LangGraph nodes that
    perform each stage of the question‑answer pipeline:

    1. Instantiate embeddings, LLM, vector store, and retriever.
    2. Extract document chunks relevant to the user’s question.
    3. Optionally format raw documents or generate per‑file answers.
    4. Combine answers (or raw text) into a final synthesized response.

    Conditional routing after document extraction is decided by
    :pyfunc:`decide_processing_path_after_extraction`.

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
    workflow.add_node("extract_documents_node", extract_documents_parallel_node)
    workflow.add_node("format_raw_documents_node", format_raw_documents_for_synthesis_node) 
    workflow.add_node("generate_answers_node", generate_individual_answers_node) 
    workflow.add_node("combine_answers_node", combine_answers_node) 

    workflow.set_entry_point("instantiate_embeddings_node")
    workflow.add_edge("instantiate_embeddings_node", "instantiate_llm_large_node")
    workflow.add_edge("instantiate_llm_large_node", "instantiate_llm_small_node")
    workflow.add_edge("instantiate_llm_small_node", "load_vectorstore_node")
    workflow.add_edge("load_vectorstore_node", "instantiate_retriever_node")
    workflow.add_edge("instantiate_retriever_node", "extract_documents_node")
    workflow.add_conditional_edges(
        "extract_documents_node",
        decide_processing_path_after_extraction,
        {
            "to_format_raw_for_synthesis": "format_raw_documents_node",
            "to_generate_individual_answers": "generate_answers_node"
        }
    )
    workflow.add_edge("format_raw_documents_node", "combine_answers_node")
    workflow.add_edge("generate_answers_node", "combine_answers_node") 
    workflow.add_edge("combine_answers_node", END) 

    return workflow.compile()
