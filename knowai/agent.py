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
from langgraph.graph import StateGraph, END

# Constants (can be overridden by KnowAIAgent constructor)
K_CHUNKS_RETRIEVER_DEFAULT = 25 
COMBINE_THRESHOLD_DEFAULT = 3
MAX_CONVERSATION_TURNS_DEFAULT = 15

# --- Content Policy Error Handling ---
CONTENT_POLICY_MESSAGE = "Due to content management policy issues with the AI provider, we are not able to provide a response to this topic. Please rephrase your question and try again."

def _is_content_policy_error(e: Exception) -> bool:
    """Checks if an exception message indicates a content policy violation."""
    error_message = str(e).lower()
    keywords = [
        "content filter", 
        "content management policy", 
        "responsible ai", 
        "safety policy",
        "prompt blocked" # Common for Azure
    ]
    return any(keyword in error_message for keyword in keywords)

# Fetch Azure credentials from environment variables (loaded by core.py)
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") 
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

logger = logging.getLogger(__name__) 

class GraphState(TypedDict):
    embeddings: Union[None, LangchainEmbeddings] 
    vectorstore_path: str 
    vectorstore: Union[None, FAISS]
    llm_large: Union[None, AzureChatOpenAI] 
    retriever: Union[None, VectorStoreRetriever] 
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
    k_chunks_retriever: int
    combine_threshold: int


def instantiate_embeddings(state: GraphState) -> GraphState:
    t_node_start = time.perf_counter()
    node_name = "instantiate_embeddings_node"
    logging.info(f"--- Starting Node: {node_name} ---")
    if not state.get("embeddings"):
        logging.info("Instantiating embeddings model")
        try:
            new_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=embeddings_deployment, azure_endpoint=azure_endpoint,
                api_key=api_key, openai_api_version=openai_api_version
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_llm_node"
    logging.info(f"--- Starting Node: {node_name} (for query generation) ---")
    if not state.get("llm_large"):
        logging.info("Instantiating large LLM model (for query generation)")
        try:
            new_llm = AzureChatOpenAI(
                temperature=0, api_key=api_key, openai_api_version=openai_api_version,
                azure_deployment=deployment, azure_endpoint=azure_endpoint,
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

def load_faiss_vectorstore(state: GraphState) -> GraphState:
    t_node_start = time.perf_counter()
    node_name = "load_vectorstore_node"
    logging.info(f"--- Starting Node: {node_name} ---")
    current_vectorstore_path = state.get("vectorstore_path") 
    embeddings = state.get("embeddings")
    
    if "vectorstore" not in state: state["vectorstore"] = None 

    if state.get("vectorstore"): logging.info("Vectorstore already exists in state.")
    elif not current_vectorstore_path: logging.error("Vectorstore path not provided in state."); state["vectorstore"] = None
    elif not embeddings: logging.error("Embeddings not instantiated."); state["vectorstore"] = None
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
    t_node_start = time.perf_counter()
    node_name = "instantiate_retriever_node"
    logging.info(f"--- Starting Node: {node_name} ---")
    if "retriever" not in state: state["retriever"] = None
    vectorstore = state.get("vectorstore")
    k_retriever = state.get("k_chunks_retriever", K_CHUNKS_RETRIEVER_DEFAULT)

    if vectorstore is None: logging.error("Vectorstore not loaded."); state["retriever"] = None
    else:
        search_kwargs = {"k": k_retriever}
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
    vectorstore: FAISS, file_name: str, query_embeddings_list: List[List[float]],
    query_list_texts: List[str], k_per_query: int
) -> tuple[str, Optional[List[Document]]]:
    logging.info(f"Retrieving for file: {file_name} using {len(query_embeddings_list)} pre-computed query embeddings.")
    retrieved_docs: List[Document] = []
    try:
        for i, query_embedding in enumerate(query_embeddings_list):
            docs_for_embedding = await vectorstore.asimilarity_search_by_vector(
                embedding=query_embedding, k=k_per_query, filter={"file": file_name}
            )
            retrieved_docs.extend(docs_for_embedding)
        unique_docs_map: Dict[tuple, Document] = {}
        for doc in retrieved_docs:
            key = (doc.metadata.get("file"), doc.metadata.get("page"), doc.page_content.strip() if hasattr(doc, 'page_content') else "")
            if key not in unique_docs_map: unique_docs_map[key] = doc
        final_unique_docs = list(unique_docs_map.values())
        logging.info(f"[{file_name}] Retrieved {len(retrieved_docs)} raw -> {len(final_unique_docs)} unique docs.")
        return file_name, final_unique_docs
    except Exception as e_retrieve:
        logging.exception(f"[{file_name}] Error during similarity search by vector: {e_retrieve}")
        return file_name, None

async def extract_documents_parallel_node(state: GraphState) -> GraphState:
    t_node_start = time.perf_counter()
    node_name = "extract_documents_node"
    logging.info(f"--- Starting Node: {node_name} (Async) ---")
    question, llm, base_retriever, vectorstore, embeddings_model, allowed_files = (
        state.get("question"), state.get("llm_large"), state.get("retriever"),
        state.get("vectorstore"), state.get("embeddings"), state.get("allowed_files")
    )
    n_alternatives = state.get("n_alternatives", 4)
    k_per_query = state.get("k_per_query", state.get("k_chunks_retriever", K_CHUNKS_RETRIEVER_DEFAULT))
    current_documents_by_file: Dict[str, List[Document]] = {}

    if not question: logging.info(f"[{node_name}] No question. Skipping extraction."); return {**state, "documents_by_file": current_documents_by_file}
    if not allowed_files: logging.info(f"[{node_name}] No files selected. Skipping extraction."); return {**state, "documents_by_file": current_documents_by_file}
    if not all([llm, base_retriever, vectorstore, embeddings_model]):
        logging.error(f"[{node_name}] Missing components for extraction. Halting."); return {**state, "documents_by_file": current_documents_by_file}

    query_list: List[str] = [question]
    try:
        logging.info(f"[{node_name}] Generating alternative queries...")
        mqr_llm_chain = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm).llm_chain # type: ignore
        llm_response = await mqr_llm_chain.ainvoke({"question": question})
        raw_queries_text = ""
        if isinstance(llm_response, dict): raw_queries_text = str(llm_response.get(mqr_llm_chain.output_key, ""))
        elif isinstance(llm_response, str): raw_queries_text = llm_response
        elif isinstance(llm_response, list): raw_queries_text = "\n".join(str(item).strip() for item in llm_response if isinstance(item, str) and str(item).strip())
        else: raw_queries_text = str(llm_response)
        
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
        query_embeddings_list = await embeddings_model.aembed_documents(query_list) # type: ignore
    except Exception as e_embed: logging.exception(f"[{node_name}] Failed to embed queries: {e_embed}"); return {**state, "documents_by_file": current_documents_by_file}
    if not query_embeddings_list or len(query_embeddings_list) != len(query_list):
        logging.error(f"[{node_name}] Query embedding failed/mismatched."); return {**state, "documents_by_file": current_documents_by_file}

    tasks = [
        _async_retrieve_docs_with_embeddings_for_file(
            vectorstore, f_name, query_embeddings_list, query_list, k_per_query # type: ignore
        ) for f_name in allowed_files # type: ignore
    ]
    if tasks:
        results = await asyncio.gather(*tasks)
        for f_name, docs in results: current_documents_by_file[f_name] = docs if docs else []
    duration_node = time.perf_counter() - t_node_start
    logging.info(f"--- Node: {node_name} finished in {duration_node:.4f} seconds ---")
    return {**state, "documents_by_file": current_documents_by_file}

async def generate_individual_answers_node(state: GraphState) -> GraphState:
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
        llm = AzureChatOpenAI(temperature=0.1, api_key=api_key, openai_api_version=openai_api_version, azure_deployment="gpt-4o", azure_endpoint=azure_endpoint, max_tokens=2000)
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

def _format_conversation_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history: return "No previous conversation history."
    return "\n\n".join([f"User: {t.get('user_question', 'N/A')}\nAssistant: {t.get('assistant_response', 'N/A')}" for t in history])

async def _async_combine_answer_chunk(
    question: str, answer_chunk_input: Union[Dict[str, str], str], llm_combiner: BaseLanguageModel,
    combination_prompt_template: PromptTemplate, chunk_name: str, conversation_history_str: str,
    is_raw_chunk: bool
) -> str:
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
            "question": question, "formatted_answers_or_raw_docs": formatted_chunk_content_for_prompt,
            "files_no_info": no_info_placeholder, "files_errors": errors_placeholder,
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
    t_node_start = time.perf_counter()
    node_name = "combine_answers_node"
    logging.info(f"--- Starting Node: {node_name} (Async) ---")

    question, individual_answers, allowed_files, conversation_history, bypass_flag, raw_docs_for_synthesis = (
        state.get("question"), state.get("individual_answers"), state.get("allowed_files"),
        state.get("conversation_history"), state.get("bypass_individual_generation", False),
        state.get("raw_documents_for_synthesis")
    )
    combine_thresh = state.get("combine_threshold", COMBINE_THRESHOLD_DEFAULT)
    output_generation: Optional[str] = "Error during synthesis."
    state_to_return = {**state}

    if not allowed_files: output_generation = "Please select files to analyze."
    elif not question: output_generation = f"Files selected: {', '.join(allowed_files) if allowed_files else 'any'}. Ask a question."
    else:
        conversation_history_str = _format_conversation_history(conversation_history)
        llm_instance = AzureChatOpenAI(temperature=0.0, api_key=api_key, openai_api_version=openai_api_version, azure_deployment="gpt-4o", azure_endpoint=azure_endpoint)
        
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
        combo_prompt = PromptTemplate(template=active_prompt_text, input_variables=["question", "formatted_answers_or_raw_docs", "files_no_info", "files_errors", "conversation_history"])
        
        no_info_list: List[str] = []
        error_list: List[str] = []
        content_llm: str = ""

        if bypass_flag:
            logging.info(f"[{node_name}] Combining raw documents.")
            content_llm = raw_docs_for_synthesis if raw_docs_for_synthesis else "No raw documents."
            if not raw_docs_for_synthesis or "No documents retrieved" in raw_docs_for_synthesis or "No files selected" in raw_docs_for_synthesis:
                output_generation = raw_docs_for_synthesis
            else:
                docs_by_file = state.get("documents_by_file", {})
                if allowed_files:
                    for af in allowed_files:
                        if not docs_by_file.get(af): no_info_list.append(f"`{af}`")
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
                    if any(CONTENT_POLICY_MESSAGE in individual_answers.values()):
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


def create_graph_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("instantiate_embeddings_node", instantiate_embeddings)
    workflow.add_node("instantiate_llm_node", instantiate_llm_large)
    workflow.add_node("load_vectorstore_node", load_faiss_vectorstore)
    workflow.add_node("instantiate_retriever_node", instantiate_retriever)
    workflow.add_node("extract_documents_node", extract_documents_parallel_node)
    workflow.add_node("format_raw_documents_node", format_raw_documents_for_synthesis_node) 
    workflow.add_node("generate_answers_node", generate_individual_answers_node) 
    workflow.add_node("combine_answers_node", combine_answers_node) 

    workflow.set_entry_point("instantiate_embeddings_node")
    workflow.add_edge("instantiate_embeddings_node", "instantiate_llm_node")
    workflow.add_edge("instantiate_llm_node", "load_vectorstore_node")
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
