import streamlit as st
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

DEBUG_CAPTURE = [] # For capturing debug timing or messages


# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")  # collapse sidebar on load

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- Logging Setup ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Keep Langchain retriever logging concise if desired, or set to INFO for more detail
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


# --- Import Vector Store Functions ---
# Attempt to import functions from build_vectorstore.py
# Provide dummy functions if import fails to prevent NameErrors, but log the error.
try:
    from build_vectorstore import load_vectorstore, list_vectorstore_files
    vectorstore_functions_available = True
except ImportError:
    logging.error("Failed to import 'build_vectorstore'. Make sure 'build_vectorstore.py' exists and is accessible.")
    st.error("Critical component 'build_vectorstore.py' not found. File listing and potential loading operations will fail.", icon="🚨")
    # Define dummy functions
    def list_vectorstore_files(vs):
        logging.warning("Using dummy list_vectorstore_files function.")
        return []
    # Define load_vectorstore if it were used directly in this script
    # def load_vectorstore(path, embeddings):
    #     logging.warning("Using dummy load_vectorstore function.")
    #     return None
    vectorstore_functions_available = False


# --- Azure OpenAI Configuration ---
load_dotenv() # Load environment variables from .env file

k_chunks_retriever = 25 # Number of chunks MultiQueryRetriever aims for IN TOTAL


# Fetch Azure credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # Default model
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large") # Default embedding model
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # API version


# --- Logo Display ---
logo_col1, logo_col2, title_col, logo_col3 = st.columns([1, 1.4, 4, 1.4]) # Adjust column ratios if needed
logo_path1 = "./img/pnnl-logo.png"
logo_path2 = "./img/gdo-logo.png"
logo_path3 = "./img/wrr-pnnl-logo.svg"

# Helper function to display logos with error checking
def display_logo(column, path, alt_text):
    """Displays an image in a Streamlit column with existence check."""
    try:
        if os.path.exists(path):
            column.image(path, use_container_width=True)
        else:
            column.warning(f"Logo not found: {path}", icon="🖼️")
            column.caption(alt_text) # Show alt text if image missing
    except Exception as e:
        logging.error(f"Error loading logo ({alt_text}) from {path}: {e}")
        column.error(f"Error loading logo: {alt_text}", icon="🚨")

with logo_col1:
    display_logo(logo_col1, logo_path1, "PNNL Logo")
with logo_col2:
    display_logo(logo_col2, logo_path2, "GDO Logo")
with logo_col3:
    display_logo(logo_col3, logo_path3, "WRR Logo") # Corrected: Pass logo_col3

# Space between logos and title
st.markdown(
    "<div style='text-align:center;'></br></br></br></div>",
    unsafe_allow_html=True
)

# --- Title with SVG Icon ---
svg_path = "./img/projectLogo.svg"
try:
    # Check if SVG file exists before trying to read
    if os.path.exists(svg_path):
        with open(svg_path, "rb") as f:
            svg_base64 = base64.b64encode(f.read()).decode()
        # Center the title using markdown and HTML, adjust margin for alignment
        st.markdown(
            f"<div style='display:flex; align-items:center; justify-content:center; margin-top: -50px;'>" # Negative margin to pull title up
            f"<img src='data:image/svg+xml;base64,{svg_base64}' width='32' height='32' style='margin-right:8px;'/>"
            f"<h2 style='margin:0;'>Wildfire Mitigation Plans Database</h2>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        logging.warning(f"SVG logo not found at {svg_path}. Displaying text title only.")
        st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True) # Fallback title without SVG
except Exception as e:
    logging.error(f"Error loading SVG logo from {svg_path}: {e}")
    st.error(f"Error loading SVG logo: {e}")
    st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True) # Fallback title

# Subtitle for the chat feature
st.markdown(
    "<div style='text-align:center;'><h4>-- AI Assistant - Proof of Concept --</h4></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;'><em>Your AI Assistant to find answers across multiple reports!</em></div>",
    unsafe_allow_html=True
)


# --- Check for Azure Credentials ---
azure_creds_valid = True
if not api_key or not azure_endpoint:
    st.error("Azure OpenAI API key or endpoint not found. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.", icon="🚨")
    azure_creds_valid = False
    logging.error("Azure credentials (key or endpoint) missing.")


# --- Load Base Retriever and List Files ---
vectorstore_path = "test_faiss_store" # Path to your FAISS index directory
base_retriever = None
vectorstore = None
files_in_store = []
embeddings = None # Define embeddings in a broader scope

if azure_creds_valid and vectorstore_functions_available:
    try:
        logging.info("Initializing Azure OpenAI Embeddings...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )
        logging.info("Embeddings initialized.")

        # Check if the vectorstore path exists and is a directory
        if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
            logging.info(f"Loading FAISS vector store from: {vectorstore_path}")
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True # Be cautious with this setting
            )
            logging.info("FAISS vector store loaded.")

            # List files using the potentially imported function
            try:
                files_in_store = list_vectorstore_files(vectorstore)
                logging.info(f"Found {len(files_in_store)} files in vector store index: {files_in_store}")
            except Exception as e:
                 logging.error(f"Error calling list_vectorstore_files: {e}")
                 st.error(f"Error listing files from vector store: {e}")
                 files_in_store = []

            # Create retriever only if files were found
            if files_in_store:
                 base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks_retriever})
                #  st.sidebar.success(f"Loaded vector store with {len(files_in_store)} file(s).")
                 logging.info("Base retriever created.")
            else:
                 st.sidebar.warning(f"Vector store loaded, but no files were found within it (using list_vectorstore_files).")
                 logging.warning("No files found in vector store index. Retriever not created.")
                 # Keep azure_creds_valid = True, but base_retriever will be None

        else:
             st.sidebar.error(f"Vector store directory not found or is not a directory: '{vectorstore_path}'. Cannot load retriever.")
             logging.error(f"Vector store path not found or invalid: {vectorstore_path}")
             azure_creds_valid = False # Cannot proceed without a store

    except Exception as e:
        st.error(f"Error loading vectorstore '{vectorstore_path}': {e}")
        logging.exception(f"Failed to load vector store from {vectorstore_path}") # Log full traceback
        azure_creds_valid = False # Treat as invalid if store loading fails
elif not azure_creds_valid:
    st.sidebar.warning("Skipping vector store load due to missing Azure credentials.")
    logging.warning("Vector store loading skipped (missing Azure credentials).")
elif not vectorstore_functions_available:
     st.sidebar.error("Skipping vector store load because 'build_vectorstore.py' could not be imported.")
     logging.error("Vector store loading skipped (build_vectorstore import failed).")


# --- Constants for Predefined Questions ---
PREDEFINED_QUESTIONS = [
    "Enter my own question...",
    "Does the plan include vegetation management?",
    "Does the plan include undergrounding?",
    "Does the plan include Public Safety Power Shutoff (PSPS)?",
    "How frequently does the utility perform asset inspections?",
    "Are there generation considerations, such as derating solar PV during smoky conditions?"
]

# --- File Selector UI (Sidebar) ---
st.sidebar.subheader("Spaceholder for Filtering Sidebar")
st.sidebar.warning(
    "Content from your https://wildfire.pnnl.gov/mitigationPlans/pages/list data filtering sidebar will appear here."
    "Items selected in your filtering will be used for QA in the AI Assistant."
)

st.sidebar.subheader("Select Source File(s)")
if not files_in_store and azure_creds_valid and vectorstore_functions_available:
     # Only show this warning if creds are valid and store was loaded, but no files found
     st.sidebar.warning("No files found in the loaded vector store index.")
elif not azure_creds_valid:
     st.sidebar.info("File selection disabled (check Azure credentials).")
elif not vectorstore_functions_available:
     st.sidebar.info("File selection disabled (vector store functions unavailable).")


file_options = ["All"] + files_in_store
# Ensure default is valid even if files_in_store is empty
default_selection = ["All"] if files_in_store else []

# Determine if the multiselect should be disabled
multiselect_disabled = not (azure_creds_valid and base_retriever and files_in_store)

selected_files_user_choice = st.sidebar.multiselect(
    "Files to Query",
    file_options,
    default=default_selection,
    help="Select the specific files to search within, or 'All' to search across the entire loaded vector store.",
    disabled=multiselect_disabled
)

# Determine the actual list of files to process based on user selection
files_to_process_in_graph = []
if files_in_store: # Only determine if there are files available
    if "All" in selected_files_user_choice or not selected_files_user_choice:
        # If 'All' is selected OR if the user cleared the selection (and 'All' was an option)
        files_to_process_in_graph = files_in_store
        if not selected_files_user_choice and "All" in file_options: # Log if defaulting due to cleared selection
             logging.info("User cleared selection, defaulting to 'All' files.")
    else:
        # User selected specific files (and not 'All')
        files_to_process_in_graph = [f for f in selected_files_user_choice if f != "All"]

    # Display info/warning based on the final list
    if files_to_process_in_graph == files_in_store and "All" not in selected_files_user_choice:
         # This case happens if user manually selects all available files
         st.sidebar.info(f"Querying all {len(files_to_process_in_graph)} available file(s).")
         logging.info(f"User selected all available files: {files_to_process_in_graph}")
    elif files_to_process_in_graph:
         st.sidebar.info(f"Will query within: {', '.join(files_to_process_in_graph)}")
         logging.info(f"User selected specific files: {files_to_process_in_graph}")
    elif not files_to_process_in_graph and selected_files_user_choice:
         # This case means user selected only 'All' but 'All' wasn't in files_in_store (shouldn't happen)
         # OR user deselected everything
         st.sidebar.warning("No specific files selected. Please select files or 'All'.")
         logging.warning("File selection resulted in an empty list.")
         # Keep files_to_process_in_graph empty, subsequent steps should handle this

else:
    # No files in store to begin with
    st.sidebar.warning("No files available in the vector store to select.")
    logging.warning("No files available in vector store for selection.")
    files_to_process_in_graph = []



# --- Helper: Token‑count Logging ---
def log_token_count(llm, prompt_or_messages, description: str = ""):
    """
    Logs and captures (via DEBUG_CAPTURE) the token count for a prompt or list of
    chat messages. Tries llm.get_num_tokens first, then falls back to tiktoken.
    """
    token_count = None

    # 1) Preferred: use LangChain's built‑in counting (if available)
    try:
        if hasattr(llm, "get_num_tokens"):
            token_count = llm.get_num_tokens(prompt_or_messages)
    except Exception as e:
        logging.debug(f"get_num_tokens failed for '{description}': {e}")

    # 2) Fallback: use tiktoken directly
    if token_count is None:
        try:
            # Derive a reasonable model name for tiktoken; default to cl100k_base
            model_name = getattr(llm, "model_name", "gpt-4o")
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            if isinstance(prompt_or_messages, (list, tuple)):
                text_for_count = "".join(str(m) for m in prompt_or_messages)
            else:
                text_for_count = str(prompt_or_messages)

            token_count = len(enc.encode(text_for_count))
        except Exception as e:
            logging.debug(f"tiktoken counting failed for '{description}': {e}")
            token_count = -1  # Indicate failure to count

    logging.info(f"Token count ({description}): {token_count}")
    # Capture in DEBUG_CAPTURE for on‑screen debugging
    DEBUG_CAPTURE.append(f"Token count ({description}): {token_count}")

# --- Caching (Retriever Creation - Kept for potential future use) ---
# Note: Using st.cache_resource for things like models or retrievers is appropriate.
# Avoid calling st.write or other UI elements directly inside cached functions.
@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def get_base_retriever_from_pdf(_file_hash, uploaded_file_name): # Use underscore for unused hash
    """Processes a single PDF, creates embeddings, builds FAISS store, returns BASE retriever."""
    # This function assumes the file content is stored in session state, which can be fragile.
    # Consider passing file bytes directly if refactoring.
    logging.info(f"Cache miss or first run for get_base_retriever_from_pdf: {uploaded_file_name} (hash: {_file_hash})")

    if not api_key or not azure_endpoint:
        logging.error("Azure credentials not available for PDF processing inside cached function.")
        # Cannot use st.error here directly. Log and return None.
        return None

    # Attempt to retrieve file from session state
    uploaded_file = st.session_state.get(f"file_{_file_hash}", None)
    if not uploaded_file:
        # Fallback: Search session state by name (less reliable)
        for key, value in st.session_state.items():
            if hasattr(value, 'name') and value.name == uploaded_file_name:
                uploaded_file = value
                logging.warning(f"Retrieved file '{uploaded_file_name}' by name lookup in session state.")
                break
        if not uploaded_file:
             logging.error(f"Could not find file content for '{uploaded_file_name}' in session state (hash: {_file_hash}).")
             return None

    try:
        logging.info(f"Starting PDF processing for: {uploaded_file.name}")
        pages_data = extract_pages_from_pdf(uploaded_file)
        if not pages_data:
            # Error/warning logged within extract_pages_from_pdf
            return None

        logging.info(f"Splitting text for {uploaded_file.name}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
        all_docs = []
        for page_num, page_text, filename in pages_data:
            page_chunks = text_splitter.split_text(page_text)
            for chunk_index, chunk in enumerate(page_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"page": page_num, "file": filename, "chunk_in_page": chunk_index}
                )
                all_docs.append(doc)

        if not all_docs:
            logging.warning(f"Could not create any text chunks from '{uploaded_file.name}'.")
            return None
        logging.info(f"Split PDF '{uploaded_file.name}' into {len(all_docs)} chunks.")

        logging.info(f"Creating embeddings for '{uploaded_file.name}' chunks...")
        # Use the globally defined embeddings if available and valid
        current_embeddings = embeddings
        if not current_embeddings and azure_creds_valid:
             logging.warning("Re-initializing embeddings inside cached function (should ideally use global).")
             current_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=embeddings_deployment, azure_endpoint=azure_endpoint,
                api_key=api_key, openai_api_version=openai_api_version
            )
        elif not current_embeddings:
             logging.error("Embeddings model not available inside cached function.")
             return None

        logging.info(f"Building FAISS index for '{uploaded_file.name}'...")
        vectorstore_pdf = FAISS.from_documents(documents=all_docs, embedding=current_embeddings)
        logging.info(f"FAISS index built successfully for '{uploaded_file.name}'.")

        # Return the retriever for this specific PDF
        return vectorstore_pdf.as_retriever(search_kwargs={'k': k_chunks_retriever}) 

    except Exception as e:
        logging.exception(f"Error processing PDF '{uploaded_file_name}' in cached function")
        # Cannot use st.error here. Log and return None.
        return None


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    """
    Represents the state passed between nodes in the LangGraph workflow.

    Attributes:
        question: The user's current question.
        generation: The final combined LLM generation (answer).
        documents: List of documents deemed relevant AFTER grading.
        original_documents: List of documents initially retrieved by the retriever.
        filtered_documents: List of documents AFTER structural filtering but BEFORE relevance grading.
        relevance_grade: The 'yes'/'no'/'skipped'/'error' decision from the grading node.
        base_retriever: The FAISS retriever object used for document lookup.
        files_to_process: List of filenames selected by the user to query within.
        individual_answers: A dictionary mapping filename to the answer generated for that specific file.
    """
    question: str
    generation: Optional[str]
    documents: List[Document] # Post-grading relevant docs
    original_documents: List[Document] # Post-retrieval docs
    filtered_documents: Optional[List[Document]] # Post-filtering docs
    relevance_grade: Optional[str] # Grading decision
    base_retriever: object # Should be FAISS retriever instance
    files_to_process: List[str]
    individual_answers: Dict[str, str]


# --- LangGraph Nodes ---
# Use logging within nodes for internal state/progress messages.
# UI updates (like st.status) should happen in the main script logic where the graph is invoked.

# Node 1: Retrieve documents using MultiQueryRetriever
def retrieve_docs_multi_query(state: GraphState) -> GraphState:
    """Retrieves documents using MultiQueryRetriever and filters by selected files."""

    logging.info("--- Starting Node: retrieve_docs_multi_query ---")

    question = state["question"]
    base_retriever = state["base_retriever"]
    files_to_process = state["files_to_process"]

    # Initialize return keys to ensure they exist in the output state
    output_state = {
        "original_documents": [],
        "filtered_documents": [], # Will be populated by the next node
        "documents": [],         # Will be populated by the grading node
        "relevance_grade": None  # Will be populated by the grading node
    }

    if not base_retriever:
        logging.error("Base retriever is None in retrieve_docs_multi_query. Cannot retrieve.")
        return {**state, **output_state} # Return current state merged with empty outputs

    logging.info(f"Processing question: '{question}' for files: {files_to_process}")

    # Handle case where Azure credentials might be missing (fallback or skip)
    if not azure_creds_valid:
         logging.warning("Azure credentials missing. Falling back to simple retrieval.")
         try:
             all_docs = base_retriever.get_relevant_documents(question)

             # Filter AFTER simple retrieval
             filtered_docs = [doc for doc in all_docs if doc.metadata.get("file") in files_to_process]
             logging.info(f"Retrieved {len(filtered_docs)} documents via simple fallback for files: {files_to_process}.")

             output_state["original_documents"] = filtered_docs
             output_state["documents"] = output_state["original_documents"]

         except Exception as e:
             logging.error(f"Error during simple fallback retrieval: {e}")

         return {**state, **output_state}

    # Proceed with MultiQueryRetriever if credentials are valid
    try:
        logging.info(f"Initializing MultiQueryRetriever for question: '{question}'")

        llm_for_queries = AzureChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=llm_for_queries
        )

        # --- Capture and log the alternative queries generated by MultiQueryRetriever ---
        alt_queries = []
        # Newer versions expose .generate_queries(); fall back to the protected method otherwise
        if hasattr(multi_query_retriever, "generate_queries"):
            try:
                alt_queries = multi_query_retriever.generate_queries(question)
            except Exception as e:
                logging.debug(f"generate_queries failed: {e}")
        elif hasattr(multi_query_retriever, "_generate_queries"):
            try:
                alt_queries = multi_query_retriever._generate_queries(question)
            except Exception as e:
                logging.debug(f"_generate_queries failed: {e}")

        if alt_queries:
            logging.info(f"MultiQueryRetriever generated {len(alt_queries)} alternative queries: {alt_queries}")
            DEBUG_CAPTURE.append(
                "MultiQueryRetriever alternative queries:\n" + "\n".join(f"- {q}" for q in alt_queries)
            )

        logging.info(f"Invoking MultiQueryRetriever...")

        # Retrieve documents based on multiple generated queries
        unique_docs_unfiltered = multi_query_retriever.invoke(input=question)

        logging.info(f"MultiQueryRetriever returned {len(unique_docs_unfiltered)} total unique chunks before file filtering.")

        # Filter the retrieved documents based on the user's file selection
        unique_docs_filtered = [
            doc for doc in unique_docs_unfiltered
            if doc.metadata.get("file") in files_to_process
        ]

        # logging.info(f"Filtered down to {len(unique_docs_filtered)} chunks based on selected file(s): {files_to_process}")

        output_state["original_documents"] = unique_docs_filtered
        output_state["documents"] = output_state["original_documents"]

    except Exception as e:
        logging.exception("Error during multi-query retrieval") # Log full traceback

    finally:
        logging.info("--- Finished Node: retrieve_docs_multi_query ---")
        return {**state, **output_state} # Merge output with input state



# Node: Generate Answer for each file (using relevant documents)
async def generate_individual_answers(state: GraphState) -> GraphState:

    t_start_node = time.time()

    logging.info("--- Starting Node: generate_individual_answers (Async) ---")
    question_async = state["question"] # Renamed
    relevant_documents_async = state["documents"] # Renamed
    files_to_process_async = state["files_to_process"] # Renamed

    individual_answers_async = { # Renamed
        filename: f"No relevant information found in the file '{filename}' to answer the question based on the retrieved, filtered, and graded context."
        for filename in files_to_process_async
    }

    if not relevant_documents_async:
        logging.warning("Generate Individual Answers (Async) called with no relevant documents. Default 'no info' messages will be used.")
        return {**state, "individual_answers": individual_answers_async, "generation": None}
    logging.info(f"Generating answers (async) based on {len(relevant_documents_async)} relevant documents across selected files.")

    if not azure_creds_valid: # Global check
        logging.warning("Skipping answer generation (async) due to missing Azure credentials.")
        for doc_file_had_relevant_docs in {doc.metadata.get("file") for doc in relevant_documents_async if doc.metadata.get("file")}:
            if doc_file_had_relevant_docs in individual_answers_async:
                 individual_answers_async[doc_file_had_relevant_docs] = "Could not generate answer due to missing credentials."
        return {**state, "individual_answers": individual_answers_async, "generation": None}

    docs_by_file_async = defaultdict(list) # Renamed
    for doc_loop_async in relevant_documents_async: # Renamed
        filename_meta_async = doc_loop_async.metadata.get("file") # Renamed
        if filename_meta_async and filename_meta_async in files_to_process_async:
            docs_by_file_async[filename_meta_async].append(doc_loop_async)

    if not docs_by_file_async:
        logging.warning(f"No relevant documents found for ANY of the selected files: {files_to_process_async} after filtering (Async). Default 'no info' messages will be used.")
        return {**state, "individual_answers": individual_answers_async, "generation": None}
    logging.info(f"Found relevant documents spread across {len(docs_by_file_async)} file(s) for async processing: {list(docs_by_file_async.keys())}")

    prompt_template_single_file_text_async = """You are an expert assistant analyzing a technical report. Your task is to answer the user's question comprehensively based *only* on the provided context chunks from a SINGLE FILE, which have been filtered and graded for relevance.

Follow these instructions carefully:
1.  Thoroughly read all provided context chunks from this file, each marked with '--- Context from Page X ---'. Pay attention to the page numbers. Synthesize information found across multiple chunks *within this file* if they relate to the same aspect of the question.
2.  Answer the user's question directly based *only* on the information present in the context from this file.
3.  Identify the most relevant information or statement(s) within the context that directly address the question.
4.  **Contextualized Quoting:** When presenting this key information, include a direct quote (enclosed in **double quotation marks**) of the most relevant sentence or phrase. **Crucially, also explain the surrounding context from the *same chunk* to clarify the quote's meaning or provide necessary background.** For example, instead of just citing, you might say: `The report discusses mitigation strategies, stating, "direct quote text..." ({filename}, Page X), which is part of a larger section detailing preventative measures.` Cite the **file name and page number in parentheses** immediately after the closing quotation mark, like this: `"direct quote text..." ({filename}, Page X)`. Use the exact filename provided.
5.  If the question asks about specific details (like financial allocations, frequencies, specific procedures) and that detail is *not* found in this file's context, explicitly state that the information is not provided *in this specific file*. Do not make assumptions or provide external knowledge.
6.  **Structure for Clarity:** Structure your answer logically for this file. Start with a direct summary answer if possible based on this file. Then, present the supporting details using the contextualized quoting method described above. Ensure the explanation connects the quote and its context clearly back to the user's original question. Conclude by addressing any parts of the question that couldn't be answered from this file's context. If no relevant information is found in the provided context *from this file* to answer the question, state that clearly and concisely.

Context from Document Chunks (File: {filename}):
{context}

Question: {question}

Detailed Answer based ONLY on File '{filename}' (with Contextualized Quotes and Citations in the format "quote..." ({filename}, Page X)):"""
    prompt_single_file_async = PromptTemplate( # Renamed
        template=prompt_template_single_file_text_async,
        input_variables=["context", "question", "filename"]
    )
    llm_generate_async = AzureChatOpenAI( # Renamed
        temperature=0.1, 
        api_key=api_key, 
        openai_api_version=openai_api_version,
        azure_deployment="gpt-4o", 
        azure_endpoint=azure_endpoint, 
        max_tokens=2000
    )
    rag_chain_single_file_async = prompt_single_file_async | llm_generate_async | StrOutputParser() # Renamed

    async def _process_file_async_inner(filename_task: str, file_docs_list_task: List[Document]): # Renamed inner function
        logging.info(f"--- Starting async processing for file (inner): {filename_task} ---")
        t_file_start = time.time()

        context_inner = "\n\n".join([f"--- Context from Page {d.metadata.get('page', 'N/A')} ---\n{d.page_content}" for d in file_docs_list_task]) # Renamed
        logging.info(f"Using {len(file_docs_list_task)} relevant chunks from '{filename_task}' for async generation (inner).")
        try:
            generation_inner = await rag_chain_single_file_async.ainvoke({ # Renamed
                "context": context_inner, "question": question_async, "filename": filename_task
            })
            logging.info(f"Successfully generated answer for {filename_task} (async inner). Took {time.time() - t_file_start:.2f}s.")
            return filename_task, generation_inner
        except Exception as e:
            logging.exception(f"Error generating answer for file {filename_task} (async inner)")
            return filename_task, f"An error occurred while generating the answer for file '{filename_task}': {e}"

    tasks_async = [] # Renamed
    for filename_loop_async, file_docs_loop_async in docs_by_file_async.items(): # Renamed
        if file_docs_loop_async:
            tasks_async.append(_process_file_async_inner(filename_loop_async, file_docs_loop_async))
        else:
            logging.warning(f"Skipping task creation for file '{filename_loop_async}' as it has no relevant documents after grouping (Async).")

    if tasks_async:
        results_async = await asyncio.gather(*tasks_async, return_exceptions=True) # Renamed
        for result_item in results_async: # Renamed
            if isinstance(result_item, Exception):
                logging.error(f"A task failed with an unhandled exception during asyncio.gather (Async): {result_item}")
            elif isinstance(result_item, tuple) and len(result_item) == 2:
                fn_res, answer_or_error_res = result_item # Renamed
                individual_answers_async[fn_res] = answer_or_error_res
                if "An error occurred" in answer_or_error_res:
                    logging.warning(f"Error message stored for {fn_res} (Async): {answer_or_error_res}")
                else:
                    logging.info(f"Successfully stored answer for {fn_res} (Async)")
            else:
                logging.error(f"Unexpected result type from asyncio.gather task (Async): {result_item}")
    else:
        logging.info("No tasks were created for async generation (e.g., no files had relevant documents) (Async).")

    logging.info("--- Finished Node: generate_individual_answers (Async) ---")
    
    DEBUG_CAPTURE.append(f"Generate individual PDF responses (Async): {round(time.time() - t_start_node, 4)} seconds")


    return {**state, "individual_answers": individual_answers_async, "generation": None}


# Node 5: Combine individual answers into a final response
def combine_answers(state: GraphState) -> GraphState:
    """Combines the individual answers into a single, synthesized response."""

    t0 = time.time()


    logging.info("--- Starting Node: combine_answers ---")
    question = state["question"]
    individual_answers = state["individual_answers"]
    files_processed_count = len(individual_answers) # Count how many files we have *any* result for

    # Initialize generation to None
    output_state = {"generation": None}

    if not individual_answers:
        logging.warning("Combine Answers node called with no individual answers dictionary. This indicates a potential graph logic error.")
        output_state["generation"] = "An internal error occurred: No individual file answers were available to combine."
        return {**state, **output_state}

    # Filter out answers that indicate errors or no relevant info found, keep track of them
    answers_to_combine = {}
    files_with_no_info = []
    files_with_errors = []
    for filename, answer in individual_answers.items():
        if "An error occurred while generating" in answer or "Could not generate answer due to missing credentials" in answer:
            files_with_errors.append(f"`{filename}`")
        elif "No relevant information found" in answer:
            files_with_no_info.append(f"`{filename}`")
        else:
            # Assume this answer has substantive content to combine
            answers_to_combine[filename] = answer

    # If NO substantive answers remain after filtering, create a summary message
    if not answers_to_combine:
        logging.info("No substantive individual answers available to combine after filtering out 'no info'/'error' messages.")
        final_msg = f"Could not find relevant information to answer the question in the analyzed sections of the {files_processed_count} selected file(s)."
        if files_with_no_info:
            final_msg += f"\nFiles checked with no relevant info found: {', '.join(files_with_no_info)}."
        if files_with_errors:
            final_msg += f"\nErrors occurred during processing for files: {', '.join(files_with_errors)}."
        output_state["generation"] = final_msg
        logging.info("--- Finished Node: combine_answers (No substantive content) ---")
        return {**state, **output_state}

    # Proceed with combination if there are substantive answers
    logging.info(f"Combining {len(answers_to_combine)} substantive answers from files: {list(answers_to_combine.keys())}")

    if not azure_creds_valid:
         logging.warning("Skipping answer combination due to missing Azure credentials.")
         # Simple concatenation as fallback
         combined = f"Could not properly combine answers due to missing credentials. Individual findings:\n\n"
         for filename, answer in answers_to_combine.items(): # Use the filtered list
              combined += f"--- Findings from {filename} ---\n{answer}\n\n"
         # Add notes about other files
         if files_with_no_info: combined += f"Files checked with no relevant info: {', '.join(files_with_no_info)}.\n"
         if files_with_errors: combined += f"Errors occurred for files: {', '.join(files_with_errors)}.\n"
         output_state["generation"] = combined.strip()
         return {**state, **output_state}

    # Format the substantive answers for the combination prompt
    formatted_answers = ""
    for filename, answer in answers_to_combine.items():
        formatted_answers += f"--- Answer based on file: {filename} ---\n{answer}\n\n"

    # Define the prompt for combining answers
    # This prompt emphasizes preserving detail, attribution, and handling missing info.
    prompt_template_combine = """You are an expert synthesis assistant. Your task is to combine multiple answers, each generated from a different source file in response to the same user question. Create a single, comprehensive, and well-structured response.

Follow these instructions VERY carefully:
1.  **Goal:** Synthesize the information from all provided file-specific answers into ONE cohesive response to the original user question.
2.  **Preserve ALL Details:** Do NOT summarize or omit any specific facts, figures, quotes, or findings mentioned in the individual answers. Ensure the final answer is as detailed as the sum of the individual answers. Pay attention to the specific citation format used in the individual answers (e.g., "quote..." (filename.pdf, Page X)) and **retain this exact format** in the final synthesized answer.
3.  **Attribute Clearly:** Explicitly mention the source file(s) for each piece of information or finding. Use parenthetical citations like `(Source: filename.pdf)` or integrate attribution naturally, e.g., `File 'report_A.pdf' states that... while 'report_B.pdf' adds...`. When incorporating direct quotes from the individual answers, **ensure their original citation (including filename and page) is kept intact** immediately following the quote.
4.  **Structure Logically:** Organize the combined answer logically based on the user's question. If the question has multiple parts, address each part, synthesizing information from relevant files for that part. Use headings or bullet points if it improves clarity. Start with a direct summary if possible, then elaborate with details from each file.
5.  **Handle Contradictions/Nuances:** If different files provide conflicting or slightly different information, present both findings and attribute them clearly (e.g., `File A reports X (FileA.pdf, Page Y), whereas File B reports Z (FileB.pdf, Page W).`). Do not try to resolve conflicts unless the context explicitly allows it.
6.  **Acknowledge Missing Info:** After presenting the synthesized information, add a concluding sentence or section summarizing which of the originally processed files did not contain relevant information or encountered errors, based on the context provided below (if any). For example: "Information on [specific topic] was not found in File C.pdf or File D.pdf." or "Processing errors occurred for File E.pdf."
7.  **Introduction and Conclusion:** Start with a brief introductory sentence acknowledging the question and the sources that provided information. End with a concise summary if appropriate, reiterating the main findings across the files.

User's Original Question:
{question}

Individual Answers Generated from Different Files (Note the citation format used within):
{formatted_answers}

Files Processed with No Relevant Info Found: {files_no_info}
Files Processed with Errors: {files_errors}

Synthesized Comprehensive Answer (Preserving all details, attributing sources, retaining original quote citations, and noting files without info/errors):"""
    prompt_combine = PromptTemplate(
        template=prompt_template_combine,
        input_variables=["question", "formatted_answers", "files_no_info", "files_errors"]
    )

    DEBUG_CAPTURE.append(f"Combine responses into answer - prep: {round(time.time() - t0, 4)} seconds")


    try:
        logging.info(f"Invoking combination LLM to synthesize answers...")

        t0 = time.time()

        # Instantiate LLM for combination
        llm_combine = AzureChatOpenAI(
            temperature=0.0, # Use low temperature for faithful combination
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment="gpt-4o",  # deployment
            azure_endpoint=azure_endpoint,
            # max_tokens=2000 # Increase tokens if combining many long answers
        )

        # Create the combination chain
        combination_chain = prompt_combine | llm_combine | StrOutputParser()

        # Prepare input for the combination prompt
        combine_input = {
            "question": question,
            "formatted_answers": formatted_answers.strip(),
            "files_no_info": ", ".join(files_with_no_info) if files_with_no_info else "None",
            "files_errors": ", ".join(files_with_errors) if files_with_errors else "None"
        }

        # Invoke the chain
        final_generation = combination_chain.invoke(combine_input)
        logging.info("Combination LLM finished. Final answer generated.")
        output_state["generation"] = final_generation

    except Exception as e:
        logging.exception("Error combining individual answers")
        # Fallback: concatenate if combination fails, include info/error files
        combined = f"An error occurred during answer synthesis. Raw findings:\n\n"
        for filename, answer in answers_to_combine.items():
             combined += f"--- Findings from {filename} ---\n{answer}\n\n"
        if files_with_no_info: combined += f"\nFiles checked with no relevant info: {', '.join(files_with_no_info)}."
        if files_with_errors: combined += f"\nErrors occurred for files: {', '.join(files_with_errors)}."
        output_state["generation"] = combined.strip()
    finally:
        logging.info("--- Finished Node: combine_answers ---")

        DEBUG_CAPTURE.append(f"Combine responses into answer - LLM: {round(time.time() - t0, 4)} seconds")

        return {**state, **output_state}


# --- LangGraph Conditional Edge ---
def decide_to_generate(state: GraphState) -> str:
    """Determines whether to proceed to generation based on relevance grading."""
    logging.info("--- Decision Node: decide_to_generate ---") # Decision Node
    relevance_grade = state.get("relevance_grade", "no") # Default to 'no' if key missing
    documents_found = state.get("documents", []) # Check the final list post-grading

    # Decide based primarily on the grade, but double-check documents list
    if relevance_grade == "yes" and documents_found:
        logging.info("Decision: Relevant documents found (Grade: YES). Proceeding to generate individual answers.")
        return "generate_individual" # Route to the node that generates answers per file
    elif relevance_grade == "skipped" and documents_found:
         logging.warning("Decision: Grading was skipped, but documents exist. Proceeding to generate individual answers (results may be less relevant).")
         return "generate_individual"
    else:
        # This covers grade 'no', 'error', 'error_upstream', or 'yes' but empty docs list
        logging.info(f"Decision: Not generating individual answers (Grade: {relevance_grade}, Relevant Docs Count: {len(documents_found)}). Ending workflow here.")
        return "end_no_relevance" # End the process


# --- Build LangGraph Workflow ---
langgraph_app = None
if azure_creds_valid and base_retriever: # Need retriever for the graph to be useful
    try:
        logging.info("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)

        # Add nodes to the graph
        workflow.add_node("retrieve", retrieve_docs_multi_query)
        workflow.add_node("generate_individual", generate_individual_answers)
        workflow.add_node("combine_answers", combine_answers)

        # Define the graph's flow (edges)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate_individual")

        # After generating individual answers, combine them
        workflow.add_edge("generate_individual", "combine_answers")

        # The final combined answer marks the end of the successful path
        workflow.add_edge("combine_answers", END)

        # Compile the graph
        langgraph_app = workflow.compile()
        # st.sidebar.success("Multi-file RAG workflow compiled.")
        logging.info("LangGraph workflow compiled successfully.")

    except Exception as e:
        st.sidebar.error(f"Failed to compile LangGraph: {e}")
        logging.exception("LangGraph compilation failed.")
        langgraph_app = None # Ensure it's None if compilation fails
elif not azure_creds_valid:
    st.sidebar.error("LangGraph compilation skipped: Missing Azure credentials.")
    logging.error("LangGraph compilation skipped (missing Azure credentials).")
elif not base_retriever:
     st.sidebar.error("LangGraph compilation skipped: Vector store not loaded or empty.")
     logging.error("LangGraph compilation skipped (base retriever not available).")


# --- Streamlit UI ---

# --- Determine if inputs should be disabled ---
# This depends on credentials, retriever, graph, and file selection
input_disabled = not (azure_creds_valid and base_retriever and langgraph_app and files_to_process_in_graph)
disabled_reason = ""
if not azure_creds_valid: disabled_reason += "Azure credentials missing. "
if not base_retriever: disabled_reason += "Vector store not loaded/empty. "
if not files_to_process_in_graph: disabled_reason += "No files selected/available in sidebar. "
if not langgraph_app: disabled_reason += "RAG workflow failed. "
disabled_reason = disabled_reason.strip()

# --- Initial Question Input Area (Main Area) ---
# Place this section before the chat history display
st.subheader("Ask Initial Question")
q_col1, q_col2 = st.columns([3, 1]) # Adjust ratio as needed

with q_col1:
    selected_question = st.selectbox(
        "Select a commonly asked question or customize your own:",
        PREDEFINED_QUESTIONS,
        index=0, # Default to "Enter my own question..."
        key="question_select",
        disabled=input_disabled,
        help=disabled_reason or "Select a predefined question or choose the first option to type your own below."
    )
    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0] or input_disabled
    custom_question = st.text_input(
        "Enter your custom question here...",
        key="custom_question_input",
        disabled=custom_question_disabled,
        placeholder="Type your question here"
    )

with q_col2:
    st.markdown("<br/>", unsafe_allow_html=True) # Add space for alignment
    ask_button = st.button(
        "Start Chat", # Changed button text slightly
        type="primary",
        use_container_width=True,
        disabled=input_disabled,
        key="start_chat_button",
        help=disabled_reason or "Click to start the chat with the selected or entered question."
    )

# Determine the final question to ask based on user input
initial_question_to_ask = ""
if selected_question == PREDEFINED_QUESTIONS[0]:
    initial_question_to_ask = custom_question.strip() if custom_question else ""
else:
    initial_question_to_ask = selected_question  # Use predefined question

st.divider() # Add a divider between input and chat history

# --- Initialize Chat History ---
# Use st.session_state to keep track of messages across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Initialized chat history in session state.")
# Show chat history and follow‑up input only after first assistant reply
show_chat_history = any(m.get("role") == "assistant" for m in st.session_state.messages)


# --- Display Existing Chat Messages ---
# Iterate through the stored messages and display them
# if show_chat_history:
st.subheader("Explore your Findings")

# Show a waiting message until the first assistant response appears
if not show_chat_history:
    st.info("Awaiting Response")


chat_container = st.container() # Use a container for chat messages

with chat_container:
    if not show_chat_history:
         # Display initial message within the chat area if history is empty
         if not input_disabled:
            #  st.info("👆 Start the conversation by asking an initial question above.")
             pass
         else:
             # Provide guidance if the app is disabled
             st.warning(f"Application is not ready. Please check the following: {disabled_reason}", icon="⚠️")
             logging.warning(f"App not ready on initial load. Reason: {disabled_reason}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display the main text content
            st.markdown(message["content"])

            # --- Display Debug Info, Context, and Individual Answers within Assistant Messages ---
            if message["role"] == "assistant":
                # Expander for Debugging Info
                if "debug_info" in message and message["debug_info"]:
                    with st.expander("Show Debugging Info"):
                        debug_info = message.get("debug_info", {})
                        orig_cnt = debug_info.get("original_doc_count", "N/A")
                        filt_cnt = debug_info.get("filtered_doc_count", "N/A")
                        rel_grade = str(debug_info.get("relevance_grade", "N/A")).upper()
                        st.markdown(f"**Initial Retrieval:** {orig_cnt} chunks found.")
                        st.markdown(f"**After Filtering:** {filt_cnt} chunks remained.")
                        st.markdown(f"**Relevance Grade:** {rel_grade}")

                # Expander for Initially Retrieved Context
                if "original_docs" in message and message["original_docs"]:
                     with st.expander(f"Show Initially Retrieved Context ({len(message['original_docs'])} chunks)"):
                          for i, doc in enumerate(message["original_docs"]):
                               page_num = doc.metadata.get('page', 'N/A')
                               file_name = doc.metadata.get("file", "N/A")
                               st.markdown(f"**Chunk {i+1} (`{file_name}` - Page {page_num}):**")
                               st.markdown(f"> {doc.page_content}") # Use blockquote
                               st.divider()

                # Expander for Individual File Answers (if generated)
                if "individual_answers" in message and message["individual_answers"]:
                     substantive_answers = {
                         fname: ans for fname, ans in message["individual_answers"].items()
                         if "No relevant information found" not in ans and "An error occurred" not in ans and "Could not generate" not in ans
                     }
                     expander_title = f"Show Individual Answers per File ({len(substantive_answers)} file(s) with substantive answers)"
                     if len(substantive_answers) < len(message["individual_answers"]):
                          expander_title += f" / {len(message['individual_answers'])} total processed"

                     with st.expander(expander_title):
                          for filename, answer in message["individual_answers"].items():
                              st.markdown(f"**Answer from `{filename}`:**")
                              st.markdown(answer)
                              st.divider()


# --- Function to Run Graph and Handle Response ---
# (This function remains unchanged from the previous version)
def run_graph_and_get_response(question_to_ask: str) -> Optional[Dict]:
    """
    Invokes the LangGraph workflow with the given question and returns results.

    Args:
        question_to_ask: The user's question.

    Returns:
        A dictionary containing the response components ('content', 'original_docs',
        'individual_answers', 'debug_info') or None if the graph fails to run.
    """
    if not langgraph_app or not base_retriever or not files_to_process_in_graph:
        # Display error in the main chat area if prerequisites fail during execution
        st.chat_message("assistant").error("Cannot process request. Workflow, retriever, or file selection is not ready.")
        logging.error("Attempted to run graph but prerequisites not met.")
        return None

    final_state = None
    response_data = {
        "content": "An unexpected error occurred.",
        "original_docs": [],
        "individual_answers": {},
        "debug_info": {}
    }

    try:
        # Use st.status for a cleaner progress indication during graph execution
        status_message = f"Analyzing '{question_to_ask[:50]}...' across {len(files_to_process_in_graph)} file(s)"
        with st.status(status_message, expanded=True) as status: # Keep expanded for visibility

            # Define the initial state for this graph run
            initial_state = {
                "question": question_to_ask,
                "base_retriever": base_retriever,
                "files_to_process": files_to_process_in_graph,
                # Initialize other keys expected by the graph state
                "generation": None,
                "documents": [],
                "original_documents": [],
                "filtered_documents": [],
                "relevance_grade": None,
                "individual_answers": {}
            }
            logging.info(f"Invoking LangGraph app with initial state for question: '{question_to_ask}'")
            # status.write("Invoking RAG workflow...") # Initial status message

            # --- Graph Invocation ---
            # Note: Nodes themselves use logging. Status updates happen here based on final state.
            # Use asynchronous invocation because the workflow contains async nodes
            final_state = asyncio.run(langgraph_app.ainvoke(initial_state))
            # -----------------------

            logging.info("LangGraph app invocation finished.")
            status.write("Processing results...") # Update status

            # Process the final state to populate the response
            final_answer = final_state.get("generation", None)
            original_docs = final_state.get('original_documents', [])
            filtered_docs = final_state.get('filtered_documents', []) # Get filtered docs
            relevance_grade = final_state.get('relevance_grade', 'N/A') # Get grade
            individual_answers = final_state.get('individual_answers', {})

            # Populate debug info
            response_data["debug_info"] = {
                "original_doc_count": len(original_docs) if original_docs else 0,
                "filtered_doc_count": len(filtered_docs) if filtered_docs else 0,
                "relevance_grade": str(relevance_grade) # Ensure string
            }

            # Populate other response parts
            response_data["original_docs"] = original_docs
            response_data["individual_answers"] = individual_answers

            # Determine final content message
            if final_answer:
                response_data["content"] = final_answer
                status.update(label="Analysis complete!", state="complete", expanded=False)
                logging.info("Graph execution successful, final answer generated.")
            elif relevance_grade == "no" or (relevance_grade == "yes" and not final_state.get("documents")):
                 no_info_message = "Could not find relevant information across the selected documents to answer the question after filtering and grading."
                 potential_combine_message = final_state.get("generation")
                 if potential_combine_message and ("Could not find relevant information" in potential_combine_message or "Errors occurred" in potential_combine_message):
                      response_data["content"] = potential_combine_message
                 else:
                      response_data["content"] = no_info_message
                 status.update(label="No relevant information found.", state="complete", expanded=False)
                 logging.info("Graph execution finished, but no relevant information found after grading.")
            else:
                response_data["content"] = final_answer or "An issue occurred during the final answer generation."
                status.update(label="Analysis finished with issues.", state="warning", expanded=False)
                logging.warning(f"Graph execution finished, but final answer might be missing or incomplete. Grade: {relevance_grade}")

    except Exception as e:
        logging.exception("Error running LangGraph workflow") # Log full traceback
        # Display error in the main chat area
        st.error(f"An error occurred during the workflow execution: {e}", icon="❌")
        response_data["content"] = f"An error occurred during processing: {e}"
        # Populate partial debug info if possible
        if final_state:
             response_data["debug_info"] = {
                "original_doc_count": len(final_state.get('original_documents', [])),
                "filtered_doc_count": len(final_state.get('filtered_documents', [])),
                "relevance_grade": str(final_state.get('relevance_grade', 'Error'))
             }
             response_data["original_docs"] = final_state.get('original_documents', [])
             response_data["individual_answers"] = final_state.get('individual_answers', {})
        # Update status if it exists
        if 'status' in locals():
             status.update(label="Workflow Error", state="error", expanded=False)
        return response_data # Return partial data even on error

    return response_data


# --- Handle Initial Question Submission (Button Click) ---
# This logic now refers to the button and inputs in the main area
if ask_button and initial_question_to_ask:
    logging.info(f"Initial question submitted via button: '{initial_question_to_ask}'")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": initial_question_to_ask})
    # Display user message immediately in the chat
    # Use the chat container defined earlier
    with chat_container:
        with st.chat_message("user"):
            st.markdown(initial_question_to_ask)

        # Get assistant response by running the graph
        with st.chat_message("assistant"):
            # Use a placeholder while running
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            response_data = run_graph_and_get_response(initial_question_to_ask)
            placeholder.empty() # Clear the placeholder

            if response_data:
                # Display main content
                st.markdown(response_data["content"])

                # Display Debug Info Expander
                with st.expander("Show Debugging Info"):
                    debug_info = response_data.get("debug_info", {})
                    orig_cnt = debug_info.get("original_doc_count", "N/A")
                    filt_cnt = debug_info.get("filtered_doc_count", "N/A")
                    rel_grade = str(debug_info.get("relevance_grade", "N/A")).upper()
                    st.markdown(f"**Initial Retrieval:** {orig_cnt} chunks found.")
                    st.markdown(f"**After Filtering:** {filt_cnt} chunks remained.")
                    st.markdown(f"**Relevance Grade:** {rel_grade}")

                # Display Initial Context Expander
                if response_data["original_docs"]:
                     with st.expander(f"Show Initially Retrieved Context ({len(response_data['original_docs'])} chunks)"):
                          for i, doc in enumerate(response_data["original_docs"]):
                               page_num = doc.metadata.get('page', 'N/A')
                               file_name = doc.metadata.get("file", "N/A")
                               st.markdown(f"**Chunk {i+1} (`{file_name}` - Page {page_num}):**")
                               st.markdown(f"> {doc.page_content}")
                               st.divider()

                # Display Individual Answers Expander
                if response_data["individual_answers"]:
                     substantive_answers = {fname: ans for fname, ans in response_data["individual_answers"].items() if "No relevant information found" not in ans and "An error occurred" not in ans and "Could not generate" not in ans}
                     expander_title = f"Show Individual Answers per File ({len(substantive_answers)} file(s) with substantive answers / {len(response_data['individual_answers'])} total processed)"
                     with st.expander(expander_title):
                          for filename, answer in response_data["individual_answers"].items():
                              st.markdown(f"**Answer from `{filename}`:**")
                              st.markdown(answer)
                              st.divider()

                # Add the complete assistant response (including debug/context data) to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["content"],
                    "original_docs": response_data["original_docs"],
                    "individual_answers": response_data["individual_answers"],
                    "debug_info": response_data["debug_info"] # Store debug info
                })
            else:
                # Handle case where run_graph_and_get_response returned None (major failure)
                error_message = "Sorry, I encountered a critical error and could not process your request."
                st.error(error_message) # Display error in the assistant message bubble
                st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Rerun to clear the input fields after successful submission?
    # This can sometimes cause issues, test carefully.
    # st.rerun()


# --- Handle Follow-up Questions via Chat Input ---
# Determine if chat input should be disabled
# Disable if app isn't ready
chat_input_disabled = input_disabled

if prompt := st.chat_input("Ask a follow-up question...", disabled=chat_input_disabled, key="follow_up_input"):
    logging.info(f"Follow-up question submitted via chat input: '{prompt}'")
    # Add user's follow-up message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user's follow-up message in the chat container
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response for the follow-up question
        with st.chat_message("assistant"):
            # Use a placeholder while running
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            response_data = run_graph_and_get_response(prompt) # Rerun the graph
            placeholder.empty() # Clear placeholder

            if response_data:
                # Display main content
                st.markdown(response_data["content"])

                # Display Debug Info Expander
                with st.expander("Show Debugging Info"):
                    debug_info = response_data.get("debug_info", {})
                    orig_cnt = debug_info.get("original_doc_count", "N/A")
                    filt_cnt = debug_info.get("filtered_doc_count", "N/A")
                    rel_grade = str(debug_info.get("relevance_grade", "N/A")).upper()
                    st.markdown(f"**Initial Retrieval:** {orig_cnt} chunks found.")
                    st.markdown(f"**After Filtering:** {filt_cnt} chunks remained.")
                    st.markdown(f"**Relevance Grade:** {rel_grade}")

                # Display Initial Context Expander
                if response_data["original_docs"]:
                     with st.expander(f"Show Initially Retrieved Context ({len(response_data['original_docs'])} chunks)"):
                          for i, doc in enumerate(response_data["original_docs"]):
                               page_num = doc.metadata.get('page', 'N/A')
                               file_name = doc.metadata.get("file", "N/A")
                               st.markdown(f"**Chunk {i+1} (`{file_name}` - Page {page_num}):**")
                               st.markdown(f"> {doc.page_content}")
                               st.divider()

                # Display Individual Answers Expander
                if response_data["individual_answers"]:
                     substantive_answers = {fname: ans for fname, ans in response_data["individual_answers"].items() if "No relevant information found" not in ans and "An error occurred" not in ans and "Could not generate" not in ans}
                     expander_title = f"Show Individual Answers per File ({len(substantive_answers)} file(s) with substantive answers / {len(response_data['individual_answers'])} total processed)"
                     with st.expander(expander_title):
                          for filename, answer in response_data["individual_answers"].items():
                              st.markdown(f"**Answer from `{filename}`:**")
                              st.markdown(answer)
                              st.divider()

                # Add the complete assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["content"],
                    "original_docs": response_data["original_docs"],
                    "individual_answers": response_data["individual_answers"],
                    "debug_info": response_data["debug_info"]
                })
            else:
                # Handle case where run_graph_and_get_response returned None
                error_message = "Sorry, I encountered a critical error and could not process your follow-up request."
                st.error(error_message) # Display error in assistant message bubble
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Log DEBUG_CAPTURE if any items were added ---
if DEBUG_CAPTURE:
    logging.info("--- Captured Debug Timings/Messages ---")
    for item in DEBUG_CAPTURE:
        logging.info(item)
    DEBUG_CAPTURE.clear() # Clear after logging for next run

# --- Footer (Sidebar) ---
st.sidebar.divider()
st.sidebar.caption("Powered by LangChain, LangGraph, Azure OpenAI, FAISS, and Streamlit")
