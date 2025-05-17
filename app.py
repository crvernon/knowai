import streamlit as st
import base64
import os
import fitz  # PyMuPDF
# import faiss # Not directly used in this snippet, but likely in build_vectorstore
import hashlib # For hashing file content if needed for caching
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
import operator
from collections import defaultdict
import asyncio # Added for asynchronous operations
import time

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


DEBUG_CAPTURE = [] # For capturing debug timing or messages

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed") # collapse sidebar on load

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
    logging.info("Successfully imported 'build_vectorstore'.")
except ImportError:
    logging.error("Failed to import 'build_vectorstore'. Make sure 'build_vectorstore.py' exists and is accessible.")
    st.error("Critical component 'build_vectorstore.py' not found. File listing and potential loading operations will fail.", icon="🚨")
    # Define dummy functions
    def list_vectorstore_files(vs): # vs is the vectorstore object
        logging.warning("Using dummy list_vectorstore_files function.")
        return []
    # Define load_vectorstore if it were used directly in this script
    # def load_vectorstore(path, embeddings_model): # Renamed to avoid conflict
    #     logging.warning("Using dummy load_vectorstore function.")
    #     return None
    vectorstore_functions_available = False


# Load credentials; make sure dotenv overwrites any system variable settings
load_dotenv(override=True)

k_chunks_retriever = 75 # Number of chunks MultiQueryRetriever aims for IN TOTAL

# Fetch Azure credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # Default model
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large") # Default embedding model
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # API version

# Check for Azure Credentials
azure_creds_valid = True
if not api_key or not azure_endpoint:
    # This error will be shown prominently in the UI if creds are missing.
    # Subsequent logic relies on azure_creds_valid.
    azure_creds_valid = False
    logging.error("Azure credentials (key or endpoint) missing. App functionality will be limited.")
    # No st.error here yet, will be handled in UI setup after logos.


# --- Logo Display ---
logo_col1, logo_col2, title_col, logo_col3 = st.columns([1, 1.4, 4, 1.4]) # Adjust column ratios if needed
logo_path1 = "./img/pnnl-logo.png"
logo_path2 = "./img/gdo-logo.png"
logo_path3 = "./img/wrr-pnnl-logo.svg"
svg_path = "./img/projectLogo.svg"


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
    display_logo(logo_col3, logo_path3, "WRR Logo")

# Space between logos and title
st.markdown(
    "<div style='text-align:center;'></br></br></br></div>",
    unsafe_allow_html=True
)

# --- Title with SVG Icon ---
try:
    if os.path.exists(svg_path):
        with open(svg_path, "rb") as f:
            svg_base64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"<div style='display:flex; align-items:center; justify-content:center; margin-top: -50px;'>"
            f"<img src='data:image/svg+xml;base64,{svg_base64}' width='32' height='32' style='margin-right:8px;'/>"
            f"<h2 style='margin:0;'>Wildfire Mitigation Plans Database</h2>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        logging.warning(f"SVG logo not found at {svg_path}. Displaying text title only.")
        st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True)
except Exception as e:
    logging.error(f"Error loading SVG logo from {svg_path}: {e}")
    # st.error(f"Error loading SVG logo: {e}") # Avoid st.error here if possible, use logging
    st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True)

# Subtitle for the chat feature
st.markdown(
    "<div style='text-align:center;'><h4>-- AI Assistant Chat --</h4></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;'><em>Your AI Assistant to find answers across multiple reports!</em></div>",
    unsafe_allow_html=True
)

# Display Azure credential error here if they were missing
if not azure_creds_valid:
    st.error("Azure OpenAI API key or endpoint not found. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables. The application's AI features will be disabled.", icon="🚨")


# --- Load Base Retriever and List Files ---
vectorstore_path = "test_faiss_store" # Path to your FAISS index directory
base_retriever = None
vectorstore = None # Initialize vectorstore to None
files_in_store = []
embeddings_model = None # Renamed from 'embeddings' to avoid conflict with Langchain's typical usage

if azure_creds_valid and vectorstore_functions_available:
    try:
        logging.info("Initializing Azure OpenAI Embeddings...")
        embeddings_model = AzureOpenAIEmbeddings( # Assign to embeddings_model
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )
        logging.info("Embeddings model initialized.")

        if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
            logging.info(f"Loading FAISS vector store from: {vectorstore_path}")
            # Use the imported load_vectorstore or FAISS.load_local directly
            # If build_vectorstore.py has its own load_vectorstore, it might encapsulate FAISS.load_local
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings_model, # Pass the initialized embeddings model
                allow_dangerous_deserialization=True,
            )
            logging.info("FAISS vector store loaded.")

            try:
                files_in_store = list_vectorstore_files(vectorstore) # Pass the loaded vectorstore
                logging.info(f"Found {len(files_in_store)} files in vector store index: {files_in_store}")
            except Exception as e:
                logging.error(f"Error calling list_vectorstore_files: {e}")
                st.error(f"Error listing files from vector store: {e}")
                files_in_store = []

            if files_in_store:
                base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks_retriever})
                logging.info("Base retriever created.")
                # st.sidebar.success(f"Loaded vector store with {len(files_in_store)} file(s).") # Moved to sidebar UI
            else:
                # st.sidebar.warning(f"Vector store loaded, but no files were found within it.") # Moved to sidebar UI
                logging.warning("No files found in vector store index. Retriever not created.")
        else:
            # st.sidebar.error(f"Vector store directory not found: '{vectorstore_path}'.") # Moved to sidebar UI
            logging.error(f"Vector store path not found or invalid: {vectorstore_path}")
            # No need to set azure_creds_valid to False here, other components might still work.
            # The base_retriever will remain None, and UI will reflect this.
    except Exception as e:
        st.error(f"Error initializing embeddings or loading vectorstore '{vectorstore_path}': {e}", icon="💾")
        logging.exception(f"Failed to load vector store from {vectorstore_path}")
        # base_retriever remains None
elif not azure_creds_valid:
    # st.sidebar.warning("Skipping vector store load due to missing Azure credentials.") # Moved to sidebar UI
    logging.warning("Vector store loading skipped (missing Azure credentials).")
elif not vectorstore_functions_available:
    # st.sidebar.error("Skipping vector store load: 'build_vectorstore.py' unavailable.") # Moved to sidebar UI
    logging.error("Vector store loading skipped (build_vectorstore import failed).")


# --- Constants for Predefined Questions ---
PREDEFINED_QUESTIONS = [
    "Enter my own question...",
    "Does the plan include vegetation management? If so, how much money is allocated to it?",
    "Does the plan include undergrounding? If so, how much money is allocated to it?",
    "Does the plan include PSPS? If so, how much money is allocated to it?",
    "How frequently does the utility perform asset inspections?",
    "Are there generation considerations, such as derating solar PV during smoky conditions?"
]

# --- File Selector UI (Sidebar) ---
st.sidebar.subheader("Spaceholder for Filtering Sidebar")
st.sidebar.warning(
    "Content from your https://wildfire.pnnl.gov/mitigationPlans/pages/list data filtering sidebar will appear here. "
    "Items selected in your filtering will be used for QA in the AI Assistant."
)

st.sidebar.subheader("Select Source File(s)")
if not azure_creds_valid:
    st.sidebar.warning("File selection disabled: Azure credentials missing.", icon="🔒")
elif not vectorstore_functions_available:
    st.sidebar.warning("File selection disabled: Vector store components unavailable.", icon="⚙️")
elif not os.path.exists(vectorstore_path) or not os.path.isdir(vectorstore_path):
    st.sidebar.error(f"File selection disabled: Vector store directory '{vectorstore_path}' not found.", icon="📁")
elif not vectorstore: # If vectorstore loaded but something went wrong (e.g. failed to load_local)
    st.sidebar.error("File selection disabled: Vector store could not be loaded.", icon="💾")
elif not files_in_store and vectorstore: # Store loaded but list_vectorstore_files found nothing
    st.sidebar.warning("Vector store loaded, but no files were found within its index.", icon="📄")
elif files_in_store:
    st.sidebar.success(f"Vector store loaded with {len(files_in_store)} file(s). Ready for selection.")


file_options = ["All"] + files_in_store if files_in_store else ["All"] # Ensure "All" is always an option if store might be empty
default_selection = ["All"] if files_in_store else []

# Determine if the multiselect should be disabled
# It's disabled if creds are bad, or retriever isn't there, or no files in store to select from
multiselect_disabled = not (azure_creds_valid and base_retriever and files_in_store)

selected_files_user_choice = st.sidebar.multiselect(
    "Files to Query",
    options=file_options,
    default=default_selection,
    help="Select specific files to search, or 'All' for the entire loaded vector store.",
    disabled=multiselect_disabled
)

# Determine the actual list of files to process based on user selection
files_to_process_in_graph = []
if files_in_store: # Only determine if there are files available
    if "All" in selected_files_user_choice or not selected_files_user_choice:
        files_to_process_in_graph = files_in_store
        if not selected_files_user_choice and "All" in file_options:
            logging.info("User cleared selection, defaulting to 'All' available files.")
    else:
        files_to_process_in_graph = [f for f in selected_files_user_choice if f != "All"]

    # Display info/warning based on the final list
    if files_to_process_in_graph == files_in_store and "All" not in selected_files_user_choice and len(files_to_process_in_graph) > 0 :
        st.sidebar.info(f"Querying all {len(files_to_process_in_graph)} available file(s) (selected individually).")
        logging.info(f"User selected all available files individually: {files_to_process_in_graph}")
    elif files_to_process_in_graph and "All" in selected_files_user_choice :
         st.sidebar.info(f"Querying all {len(files_to_process_in_graph)} available file(s) (selected 'All').")
         logging.info(f"User selected 'All', processing: {files_to_process_in_graph}")
    elif files_to_process_in_graph:
        st.sidebar.info(f"Will query within: {', '.join(files_to_process_in_graph)}")
        logging.info(f"User selected specific files: {files_to_process_in_graph}")
    elif not files_to_process_in_graph and selected_files_user_choice: # e.g. user deselects everything
        st.sidebar.warning("No specific files selected. Please select files or 'All'. Querying will be disabled.", icon="⚠️")
        logging.warning("File selection resulted in an empty list. Querying will be disabled.")
    elif not files_to_process_in_graph and not selected_files_user_choice and not multiselect_disabled: # No files selected, not disabled
        st.sidebar.warning("No files selected for query. Please choose from the list.", icon="⚠️")
        logging.warning("No files selected by user for query.")

elif not multiselect_disabled: # No files in store, but controls are not disabled (e.g. store loaded but empty)
    st.sidebar.warning("No files available in the vector store to select for query.", icon="📄")
    logging.warning("No files available in vector store for selection by user.")
# If multiselect_disabled is true, other messages cover the reason.


# --- Helper Functions (PDF Processing - Kept for potential future use, e.g., ad-hoc uploads) ---
def extract_pages_from_pdf(pdf_file_obj): # Renamed from pdf_file to avoid conflict
    """Extracts text page by page from an uploaded PDF file object, returning a list of (page_number, text, filename)."""
    pages_content = []
    file_name = pdf_file_obj.name if hasattr(pdf_file_obj, 'name') else "Unknown Filename"
    try:
        pdf_bytes = pdf_file_obj.getvalue()
        if not pdf_bytes:
            logging.error(f"Uploaded file '{file_name}' is empty.")
            st.error(f"Uploaded file '{file_name}' is empty.") # This would show up if used with st.file_uploader
            return None
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logging.info(f"Processing PDF '{file_name}' with {len(doc)} pages.")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text and text.strip():
                pages_content.append((page_num + 1, text, file_name))
        doc.close()
        if not pages_content:
            logging.warning(f"Could not extract any text content from PDF '{file_name}'.")
            # st.warning(f"Could not extract any text from '{file_name}'.")
            return None
        logging.info(f"Extracted text from {len(pages_content)} pages in '{file_name}'.")
        return pages_content
    except fitz.fitz.FileDataError:
        logging.error(f"Invalid or corrupted PDF file: {file_name}")
        # st.error(f"Invalid or corrupted PDF file: {file_name}", icon="📄")
        return None
    except Exception as e:
        logging.exception(f"Error reading PDF '{file_name}'")
        # st.error(f"Error reading PDF '{file_name}': {e}", icon="📄")
        return None

# --- Caching (Retriever Creation - Kept for potential future use if uploading files) ---
@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def get_base_retriever_from_pdf(_file_hash, uploaded_file_name_cache): # Use underscore for unused hash
    """Processes a single PDF, creates embeddings, builds FAISS store, returns BASE retriever."""
    t0 = time.time() # Corrected: time.time()
    logging.info(f"Cache miss or first run for get_base_retriever_from_pdf: {uploaded_file_name_cache} (hash: {_file_hash})")

    if not api_key or not azure_endpoint: # Check global Azure creds
        logging.error("Azure credentials not available for PDF processing inside cached function.")
        return None

    uploaded_file_cache = st.session_state.get(f"file_{_file_hash}", None)
    if not uploaded_file_cache:
        for key, value in st.session_state.items():
            if hasattr(value, 'name') and value.name == uploaded_file_name_cache:
                uploaded_file_cache = value
                logging.warning(f"Retrieved file '{uploaded_file_name_cache}' by name lookup in session state for cache.")
                break
        if not uploaded_file_cache:
            logging.error(f"Could not find file content for '{uploaded_file_name_cache}' in session state (hash: {_file_hash}) for cache.")
            return None
    try:
        logging.info(f"Starting PDF processing for (cache): {uploaded_file_cache.name}")
        pages_data = extract_pages_from_pdf(uploaded_file_cache)
        if not pages_data:
            return None

        logging.info(f"Splitting text for (cache) {uploaded_file_cache.name}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
        all_docs = []
        for page_num, page_text, filename_meta in pages_data: # Renamed filename to filename_meta
            page_chunks = text_splitter.split_text(page_text)
            for chunk_index, chunk in enumerate(page_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"page": page_num, "file": filename_meta, "chunk_in_page": chunk_index}
                )
                all_docs.append(doc)
        if not all_docs:
            logging.warning(f"Could not create any text chunks from (cache) '{uploaded_file_cache.name}'.")
            return None
        logging.info(f"Split PDF (cache) '{uploaded_file_cache.name}' into {len(all_docs)} chunks.")

        logging.info(f"Creating embeddings for (cache) '{uploaded_file_cache.name}' chunks...")
        # Use the globally defined embeddings_model if available
        current_embeddings_cache = embeddings_model # Use the globally loaded one
        if not current_embeddings_cache and azure_creds_valid: # Fallback if global somehow not set but creds are ok
            logging.warning("Re-initializing embeddings inside cached function (should use global).")
            current_embeddings_cache = AzureOpenAIEmbeddings(
                azure_deployment=embeddings_deployment, 
                azure_endpoint=azure_endpoint,
                api_key=api_key, 
                openai_api_version=openai_api_version
            )
        elif not current_embeddings_cache:
            logging.error("Embeddings model not available inside cached function.")
            return None

        logging.info(f"Building FAISS index for (cache) '{uploaded_file_cache.name}'...")
        vectorstore_pdf = FAISS.from_documents(documents=all_docs, embedding=current_embeddings_cache)
        logging.info(f"FAISS index built successfully for (cache) '{uploaded_file_cache.name}'.")
        DEBUG_CAPTURE.append(f"Initialize vectorstore (cache): {round(time.time() - t0, 4)} seconds")
        return vectorstore_pdf.as_retriever(search_kwargs={'k': k_chunks_retriever}) 
    except Exception as e:
        logging.exception(f"Error processing PDF '{uploaded_file_name_cache}' in cached function")
        return None


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    documents: List[Document] 
    original_documents: List[Document] 
    filtered_documents: Optional[List[Document]] 
    relevance_grade: Optional[str] 
    base_retriever: object 
    files_to_process: List[str]
    individual_answers: Dict[str, str]


# --- LangGraph Nodes ---

# Node 1: Retrieve documents using MultiQueryRetriever
def retrieve_docs_multi_query(state: GraphState) -> GraphState:
    t0 = time.time()
    logging.info("--- Starting Node: retrieve_docs_multi_query ---")
    question = state["question"]
    current_base_retriever = state["base_retriever"] # Renamed to avoid conflict
    files_to_process_node = state["files_to_process"] # Renamed

    output_state_node = { # Renamed
        "original_documents": [], "filtered_documents": [], 
        "documents": [], "relevance_grade": None
    }
    if not current_base_retriever:
        logging.error("Base retriever is None in retrieve_docs_multi_query. Cannot retrieve.")
        return {**state, **output_state_node}
    logging.info(f"Processing question: '{question}' for files: {files_to_process_node}")

    if not azure_creds_valid: # Global check
        logging.warning("Azure credentials missing. Falling back to simple retrieval in retrieve_docs_multi_query.")
        try:
            all_retrieved_docs = current_base_retriever.get_relevant_documents(question) # Renamed
            # Filter AFTER simple retrieval
            filtered_retrieved_docs = [doc for doc in all_retrieved_docs if doc.metadata.get("file") in files_to_process_node] # Renamed
            logging.info(f"Retrieved {len(filtered_retrieved_docs)} documents via simple fallback for files: {files_to_process_node}.")
            output_state_node["original_documents"] = filtered_retrieved_docs
        except Exception as e:
            logging.error(f"Error during simple fallback retrieval in retrieve_docs_multi_query: {e}")
        return {**state, **output_state_node}
    try:
        logging.info(f"Initializing MultiQueryRetriever for question: '{question}'")
        llm_for_queries = AzureChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            openai_api_version=openai_api_version,
            azure_deployment="gpt-4o-mini", 
            azure_endpoint=azure_endpoint,
        )
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=current_base_retriever, llm=llm_for_queries
        )
        logging.info(f"Invoking MultiQueryRetriever...")
        unique_docs_unfiltered = multi_query_retriever.invoke(input=question)
        logging.info(f"MultiQueryRetriever returned {len(unique_docs_unfiltered)} total unique chunks before file filtering.")
        unique_docs_filtered = [
            doc for doc in unique_docs_unfiltered
            if doc.metadata.get("file") in files_to_process_node
        ]
        logging.info(f"Filtered down to {len(unique_docs_filtered)} chunks based on selected file(s): {files_to_process_node}")
        output_state_node["original_documents"] = unique_docs_filtered
    except Exception as e:
        logging.exception("Error during multi-query retrieval in retrieve_docs_multi_query")
    finally:
        logging.info("--- Finished Node: retrieve_docs_multi_query ---")
    DEBUG_CAPTURE.append(f"Conduct MultiQuery: {round(time.time() - t0, 4)} seconds")
    return {**state, **output_state_node}


# Node 2: Filter structural documents
class FilteredDocs(BaseModel):
    keep_indices: List[int] = Field(
        description="List of zero-based indices of the documents that should be kept (i.e., are not primarily structural like ToC, headers, footers, or simple reference lists)."
    )

def filter_documents(state: GraphState) -> GraphState:
    t0 = time.time()
    logging.info("--- Starting Node: filter_documents ---")
    original_documents_node = state["original_documents"] # Renamed
    question_node = state["question"] # Renamed

    output_state_filter = { # Renamed
        "filtered_documents": [], "documents": [], "relevance_grade": None
    }
    if not original_documents_node:
        logging.info("No documents received from retrieval node to filter.")
        return {**state, **output_state_filter}
    logging.info(f"Attempting to filter {len(original_documents_node)} retrieved chunks for structural content.")

    if not azure_creds_valid: # Global check
        logging.warning("Skipping structural filtering due to missing Azure credentials.")
        output_state_filter["filtered_documents"] = original_documents_node
        return {**state, **output_state_filter}
    try:
        filtering_llm = AzureChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            openai_api_version=openai_api_version,
            azure_deployment="gpt-4o-mini", 
            azure_endpoint=azure_endpoint,
        ).bind_tools([FilteredDocs], tool_choice="FilteredDocs")
        doc_context = ""
        for i, doc_loop in enumerate(original_documents_node): # Renamed doc to doc_loop
            filename_meta = doc_loop.metadata.get('file', 'Unknown File') # Renamed
            page_num_meta = doc_loop.metadata.get('page', 'N/A') # Renamed
            content_preview = (doc_loop.page_content[:500] + '...') if len(doc_loop.page_content) > 500 else doc_loop.page_content
            doc_context += f"--- Document Index {i} (File: {filename_meta}, Page: {page_num_meta}) ---\n{content_preview}\n\n"
        
        filtering_prompt_template_text = """You are an expert document analyst. Your task is to identify document chunks that are primarily structural elements like Table of Contents entries, page headers/footers, indices, reference lists, or lists of figures/tables. These structural elements are unlikely to contain direct answers to specific content-based questions about technical reports.

Analyze the following document chunks provided below, each marked with a 'Document Index', 'File', and 'Page'. The user's question is: "{question}"

Document Chunks:
{doc_context}

Based *only* on the content of each chunk, identify the indices of the documents that ARE LIKELY TO CONTAIN SUBSTANTIVE CONTENT (prose, data, analysis, findings, methods, explanations) relevant to potentially answering the user's question. Ignore chunks that are just lists of section titles with page numbers (like a ToC), repetitive headers/footers, bibliographies, or cover page elements unless they uniquely contain relevant information not present elsewhere.

Use the 'FilteredDocs' tool to return the list of zero-based indices of the documents to **keep**. Return ONLY the indices to keep. If all documents appear structural or irrelevant, return an empty list.
"""
        filtering_prompt = PromptTemplate(
            template=filtering_prompt_template_text, # Use the variable
            input_variables=["doc_context", "question"],
        )
        filtering_chain = filtering_prompt | filtering_llm
        logging.info("Invoking filtering LLM...")
        response = filtering_chain.invoke({"doc_context": doc_context, "question": question_node})
        logging.info("Filtering LLM response received.")
        kept_indices = []
        if response.tool_calls and response.tool_calls[0]['name'] == 'FilteredDocs':
            kept_indices = response.tool_calls[0]['args'].get('keep_indices', [])
            valid_indices = [i for i in kept_indices if isinstance(i, int) and 0 <= i < len(original_documents_node)]
            if len(valid_indices) != len(kept_indices):
                logging.warning(f"Filtering LLM returned invalid indices. Original: {kept_indices}, Validated: {valid_indices}")
            kept_indices = valid_indices
            logging.info(f"Filtering LLM identified {len(kept_indices)} documents to keep out of {len(original_documents_node)}.")
        else:
            logging.warning(f"Filtering LLM did not return 'FilteredDocs' tool call. Response: {response}. Keeping all documents.")
            output_state_filter["filtered_documents"] = original_documents_node # Pass through on error
            return {**state, **output_state_filter}
        
        filtered_docs_list = [original_documents_node[i] for i in kept_indices] # Renamed
        logging.info(f"Filtered down to {len(filtered_docs_list)} potentially substantive documents.")
        output_state_filter["filtered_documents"] = filtered_docs_list
    except Exception as e:
        logging.exception("Error during document filtering")
        logging.warning("Filtering failed. Passing all original documents to grading.")
        output_state_filter["filtered_documents"] = original_documents_node # Pass through on error
    finally:
        logging.info("--- Finished Node: filter_documents ---")
    DEBUG_CAPTURE.append(f"Filtered documents: {round(time.time() - t0, 4)} seconds")
    return {**state, **output_state_filter}


# Node 3: Grade documents for relevance
def grade_documents(state: GraphState) -> GraphState:
    t0 = time.time()
    logging.info("--- Starting Node: grade_documents ---")
    question_grade = state["question"] # Renamed
    documents_to_grade = state["filtered_documents"]

    output_state_grade = {"documents": [], "relevance_grade": "no"} # Renamed
    if documents_to_grade is None:
        logging.warning("Documents to grade is None (upstream error). Setting to empty list.")
        documents_to_grade = []
        output_state_grade["relevance_grade"] = "error_upstream"
        return {**state, **output_state_grade}
    if not documents_to_grade:
        logging.info("No documents remaining after filtering to grade.")
        return {**state, **output_state_grade}
    logging.info(f"Attempting to grade relevance of {len(documents_to_grade)} filtered documents for question: '{question_grade}'")

    if not azure_creds_valid: # Global check
        logging.warning("Skipping relevance grading due to missing Azure credentials.")
        output_state_grade["documents"] = documents_to_grade
        output_state_grade["relevance_grade"] = "skipped"
        return {**state, **output_state_grade}
    try:
        llm_grader = AzureChatOpenAI(
            temperature=0, 
            api_key=api_key, 
            openai_api_version=openai_api_version,
            azure_deployment="gpt-4o-mini", 
            azure_endpoint=azure_endpoint,
        )
        grading_function_schema = { # Renamed
            "name": "grade_relevance",
            "description": "Determine if the provided document chunks collectively contain information relevant to answering the user's question.",
            "parameters": {
                "type": "object",
                "properties": {"relevant": {"type": "string", "enum": ["yes", "no"], "description": "Output 'yes' if relevant, 'no' otherwise."}},
                "required": ["relevant"]
            },
        }
        grading_prompt_template_text = """You are a strict grader assessing the relevance of pre-filtered document chunks to a specific user question. Your goal is to determine if these chunks contain information that **directly answers** the question.

User Question:
{question}

Potentially Relevant Document Chunks (pre-filtered for structural content):
{documents}

Instructions:
1. Read the User Question carefully.
2. Analyze the provided Document Chunks.
3. Decide if the information within these chunks, taken together, can **directly** address and answer the user's question. Do not consider tangential or background information as sufficient unless it's essential to the answer.
4. Respond using the 'grade_relevance' function call. Output 'yes' if direct answering information is present, otherwise output 'no'. Be liberal; if uncertain, lean towards 'yes'.
"""
        grading_prompt = PromptTemplate(
            template=grading_prompt_template_text, # Use the variable
            input_variables=["question", "documents"],
        )
        doc_texts_with_meta = "\n\n".join([f"--- File: {d.metadata.get('file', 'Unknown')}, Page: {d.metadata.get('page', 'N/A')} ---\n{d.page_content}" for d in documents_to_grade])
        grader_chain = grading_prompt | llm_grader.bind_tools([grading_function_schema], tool_choice="grade_relevance")
        logging.info("Invoking grading LLM...")
        response = grader_chain.invoke({"question": question_grade, "documents": doc_texts_with_meta})
        logging.info("Grading LLM response received.")
        is_relevant_grade = "no" # Renamed
        if response.tool_calls and response.tool_calls[0]['name'] == 'grade_relevance':
            is_relevant_grade = response.tool_calls[0]['args'].get('relevant', 'no')
            logging.info(f"Relevance Grade determined by LLM: {is_relevant_grade.upper()}")
        else:
            logging.warning(f"Grader LLM did not return expected format. Response: {response}. Assuming 'no' relevance.")
        
        output_state_grade["relevance_grade"] = is_relevant_grade
        if is_relevant_grade == "yes":
            logging.info("Decision: Relevant documents found. Passing them to the next step.")
            output_state_grade["documents"] = documents_to_grade
        else:
            logging.info("Decision: No relevant documents found after grading.")
    except Exception as e:
        logging.exception("Error during document grading")
        output_state_grade["documents"] = []
        output_state_grade["relevance_grade"] = "error"
    finally:
        logging.info("--- Finished Node: grade_documents ---")
    DEBUG_CAPTURE.append(f"Grading documents: {round(time.time() - t0, 4)} seconds")
    return {**state, **output_state_grade}


# Node 4: Generate Answer for each file (using relevant documents) - ASYNC Version
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
    t0 = time.time()
    logging.info("--- Starting Node: combine_answers ---")
    question_combine = state["question"] # Renamed
    individual_answers_combine = state["individual_answers"] # Renamed
    files_processed_count = len(individual_answers_combine)

    output_state_combine = {"generation": None} # Renamed
    if not individual_answers_combine:
        logging.warning("Combine Answers node called with no individual answers dictionary.")
        output_state_combine["generation"] = "An internal error occurred: No individual file answers were available to combine."
        return {**state, **output_state_combine}

    answers_to_combine_dict = {} # Renamed
    files_with_no_info_list = [] # Renamed
    files_with_errors_list = [] # Renamed
    for filename_loop_combine, answer_loop_combine in individual_answers_combine.items(): # Renamed
        if "An error occurred while generating" in answer_loop_combine or "Could not generate answer due to missing credentials" in answer_loop_combine:
            files_with_errors_list.append(f"`{filename_loop_combine}`")
        elif "No relevant information found" in answer_loop_combine:
            files_with_no_info_list.append(f"`{filename_loop_combine}`")
        else:
            answers_to_combine_dict[filename_loop_combine] = answer_loop_combine
    
    if not answers_to_combine_dict:
        logging.info("No substantive individual answers available to combine after filtering.")
        final_msg_combine = f"Could not find relevant information to answer the question in the analyzed sections of the {files_processed_count} selected file(s)." # Renamed
        if files_with_no_info_list:
            final_msg_combine += f"\nFiles checked with no relevant info found: {', '.join(files_with_no_info_list)}."
        if files_with_errors_list:
            final_msg_combine += f"\nErrors occurred during processing for files: {', '.join(files_with_errors_list)}."
        output_state_combine["generation"] = final_msg_combine
        logging.info("--- Finished Node: combine_answers (No substantive content) ---")
        return {**state, **output_state_combine}

    logging.info(f"Combining {len(answers_to_combine_dict)} substantive answers from files: {list(answers_to_combine_dict.keys())}")
    if not azure_creds_valid: # Global check
        logging.warning("Skipping answer combination due to missing Azure credentials.")
        combined_fallback = f"Could not properly combine answers due to missing credentials. Individual findings:\n\n" # Renamed
        for filename_fb, answer_fb in answers_to_combine_dict.items(): # Renamed
            combined_fallback += f"--- Findings from {filename_fb} ---\n{answer_fb}\n\n"
        if files_with_no_info_list: combined_fallback += f"Files checked with no relevant info: {', '.join(files_with_no_info_list)}.\n"
        if files_with_errors_list: combined_fallback += f"Errors occurred for files: {', '.join(files_with_errors_list)}.\n"
        output_state_combine["generation"] = combined_fallback.strip()
        return {**state, **output_state_combine}

    formatted_answers_text = "" # Renamed
    for filename_fmt, answer_fmt in answers_to_combine_dict.items(): # Renamed
        formatted_answers_text += f"--- Answer based on file: {filename_fmt} ---\n{answer_fmt}\n\n"
    
    prompt_template_combine_text = """You are an expert synthesis assistant. Your task is to combine multiple answers, each generated from a different source file in response to the same user question. Create a single, comprehensive, and well-structured response.

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
        template=prompt_template_combine_text, # Use the variable
        input_variables=["question", "formatted_answers", "files_no_info", "files_errors"]
    )

    DEBUG_CAPTURE.append(f"Combine responses into answer - prep: {round(time.time() - t0, 4)} seconds")


    try:
        logging.info(f"Invoking combination LLM to synthesize answers...")
        t0 = time.time()


        llm_combine = AzureChatOpenAI(
            temperature=0.0, 
            api_key=api_key, 
            openai_api_version=openai_api_version,
            azure_deployment="gpt-4o-mini", 
            azure_endpoint=azure_endpoint, 
            # max_tokens=2000
        )

        combination_chain = prompt_combine | llm_combine | StrOutputParser()
        combine_input_dict = { # Renamed
            "question": question_combine,
            "formatted_answers": formatted_answers_text.strip(),
            "files_no_info": ", ".join(files_with_no_info_list) if files_with_no_info_list else "None",
            "files_errors": ", ".join(files_with_errors_list) if files_with_errors_list else "None"
        }
        final_generation_text = combination_chain.invoke(combine_input_dict) # Renamed

        logging.info(combine_input_dict)
        logging.info(prompt_combine)

        logging.info("Combination LLM finished. Final answer generated.")

        output_state_combine["generation"] = final_generation_text

    except Exception as e:
        logging.exception("Error combining individual answers")
        combined_error_fallback = f"An error occurred during answer synthesis. Raw findings:\n\n" # Renamed
        for filename_err_fb, answer_err_fb in answers_to_combine_dict.items(): # Renamed
            combined_error_fallback += f"--- Findings from {filename_err_fb} ---\n{answer_err_fb}\n\n"
        if files_with_no_info_list: combined_error_fallback += f"\nFiles checked with no relevant info: {', '.join(files_with_no_info_list)}."
        if files_with_errors_list: combined_error_fallback += f"\nErrors occurred for files: {', '.join(files_with_errors_list)}."
        output_state_combine["generation"] = combined_error_fallback.strip()
    finally:
        logging.info("--- Finished Node: combine_answers ---")

        DEBUG_CAPTURE.append(f"Combine responses into answer - LLM: {round(time.time() - t0, 4)} seconds")

    return {**state, **output_state_combine}


# --- LangGraph Conditional Edge ---
def decide_to_generate(state: GraphState) -> str:
    logging.info("--- Decision Node: decide_to_generate ---")
    relevance_grade_decision = state.get("relevance_grade", "no") # Renamed
    documents_found_decision = state.get("documents", []) # Renamed

    if relevance_grade_decision == "yes" and documents_found_decision:
        logging.info("Decision: Relevant documents found (Grade: YES). Proceeding to generate individual answers.")
        return "generate_individual"
    elif relevance_grade_decision == "skipped" and documents_found_decision:
        logging.warning("Decision: Grading was skipped, but documents exist. Proceeding to generate individual answers (results may be less relevant).")
        return "generate_individual"
    else:
        logging.info(f"Decision: Not generating individual answers (Grade: {relevance_grade_decision}, Relevant Docs Count: {len(documents_found_decision)}). Ending workflow here.")
        return "end_no_relevance"


# --- Build LangGraph Workflow ---
langgraph_app = None
if azure_creds_valid and base_retriever: # Global checks
    try:
        logging.info("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve_docs_multi_query)
        workflow.add_node("filter_documents", filter_documents)
        workflow.add_node("grade_documents", grade_documents)
        # Use the ASYNC version of generate_individual_answers
        workflow.add_node("generate_individual", generate_individual_answers)
        workflow.add_node("combine_answers", combine_answers)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "filter_documents")
        workflow.add_edge("filter_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents", decide_to_generate,
            {"generate_individual": "generate_individual", "end_no_relevance": END}
        )
        workflow.add_edge("generate_individual", "combine_answers")
        workflow.add_edge("combine_answers", END)

        langgraph_app = workflow.compile()
        logging.info("LangGraph workflow compiled successfully.")
        # st.sidebar.success("Multi-file RAG workflow compiled.") # UI element, moved to UI section
    except Exception as e:
        # st.sidebar.error(f"Failed to compile LangGraph: {e}") # UI element
        logging.exception("LangGraph compilation failed.")
        langgraph_app = None
# UI messages for LangGraph status will be in the sidebar section or main UI
# elif not azure_creds_valid:
#     # st.sidebar.error("LangGraph compilation skipped: Missing Azure credentials.")
#     logging.error("LangGraph compilation skipped (missing Azure credentials).")
# elif not base_retriever:
#     # st.sidebar.error("LangGraph compilation skipped: Vector store not loaded or empty.")
#     logging.error("LangGraph compilation skipped (base retriever not available).")

# Display LangGraph compilation status in sidebar
if langgraph_app and azure_creds_valid and base_retriever:
    st.sidebar.success("AI Assistant workflow ready.", icon="✅")
elif not azure_creds_valid:
    st.sidebar.error("AI Assistant workflow disabled: Azure credentials missing.", icon="🔒")
elif not base_retriever : # Covers case where vectorstore might be loaded but retriever failed, or store not found
    st.sidebar.error("AI Assistant workflow disabled: Vector store/retriever not ready.", icon="💾")
elif not langgraph_app: # If creds and retriever were fine, but graph failed to compile
    st.sidebar.error("AI Assistant workflow disabled: Compilation failed.", icon="❌")


# --- Streamlit UI ---

# --- Determine if inputs should be disabled ---
input_disabled = not (azure_creds_valid and base_retriever and langgraph_app and files_to_process_in_graph)
disabled_reason = ""
if not azure_creds_valid: disabled_reason += "Azure credentials missing. "
if not base_retriever: disabled_reason += "Vector store not loaded/empty. "
if not files_to_process_in_graph: disabled_reason += "No files selected/available in sidebar. "
if not langgraph_app: disabled_reason += "RAG workflow failed. " # Or "not ready"
disabled_reason = disabled_reason.strip()

# --- Initial Question Input Area (Main Area) ---
st.subheader("Ask Initial Question")
q_col1, q_col2 = st.columns([3, 1])

with q_col1:
    selected_question = st.selectbox(
        "Select a commonly asked question or customize your own:",
        PREDEFINED_QUESTIONS, index=0, key="question_select",
        disabled=input_disabled,
        help=disabled_reason or "Select or type your question."
    )
    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0] or input_disabled
    custom_question = st.text_input(
        "Enter your custom question here:", key="custom_question_input",
        disabled=custom_question_disabled,
        placeholder="Type question here..." if not custom_question_disabled else (disabled_reason or "Select 'Enter my own...' above")
    )
with q_col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    ask_button = st.button(
        "Start Chat", type="primary", use_container_width=True,
        disabled=input_disabled, key="start_chat_button",
        help=disabled_reason or "Click to start the chat."
    )

initial_question_to_ask = ""
if selected_question == PREDEFINED_QUESTIONS[0]:
    initial_question_to_ask = custom_question.strip()
else:
    initial_question_to_ask = selected_question
st.divider()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Initialized chat history in session state.")

# --- Display Existing Chat Messages ---
st.subheader("Explore your Findings")
chat_container = st.container(height=600, border=False) # Example: Set height for scrollability
with chat_container:
    if not st.session_state.messages:
        if not input_disabled:
            st.info("👆 Start the conversation by asking an initial question above.")
        else:
            st.warning(f"Application is not ready. Please check the following: {disabled_reason}", icon="⚠️")
            logging.warning(f"App not ready on initial load. Reason: {disabled_reason}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "debug_info" in message and message["debug_info"]:
                    with st.expander("Show Debugging Info"):
                        st.markdown(f"**Initial Retrieval:** {message['debug_info'].get('original_doc_count', 'N/A')} chunks found.")
                        st.markdown(f"**After Filtering:** {message['debug_info'].get('filtered_doc_count', 'N/A')} chunks remained.")
                        st.markdown(f"**Relevance Grade:** {str(message['debug_info'].get('relevance_grade', 'N/A')).upper()}")
                if "original_docs" in message and message["original_docs"]:
                    with st.expander(f"Show Initially Retrieved Context ({len(message['original_docs'])} chunks)"):
                        for i, doc_item_hist in enumerate(message["original_docs"]): # Renamed
                            page_num_hist = doc_item_hist.metadata.get('page', 'N/A') # Renamed
                            file_name_hist = doc_item_hist.metadata.get("file", "N/A") # Renamed
                            st.markdown(f"**Chunk {i+1} (`{file_name_hist}` - Page {page_num_hist}):**")
                            st.markdown(f"> {doc_item_hist.page_content}")
                            st.divider()
                if "individual_answers" in message and message["individual_answers"]:
                    substantive_answers_hist = { # Renamed
                        fname: ans for fname, ans in message["individual_answers"].items()
                        if not ("No relevant information found" in ans or "An error occurred" in ans or "Could not generate" in ans or "no relevant context was identified" in ans)
                    }
                    expander_title_hist = f"Show Individual Answers per File ({len(substantive_answers_hist)} file(s) with substantive answers)" # Renamed
                    total_processed_hist = len(message["individual_answers"]) # Renamed
                    if len(substantive_answers_hist) < total_processed_hist:
                        expander_title_hist += f" / {total_processed_hist} total processed"
                    with st.expander(expander_title_hist):
                        if message["individual_answers"]:
                            for filename_hist, answer_hist in message["individual_answers"].items(): # Renamed
                                st.markdown(f"**Answer from `{filename_hist}`:**")
                                st.markdown(answer_hist)
                                st.divider()
                        else:
                            st.markdown("No individual answers were recorded in this message.")


# --- Function to Run Graph and Handle Response --- ASYNC Version
async def run_graph_and_get_response(question_to_ask_async: str) -> Optional[Dict]: # Renamed
    if not langgraph_app or not base_retriever or not files_to_process_in_graph: # Global checks
        logging.error("Attempted to run graph but prerequisites not met (run_graph_and_get_response).")
        return {
            "content": "Cannot process request. Workflow, retriever, or file selection is not ready. Please check setup.",
            "original_docs": [], "individual_answers": {}, 
            "debug_info": {"relevance_grade": "SETUP_ERROR"}
        }
    final_state_async = None # Renamed
    response_data_async = { # Renamed
        "content": "An unexpected error occurred during async workflow execution.",
        "original_docs": [], "individual_answers": {}, "debug_info": {}
    }
    try:
        logging.info(f"Preparing to run graph (async) for: '{question_to_ask_async[:50]}...' across {len(files_to_process_in_graph)} file(s)")
        initial_state_dict_async = { # Renamed
            "question": question_to_ask_async, "base_retriever": base_retriever,
            "files_to_process": files_to_process_in_graph, "generation": None,
            "documents": [], "original_documents": [], "filtered_documents": [],
            "relevance_grade": None, "individual_answers": {}
        }
        logging.info(f"Invoking LangGraph app (async) with initial state for question: '{question_to_ask_async}'")
        final_state_async = await langgraph_app.ainvoke(initial_state_dict_async)
        logging.info("LangGraph app invocation (async) finished.")
        logging.info("Processing results from async graph execution...")

        final_answer_async = final_state_async.get("generation", None) # Renamed
        original_docs_async = final_state_async.get('original_documents', []) # Renamed
        filtered_docs_async = final_state_async.get('filtered_documents', []) # Renamed
        relevance_grade_async = final_state_async.get('relevance_grade', 'N/A') # Renamed
        individual_answers_from_state_async = final_state_async.get('individual_answers', {}) # Renamed

        response_data_async["debug_info"] = {
            "original_doc_count": len(original_docs_async) if original_docs_async else 0,
            "filtered_doc_count": len(filtered_docs_async) if filtered_docs_async else 0,
            "relevance_grade": str(relevance_grade_async)
        }
        response_data_async["original_docs"] = original_docs_async
        response_data_async["individual_answers"] = individual_answers_from_state_async

        if final_answer_async:
            response_data_async["content"] = final_answer_async
            logging.info("Graph execution successful (async), final answer generated.")
        elif relevance_grade_async == "no" or \
             (relevance_grade_async == "yes" and not final_state_async.get("documents")) or \
             (relevance_grade_async == "skipped" and not final_state_async.get("documents")):
            potential_combine_message_async = final_state_async.get("generation") # Renamed
            if potential_combine_message_async and ("Could not find relevant information" in potential_combine_message_async or "Errors occurred" in potential_combine_message_async or "No substantive individual answers available" in potential_combine_message_async) :
                 response_data_async["content"] = potential_combine_message_async
                 logging.info(f"Graph execution (async) finished. Combine_answers provided specific message: {potential_combine_message_async}")
            else:
                no_info_message_async = "Could not find relevant information across the selected documents to answer the question after filtering and grading." # Renamed
                response_data_async["content"] = no_info_message_async
                logging.info(f"Graph execution (async) finished, but no relevant information found after grading (Grade: {relevance_grade_async}).")
        else:
            response_data_async["content"] = final_answer_async or "An issue occurred during the final answer generation. The graph might have ended prematurely or the 'generation' field was not populated."
            logging.warning(f"Graph execution (async) finished, but final answer might be missing or incomplete. Grade: {relevance_grade_async}, Final Answer: {final_answer_async}")
    except Exception as e:
        logging.exception("Error running LangGraph workflow (async)")
        response_data_async["content"] = f"An critical error occurred during the asynchronous workflow execution: {e}"
        if final_state_async:
            response_data_async["debug_info"] = {
                "original_doc_count": len(final_state_async.get('original_documents', [])),
                "filtered_doc_count": len(final_state_async.get('filtered_documents', [])),
                "relevance_grade": str(final_state_async.get('relevance_grade', 'WORKFLOW_ERROR'))
            }
            response_data_async["original_docs"] = final_state_async.get('original_documents', [])
            response_data_async["individual_answers"] = final_state_async.get('individual_answers', {})
        else:
             response_data_async["debug_info"]["relevance_grade"] = "WORKFLOW_ERROR_EARLY"
        return response_data_async
    return response_data_async


# --- Handle Initial Question Submission (Button Click) --- ASYNC Call
if ask_button and initial_question_to_ask:
    logging.info(f"Initial question submitted via button: '{initial_question_to_ask}'")
    st.session_state.messages.append({"role": "user", "content": initial_question_to_ask})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(initial_question_to_ask)
        with st.chat_message("assistant"):
            placeholder_ui_init = st.empty() # Renamed
            response_data_ui_init = None # Renamed
            status_label_init = f"Analyzing '{initial_question_to_ask[:30]}...' across {len(files_to_process_in_graph)} file(s)" # Renamed
            with st.status(status_label_init, expanded=True) as status_ui_init: # Renamed
                placeholder_ui_init.markdown("🧠 Thinking...")
                status_ui_init.write("Invoking AI Assistant workflow...")
                try:
                    response_data_ui_init = asyncio.run(run_graph_and_get_response(initial_question_to_ask)) # ASYNC CALL
                    grade_check_init = response_data_ui_init.get("debug_info", {}).get("relevance_grade") # Renamed
                    content_check_init = response_data_ui_init.get("content", "").lower() # Renamed
                    if grade_check_init == "SETUP_ERROR": status_ui_init.update(label="Setup Error", state="error", expanded=False)
                    elif "error occurred" in content_check_init: status_ui_init.update(label="Processing Error", state="error", expanded=False)
                    elif "could not find relevant information" in content_check_init or "no relevant information found" in content_check_init: status_ui_init.update(label="No relevant information found.", state="complete", expanded=False)
                    elif response_data_ui_init: status_ui_init.update(label="Analysis complete!", state="complete", expanded=False)
                    else: status_ui_init.update(label="Unexpected error.", state="error", expanded=False)
                except Exception as e:
                    logging.exception("Exception during asyncio.run(run_graph_and_get_response) for initial q")
                    response_data_ui_init = {"content": f"A critical error occurred: {e}", "original_docs": [], "individual_answers": {}, "debug_info": {"relevance_grade":"ASYNC_RUN_ERROR"}}
                    status_ui_init.update(label="Critical Error", state="error", expanded=False)
            placeholder_ui_init.empty()

            if response_data_ui_init:
                st.markdown(response_data_ui_init["content"])
                # Display expanders (same logic as before, ensure variable names match response_data_ui_init)
                if "debug_info" in response_data_ui_init and response_data_ui_init["debug_info"]:
                    with st.expander("Show Debugging Info (Initial)"): # Added (Initial) for clarity
                        st.markdown(f"**Initial Retrieval:** {response_data_ui_init['debug_info'].get('original_doc_count', 'N/A')} chunks.")
                        st.markdown(f"**After Filtering:** {response_data_ui_init['debug_info'].get('filtered_doc_count', 'N/A')} chunks.")
                        st.markdown(f"**Relevance Grade:** {str(response_data_ui_init['debug_info'].get('relevance_grade', 'N/A')).upper()}")
                if "original_docs" in response_data_ui_init and response_data_ui_init["original_docs"]:
                     with st.expander(f"Show Initially Retrieved Context ({len(response_data_ui_init['original_docs'])} chunks) (Initial)"):
                        for i, doc_item_exp in enumerate(response_data_ui_init["original_docs"]):
                            st.markdown(f"**Chunk {i+1} (`{doc_item_exp.metadata.get('file', 'N/A')}` - Pg {doc_item_exp.metadata.get('page', 'N/A')}):**")
                            st.markdown(f"> {doc_item_exp.page_content}")
                            st.divider()
                if "individual_answers" in response_data_ui_init and response_data_ui_init["individual_answers"]:
                    # Filter substantive answers for the expander title
                    sub_ans_init = {k: v for k, v in response_data_ui_init["individual_answers"].items() if not ("No relevant information found" in v or "An error occurred" in v or "Could not generate" in v)}
                    exp_title_init = f"Show Individual Answers per File ({len(sub_ans_init)} substantive / {len(response_data_ui_init['individual_answers'])} total) (Initial)"
                    with st.expander(exp_title_init):
                        for fname, ans in response_data_ui_init["individual_answers"].items():
                            st.markdown(f"**Answer from `{fname}`:**"); st.markdown(ans); st.divider()

                st.session_state.messages.append({
                    "role": "assistant", "content": response_data_ui_init["content"],
                    "original_docs": response_data_ui_init.get("original_docs", []),
                    "individual_answers": response_data_ui_init.get("individual_answers", {}),
                    "debug_info": response_data_ui_init.get("debug_info", {})
                })
            else:
                error_msg_ui_init = "Critical error processing initial question (no response data)." # Renamed
                st.error(error_msg_ui_init)
                st.session_state.messages.append({"role": "assistant", "content": error_msg_ui_init})
    # Consider st.rerun() if inputs should be cleared, but test carefully.

# --- Handle Follow-up Questions via Chat Input --- ASYNC Call
chat_input_disabled = input_disabled or not st.session_state.messages # Re-evaluate for chat input
if prompt_follow_up := st.chat_input("Ask a follow-up question...", disabled=chat_input_disabled, key="follow_up_input"): # Renamed
    logging.info(f"Follow-up question submitted via chat input: '{prompt_follow_up}'")
    st.session_state.messages.append({"role": "user", "content": prompt_follow_up})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt_follow_up)
        with st.chat_message("assistant"):
            placeholder_ui_fu = st.empty() # Renamed
            response_data_ui_fu = None # Renamed
            status_label_fu = f"Analyzing '{prompt_follow_up[:30]}...' across {len(files_to_process_in_graph)} file(s)" # Renamed
            with st.status(status_label_fu, expanded=True) as status_ui_fu: # Renamed
                placeholder_ui_fu.markdown("🧠 Thinking...")
                status_ui_fu.write("Invoking AI Assistant workflow for follow-up...")
                try:
                    response_data_ui_fu = asyncio.run(run_graph_and_get_response(prompt_follow_up)) # ASYNC CALL
                    grade_check_fu = response_data_ui_fu.get("debug_info", {}).get("relevance_grade") # Renamed
                    content_check_fu = response_data_ui_fu.get("content", "").lower() # Renamed
                    if grade_check_fu == "SETUP_ERROR": status_ui_fu.update(label="Setup Error", state="error", expanded=False)
                    elif "error occurred" in content_check_fu: status_ui_fu.update(label="Processing Error", state="error", expanded=False)
                    elif "could not find relevant information" in content_check_fu or "no relevant information found" in content_check_fu: status_ui_fu.update(label="No relevant information found.", state="complete", expanded=False)
                    elif response_data_ui_fu: status_ui_fu.update(label="Analysis complete!", state="complete", expanded=False)
                    else: status_ui_fu.update(label="Unexpected error.", state="error", expanded=False)
                except Exception as e:
                    logging.exception("Exception during asyncio.run(run_graph_and_get_response) for follow-up q")
                    response_data_ui_fu = {"content": f"A critical error occurred: {e}", "original_docs": [], "individual_answers": {}, "debug_info": {"relevance_grade":"ASYNC_RUN_ERROR"}}
                    status_ui_fu.update(label="Critical Error", state="error", expanded=False)
            placeholder_ui_fu.empty()

            if response_data_ui_fu:
                st.markdown(response_data_ui_fu["content"])
                # Display expanders (same logic, ensure variable names match response_data_ui_fu)
                if "debug_info" in response_data_ui_fu and response_data_ui_fu["debug_info"]:
                    with st.expander("Show Debugging Info (Follow-up)"): # Added (Follow-up)
                        st.markdown(f"**Initial Retrieval:** {response_data_ui_fu['debug_info'].get('original_doc_count', 'N/A')} chunks.")
                        st.markdown(f"**After Filtering:** {response_data_ui_fu['debug_info'].get('filtered_doc_count', 'N/A')} chunks.")
                        st.markdown(f"**Relevance Grade:** {str(response_data_ui_fu['debug_info'].get('relevance_grade', 'N/A')).upper()}")
                if "original_docs" in response_data_ui_fu and response_data_ui_fu["original_docs"]:
                     with st.expander(f"Show Initially Retrieved Context ({len(response_data_ui_fu['original_docs'])} chunks) (Follow-up)"):
                        for i, doc_item_exp_fu in enumerate(response_data_ui_fu["original_docs"]):
                            st.markdown(f"**Chunk {i+1} (`{doc_item_exp_fu.metadata.get('file', 'N/A')}` - Pg {doc_item_exp_fu.metadata.get('page', 'N/A')}):**")
                            st.markdown(f"> {doc_item_exp_fu.page_content}")
                            st.divider()
                if "individual_answers" in response_data_ui_fu and response_data_ui_fu["individual_answers"]:
                    sub_ans_fu = {k: v for k, v in response_data_ui_fu["individual_answers"].items() if not ("No relevant information found" in v or "An error occurred" in v or "Could not generate" in v)}
                    exp_title_fu = f"Show Individual Answers per File ({len(sub_ans_fu)} substantive / {len(response_data_ui_fu['individual_answers'])} total) (Follow-up)"
                    with st.expander(exp_title_fu):
                        for fname, ans in response_data_ui_fu["individual_answers"].items():
                            st.markdown(f"**Answer from `{fname}`:**"); st.markdown(ans); st.divider()

                st.session_state.messages.append({
                    "role": "assistant", "content": response_data_ui_fu["content"],
                    "original_docs": response_data_ui_fu.get("original_docs", []),
                    "individual_answers": response_data_ui_fu.get("individual_answers", {}),
                    "debug_info": response_data_ui_fu.get("debug_info", {})
                })
            else:
                error_msg_ui_fu = "Critical error processing follow-up question (no response data)." # Renamed
                st.error(error_msg_ui_fu)
                st.session_state.messages.append({"role": "assistant", "content": error_msg_ui_fu})

# --- Log DEBUG_CAPTURE if any items were added ---
if DEBUG_CAPTURE:
    logging.info("--- Captured Debug Timings/Messages ---")
    for item in DEBUG_CAPTURE:
        logging.info(item)
    DEBUG_CAPTURE.clear() # Clear after logging for next run

# --- Footer (Sidebar) ---
st.sidebar.divider()
st.sidebar.caption("Powered by LangChain, LangGraph, Azure OpenAI, FAISS, and Streamlit")
