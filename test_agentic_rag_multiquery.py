import streamlit as st
import base64
import os

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

import fitz  # PyMuPDF
import faiss
import os
import hashlib
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
import operator
from collections import defaultdict

# Import vectorstore loading function
from build_vectorstore import load_vectorstore
from build_vectorstore import list_vectorstore_files
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

# --- Logging Setup ---
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# --- Azure OpenAI Configuration ---
load_dotenv()

k_chunks = 50 # Number of chunks MultiQueryRetriever aims for IN TOTAL across all its generated queries

api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")


# --- Logo Display ---
logo_col1, logo_col2, title_col = st.columns([1, 1, 4])
logo_path1 = "./img/pnnl-logo.png"
logo_path2 = "./img/gdo-logo.png"
# svg_path = "./img/projectLogo.svg" # Defined later for title

try:
    with logo_col1:
        st.image(logo_path1, use_container_width=True)
except FileNotFoundError:
    with logo_col1:
        st.warning(f"PNNL logo not found at {logo_path1}", icon="🖼️")
except Exception as e:
    with logo_col1:
        st.error(f"Error loading PNNL logo: {e}", icon="🚨")

try:
    with logo_col2:
        st.image(logo_path2, use_container_width=True)
except FileNotFoundError:
    with logo_col2:
        st.warning(f"GDO logo not found at {logo_path2}", icon="🖼️")
except Exception as e:
    with logo_col2:
        st.error(f"Error loading GDO logo: {e}", icon="🚨")


st.divider()


# --- Check for Azure Credentials ---
azure_creds_valid = True
if not api_key or not azure_endpoint:
    st.error("Azure OpenAI API key or endpoint not found. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.", icon="🚨")
    azure_creds_valid = False


# --- Load Base Retriever and List Files ---
vectorstore_path = "test_faiss_store"
base_retriever = None
vectorstore = None
files_in_store = []

if azure_creds_valid:
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )
        # Load the vectorstore first to list files
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        files_in_store = list_vectorstore_files(vectorstore)
        # Create the base retriever from the loaded vectorstore
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})
        st.sidebar.success(f"Loaded vector store with {len(files_in_store)} file(s).")
    except Exception as e:
        st.error(f"Error loading vectorstore '{vectorstore_path}': {e}")
        azure_creds_valid = False # Treat as invalid if store loading fails
else:
    st.sidebar.warning("Skipping vector store load due to missing Azure credentials.")


# Constants for predefined questions
PREDEFINED_QUESTIONS = [
    "Enter my own question...",
    "Does the plan include vegetation management? If so, how much money is allocated to it?",
    "Does the plan include undergrounding? If so, how much money is allocated to it?",
    "Does the plan include PSPS? If so, how much money is allocated to it?",
    "How frequently does the utility perform asset inspections?",
    "Are there generation considerations, such as derating solar PV during smoky conditions?"
]

# --- File Selector UI ---
st.sidebar.subheader("Select Source File(s)")
file_options = ["All"] + files_in_store
selected_files_user_choice = st.sidebar.multiselect(
    "Files to Query",
    file_options,
    default=["All"],
    help="Select the specific files to search within, or 'All' to search across the entire loaded vector store."
)

# Determine the actual list of files to process based on user selection
files_to_process_in_graph = []
if "All" in selected_files_user_choice or not selected_files_user_choice:
    files_to_process_in_graph = files_in_store # Use all files if "All" or empty selection
else:
    files_to_process_in_graph = [f for f in selected_files_user_choice if f != "All"]

if azure_creds_valid and not files_to_process_in_graph and files_in_store:
     st.sidebar.warning("No specific files selected, and 'All' was not chosen. Defaulting to search all available files.")
     files_to_process_in_graph = files_in_store
elif not files_in_store and azure_creds_valid:
    st.sidebar.error("No files found in the vector store index.")
    azure_creds_valid = False # Cannot proceed without files
elif files_to_process_in_graph:
    st.sidebar.info(f"Will query within: {', '.join(files_to_process_in_graph)}")


# --- Helper Functions (PDF Processing - Unchanged, kept for potential future use) ---
def extract_pages_from_pdf(pdf_file):
    """Extracts text page by page from an uploaded PDF file, returning a list of (page_number, text)."""
    pages_content = []
    try:
        pdf_bytes = pdf_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text:
                # Include filename in metadata from the start
                pages_content.append((page_num + 1, text, pdf_file.name))
        doc.close()
        return pages_content
    except Exception as e:
        st.error(f"Error reading PDF: {e}", icon="📄")
        return None

# --- Caching (Retriever Creation - Kept for potential future use, but not primary path now) ---
@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def get_base_retriever_from_pdf(file_hash, uploaded_file_name): # Pass filename for metadata
    """Processes PDF, creates Azure embeddings, builds FAISS store, returns BASE retriever."""
    # ... (rest of the function remains largely the same, but ensure metadata includes filename)
    if not api_key or not azure_endpoint:
        st.error("Azure credentials not available for PDF processing.", icon="🚨")
        return None

    uploaded_file = st.session_state.get(f"file_{file_hash}", None)
    if not uploaded_file:
        st.error("Could not find file content in session state for processing.", icon="🔒")
        return None
    try:
        st.write(f"Processing file: {uploaded_file.name}") # Use the actual filename
        pages_data = extract_pages_from_pdf(uploaded_file) # Will include filename
        if not pages_data:
            st.warning("Could not extract text from the PDF.", icon="⚠️")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        all_docs = []
        for page_num, page_text, filename in pages_data: # Unpack filename
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                 # Add filename to metadata here
                doc = Document(page_content=chunk, metadata={"page": page_num, "file": filename})
                all_docs.append(doc)

        # ... (rest of the function: check all_docs, instantiate embeddings, build FAISS)
        if not all_docs:
            st.warning("Could not create any text chunks from the PDF.", icon="⚠️")
            return None
        st.write(f"Split PDF into {len(all_docs)} chunks across {len(pages_data)} pages.")

        st.write(f"Translating the user query into embeddings...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )

        st.write("Building FAISS index with Azure embeddings...")
        vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)
        st.write("FAISS index built successfully.")

        return vectorstore.as_retriever(search_kwargs={'k': 50})
    except Exception as e:
        st.error(f"Error processing PDF and building vector store: {e}", icon="❌")
        st.exception(e)
        return None


# --- LangGraph State Definition (MODIFIED) ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The final combined LLM generation.
        documents: List of documents filtered and graded as relevant across all selected files.
        original_documents: List of documents initially retrieved by MultiQueryRetriever.
        base_retriever: The base FAISS retriever object.
        files_to_process: List of filenames selected by the user (or all files).
        individual_answers: A dictionary mapping filename to the answer generated for that file.
    """
    question: str
    generation: Optional[str] # Final combined answer
    documents: List[Document] # Relevant docs across all files
    original_documents: List[Document] # Initially retrieved docs
    base_retriever: object
    files_to_process: List[str] # Files determined from user selection
    individual_answers: Dict[str, str] # File-specific answers


# --- LangGraph Nodes ---

# Node 1: Retrieve documents using MultiQueryRetriever
def retrieve_docs_multi_query(state: GraphState) -> GraphState:
    """
    Uses MultiQueryRetriever, filters by the files_to_process list stored in the state.
    Stores combined, unique documents in 'original_documents'.
    """
    st.write(f"--- Retrieving Documents using Multi-Query ---")
    question = state["question"]
    base_retriever = state["base_retriever"]
    # files_to_process is already set in the initial state
    files_to_process = state["files_to_process"]

    if not base_retriever:
        st.error("Base retriever not available.")
        return {"original_documents": [], "question": question, "files_to_process": files_to_process}

    if not azure_creds_valid:
         st.warning("Skipping multi-query generation due to missing Azure credentials.")
         try:
             st.write("Falling back to simple retrieval...")
             documents = base_retriever.get_relevant_documents(question)
             # Filter even in fallback
             filtered_docs = [doc for doc in documents if doc.metadata.get("file") in files_to_process]
             st.write(f"Retrieved {len(filtered_docs)} documents via simple fallback from {files_to_process}.")
             return {"original_documents": filtered_docs, "question": question, "files_to_process": files_to_process}
         except Exception as e:
             st.error(f"Error during simple fallback retrieval: {e}")
             return {"original_documents": [], "question": question, "files_to_process": files_to_process}

    try:
        st.write(f"Generating query variations for: '{question}'")
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

        st.write(f"Retrieving relevant chunks from {len(files_to_process)} file(s)...")
        # Retrieve potentially relevant documents across the entire base store first
        unique_docs_unfiltered = multi_query_retriever.get_relevant_documents(query=question)
        st.write(f"Retrieved {len(unique_docs_unfiltered)} total unique chunks before file filtering.")

        # Filter the retrieved documents to include only those from the selected files
        unique_docs_filtered = [
            doc for doc in unique_docs_unfiltered
            if doc.metadata.get("file") in files_to_process
        ]
        st.write(f"Filtered down to {len(unique_docs_filtered)} chunks from the selected file(s): {', '.join(files_to_process)}")

        # Store these initial, file-filtered documents
        return {"original_documents": unique_docs_filtered, "question": question, "files_to_process": files_to_process}

    except Exception as e:
        st.error(f"Error during multi-query retrieval: {e}")
        st.exception(e)
        return {"original_documents": [], "question": question, "files_to_process": files_to_process}


# Node 2: Filter structural documents (Unchanged logic, operates on 'original_documents')
class FilteredDocs(BaseModel):
    """Schema for the function call to identify indices of documents to keep."""
    keep_indices: List[int] = Field(description="List of zero-based indices of the documents that should be kept (i.e., are not structural or irrelevant).")

def filter_documents(state: GraphState) -> GraphState:
    """
    Filters the initially retrieved documents (from MultiQuery, already file-filtered)
    to remove structurally irrelevant ones. Operates on 'original_documents'.
    """
    st.write("--- Filtering Structurally Irrelevant Documents ---")
    original_documents = state["original_documents"] # Docs from MultiQuery, filtered by file
    question = state["question"]

    if not original_documents:
        st.write("No documents retrieved to filter.")
        # Ensure 'documents' key exists, even if empty
        return {"documents": []}

    if not azure_creds_valid:
         st.warning("Skipping filtering due to missing Azure credentials.")
         return {"documents": original_documents} # Pass original docs through

    try:
        st.write(f"Filtering {len(original_documents)} retrieved chunks for structural content...")
        filtering_llm = AzureChatOpenAI(
            temperature=0,
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        ).bind_tools([FilteredDocs], tool_choice="FilteredDocs")

        doc_context = ""
        for i, doc in enumerate(original_documents):
            # Include filename in context for the LLM
            filename = doc.metadata.get('file', 'Unknown File')
            page_num = doc.metadata.get('page', 'N/A')
            doc_context += f"--- Document Index {i} (File: {filename}, Page: {page_num}) ---\n{doc.page_content}\n\n"

        filtering_prompt = PromptTemplate(
            template="""You are an expert document analyst. Your task is to identify document chunks that are primarily structural elements like Table of Contents entries, page headers/footers, indices, or reference lists, which are unlikely to contain direct answers to content-based questions about technical reports.

            Analyze the following document chunks provided below, each marked with a 'Document Index', 'File', and 'Page'.

            Document Chunks:
            {doc_context}

            Based on the content of each chunk, identify the indices of the documents that ARE LIKELY TO CONTAIN SUBSTANTIVE CONTENT (prose, data, analysis, findings) and should be kept for further analysis. Ignore chunks that are just lists of section titles with page numbers (like a ToC), repetitive headers/footers, or bibliographies unless they uniquely contain relevant information not present elsewhere.

            Use the 'FilteredDocs' tool to return the list of indices to keep. Return ONLY the indices to keep.
            """,
            input_variables=["doc_context"],
        )

        filtering_chain = filtering_prompt | filtering_llm
        response = filtering_chain.invoke({"doc_context": doc_context})

        kept_indices = []
        if response.tool_calls and response.tool_calls[0]['name'] == 'FilteredDocs':
            kept_indices = response.tool_calls[0]['args'].get('keep_indices', [])
            st.write(f"LLM identified {len(kept_indices)} documents to keep out of {len(original_documents)}.")
        else:
             st.warning("Filtering LLM did not return expected format. Keeping all documents for safety.")
             # Pass original docs through if filtering fails
             return {"documents": original_documents}

        filtered_docs = [original_documents[i] for i in kept_indices if i < len(original_documents)]
        st.write(f"Filtered down to {len(filtered_docs)} potentially substantive documents.")
        # Update the 'documents' key with the filtered list
        return {"documents": filtered_docs}

    except Exception as e:
        st.error(f"Error during document filtering: {e}")
        st.exception(e)
        st.warning("Filtering failed. Passing all original documents to grading.")
        # Pass original docs through if filtering errors
        return {"documents": original_documents}


# Node 3: Grade documents for relevance (Unchanged logic, operates on filtered 'documents')
def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the *filtered* documents are relevant to the question using Azure LLM.
    Operates on the 'documents' list produced by the filter_documents node.
    """
    st.write("--- Grading Filtered Document Relevance ---")
    question = state["question"]
    documents = state["documents"] # Use the potentially filtered documents

    if not documents:
         st.write("No documents remaining after filtering to grade.")
         # Ensure 'documents' key exists, even if empty
         return {"documents": []}

    if not azure_creds_valid:
         st.warning("Skipping grading due to missing Azure credentials.")
         # Assume relevant if cannot grade, pass through
         return {"documents": documents}

    try:
        st.write(f"Grading relevance of {len(documents)} filtered documents...")
        llm = AzureChatOpenAI(
            temperature=0,
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        )

        grading_function = {
            "name": "grade_relevance",
            "description": "Determine if the provided document chunks are relevant to the user's question.",
            "parameters": { "type": "object", "properties": { "relevant": { "type": "string", "enum": ["yes", "no"], "description": "Whether the documents contain information relevant to answering the question."}}, "required": ["relevant"]},
        }

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of potentially relevant document chunks to a user question. These chunks have already been pre-filtered to remove obvious structural elements.
            Analyze the following document chunks based on the user question:
            \n ------- \n
            User Question: {question}
            \n ------- \n
            Document Chunks (from potentially multiple files):
            {documents}
            \n ------- \n
            Determine if these remaining chunks collectively contain information that can directly answer the user's question. Respond using the 'grade_relevance' function call. Ensure your response strictly adheres to the function call format.
            """,
            input_variables=["question", "documents"],
        )

        # Include file and page in the context passed to the grader
        doc_texts_with_meta = "\n\n".join([f"--- File: {d.metadata.get('file', 'Unknown')}, Page: {d.metadata.get('page', 'N/A')} ---\n{d.page_content}" for d in documents])
        grader_chain = prompt | llm.bind_tools([grading_function], tool_choice="grade_relevance")
        response = grader_chain.invoke({"question": question, "documents": doc_texts_with_meta})

        is_relevant = "no"
        if response.tool_calls and response.tool_calls[0]['name'] == 'grade_relevance':
            is_relevant = response.tool_calls[0]['args'].get('relevant', 'no')
            st.write(f"Relevance Grade: {is_relevant.upper()}")
        else:
             st.warning("Grader LLM did not return expected function call format. Assuming 'no' relevance.")
             is_relevant = "no"

        if is_relevant == "yes":
            st.write("Decision: Relevant documents found across selected files. Proceeding to generate individual answers.")
            # Keep the relevant documents in the state
            return {"documents": documents}
        else:
            st.write("Decision: No relevant documents found after filtering and grading. Skipping generation.")
            # Clear documents if not relevant
            return {"documents": []}

    except Exception as e:
        st.error(f"Error during document grading: {e}")
        st.exception(e)
        # Clear documents on error to prevent proceeding
        return {"documents": []}


# Node 4: Generate Answer for each file (PROMPT MODIFIED)
def generate_individual_answers(state: GraphState) -> GraphState:
    """
    Generates an answer FOR EACH FILE based on its relevant documents.
    Uses the filtered and graded 'documents' list.
    Stores results in 'individual_answers'.
    """
    st.write("--- Generating Answer per File ---")
    question = state["question"]
    documents = state["documents"] # Filtered and graded documents
    files_to_process = state["files_to_process"]
    individual_answers = {}

    if not documents:
        st.warning("Generate Individual Answers called with no relevant documents.")
        return {"individual_answers": {}}

    if not azure_creds_valid:
         st.warning("Skipping generation due to missing Azure credentials.")
         for filename in files_to_process:
             individual_answers[filename] = "Could not generate answer due to missing credentials."
         return {"individual_answers": individual_answers}

    # Group documents by filename
    docs_by_file = defaultdict(list)
    for doc in documents:
        filename = doc.metadata.get("file")
        if filename:
            docs_by_file[filename].append(doc)

    st.write(f"Found relevant documents in {len(docs_by_file)} out of {len(files_to_process)} selected file(s).")

    # Define the prompt template for generating answer from a single file's context
    # *** MODIFIED INSTRUCTION #4 FOR CITATION FORMAT ***
    prompt_template_single_file = """You are an expert assistant analyzing a technical report. Your task is to answer the user's question comprehensively based *only* on the provided context chunks from a SINGLE FILE, which have been filtered for relevance.

    Follow these instructions carefully:
    1.  Thoroughly read all provided context chunks from this file, each marked with '--- Context from Page X ---'. Pay attention to the page numbers. Synthesize information found across multiple chunks *within this file* if they relate to the same aspect of the question.
    2.  Answer the user's question directly based *only* on the information present in the context from this file.
    3.  Identify the most relevant information or statement(s) within the context that directly address the question.
    4.  **Contextualized Quoting:** When presenting this key information, include a direct quote (enclosed in **double quotation marks**) of the most relevant sentence or phrase. **Crucially, also explain the surrounding context from the *same chunk* to clarify the quote's meaning or provide necessary background.** For example, instead of just citing, you might say: `The report discusses mitigation strategies, stating, "direct quote text..." ({filename}, Page X), which is part of a larger section detailing preventative measures.` Cite the **file name and page number in parentheses** immediately after the closing quotation mark, like this: `"direct quote text..." ({filename}, Page X)`.
    5.  If the question asks about specific details (like financial allocations, frequencies, specific procedures) and that detail is *not* found in this file's context, explicitly state that the information is not provided *in this specific file*. Do not make assumptions or provide external knowledge.
    6.  **Structure for Clarity:** Structure your answer logically for this file. Start with a direct summary answer if possible based on this file. Then, present the supporting details using the contextualized quoting method described above. Ensure the explanation connects the quote and its context clearly back to the user's original question. Conclude by addressing any parts of the question that couldn't be answered from this file's context. If no relevant information is found in the provided context *from this file* to answer the question, state that clearly.

    Context from Document Chunks (File: {filename}):
    {context}

    Question: {question}

    Detailed Answer based ONLY on File '{filename}' (with Contextualized Quotes and Citations in the format "quote..." (filename.pdf, Page X)):"""
    prompt_single_file = PromptTemplate(template=prompt_template_single_file, input_variables=["context", "question", "filename"])

    # Instantiate LLM for generation
    llm_generate = AzureChatOpenAI(
        temperature=0.1,
        api_key=api_key,
        openai_api_version=openai_api_version,
        azure_deployment=deployment,
        azure_endpoint=azure_endpoint,
        # max_tokens=1000 # Optional: adjust if needed
    )

    # Create the chain for single-file generation
    rag_chain_single_file = prompt_single_file | llm_generate | StrOutputParser()

    # Iterate through the files we intended to process
    for filename in files_to_process:
        st.write(f"--- Generating answer for file: {filename} ---")
        if filename in docs_by_file:
            file_docs = docs_by_file[filename]
            context = "\n\n".join([f"--- Context from Page {d.metadata.get('page', 'N/A')} ---\n{d.page_content}" for d in file_docs])
            st.write(f"Using {len(file_docs)} relevant chunks from this file.")
            try:
                # Invoke the chain for this specific file
                generation = rag_chain_single_file.invoke({
                    "context": context,
                    "question": question,
                    "filename": filename # Pass filename for the prompt
                })
                individual_answers[filename] = generation
                st.write(f"Generated answer for {filename}.")
            except Exception as e:
                st.error(f"Error generating answer for file {filename}: {e}")
                individual_answers[filename] = f"An error occurred while generating the answer for this file: {e}"
        else:
            # If a file was selected but had no relevant docs after filtering/grading
            st.write(f"No relevant documents found for file: {filename} after filtering/grading.")
            individual_answers[filename] = f"No relevant information found in the file '{filename}' to answer the question based on the retrieved context."

    # Return the dictionary of individual answers
    return {"individual_answers": individual_answers}


# Node 5: Combine individual answers (Unchanged)
def combine_answers(state: GraphState) -> GraphState:
    """
    Combines the individual answers generated for each file into a single, comprehensive answer.
    """
    st.write("--- Combining Answers from Individual Files ---")
    question = state["question"]
    individual_answers = state["individual_answers"]

    if not individual_answers:
        st.warning("Combine Answers node called with no individual answers to combine.")
        return {"generation": "No individual answers were generated to combine."}

    if not azure_creds_valid:
         st.warning("Skipping combination due to missing Azure credentials.")
         # Simple concatenation as fallback
         combined = f"Could not combine answers due to missing credentials. Individual findings:\n\n"
         for filename, answer in individual_answers.items():
              combined += f"--- Findings from {filename} ---\n{answer}\n\n"
         return {"generation": combined.strip()}

    # Format individual answers for the prompt
    formatted_answers = ""
    for filename, answer in individual_answers.items():
        formatted_answers += f"--- Answer based on file: {filename} ---\n{answer}\n\n"

    prompt_template_combine = """You are an expert synthesis assistant. Your task is to combine multiple answers, each generated from a different source file in response to the same user question. Create a single, comprehensive, and well-structured response.

    Follow these instructions VERY carefully:
    1.  **Goal:** Synthesize the information from all provided file-specific answers into ONE cohesive response to the original user question.
    2.  **Preserve ALL Details:** Do NOT summarize or omit any specific facts, figures, quotes, or findings mentioned in the individual answers. Ensure the final answer is as detailed as the sum of the individual answers. Pay attention to the specific citation format used in the individual answers (e.g., "quote..." (filename.pdf, Page X)).
    3.  **Attribute Clearly:** Explicitly mention the source file(s) for each piece of information or finding. Use parenthetical citations like `(Source: filename.pdf)` or integrate attribution naturally, e.g., `File 'report_A.pdf' states that... while 'report_B.pdf' adds...`. When incorporating direct quotes from the individual answers, retain their original citation format.
    4.  **Structure Logically:** Organize the combined answer logically based on the user's question. If the question has multiple parts, address each part, synthesizing information from relevant files for that part. Use headings or bullet points if it improves clarity.
    5.  **Handle Contradictions/Nuances:** If different files provide conflicting or slightly different information, present both findings and attribute them clearly (e.g., `File A reports X, whereas File B reports Y.`). Do not try to resolve conflicts unless the context explicitly allows it.
    6.  **Acknowledge Missing Info:** If an individual answer explicitly stated that information was *not* found in a specific file, reflect that in the combined answer where appropriate (e.g., `While File A provided details on X, File B did not contain information on this topic.`).
    7.  **Introduction and Conclusion:** Start with a brief introductory sentence acknowledging the question and the sources consulted. End with a concise summary if appropriate, reiterating the main findings across the files.

    User's Original Question:
    {question}

    Individual Answers Generated from Different Files (Note the citation format used within):
    {formatted_answers}

    Synthesized Comprehensive Answer (Preserving all details and attributing sources, retaining original quote citations):"""
    prompt_combine = PromptTemplate(template=prompt_template_combine, input_variables=["question", "formatted_answers"])

    try:
        st.write(f"Synthesizing answers from {len(individual_answers)} file(s)...")
        # Instantiate LLM for combination
        llm_combine = AzureChatOpenAI(
            temperature=0.0, # Low temp for faithful combination
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
            # max_tokens=2000 # May need more tokens for combined answer
        )

        # Create the combination chain
        combination_chain = prompt_combine | llm_combine | StrOutputParser()

        # Invoke the chain
        final_generation = combination_chain.invoke({
            "question": question,
            "formatted_answers": formatted_answers.strip() # Remove trailing newlines
        })
        st.write("Combined answer generated.")
        return {"generation": final_generation}

    except Exception as e:
        st.error(f"Error combining individual answers: {e}")
        st.exception(e)
        # Fallback: just concatenate if combination fails
        combined = f"An error occurred during synthesis. Raw findings:\n\n"
        for filename, answer in individual_answers.items():
             combined += f"--- Findings from {filename} ---\n{answer}\n\n"
        return {"generation": combined.strip()}


# --- LangGraph Conditional Edge (Unchanged) ---
def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to proceed to generating individual answers based on document relevance.
    Operates on the 'documents' list after grading.
    """
    st.write("--- Checking Relevance for Generation ---")
    # Check the 'documents' list which would have been cleared by 'grade_documents' if not relevant
    documents = state.get("documents", []) # Use .get for safety

    if not documents:
        st.write("Decision: No relevant documents found after grading. Ending.")
        return "end_no_relevance"
    else:
        st.write(f"Decision: {len(documents)} relevant documents found. Proceeding to generate individual answers.")
        # Route to the node that generates answers per file
        return "generate_individual"

# --- Build LangGraph (Unchanged Structure) ---

# Define the workflow graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve_docs_multi_query)
workflow.add_node("filter_documents", filter_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_individual", generate_individual_answers) # Node with updated prompt
workflow.add_node("combine_answers", combine_answers) # New node

# Build graph edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "filter_documents")
workflow.add_edge("filter_documents", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate, # Decision based on graded docs
    {
        # If relevant, go to the node that generates answers per file
        "generate_individual": "generate_individual",
        # If not relevant, end the process
        "end_no_relevance": END
    }
)
# After generating individual answers, combine them
workflow.add_edge("generate_individual", "combine_answers")
# The final combined answer marks the end
workflow.add_edge("combine_answers", END)

# Compile the graph
langgraph_app = None
if azure_creds_valid and base_retriever: # Ensure retriever is loaded too
    try:
        langgraph_app = workflow.compile()
        st.sidebar.success("Multi-file RAG workflow compiled.")
    except Exception as e:
        st.error(f"Failed to compile LangGraph: {e}")
        st.exception(e)
        langgraph_app = None # Ensure it's None if compilation fails
elif not azure_creds_valid:
    st.sidebar.error("LangGraph compilation skipped due to missing Azure credentials.")
elif not base_retriever:
     st.sidebar.error("LangGraph compilation skipped because the base vector store could not be loaded.")


# --- Streamlit UI (Unchanged) ---

# --- Inline SVG Logo with Title ---
svg_path = "./img/projectLogo.svg"
try:
    with open(svg_path, "rb") as f:
        svg_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<div style='display:flex; align-items:center; justify-content:center;'>"
        f"<img src='data:image/svg+xml;base64,{svg_base64}' width='32' height='32' style='margin-right:8px;'/>"
        f"<h3 style='margin:0;'>Wildfire Mitigation Plans Database</h3>"
        f"</div>",
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Error loading SVG logo: {e}")

st.markdown(
    "<div style='text-align:center;'><h4>-- AI Assistant Feature Demonstration --</h4></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;'><em>Your AI Assistant to find answers across multiple reports!</em></div>",
    unsafe_allow_html=True
)

# --- Question Input ---
st.divider()
st.subheader("Ask a Question")

col1, col2 = st.columns([3, 1])

with col1:
    selected_question = st.selectbox(
        "Select a predefined question or choose 'Enter my own question...':",
        PREDEFINED_QUESTIONS,
        index=0,
        key="question_select",
        # Disable if creds invalid OR if graph failed to compile OR no files to process
        disabled=not azure_creds_valid or not langgraph_app or not files_to_process_in_graph
    )
    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0] or not azure_creds_valid or not langgraph_app or not files_to_process_in_graph
    custom_question = st.text_input(
        "Enter your custom question here:",
        key="custom_question_input",
        disabled=custom_question_disabled,
        placeholder="Type your question..." if azure_creds_valid and langgraph_app and files_to_process_in_graph else "Disabled (check credentials, file selection, and vector store)."
    )

with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    # Disable button if base_retriever/langgraph_app missing, creds invalid, or no files selected
    ask_button_disabled = not base_retriever or not langgraph_app or not azure_creds_valid or not files_to_process_in_graph
    ask_button = st.button("Get Answer", type="primary", use_container_width=True, disabled=ask_button_disabled)

if selected_question == PREDEFINED_QUESTIONS[0]:
    final_question = custom_question
else:
    final_question = selected_question

# --- Answer Generation and Display ---
st.divider()
st.subheader("Answer")

answer_placeholder = st.empty()
context_placeholder = st.empty() # For initially retrieved docs
individual_answers_placeholder = st.empty() # For file-specific answers

# Only run if button clicked AND prerequisites met
if ask_button and base_retriever and final_question and langgraph_app and azure_creds_valid and files_to_process_in_graph:
    answer_placeholder.info(f"Starting analysis for '{final_question}' across {len(files_to_process_in_graph)} file(s)...", icon="⏳")
    context_placeholder.empty()
    individual_answers_placeholder.empty()

    # Monkey-patch st.write for progress updates
    progress_messages = []
    _original_st_write = st.write
    def progress_write(msg, *args, **kwargs):
        progress_messages.append(msg)
        answer_placeholder.info(f"⏳ {msg}")
    st.write = progress_write

    try:
        # Define the initial state, including the list of files to process
        initial_state = {
            "question": final_question,
            "base_retriever": base_retriever,
            "files_to_process": files_to_process_in_graph,
            # Initialize other fields that might be expected by nodes
            "documents": [],
            "original_documents": [],
            "individual_answers": {},
            "generation": None
        }
        # Invoke the LangGraph app
        final_state = langgraph_app.invoke(initial_state)

    except Exception as e:
        st.write = _original_st_write # Restore original write on error
        answer_placeholder.error(f"Error running LangGraph workflow: {e}", icon="❌")
        st.exception(e)
        final_state = None # Ensure no further processing happens

    finally:
        # ALWAYS restore st.write
        st.write = _original_st_write

    # --- Display Results ---
    if final_state:
        # Display the final combined answer
        final_answer = final_state.get("generation", None)
        if final_answer:
             # Use success for the final combined answer
             answer_placeholder.success(final_answer)
        elif not final_state.get("documents"): # Check if grading decided no relevance
            answer_placeholder.warning("Could not find relevant information across the selected documents to answer the question after filtering and grading.", icon="⚠️")
        else:
            answer_placeholder.error("An unexpected issue occurred. No final answer was generated.", icon="🚨") # Should ideally not happen if graph runs

        # Display the individual answers (if generated)
        individual_answers = final_state.get('individual_answers', {})
        if individual_answers:
             with individual_answers_placeholder.expander(f"Show Individual Answers per File ({len(individual_answers)} files)"):
                 for filename, answer in individual_answers.items():
                     st.markdown(f"**Answer from `{filename}`:**")
                     st.markdown(answer) # Display markdown formatted answer
                     st.divider()

        # Display the initially retrieved context (before filtering/grading)
        original_docs_retrieved = final_state.get('original_documents', [])
        if original_docs_retrieved:
            with context_placeholder.expander(f"Show Initially Retrieved Context ({len(original_docs_retrieved)} chunks across selected files)"):
                for i, doc in enumerate(original_docs_retrieved):
                    page_num = doc.metadata.get('page', 'N/A')
                    file_name = doc.metadata.get("file", "N/A")
                    st.markdown(f"**Chunk {i+1} (`{file_name}` - Page {page_num}):**")
                    st.markdown(f"> {doc.page_content}") # Use blockquote
                    st.divider()
        # Add message if no initial docs were retrieved but processing happened
        elif not final_state.get("generation") and not individual_answers : # Only if no answer AND no individual answers
             context_placeholder.info("No documents were retrieved initially by MultiQuery for the selected files.")


# Handle other button click conditions / initial state
elif ask_button:
    if not azure_creds_valid:
         answer_placeholder.error("Cannot get answer: Azure credentials missing.", icon="🚨")
    elif not base_retriever:
        answer_placeholder.warning("Cannot get answer: Vector store not loaded.", icon="⚠️")
    elif not files_to_process_in_graph:
         answer_placeholder.warning("Cannot get answer: No files selected or available to query.", icon="⚠️")
    elif not langgraph_app:
         answer_placeholder.error("Cannot get answer: RAG workflow failed to compile.", icon="🚨")
    elif not final_question:
         answer_placeholder.warning("Please enter or select a question.", icon="⚠️")
# Initial state message
else:
    if azure_creds_valid and base_retriever and files_to_process_in_graph and langgraph_app:
        answer_placeholder.info("Select or enter a question and click 'Get Answer' to search within the selected files.")
    elif not azure_creds_valid:
         answer_placeholder.info("Please configure Azure credentials to enable the application.")
    elif not base_retriever:
         answer_placeholder.info("Waiting for the vector store to load...") # Or show error if loading failed
    elif not files_to_process_in_graph:
         answer_placeholder.info("Please select files to query in the sidebar, or ensure files exist in the vector store.")
    elif not langgraph_app:
         answer_placeholder.info("Waiting for the RAG workflow to compile...") # Or show error if compilation failed


# --- Footer ---
# st.divider()
# st.caption("Powered by LangChain, LangGraph, Azure OpenAI, FAISS, and Streamlit")
