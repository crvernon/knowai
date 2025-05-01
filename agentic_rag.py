import streamlit as st

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

import fitz  # PyMuPDF
import faiss
import os
import hashlib
# Updated OpenAI/Azure imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS # Using community import
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # For function calling output schema
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence
import operator

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

# --- Azure OpenAI Configuration ---
# Load environment variables (especially Azure credentials)
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
# Use a default deployment name if not set, or handle error if critical
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # Defaulting to gpt-4o deployment name
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large") # Defaulting embeddings deployment
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Defaulting API version

# --- Check for Azure Credentials (AFTER set_page_config) ---
azure_creds_valid = True
if not api_key or not azure_endpoint:
    st.error("Azure OpenAI API key or endpoint not found. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.", icon="🚨")
    azure_creds_valid = False
    # Consider stopping execution if credentials are required for core functionality
    # st.stop() # Uncomment if the app cannot proceed without credentials

# Constants for predefined questions
PREDEFINED_QUESTIONS = [
    "Enter my own question...",
    "Does the plan include vegetation management? If so, how much money is allocated to it?",
    "Does the plan include undergrounding? If so, how much money is allocated to it?",
    "Does the plan include PSPS? If so, how much money is allocated to it?",
    "How frequently does the utility perform asset inspections?",
    "Are there generation considerations, such as derating solar PV during smoky conditions?"
]

# --- Helper Functions (PDF Processing - Unchanged) ---

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
                pages_content.append((page_num + 1, text))
        doc.close()
        return pages_content
    except Exception as e:
        st.error(f"Error reading PDF: {e}", icon="📄")
        return None

# --- Caching (Retriever Creation - Modified for Azure & k=50) ---
@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def get_retriever_from_pdf(file_content_hash):
    """Processes PDF, creates Azure embeddings, builds FAISS store, returns retriever."""
    # Ensure Azure creds are available before proceeding within cached function if needed
    # Note: This check might be redundant if the app stops earlier, but good practice.
    if not api_key or not azure_endpoint:
        st.error("Azure credentials not available for PDF processing.", icon="🚨")
        return None

    uploaded_file = st.session_state.get(f"file_{file_content_hash}", None)
    if not uploaded_file:
        st.error("Could not find file content in session state for processing.", icon="🔒")
        return None
    try:
        st.write(f"Processing file: {uploaded_file.name}")
        pages_data = extract_pages_from_pdf(uploaded_file)
        if not pages_data:
            st.warning("Could not extract text from the PDF.", icon="⚠️")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, length_function=len
        )
        all_docs = []
        for page_num, page_text in pages_data:
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                doc = Document(page_content=chunk, metadata={"page": page_num})
                all_docs.append(doc)

        if not all_docs:
            st.warning("Could not create any text chunks from the PDF.", icon="⚠️")
            return None
        st.write(f"Split PDF into {len(all_docs)} chunks across {len(pages_data)} pages.")

        # --- Instantiate Azure Embeddings ---
        st.write(f"Using Azure embeddings deployment: {embeddings_deployment}")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment, # Use specific deployment name for embeddings
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )

        st.write("Building FAISS index with Azure embeddings...")
        # Using the community import for FAISS
        vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)
        st.write("FAISS index built successfully.")
        # *** Set k=50 ***
        return vectorstore.as_retriever(search_kwargs={'k': 50})
    except Exception as e:
        st.error(f"Error processing PDF and building vector store: {e}", icon="❌")
        st.exception(e) # Show full traceback for debugging
        return None

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents (potentially filtered)
        original_documents: list of documents retrieved initially
        retriever: The FAISS retriever object (passed at initialization)
    """
    question: str
    generation: str
    documents: List[Document]
    original_documents: List[Document] # Store initially retrieved docs
    retriever: object # Store the retriever itself in the state

# --- LangGraph Nodes (Modified for Azure LLM & Filtering) ---

def retrieve_docs(state: GraphState) -> GraphState:
    """
    Retrieves documents from the vector store based on the question. Stores them in 'original_documents'.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Adds 'original_documents' key to state.
    """
    st.write(f"--- Retrieving Top {state['retriever'].search_kwargs.get('k', 'N/A')} Documents ---")
    question = state["question"]
    retriever = state["retriever"] # Get retriever from state
    try:
        # Retrieve k=50 documents
        documents = retriever.get_relevant_documents(question)
        st.write(f"Retrieved {len(documents)} documents initially.")
        # Store these initial documents separately
        return {"original_documents": documents, "question": question}
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        st.exception(e)
        return {"original_documents": [], "question": question} # Return empty list on error

# --- NEW NODE: Filter Documents ---
class FilteredDocs(BaseModel):
    """Schema for the function call to identify indices of documents to keep."""
    keep_indices: List[int] = Field(description="List of zero-based indices of the documents that should be kept (i.e., are not structural or irrelevant).")

def filter_documents(state: GraphState) -> GraphState:
    """
    Filters the initially retrieved documents to remove structurally irrelevant ones (ToC, etc.).

    Args:
        state (dict): The current graph state including 'original_documents'.

    Returns:
        dict: Updates 'documents' key in state with the filtered list.
    """
    st.write("--- Filtering Structurally Irrelevant Documents ---")
    original_documents = state["original_documents"]
    question = state["question"] # Keep question for context if needed

    if not original_documents:
        st.write("No documents to filter.")
        return {"documents": []} # Pass empty list forward

    # Ensure Azure creds are valid before making LLM call
    if not azure_creds_valid:
         st.warning("Skipping filtering due to missing Azure credentials.")
         return {"documents": original_documents} # Pass original docs if no creds

    try:
        # --- Instantiate Azure LLM for Filtering ---
        st.write(f"Using Azure chat deployment for filtering: {deployment}")
        filtering_llm = AzureChatOpenAI(
            temperature=0, # Low temperature for deterministic filtering
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        ).bind_tools([FilteredDocs], tool_choice="FilteredDocs") # Bind the function call schema

        # Prepare context for the filtering prompt
        doc_context = ""
        for i, doc in enumerate(original_documents):
            doc_context += f"--- Document Index {i} (Page {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}\n\n"

        # Create a prompt for the filtering LLM
        filtering_prompt = PromptTemplate(
            template="""You are an expert document analyst. Your task is to identify document chunks that are primarily structural elements like Table of Contents entries, page headers/footers, indices, or reference lists, which are unlikely to contain direct answers to content-based questions about a technical report.

            Analyze the following document chunks provided below, each marked with a 'Document Index'.

            Document Chunks:
            {doc_context}

            Based on the content of each chunk, identify the indices of the documents that ARE LIKELY TO CONTAIN SUBSTANTIVE CONTENT (prose, data, analysis, findings) and should be kept for further analysis. Ignore chunks that are just lists of section titles with page numbers (like a ToC), repetitive headers/footers, or bibliographies unless they uniquely contain relevant information not present elsewhere.

            Use the 'FilteredDocs' tool to return the list of indices to keep. Return ONLY the indices to keep.
            """,
            input_variables=["doc_context"],
        )

        # Filtering Chain
        filtering_chain = filtering_prompt | filtering_llm

        # Invoke the chain
        response = filtering_chain.invoke({"doc_context": doc_context})

        # Process the LLM response
        kept_indices = []
        if response.tool_calls and response.tool_calls[0]['name'] == 'FilteredDocs':
            kept_indices = response.tool_calls[0]['args'].get('keep_indices', [])
            st.write(f"LLM identified {len(kept_indices)} documents to keep out of {len(original_documents)}.")
        else:
             st.warning("Filtering LLM did not return expected format. Keeping all documents for safety.")
             # Keep all original documents if filtering fails
             return {"documents": original_documents}

        # Create the filtered list
        filtered_docs = [original_documents[i] for i in kept_indices if i < len(original_documents)] # Ensure index is valid
        st.write(f"Filtered down to {len(filtered_docs)} documents.")
        return {"documents": filtered_docs} # Update state with filtered docs

    except Exception as e:
        st.error(f"Error during document filtering: {e}")
        st.exception(e)
        # If filtering fails, pass all original documents to the next step
        st.warning("Filtering failed. Passing all original documents to grading.")
        return {"documents": original_documents}


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the *filtered* documents are relevant to the question using Azure LLM.

    Args:
        state (dict): The current graph state including 'documents' (filtered list).

    Returns:
        dict: Updates 'documents' key based on relevance grade.
    """
    st.write("--- Grading Filtered Document Relevance (Azure LLM) ---")
    question = state["question"]
    documents = state["documents"] # Use the potentially filtered documents

    if not documents:
         st.write("No documents remaining after filtering to grade.")
         return {"documents": [], "question": question} # Skip grading if no docs

    # Ensure Azure creds are valid before making LLM call
    if not azure_creds_valid:
         st.warning("Skipping grading due to missing Azure credentials.")
         # If we can't grade, should we assume relevant or irrelevant? Assume relevant for now.
         return {"documents": documents, "question": question}

    try:
        # --- Instantiate Azure LLM for Grading ---
        st.write(f"Using Azure chat deployment for grading: {deployment}")
        llm = AzureChatOpenAI(
            temperature=0,
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        )

        # Define the function structure for the LLM to call
        grading_function = {
            "name": "grade_relevance",
            "description": "Determine if the provided document chunks are relevant to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relevant": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Whether the documents contain information relevant to answering the question.",
                    }
                },
                "required": ["relevant"],
            },
        }

        # Create a prompt for the LLM
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of potentially relevant document chunks to a user question. These chunks have already been pre-filtered to remove obvious structural elements.
            Analyze the following document chunks based on the user question:
            \n ------- \n
            User Question: {question}
            \n ------- \n
            Document Chunks:
            {documents}
            \n ------- \n
            Determine if these remaining chunks contain information that can directly answer the user's question. Respond using the 'grade_relevance' function call. Ensure your response strictly adheres to the function call format.
            """,
            input_variables=["question", "documents"],
        )

        # Format documents for the prompt
        doc_texts = "\n\n".join([d.page_content for d in documents])

        # Chain for grading
        grader_chain = prompt | llm.bind_tools([grading_function], tool_choice="grade_relevance")

        # Invoke the chain
        response = grader_chain.invoke({"question": question, "documents": doc_texts})

        # Process the LLM response
        is_relevant = "no" # Default to not relevant
        if response.tool_calls and response.tool_calls[0]['name'] == 'grade_relevance':
            is_relevant = response.tool_calls[0]['args'].get('relevant', 'no')
            st.write(f"Relevance Grade: {is_relevant.upper()}")
        else:
             st.warning("Grader LLM did not return expected function call format.")
             is_relevant = "no"


        if is_relevant == "yes":
            st.write("Decision: Documents are relevant. Proceeding to generation.")
            # Keep the currently filtered & relevant documents
            return {"documents": documents, "question": question}
        else:
            st.write("Decision: Documents not relevant enough. Skipping generation.")
            # Clear the documents list as they are not relevant
            return {"documents": [], "question": question}

    except Exception as e:
        st.error(f"Error during document grading: {e}")
        st.exception(e)
        return {"documents": [], "question": question} # Halt on error


def generate_answer(state: GraphState) -> GraphState:
    """
    Generates an answer using the final list of relevant documents, using Azure LLM.

    Args:
        state (dict): The current graph state including 'documents' (filtered and graded list).

    Returns:
        dict: New key added to state, generation, that contains LLM generation.
    """
    st.write("--- Generating Answer (Azure LLM) ---")
    question = state["question"]
    documents = state["documents"] # These are the filtered and graded documents

    if not documents:
        st.warning("Generation called with no relevant documents.")
        return {"generation": None} # Explicitly set generation to None

    # Ensure Azure creds are valid before making LLM call
    if not azure_creds_valid:
         st.warning("Skipping generation due to missing Azure credentials.")
         return {"generation": "Could not generate answer due to missing credentials."}

    try:
        # Format documents for the prompt
        context = "\n\n".join([f"Page {d.metadata.get('page', 'N/A')}:\n{d.page_content}" for d in documents])

        # Prompt template asking for quotes and page number citations.
        prompt_template = """You are an assistant analyzing a technical report. Your task is to answer the user's question based *only* on the provided context chunks, which have been filtered for relevance.

        Follow these instructions carefully:
        1.  Thoroughly read the provided context chunks, paying attention to the page numbers cited before each chunk.
        2.  Answer the user's question directly based *only* on the information present in the context.
        3.  Summarize the key points from the context that directly address the question.
        4.  **Crucially:** When providing supporting details or evidence, include direct quotes from the relevant context chunk. After each quote, cite the page number in parentheses, like this: `"The relevant text snippet..." (Page X)`. Use the page number provided before the chunk in the context section.
        5.  If the question asks about specific details (like financial allocations, frequencies, specific procedures) and that detail is *not* found in the context, explicitly state that the information is not provided in the context. Do not make assumptions or provide external knowledge.
        6.  Structure your answer clearly. Start with a direct answer if possible, then provide the supporting details and quotes with page number citations. Conclude by addressing any parts of the question that couldn't be answered from the context. If no relevant information is found in the provided context to answer the question, state that clearly.

        Context from Document (with Page Numbers):
        {context}

        Question: {question}

        Detailed Answer with Quotes and Citations:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # --- Instantiate Azure LLM for Generation ---
        st.write(f"Using Azure chat deployment for generation: {deployment}")
        llm = AzureChatOpenAI(
            temperature=0.1,
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        )

        # RAG generation chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run the chain
        generation = rag_chain.invoke({"context": context, "question": question})
        st.write("Answer generated.")
        return {"generation": generation}

    except Exception as e:
        st.error(f"Error during answer generation: {e}")
        st.exception(e)
        return {"generation": f"An error occurred while generating the answer: {e}"}


# --- LangGraph Conditional Edge ---

def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer or end the process based on document relevance
    (i.e., if the 'documents' list is non-empty after grading).

    Args:
        state (dict): The current graph state.

    Returns:
        str: Decision ("generate" or "end_no_relevance").
    """
    st.write("--- Checking Relevance for Generation ---")
    # Decision is based on the 'documents' list *after* the grading node potentially cleared it.
    documents = state["documents"]

    if not documents:
        st.write("Decision: No relevant documents found after grading. Ending.")
        return "end_no_relevance"
    else:
        st.write("Decision: Relevant documents found. Proceeding to generate.")
        return "generate"

# --- Build LangGraph ---

# Define the workflow graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("filter_documents", filter_documents) # Add new filter node
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate_answer)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "filter_documents") # Retrieve -> Filter
workflow.add_edge("filter_documents", "grade_documents") # Filter -> Grade
workflow.add_conditional_edges(
    "grade_documents", # Condition based on grading result
    decide_to_generate,
    {
        "generate": "generate", # If relevant, go to generate
        "end_no_relevance": END # If not relevant, end
    }
)
workflow.add_edge("generate", END)

# Compile the graph only if Azure credentials are valid
langgraph_app = None
if azure_creds_valid:
    try:
        langgraph_app = workflow.compile()
        # st.success("LangGraph compiled successfully.")
    except Exception as e:
        st.error(f"Failed to compile LangGraph: {e}")
        st.exception(e)
        # langgraph_app remains None


# --- Streamlit UI ---

# Title and Markdown should come after set_page_config and credential check
st.title("📄 Agentic AI Approach with RAG")
st.markdown("""
Upload a technical report (PDF) and ask questions about its content.
This version uses **Azure OpenAI** via LangGraph, retrieves **50 chunks**, **filters** for structural irrelevance, grades for question relevance, and generates answers.
Answers include direct quotes with page number citations.
""")

# --- File Upload ---
# Only show uploader if creds are valid, otherwise PDF processing won't work
if azure_creds_valid:
    uploaded_file = st.file_uploader("Choose a PDF file (max 8MB recommended)", type="pdf")
else:
    uploaded_file = None
    st.warning("File uploader disabled until Azure credentials are provided.")


# --- Session State Initialization ---
if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = {}

retriever = None
# Only attempt to get retriever if Azure creds are valid and file is uploaded
if azure_creds_valid and uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()

    if file_hash not in st.session_state.file_hashes:
        st.session_state.file_hashes[file_hash] = uploaded_file.name
        st.session_state[f"file_{file_hash}"] = uploaded_file

    retriever = get_retriever_from_pdf(file_hash)


# Clear session state if no file is present
if uploaded_file is None:
    st.session_state.file_hashes = {}
    keys_to_delete = [key for key in st.session_state if key.startswith("file_")]
    for key in keys_to_delete:
        del st.session_state[key]


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
        disabled=not azure_creds_valid # Disable if no creds
    )
    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0] or not azure_creds_valid
    custom_question = st.text_input(
        "Enter your custom question here:",
        key="custom_question_input",
        disabled=custom_question_disabled,
        placeholder="Type your question if you selected the first option above..." if azure_creds_valid else "Disabled due to missing credentials"
    )

with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    # Disable button if retriever or compiled graph is not available, or if creds are missing
    ask_button_disabled = not retriever or not langgraph_app or not azure_creds_valid
    ask_button = st.button("Get Answer", type="primary", use_container_width=True, disabled=ask_button_disabled)

if selected_question == PREDEFINED_QUESTIONS[0]:
    final_question = custom_question
else:
    final_question = selected_question

# --- Answer Generation and Display (Using LangGraph) ---
st.divider()
st.subheader("Answer")

answer_placeholder = st.empty()
context_placeholder = st.empty()

# Only run if button clicked AND prerequisites met
if ask_button and retriever and final_question and langgraph_app and azure_creds_valid:
    answer_placeholder.info("Running graph with Azure OpenAI (Retrieval -> Filter -> Grade -> Generate)...", icon="⏳")
    context_placeholder.empty()

    try:
        # Define the initial state to pass to the graph
        initial_state = {"question": final_question, "retriever": retriever}

        # Invoke the LangGraph app
        final_state = langgraph_app.invoke(initial_state)

        # Extract the final answer
        answer = final_state.get("generation", None) # Will be None if generation node wasn't reached

        if answer:
             answer_placeholder.success(answer) # Display the generated answer
        else:
            # Handle the case where generation was skipped (due to filtering or grading)
            answer_placeholder.warning("Could not find relevant information in the document to answer the question after filtering and grading.", icon="⚠️")


        # Display the context: Show the *original* 50 docs retrieved for transparency
        original_docs_retrieved = final_state.get('original_documents', [])

        if original_docs_retrieved:
            with context_placeholder.expander(f"Show Initially Retrieved Context ({len(original_docs_retrieved)} chunks)"):
                # Optionally show the filtered docs too, if needed for debugging
                # filtered_docs_state = final_state.get('documents', []) # Docs after filtering/grading
                # st.write(f"Documents remaining after filtering/grading: {len(filtered_docs_state)}")

                for i, doc in enumerate(original_docs_retrieved):
                    page_num = doc.metadata.get('page', 'N/A')
                    st.markdown(f"**Chunk {i+1} (Initial Retrieval - Page {page_num}):**")
                    st.markdown(f"> {doc.page_content}")
                    st.divider()
        elif final_state.get("generation") is None:
             context_placeholder.info("No documents were retrieved initially.")


    except Exception as e:
        answer_placeholder.error(f"Error running LangGraph: {e}", icon="❌")
        st.exception(e) # Show full traceback in Streamlit for debugging

# Handle other button click conditions
elif ask_button and not retriever:
    answer_placeholder.warning("Please upload and process a PDF file first (ensure Azure credentials are set).", icon="⚠️")
elif ask_button and not final_question:
     answer_placeholder.warning("Please enter or select a question.", icon="⚠️")
elif ask_button and (not langgraph_app or not azure_creds_valid):
     if not azure_creds_valid:
         answer_placeholder.error("Cannot get answer: Azure credentials missing.", icon="🚨")
     elif not langgraph_app:
         answer_placeholder.error("Cannot get answer: LangGraph application failed to compile.", icon="🚨")
# Initial state message
else:
    if azure_creds_valid:
        answer_placeholder.info("Upload a PDF and ask a question to get started.")
    # The missing credential message is handled by the initial check


# --- Footer ---
st.divider()
st.caption("Powered by LangChain, LangGraph, Azure OpenAI, FAISS, and Streamlit")
