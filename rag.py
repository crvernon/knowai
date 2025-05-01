import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
import hashlib
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document class
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = "gpt-4o"
openai_api_version = "2024-02-01"


# --- Configuration ---
# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Check for OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to a .env file.", icon="🚨")
    st.stop()

# Constants for predefined questions
PREDEFINED_QUESTIONS = [
    "Enter my own question...",
    "Does the plan include vegetation management? If so, how much money is allocated to it?",
    "Does the plan include undergrounding? If so, how much money is allocated to it?",
    "Does the plan include PSPS? If so, how much money is allocated to it?",
    "How frequently does the utility perform asset inspections?",
    "Are there generation considerations, such as derating solar PV during smoky conditions?"
]

# --- Helper Functions ---

def extract_pages_from_pdf(pdf_file):
    """Extracts text page by page from an uploaded PDF file, returning a list of (page_number, text)."""
    pages_content = []
    try:
        # Read bytes from uploaded file
        pdf_bytes = pdf_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text: # Only add pages with actual text content
                 # Store page number (1-based index) and text
                pages_content.append((page_num + 1, text))
        doc.close()
        return pages_content
    except Exception as e:
        st.error(f"Error reading PDF: {e}", icon="📄")
        return None

# --- Caching ---
@st.cache_resource(show_spinner="Processing PDF and building knowledge base...")
def get_retriever_from_pdf(file_content_hash):
    """
    Processes the PDF content (identified by its hash):
    1. Extracts text page by page.
    2. Splits text into chunks, preserving page number metadata.
    3. Creates embeddings using OpenAI.
    4. Builds a FAISS vector store using LangChain Documents.
    5. Returns a retriever object.

    Args:
        file_content_hash (str): The MD5 hash of the uploaded PDF file's content.

    Returns:
        FAISS retriever object or None if processing fails.
    """
    uploaded_file = st.session_state.get(f"file_{file_content_hash}", None)
    if not uploaded_file:
        st.error("Could not find file content in session state for processing.", icon="🔒")
        return None

    try:
        st.write(f"Processing file: {uploaded_file.name}")
        # 1. Extract Text Page by Page
        pages_data = extract_pages_from_pdf(uploaded_file)
        if not pages_data:
            st.warning("Could not extract text from the PDF.", icon="⚠️")
            return None

        # 2. Split Text and Create Documents with Metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )

        all_docs = []
        for page_num, page_text in pages_data:
            # Split text of the current page
            page_chunks = text_splitter.split_text(page_text)
            # Create Document objects for each chunk, adding page number to metadata
            for chunk in page_chunks:
                doc = Document(page_content=chunk, metadata={"page": page_num})
                all_docs.append(doc)

        if not all_docs:
            st.warning("Could not create any text chunks from the PDF.", icon="⚠️")
            return None
        st.write(f"Split PDF into {len(all_docs)} chunks across {len(pages_data)} pages.")

        # 3. Create Embeddings
        if not os.getenv("OPENAI_API_KEY"):
             st.error("OpenAI API key is missing.", icon="🚨")
             return None
        # embeddings = OpenAIEmbeddings()
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )

        # 4. Build FAISS Vector Store from Documents
        st.write("Building FAISS index...")
        # Use from_documents to include metadata
        vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)
        st.write("FAISS index built successfully.")

        # 5. Return Retriever
        return vectorstore.as_retriever(search_kwargs={'k': 20}) # Retrieve top 5 relevant chunks

    except Exception as e:
        st.error(f"Error processing PDF and building vector store: {e}", icon="❌")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 Standard RAG Approach")
st.markdown("""
Upload a technical report (PDF) and ask questions about its content.
The system uses OpenAI's GPT-4o and embeddings to find relevant information within the document. Page numbers are shown for context.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a PDF file (max 8MB recommended)", type="pdf")

# --- Session State Initialization ---
if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = {}

retriever = None
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()

    if file_hash not in st.session_state.file_hashes:
        st.session_state.file_hashes[file_hash] = uploaded_file.name
        st.session_state[f"file_{file_hash}"] = uploaded_file

    retriever = get_retriever_from_pdf(file_hash)

else:
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
        key="question_select"
    )

    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0]
    custom_question = st.text_input(
        "Enter your custom question here:",
        key="custom_question_input",
        disabled=custom_question_disabled,
        placeholder="Type your question if you selected the first option above..."
    )

with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    ask_button = st.button("Get Answer", type="primary", use_container_width=True, disabled=(not retriever))

if selected_question == PREDEFINED_QUESTIONS[0]:
    final_question = custom_question
else:
    final_question = selected_question

# --- Answer Generation and Display ---
st.divider()
st.subheader("Answer")

answer_placeholder = st.empty()
context_placeholder = st.empty()

if ask_button and retriever and final_question:
    answer_placeholder.info("Generating answer...", icon="⏳")
    context_placeholder.empty()

    try:
        # llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        llm = AzureChatOpenAI(
            model_name="gpt-4o", 
            temperature=0.1, 
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
        )

        prompt_template = """You are an assistant analyzing a technical report. Your task is to answer the user's question based *only* on the provided context chunks.

        Follow these instructions carefully:
        1.  Thoroughly read the provided context chunks.
        2.  Answer the user's question directly based *only* on the information present in the context.
        3.  Summarize the key points from the context that directly address the question.
        4.  If the context contains specific relevant sentences or phrases (e.g., definitions, descriptions of actions, specific numbers), include them in your answer, perhaps as quotes, to provide detail and evidence.
        5.  If the question asks about specific details (like financial allocations, frequencies, specific procedures) and that detail is *not* found in the context, explicitly state that the information is not provided in the context. Do not make assumptions or provide external knowledge.
        6.  Structure your answer clearly. Start with a direct answer if possible, then provide the supporting details/quotes from the context. Conclude by addressing any parts of the question that couldn't be answered from the context.

        Context:
        {context}

        Question: {question}

        Detailed Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(final_question)
        answer_placeholder.success(answer)

        # *** MODIFIED CONTEXT DISPLAY ***
        # Retrieve context again to display it with page numbers
        relevant_docs = retriever.get_relevant_documents(final_question)
        with context_placeholder.expander("Show Context Used"):
            for i, doc in enumerate(relevant_docs):
                # Access page number from metadata
                page_num = doc.metadata.get('page', 'N/A') # Get page number, default to N/A if not found
                st.markdown(f"**Chunk {i+1} (from Page {page_num}):**") # Display page number
                st.markdown(f"> {doc.page_content}")
                st.divider()

    except Exception as e:
        answer_placeholder.error(f"Error generating answer: {e}", icon="❌")

elif ask_button and not retriever:
    answer_placeholder.warning("Please upload and process a PDF file first.", icon="⚠️")
elif ask_button and not final_question:
     answer_placeholder.warning("Please enter or select a question.", icon="⚠️")
else:
    answer_placeholder.info("Upload a PDF and ask a question to get started.")

# --- Footer ---
st.divider()
st.caption("Powered by LangChain, OpenAI, FAISS, and Streamlit")
