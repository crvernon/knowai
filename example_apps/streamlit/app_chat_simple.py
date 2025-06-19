import streamlit as st
import os
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Optional, Any
import pickle

# Assuming 'knowai' package is installed or in PYTHONPATH
try:
    from knowai import KnowAIAgent
except ImportError:
    st.error("Failed to import KnowAIAgent. Ensure 'knowai' package is installed and in your PYTHONPATH.")
    st.stop()

# --- Page Config ---
st.set_page_config(layout="centered", page_title="KnowAI Simple Chat")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s')

# --- Configuration (Update these paths as needed) ---
# IMPORTANT: Replace this with the ACTUAL path to your FAISS vector store
VECTORSTORE_PATH = Path(__file__).resolve().parents[2] /  "vectorstores" / "test_faiss_store"

logging.info(VECTORSTORE_PATH)

# IMPORTANT: Path to your .env file
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), ".env") 

# --- Title ---
st.title("KnowAI Simple Chat Assistant")
st.caption("Ask questions about your selected documents.")

# --- Agent Initialization (once per session) ---
if "simple_agent" not in st.session_state:
    st.session_state.simple_agent = None
    st.session_state.simple_agent_initialized = False
    st.session_state.simple_initialization_error = None
    with st.spinner("Initializing AI Agent... Please wait."):
        try:
            # Ensure vector store path exists (dummy for example if not)
            if not os.path.exists(VECTORSTORE_PATH):
                os.makedirs(VECTORSTORE_PATH, exist_ok=True)
                # Create dummy index files for FAISS.load_local to not immediately fail on path check
                # This DOES NOT create a functional index. Replace with a real one.
                with open(os.path.join(VECTORSTORE_PATH, "index.faiss"), "w") as f: f.write("")
                with open(os.path.join(VECTORSTORE_PATH, "index.pkl"), "w") as f: f.write("")
                logging.info(f"Created dummy vector store directory/files at {VECTORSTORE_PATH}.")
            
            st.session_state.simple_agent = KnowAIAgent(
                vectorstore_path=VECTORSTORE_PATH,
                env_file_path=ENV_FILE_PATH
            )
            st.session_state.simple_agent_initialized = True
            logging.info("KnowAIAgent (Simple) initialized successfully.")
        except Exception as e:
            st.session_state.simple_initialization_error = f"Failed to initialize AI Agent: {e}"
            logging.exception("Error initializing KnowAIAgent (Simple):")

if not st.session_state.get("simple_agent_initialized", False):
    if st.session_state.get("simple_initialization_error"):
        st.error(f"Agent Initialization Error: {st.session_state.simple_initialization_error}")
    else:
        st.warning("AI Agent is still initializing...")
    st.stop()

# --- Session State for Chat ---
if "simple_messages" not in st.session_state:
    st.session_state.simple_messages = [] # Stores {"role": "user/assistant", "content": "..."}
if "simple_selected_files" not in st.session_state:
    st.session_state.simple_selected_files = []
if "simple_available_files" not in st.session_state: #filename read from vectorstore
    index_pkl_path = VECTORSTORE_PATH / "index.pkl"
    
    if index_pkl_path.exists():
        try:
            with open(index_pkl_path, "rb") as f:
                data = pickle.load(f)
            
            filenames = set()
            
            # Handle your tuple with InMemoryDocstore
            if isinstance(data, tuple) and len(data) >= 1:
                docstore = data[0]
                docs_dict = getattr(docstore, "_dict", {})
                for doc in docs_dict.values():
                    metadata = getattr(doc, "metadata", {})
                    file_name = metadata.get("file") or metadata.get("source")
                    if file_name and file_name.endswith(".pdf"):
                        filenames.add(os.path.basename(file_name))
            else:
                logging.warning("index.pkl data format is unexpected.")

            st.session_state.simple_available_files = sorted(list(filenames))
            logging.info(f"Extracted {len(filenames)} filenames from index.pkl.")

        except Exception as e:
            logging.exception("Failed to parse index.pkl for file list.")
            st.session_state.simple_available_files = []
    else:
        logging.warning(f"index.pkl not found at {index_pkl_path}.")
        st.session_state.simple_available_files = []
        
# --- UI: Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.session_state.simple_selected_files = st.multiselect(
        "1. Select Files:",
        options=st.session_state.simple_available_files,
        default=st.session_state.simple_selected_files,
        key="simple_file_selector"
    )
    st.session_state.simple_bypass_gen = st.checkbox(
        "Bypass individual answer generation", 
        value=st.session_state.get("simple_bypass_gen", False),
        key="simple_bypass_checkbox",
        help="Synthesize directly from all retrieved document chunks."
    )
    if st.button("Clear Chat", key="simple_clear_chat"):
        st.session_state.simple_messages = []
        if hasattr(st.session_state.simple_agent, 'session_state') and st.session_state.simple_agent.session_state is not None:
             st.session_state.simple_agent.session_state['conversation_history'] = []
        logging.info("Simple chat history cleared.")
        st.rerun()

# --- Display Chat Messages ---
for message in st.session_state.simple_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Asynchronous Function to Handle Query ---
async def handle_simple_query(question: str):
    if not st.session_state.simple_selected_files:
        st.warning("Please select at least one file in the sidebar.")
        return

    st.session_state.simple_messages.append({"role": "user", "content": question})
    with st.chat_message("user"): # Display user message immediately
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            # Call the agent's process_turn method
            agent_response_data = await st.session_state.simple_agent.process_turn(
                user_question=question,
                selected_files=st.session_state.simple_selected_files,
                bypass_individual_gen=st.session_state.simple_bypass_gen
            )
            final_answer = agent_response_data.get("generation", "Sorry, I could not generate a response.")
            
            placeholder.empty()
            st.markdown(final_answer)
            st.session_state.simple_messages.append({"role": "assistant", "content": final_answer})
        
        except Exception as e:
            placeholder.empty()
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            st.session_state.simple_messages.append({"role": "assistant", "content": error_msg})
            logging.exception("Error during agent.process_turn in simple app:")
    # No st.rerun() here to allow chat input to clear naturally. 
    # Streamlit handles reruns when chat_input is used.

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about the selected files...", key="simple_chat_input", disabled=(not st.session_state.simple_agent_initialized)):
    asyncio.run(handle_simple_query(prompt))

