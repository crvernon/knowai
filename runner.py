import streamlit as st
import base64
import os
import time
import asyncio
import logging
from typing import List, Dict, Optional, Any

# Assuming 'knowai' package is installed or in PYTHONPATH
try:
    from knowai import KnowAIAgent
except ImportError:
    st.error("Failed to import KnowAIAgent. Ensure 'knowai' package is installed and in your PYTHONPATH.")
    st.stop()

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>""", unsafe_allow_html=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s')
# logging.getLogger("langchain").setLevel(logging.WARNING) # Example to make langchain less verbose
# logging.getLogger("langgraph").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Asset Paths (Update these to your actual paths) ---
LOGO_PATH_PNNL = "./img/pnnl-logo.png"
LOGO_PATH_GDO = "./img/gdo-logo.png"
LOGO_PATH_WRR = "./img/wrr-pnnl-logo.svg"
SVG_PATH_TITLE = "./img/projectLogo.svg"
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "test_faiss_store") # IMPORTANT
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), ".env") # IMPORTANT

# --- Helper function to display logos ---
def display_logo(column, path, alt_text):
    try:
        if os.path.exists(path): column.image(path, use_container_width=True)
        else: column.warning(f"Logo not found: {path}", icon="🖼️")
    except Exception as e: logging.error(f"Error loading logo {path}: {e}"); column.error("Logo error.")

# --- UI: Logos and Title ---
logo_col1, logo_col2, _, logo_col3 = st.columns([1, 1.4, 4, 1.4])
with logo_col1: display_logo(logo_col1, LOGO_PATH_PNNL, "PNNL")
with logo_col2: display_logo(logo_col2, LOGO_PATH_GDO, "GDO")
with logo_col3: display_logo(logo_col3, LOGO_PATH_WRR, "WRR")

st.markdown("<div style='text-align:center;'></br></br></br></div>", unsafe_allow_html=True)
try:
    if os.path.exists(SVG_PATH_TITLE):
        with open(SVG_PATH_TITLE, "rb") as f: svg_base64 = base64.b64encode(f.read()).decode()
        st.markdown(f"<div style='display:flex; align-items:center; justify-content:center; margin-top: -50px;'><img src='data:image/svg+xml;base64,{svg_base64}' width='32' height='32' style='margin-right:8px;'/><h2 style='margin:0;'>Wildfire Mitigation Plans Database</h2></div>", unsafe_allow_html=True)
    else: st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True)
except Exception as e: logging.error(f"SVG Error: {e}"); st.markdown("<h2 style='text-align:center; margin-top: -50px;'>Wildfire Mitigation Plans Database</h2>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><h4>-- AI Assistant (KnowAI Powered) --</h4><em>Your AI Assistant to find answers across multiple reports!</em></div>", unsafe_allow_html=True)


# --- Agent Initialization (once per session) ---
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.agent_initialized = False
    st.session_state.initialization_error = None
    try:
        # Ensure vector store path exists (dummy for example if not)
        if not os.path.exists(VECTORSTORE_PATH):
            os.makedirs(VECTORSTORE_PATH, exist_ok=True)
            with open(os.path.join(VECTORSTORE_PATH, "index.faiss"), "w") as f: f.write("")
            with open(os.path.join(VECTORSTORE_PATH, "index.pkl"), "w") as f: f.write("")
            logging.info(f"Created dummy vector store directory/files at {VECTORSTORE_PATH} for first run.")
        
        st.session_state.agent = KnowAIAgent(
            vectorstore_path=VECTORSTORE_PATH,
            env_file_path=ENV_FILE_PATH
        )
        st.session_state.agent_initialized = True
        logging.info("KnowAIAgent initialized successfully.")
    except Exception as e:
        st.session_state.initialization_error = f"Failed to initialize AI Agent: {e}"
        logging.exception("Error initializing KnowAIAgent:")
        st.error(st.session_state.initialization_error)

if not st.session_state.get("agent_initialized", False):
    if st.session_state.get("initialization_error"):
        st.error(f"Agent Initialization Error: {st.session_state.initialization_error}")
    else:
        st.warning("AI Agent is initializing...")
    st.stop() # Stop further execution if agent isn't ready

# --- Session State Initialization for Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores {"role": "user/assistant", "content": "...", "data": {}}
if "selected_files" not in st.session_state:
    st.session_state.selected_files = [] # List of filenames
if "available_files" not in st.session_state: # Placeholder for actual file discovery
    st.session_state.available_files = ["ID_OR_Idaho_Power_2022.pdf", "OR_City_of_Bandon_2024.pdf", "Example_Doc_A.pdf", "Example_Doc_B.pdf", "Example_Doc_C.pdf"] 
    # In a real app, populate this by scanning a directory or from a database

# --- UI: Sidebar for File Selection & Options ---
with st.sidebar:
    st.header("Configuration")
    st.session_state.selected_files = st.multiselect(
        "Select Files to Analyze:",
        options=st.session_state.available_files,
        default=st.session_state.selected_files, # Persist selection
        help="Choose the documents you want the AI to consider."
    )
    st.session_state.bypass_individual_gen = st.checkbox(
        "Bypass individual answer generation (direct synthesis from raw docs)", 
        value=st.session_state.get("bypass_individual_gen", False), # Persist selection
        help="If checked, skips generating answers per file and directly synthesizes from all retrieved document chunks. Can be faster but might be less detailed for per-file insights."
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        # Also clear agent's internal conversation history
        if hasattr(st.session_state.agent, 'session_state') and st.session_state.agent.session_state is not None:
            st.session_state.agent.session_state['conversation_history'] = []
        logging.info("Chat history cleared.")
        st.rerun()

# --- UI: Main Area for Initial Question ---
st.subheader("Ask Initial Question")
PREDEFINED_QUESTIONS = [
    "Enter my own question...", "What are common vegetation management practices?",
    "Discuss Public Safety Power Shutoff (PSPS) strategies.", "Summarize asset inspection frequencies."
]
q_col1, q_col2 = st.columns([3, 1])
with q_col1:
    selected_question = st.selectbox("Quick Questions:", PREDEFINED_QUESTIONS, index=0, key="question_select")
    custom_question_disabled = selected_question != PREDEFINED_QUESTIONS[0]
    custom_question = st.text_input("Your Question:", key="custom_question_input", disabled=custom_question_disabled, placeholder="Type here if 'Enter my own question...' is selected")
with q_col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    start_chat_button = st.button("Start Chat", type="primary", use_container_width=True, key="start_chat_button")

initial_question_to_ask = (custom_question.strip() if selected_question == PREDEFINED_QUESTIONS[0] else selected_question) if start_chat_button else ""

st.divider()

# --- UI: Chat History Display ---
st.subheader("Chat")
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info("Select files and ask a question to begin.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display expanders for assistant messages if data exists
            if message["role"] == "assistant" and "data" in message and message["data"]:
                agent_response_data = message["data"]
                
                # Expander for Individual File Answers
                individual_answers = agent_response_data.get("individual_answers")
                if individual_answers and not agent_response_data.get("bypass_individual_generation"):
                     substantive_answers = {fname: ans for fname, ans in individual_answers.items() if "No relevant information found" not in ans and "An error occurred" not in ans}
                     exp_title = f"Individual Answers ({len(substantive_answers)} file(s) with info / {len(individual_answers)} total)"
                     with st.expander(exp_title):
                          for fname, ans in individual_answers.items():
                              st.markdown(f"**`{fname}`:**\n{ans}")
                              st.divider()
                
                # Expander for Retrieved Documents (Context)
                docs_by_file = agent_response_data.get("documents_by_file")
                raw_docs_text = agent_response_data.get("raw_documents_for_synthesis")

                if bypass_flag := agent_response_data.get("bypass_individual_generation"):
                    if raw_docs_text and raw_docs_text != "No documents were retrieved for the selected files and question." and raw_docs_text != "No files were selected for processing.":
                        with st.expander("Show Raw Extracted Context (Bypass Mode)"):
                            st.text_area("Raw Context", value=raw_docs_text, height=300, disabled=True)
                elif docs_by_file: # Standard path, show documents by file
                    total_chunks = sum(len(doc_list) for doc_list in docs_by_file.values())
                    with st.expander(f"Show Retrieved Context ({total_chunks} chunks across {len(docs_by_file)} files)"):
                        for filename, doc_list in docs_by_file.items():
                            if doc_list:
                                st.markdown(f"**From `{filename}` ({len(doc_list)} chunks):**")
                                for i, doc in enumerate(doc_list[:3]): # Show first 3 chunks per file
                                    page = doc.metadata.get('page', 'N/A')
                                    st.caption(f"Chunk {i+1} (Page {page}):")
                                    st.markdown(f"> {doc.page_content}")
                                if len(doc_list) > 3: st.caption("...")
                                st.divider()


# --- Asynchronous Function to Handle Query Processing ---
async def handle_user_query(question: str):
    if not st.session_state.selected_files:
        st.warning("Please select at least one file to analyze.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container: # Redisplay user message in the flow
        with st.chat_message("user"):
            st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            agent_response_data = await st.session_state.agent.process_turn(
                user_question=question,
                selected_files=st.session_state.selected_files,
                bypass_individual_gen=st.session_state.bypass_individual_gen
            )
            placeholder.empty()
            st.markdown(agent_response_data["generation"])
            st.session_state.messages.append({
                "role": "assistant", 
                "content": agent_response_data["generation"],
                "data": agent_response_data # Store all returned data for expanders
            })
        except Exception as e:
            placeholder.empty()
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "data": {}})
            logging.exception("Error during agent.process_turn:")
    st.rerun() # Rerun to update the display of messages and clear input


# --- Handle Initial Question or Follow-up ---
if initial_question_to_ask: # From "Start Chat" button
    asyncio.run(handle_user_query(initial_question_to_ask))
    # Clear the initial question inputs after processing to prevent resubmission on non-chat reruns
    st.session_state.custom_question_input = "" 
    # Note: selectbox doesn't have a direct clear, will reset on next interaction or rerun if needed

if follow_up_prompt := st.chat_input("Ask a follow-up question...", key="follow_up_chat_input", disabled=(not st.session_state.agent_initialized)):
    asyncio.run(handle_user_query(follow_up_prompt))

