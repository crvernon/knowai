"""
Prompts used throughout the KnowAI agent.

This module contains all the prompt templates used by the KnowAI agent,
centralized for easy maintenance and customization.
"""

from langchain_core.prompts import PromptTemplate


# Individual File Answer Generation Prompt
INDIVIDUAL_ANSWER_PROMPT = """You are an expert assistant. Answer the user's question based ONLY on the provided context from a SINGLE FILE.
Context from File '{filename}' (Chunks from Pages X, Y, Z...):
{context}
Question: {question}
Detailed Answer (with citations like "quote..." ({filename}, Page X)):
IMPORTANT: Always use the exact filename '{filename}' in citations, never use generic terms like "file.pdf"."""

INDIVIDUAL_ANSWER_TEMPLATE = PromptTemplate(
    template=INDIVIDUAL_ANSWER_PROMPT,
    input_variables=["context", "question", "filename"]
)


def get_individual_answer_template_for_model(is_nano_model: bool = False) -> PromptTemplate:
    """
    Get the appropriate individual answer template based on the model type.
    
    Parameters
    ----------
    is_nano_model : bool
        Whether the model is a nano/small model that needs more explicit instructions
        
    Returns
    -------
    PromptTemplate
        The appropriate prompt template for individual answer generation
    """
    if is_nano_model:
        # More explicit prompt for nano models
        nano_prompt = """You are an expert assistant. Answer the user's question based ONLY on the provided context from a SINGLE FILE.
Context from File '{filename}' (Chunks from Pages X, Y, Z...):
{context}
Question: {question}
Detailed Answer (with citations like "quote..." ({filename}, Page X)):
CRITICAL INSTRUCTIONS FOR NANO MODEL:
1. ALWAYS use the exact filename '{filename}' in every citation
2. NEVER use generic terms like "file.pdf", "document.pdf", or "the file"
3. Format citations as: "quoted text..." ({filename}, Page X)
4. The filename '{filename}' must appear exactly as shown in every citation"""
        
        return PromptTemplate(
            template=nano_prompt,
            input_variables=["context", "question", "filename"]
        )
    else:
        return INDIVIDUAL_ANSWER_TEMPLATE


# Synthesis Prompts
PROCESSED_ANSWERS_SYNTHESIS_PROMPT = """You are an expert synthesis assistant. Combine PRE-PROCESSED answers.
Conversation History: {conversation_history}
User's CURRENT Question: {question}
Individual PRE-PROCESSED Answers: {formatted_answers_or_raw_docs}
Files with No Relevant Info: {files_no_info}
Files with Errors: {files_errors}
Instructions: Synthesize, preserve details & citations. IMPORTANT: Always use the exact filename from the source (e.g., "quote..." (actual_filename.pdf, Page X)), never use generic terms like "file.pdf". Attribute. Structure. Handle contradictions. Acknowledge files with no info/errors.
Synthesized Answer:"""

RAW_DOCUMENTS_SYNTHESIS_PROMPT = """You are an expert AI assistant. Answer CURRENT question based ONLY on RAW text chunks.
Conversation History: {conversation_history}
User's CURRENT Question: {question}
RAW Text Chunks: {formatted_answers_or_raw_docs}
Files with No Relevant Info (no chunks extracted): {files_no_info}
Files with Errors (extraction errors): {files_errors}
Instructions: Read raw text. Answer ONLY from raw text. Quote with citations. IMPORTANT: Always use the exact filename from the source (e.g., "quote..." (actual_filename.pdf, Page X)), never use generic terms like "file.pdf". If info not found, state it. Structure logically.
Synthesized Answer from RAW Docs:"""


def get_synthesis_prompt_template(bypass_individual_generation: bool) -> PromptTemplate:
    """
    Get the appropriate synthesis prompt template based on the processing mode.
    
    Parameters
    ----------
    bypass_individual_generation : bool
        Whether to use raw documents mode (True) or processed answers mode (False)
        
    Returns
    -------
    PromptTemplate
        The appropriate prompt template for synthesis
    """
    if bypass_individual_generation:
        template = RAW_DOCUMENTS_SYNTHESIS_PROMPT
    else:
        template = PROCESSED_ANSWERS_SYNTHESIS_PROMPT
    
    return PromptTemplate(
        template=template,
        input_variables=[
            "question", 
            "formatted_answers_or_raw_docs", 
            "files_no_info", 
            "files_errors", 
            "conversation_history"
        ]
    )


# Content Policy Error Message
CONTENT_POLICY_MESSAGE = "Due to content management policy issues with the AI provider, we are not able to provide a response to this topic. Please rephrase your question and try again."


# Progress Messages for User Feedback
PROGRESS_MESSAGES = {
    "initialization": {
        "embeddings": "Setting up AI models...",
        "llm_large": "Initializing language models...",
        "llm_small": "Setting up query generation models...",
        "vectorstore": "Loading document database...",
        "retriever": "Setting up document search engine..."
    },
    "query_generation": {
        "multi_queries": "Generating search queries..."
    },
    "document_retrieval": {
        "extraction": "Searching documents for relevant information..."
    },
    "answer_generation": {
        "individual_answers": "Generating answers for each document..."
    },
    "document_preparation": {
        "format_raw": "Preparing documents for analysis..."
    },
    "synthesis": {
        "combine_answers": "Synthesizing final response..."
    }
}


def get_progress_message(stage: str, node: str) -> str:
    """
    Get a user-friendly progress message for a given stage and node.
    
    Parameters
    ----------
    stage : str
        The processing stage (e.g., "initialization", "query_generation")
    node : str
        The specific node name
        
    Returns
    -------
    str
        A user-friendly progress message
    """
    stage_messages = PROGRESS_MESSAGES.get(stage, {})
    
    # Map node names to message keys
    node_to_key = {
        "instantiate_embeddings_node": "embeddings",
        "instantiate_llm_large_node": "llm_large", 
        "instantiate_llm_small_node": "llm_small",
        "load_vectorstore_node": "vectorstore",
        "instantiate_retriever_node": "retriever",
        "generate_multi_queries_node": "multi_queries",
        "extract_documents_node": "extraction",
        "generate_answers_node": "individual_answers",
        "format_raw_documents_for_synthesis_node": "format_raw",
        "combine_answers_node": "combine_answers"
    }
    
    key = node_to_key.get(node, node)
    return stage_messages.get(key, f"Processing {node}...") 