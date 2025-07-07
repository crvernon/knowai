"""
Prompts used throughout the KnowAI agent.

This module contains all the prompt templates used by the KnowAI agent,
centralized for easy maintenance and customization.
"""

from langchain_core.prompts import PromptTemplate


# Synthesis Prompt for Raw Documents
RAW_DOCUMENTS_SYNTHESIS_PROMPT = (
    "You are an expert AI assistant. Answer CURRENT question based ONLY on RAW text chunks.\n"
    "Conversation History: {conversation_history}\n"
    "User's CURRENT Question: {question}\n"
    "RAW Text Chunks: {formatted_answers_or_raw_docs}\n"
    "Files with No Relevant Info (no chunks extracted): {files_no_info}\n"
    "Files with Errors (extraction errors): {files_errors}\n"
    "Instructions: Read raw text. Answer ONLY from raw text. Quote with citations. "
    "IMPORTANT: Always use the exact filename from the source "
    "(e.g., \"quote...\" (actual_filename.pdf, Page X)), never use generic terms like "
    "\"file.pdf\". If info not found, clearly state which files had no relevant content. "
    "Structure logically.\n"
    "Synthesized Answer from RAW Docs:"
)


def get_synthesis_prompt_template() -> PromptTemplate:
    """
    Get the synthesis prompt template for raw documents.
    
    Returns
    -------
    PromptTemplate
        The prompt template for synthesis
    """
    return PromptTemplate(
        template=RAW_DOCUMENTS_SYNTHESIS_PROMPT,
        input_variables=[
            "question", 
            "formatted_answers_or_raw_docs", 
            "files_no_info", 
            "files_errors", 
            "conversation_history"
        ]
    )


# Content Policy Error Message
CONTENT_POLICY_MESSAGE = (
    "Due to content management policy issues with the AI provider, we are not able "
    "to provide a response to this topic. Please rephrase your question and try again."
)


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
    "document_preparation": {
        "format_raw": "Preparing documents for analysis..."
    },
    "synthesis": {
        "process_batches": "Processing documents in batches...",
        "combine_answers": "Synthesizing final response..."
    },
    "individual_processing": {
        "process_individual_files": "Processing files individually...",
        "individual_processing_file_1_2": "Processing file 1 of 2...",
        "individual_processing_file_2_2": "Processing file 2 of 2...",
        "individual_processing_file_1_3": "Processing file 1 of 3...",
        "individual_processing_file_2_3": "Processing file 2 of 3...",
        "individual_processing_file_3_3": "Processing file 3 of 3..."
    },
    "routing_to_individual_processing": {
        "routing": "Routing to individual file processing..."
    },
    "processing_in_batches": {
        "processing": "Processing documents in batches..."
    },
    "processing_batch_1_2": {
        "processing": "Processing batch 1 of 2..."
    },
    "processing_batch_2_2": {
        "processing": "Processing batch 2 of 2..."
    },
    "processing_batch_1_3": {
        "processing": "Processing batch 1 of 3..."
    },
    "processing_batch_2_3": {
        "processing": "Processing batch 2 of 3..."
    },
    "processing_batch_3_3": {
        "processing": "Processing batch 3 of 3..."
    },
    "consolidating_individual_responses": {
        "consolidating": "Consolidating individual file responses..."
    },
    "combining_batch_results": {
        "combining": "Combining batch results..."
    },
    "traditional_synthesis": {
        "synthesizing": "Synthesizing final response..."
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
        "format_raw_documents_for_synthesis_node": "format_raw",
        "process_batches_node": "process_batches",
        "process_individual_files_node": "process_individual_files",
        "combine_answers_node": "combine_answers"
    }
    
    key = node_to_key.get(node, node)
    return stage_messages.get(key, f"Processing {node}...") 