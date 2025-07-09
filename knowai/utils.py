import logging
import tiktoken


def get_tokenizer(encoding: str = "cl100k_base"):
    """
    Get the tiktoken tokenizer for accurate token counting.
    
    Returns
    -------
    Optional[tiktoken.Encoding]
        tiktoken tokenizer if available, None otherwise.
    """
    try:
        # Use cl100k_base encoding which is used by GPT-4
        return tiktoken.get_encoding(encoding)
    except Exception as e:
        logging.warning(f"Failed to initialize tiktoken: {e}.")
        return None


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string using tiktoken.
    
    Parameters
    ----------
    text : str
        The text to estimate tokens for.
        
    Returns
    -------
    int
        Estimated number of tokens.
    """
    if not text:
        return 0
    
    try:
        tokenizer = get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"tiktoken failed: {e}.")


def estimate_synthesis_tokens(
    question: str,
    content: str,
    conversation_history: str,
    files_no_info: str,
    files_errors: str,
    conversation_history_buffer: int = 5000
) -> int:
    """
    Estimate the total number of tokens required for a synthesis operation using tiktoken.

    This function calculates the sum of tokens for the user's question, the document content
    to be synthesized, the formatted conversation history, a list of files with no information,
    and a list of files with errors. It also adds a buffer (default: 5000 tokens) to account
    for the expected length of the model's response or additional system prompt overhead.

    Parameters
    ----------
    question : str
        The user's question to be answered.
    content : str
        The document content that will be synthesized by the model.
    conversation_history : str
        The formatted conversation history to provide context.
    files_no_info : str
        A string listing files for which no information was found.
    files_errors : str
        A string listing files that encountered errors during processing.
    conversation_history_buffer : int, optional
        A buffer (in tokens) added to the total to account for the model's response
        or system prompt overhead. Default is 5000.

    Returns
    -------
    int
        The estimated total number of tokens required for the synthesis operation.
    """
    # Estimate tokens for each component
    question_tokens = estimate_tokens(question)
    content_tokens = estimate_tokens(content)
    history_tokens = estimate_tokens(conversation_history)
    no_info_tokens = estimate_tokens(files_no_info)
    errors_tokens = estimate_tokens(files_errors)
    
    # Add system prompt and response buffer
    total_tokens = (
        question_tokens +
        content_tokens +
        history_tokens +
        no_info_tokens +
        errors_tokens +
        conversation_history_buffer  # Buffer for response
    )
    
    return total_tokens
