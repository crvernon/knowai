import logging
from typing import Optional, Callable

# Error Messages
CONTENT_POLICY_MESSAGE = (
    "Due to content management policy issues with the AI provider, we are not able "
    "to provide a response to this topic. Please rephrase your question and try again."
)

RATE_LIMIT_MESSAGE = (
    "⚠️ Rate limit reached: The OpenAI LLM instance has reached its rate limit. "
    "Please try your query again in 1 minute."
)

TOKEN_LIMIT_MESSAGE = (
    "⚠️ Token limit exceeded: The prompt token limit has exceeded the maximum for the LLM instance. "
    "Please contact the site administrator for assistance."
)


class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass


class TokenLimitError(Exception):
    """Custom exception for token limit errors."""
    pass


def is_content_policy_error(e: Exception) -> bool:
    """
    Determine whether an exception message indicates an AI content‑policy
    violation.

    Parameters
    ----------
    e : Exception
        Exception raised by the LLM provider.

    Returns
    -------
    bool
        ``True`` if the exception message contains any keyword that signals
        a policy‑related block; otherwise ``False``.
    """
    error_message = str(e).lower()
    keywords = [
        "content filter",
        "content management policy",
        "responsible ai",
        "safety policy",
        "prompt blocked"  # Common for Azure
    ]
    return any(keyword in error_message for keyword in keywords)


def is_rate_limit_error(e: Exception) -> bool:
    """
    Determine whether an exception indicates a rate limit (429) error.

    Parameters
    ----------
    e : Exception
        Exception raised by the LLM provider.

    Returns
    -------
    bool
        ``True`` if the exception indicates a rate limit error; otherwise ``False``.
    """
    # Check status code if available
    if hasattr(e, 'status_code') and e.status_code == 429:
        return True
    
    # Check error message for rate limit indicators
    error_message = str(e).lower()
    rate_limit_keywords = [
        "rate limit",
        "too many requests",
        "429",
        "quota exceeded",
        "rate exceeded"
    ]
    return any(keyword in error_message for keyword in rate_limit_keywords)


def is_token_limit_error(e: Exception) -> bool:
    """
    Determine whether an exception indicates a token limit exceeded (400) error.

    Parameters
    ----------
    e : Exception
        Exception raised by the LLM provider.

    Returns
    -------
    bool
        ``True`` if the exception indicates a token limit error; otherwise ``False``.
    """
    error_message = str(e).lower()
    token_limit_keywords = [
        "token",
        "context length",
        "maximum context length",
        "input too long",
        "prompt too long",
        "context window"
    ]
    
    # Check if it's specifically a token limit error by keywords
    if any(keyword in error_message for keyword in token_limit_keywords):
        return True
    
    # Check status code if available (400 with any message)
    if hasattr(e, 'status_code') and e.status_code == 400:
        return True
    
    return False


def handle_llm_error(
    e: Exception, 
    streaming_callback: Optional[Callable[[str], None]] = None,
    node_name: str = "unknown"
) -> str:
    """
    Handle LLM errors and provide appropriate user feedback via callback.
    
    Parameters
    ----------
    e : Exception
        The exception that occurred.
    streaming_callback : Optional[Callable[[str], None]]
        Callback function to inform the user of the error.
    node_name : str
        Name of the node where the error occurred.
        
    Returns
    -------
    str
        Error message to return to the user.
    """
    if is_rate_limit_error(e):
        logging.warning(f"[{node_name}] Rate limit error: {e}")
        
        if streaming_callback:
            streaming_callback(RATE_LIMIT_MESSAGE)
        
        # Raise custom exception to stop workflow
        raise RateLimitError(RATE_LIMIT_MESSAGE)
    
    elif is_token_limit_error(e):
        logging.error(f"[{node_name}] Token limit error: {e}")
        
        if streaming_callback:
            streaming_callback(TOKEN_LIMIT_MESSAGE)
        
        # Raise custom exception to stop workflow
        raise TokenLimitError(TOKEN_LIMIT_MESSAGE)
    
    elif is_content_policy_error(e):
        logging.warning(f"[{node_name}] Content policy violation: {e}")
        return CONTENT_POLICY_MESSAGE
    
    else:
        # Generic error handling
        error_message = f"An error occurred during processing: {str(e)}"
        logging.error(f"[{node_name}] Generic error: {e}")
        
        if streaming_callback:
            streaming_callback(error_message)
        
        return error_message
