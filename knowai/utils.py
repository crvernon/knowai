



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