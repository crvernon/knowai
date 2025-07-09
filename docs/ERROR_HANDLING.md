# Error Handling in KnowAI

This document describes the error handling mechanisms implemented in KnowAI for managing various types of errors that can occur during LLM processing.

## Overview

KnowAI implements comprehensive error handling for different types of errors that can occur when interacting with OpenAI/Azure OpenAI services. The system is designed to:

1. **Detect specific error types** based on status codes and error messages
2. **Provide user-friendly error messages** via callbacks
3. **Stop workflow execution** for critical errors (rate limits, token limits)
4. **Continue processing** for non-critical errors (content policy violations)

## Error Types

### 1. Rate Limit Errors (429)

**Detection**: 
- Status code 429
- Error messages containing: "rate limit", "too many requests", "429", "quota exceeded", "rate exceeded"

**Behavior**:
- Workflow stops immediately
- User receives message: "⚠️ Rate limit reached: The OpenAI LLM instance has reached its rate limit. Please try your query again in 1 minute."
- Error is logged as a warning

**User Action**: Wait 1 minute and retry the query.

### 2. Token Limit Errors (400)

**Detection**:
- Status code 400
- Error messages containing: "token", "context length", "maximum context length", "input too long", "prompt too long", "context window"

**Behavior**:
- Workflow stops immediately
- User receives message: "⚠️ Token limit exceeded: The prompt token limit has exceeded the maximum for the LLM instance. Please contact the site administrator for assistance."
- Error is logged as an error

**User Action**: Contact site administrator for assistance.

### 3. Content Policy Errors

**Detection**:
- Error messages containing: "content filter", "content management policy", "responsible ai", "safety policy", "prompt blocked"

**Behavior**:
- Workflow continues with fallback behavior
- User receives message: "Due to content management policy issues with the AI provider, we are not able to provide a response to this topic. Please rephrase your question and try again."
- Error is logged as a warning

**User Action**: Rephrase the question and try again.

### 4. Generic Errors

**Detection**: Any error that doesn't match the above categories

**Behavior**:
- Workflow continues with fallback behavior
- User receives generic error message
- Error is logged as an error

**User Action**: Check logs for details and contact support if needed.

## Implementation Details

### Error Detection Functions

```python
def _is_rate_limit_error(e: Exception) -> bool:
    """Detect rate limit errors (429)."""

def _is_token_limit_error(e: Exception) -> bool:
    """Detect token limit errors (400)."""

def _is_content_policy_error(e: Exception) -> bool:
    """Detect content policy violations."""
```

### Error Handling Function

```python
def _handle_llm_error(
    e: Exception, 
    streaming_callback: Optional[Callable[[str], None]] = None,
    node_name: str = "unknown"
) -> str:
    """Handle LLM errors and provide appropriate user feedback."""
```

### Custom Exceptions

```python
class RateLimitError(Exception):
    """Custom exception for rate limit errors."""

class TokenLimitError(Exception):
    """Custom exception for token limit errors."""
```

## Integration Points

### 1. Core Processing

The error handling is integrated into the main processing workflow in `knowai/core.py`:

```python
try:
    updated_state = await self.graph_app.ainvoke(self.session_state)
except RateLimitError as e:
    # Handle rate limit error
    return {"generation": str(e), ...}
except TokenLimitError as e:
    # Handle token limit error
    return {"generation": str(e), ...}
```

### 2. Streaming API

The streaming API in `knowai/cli.py` also handles these exceptions:

```python
try:
    await processing_task
except RateLimitError as e:
    yield f"data: {str(e)}\n\n"
except TokenLimitError as e:
    yield f"data: {str(e)}\n\n"
```

### 3. Individual File Processing

Error handling is applied to individual file processing in `process_individual_files_node`:

```python
except Exception as e:
    error_msg = _handle_llm_error(e, None, f"process_individual_files_node_{filename}")
    return filename, error_msg
```

## Testing

A comprehensive test suite is available in `scripts/test_error_handling.py` that verifies:

1. Error detection accuracy
2. Error handling behavior
3. Callback functionality
4. Exception raising for critical errors

Run the tests with:

```bash
python scripts/test_error_handling.py
```

## Configuration

Error messages can be customized by modifying the constants in `knowai/prompts.py`:

```python
RATE_LIMIT_MESSAGE = "⚠️ Rate limit reached: ..."
TOKEN_LIMIT_MESSAGE = "⚠️ Token limit exceeded: ..."
CONTENT_POLICY_MESSAGE = "Due to content management policy issues..."
```

## Best Practices

1. **Always use the error handling functions** instead of catching exceptions directly
2. **Provide meaningful node names** for better error tracking
3. **Use streaming callbacks** when available for real-time user feedback
4. **Monitor logs** for error patterns and system health
5. **Test error scenarios** regularly to ensure proper handling

## Troubleshooting

### Common Issues

1. **Error not detected correctly**: Check if the error message or status code matches the detection criteria
2. **Workflow doesn't stop**: Ensure the error is being raised as a custom exception
3. **Callback not called**: Verify the streaming callback is properly passed to the error handling function

### Debugging

Enable debug logging to see detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The error handling system logs detailed information about each error, including:
- Error type and message
- Status codes (if available)
- API responses (if available)
- Node where the error occurred 