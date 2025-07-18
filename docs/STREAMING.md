# KnowAI Streaming Responses

This document explains how to use KnowAI's streaming response functionality, which provides real-time token streaming for a more responsive user experience.

## Overview

KnowAI now supports streaming responses from the LLM, allowing users to see the response being generated in real-time rather than waiting for the complete response. This is particularly useful for long responses or when you want to provide immediate feedback to users.

## How It Works

### Core Streaming

The streaming functionality works by:

1. **Callback Mechanism**: A callback function is provided to `process_turn()` that gets called for each token as it's generated
2. **Async Streaming**: Uses LangChain's `astream()` method to stream tokens from the LLM
3. **Real-time Updates**: Tokens are delivered to the callback function immediately as they're generated
4. **Complete Response**: The full response is still returned in the result dictionary

### API Streaming

The API provides a streaming endpoint that uses Server-Sent Events (SSE) to stream responses over HTTP.

## Usage Examples

### 1. Core Library Usage

```python
from knowai.core import KnowAIAgent

# Initialize the agent
agent = KnowAIAgent(vectorstore_path="/path/to/vectorstore")

# Define a streaming callback
def stream_callback(token: str):
    """Called for each token as it's generated."""
    print(token, end='', flush=True)

# Process with streaming
result = await agent.process_turn(
    user_question="What are the vegetation management strategies?",
    selected_files=["file1.pdf", "file2.pdf"],
    streaming_callback=stream_callback  # Enable streaming
)

# The response was streamed via callback, but is also available in result
print(f"\nComplete response: {result['generation']}")
```

### 2. API Usage

#### Initialize Session
```bash
curl -X POST "http://localhost:8000/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "vectorstore_s3_uri": "/path/to/vectorstore",
    "max_conversation_turns": 20
  }'
```

#### Stream Response
```bash
curl -X POST "http://localhost:8000/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "question": "What are the vegetation management strategies?",
    "selected_files": ["file1.pdf", "file2.pdf"]
  }' \
  --no-buffer
```

### 3. JavaScript Client Example

```javascript
// Initialize session
const initResponse = await fetch('/initialize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    vectorstore_s3_uri: '/path/to/vectorstore'
  })
});
const { session_id } = await initResponse.json();

// Stream response
const streamResponse = await fetch('/ask-stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: session_id,
    question: 'What are the vegetation management strategies?',
    selected_files: ['file1.pdf', 'file2.pdf']
  })
});

// Process streaming response
const reader = streamResponse.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') {
        console.log('Streaming completed');
        break;
      } else if (data) {
        // Display the token
        console.log(data);
      }
    }
  }
}
```

## API Endpoints

### POST /ask-stream

Streams the LLM response in real-time using Server-Sent Events.

**Request Body:**
```json
{
  "session_id": "string",
  "question": "string",
  "selected_files": ["string"],
  "bypass_individual_gen": false,
  "n_alternatives_override": null,
  "k_per_query_override": null
}
```

**Response:**
- Content-Type: `text/event-stream`
- Format: Server-Sent Events with `data:` prefix
- End marker: `data: [DONE]`

## Configuration

### Streaming Callback

The streaming callback function should have the signature:
```python
def streaming_callback(token: str) -> None:
    """Handle a single token from the streaming response."""
    pass
```

### Performance Considerations

1. **Token Granularity**: Tokens are streamed as they're generated by the LLM
2. **Network Overhead**: Streaming adds minimal overhead compared to waiting for complete responses
3. **Memory Usage**: Streaming can reduce memory usage for very long responses
4. **User Experience**: Provides immediate feedback and reduces perceived latency

## Error Handling

### Content Policy Violations

If the LLM encounters a content policy violation during streaming:
- The streaming will stop immediately
- The callback will receive the content policy message
- The complete response will contain the policy message

### Network Issues

For API streaming:
- Connection timeouts are handled gracefully
- Keepalive messages are sent to maintain connection
- The stream ends with `[DONE]` marker when complete

## Testing

### Test Scripts

1. **Basic Streaming Test**: `scripts/test_streaming.py`
   - Tests the core streaming functionality
   - Demonstrates usage patterns

2. **API Streaming Test**: `scripts/test_streaming_client.py`
   - Tests the streaming API endpoint
   - Requires a running KnowAI server

### Running Tests

```bash
# Test core streaming
python scripts/test_streaming.py

# Test API streaming (requires server)
python scripts/test_streaming_client.py
```

## Integration with Existing Code

### Backward Compatibility

- Streaming is **opt-in** - existing code continues to work unchanged
- If no `streaming_callback` is provided, behavior is identical to before
- All existing API endpoints remain functional

### Migration Guide

To add streaming to existing code:

1. **Core Library**: Add `streaming_callback` parameter to `process_turn()` calls
2. **API**: Use `/ask-stream` endpoint instead of `/ask` for streaming responses
3. **UI**: Update frontend to handle Server-Sent Events for streaming responses

## Troubleshooting

### Common Issues

1. **No Streaming**: Ensure `streaming_callback` is provided to `process_turn()`
2. **API Connection**: Check that the server supports streaming endpoints
3. **Token Display**: Ensure your callback function handles tokens appropriately
4. **Network Timeouts**: Increase timeout values for long responses

### Debug Mode

Enable debug logging to see streaming details:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements to the streaming functionality:

1. **Progress Indicators**: Show processing progress alongside streaming
2. **Partial Results**: Stream intermediate results from individual file processing
3. **Streaming Controls**: Pause/resume streaming functionality
4. **Rate Limiting**: Control streaming speed for better UX
5. **Error Recovery**: Resume streaming after network interruptions

## Examples

See the following files for complete examples:
- `scripts/test_streaming.py` - Core streaming examples
- `scripts/test_streaming_client.py` - API streaming examples
- `example_apps/` - Integration examples with different frameworks 