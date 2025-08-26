# Individual File Processing

This document explains the new individual file processing feature in KnowAI, which allows each file to be processed separately before consolidating the responses.

## Overview

KnowAI now supports two processing modes:

1. **Traditional Batch Processing** (default): All documents from all files are combined and processed together
2. **Individual File Processing**: Each file is processed separately by the LLM, then all responses are consolidated

## When to Use Individual File Processing

Individual file processing is beneficial when:

- You want to ensure each file gets equal attention from the LLM
- Files contain distinct topics that should be analyzed separately
- You want to see how each file contributes to the final answer
- You're dealing with large files that might benefit from focused analysis

## How It Works

### Traditional Mode (Default)
```
Files → Document Retrieval → Combined Processing → Single Response
```

### Individual File Mode
```
Files → Document Retrieval → Parallel Individual Processing → Consolidation → Single Response
```

## Usage

### Constructor Parameter

```python
from knowai.core import KnowAIAgent

# Enable individual file processing
agent = KnowAIAgent(
    vectorstore_path="./vectorstores/my_vectorstore",
    process_files_individually=True
)
```

### Per-Request Parameter

```python
# Enable for a specific request
result = await agent.process_turn(
    user_question="What are the main strategies?",
    selected_files=["file1.pdf", "file2.pdf"],
    process_files_individually=True
)
```

## Processing Stages

When individual file processing is enabled, the workflow follows these stages:

1. **Initialization**: Load models and vectorstore
2. **Query Generation**: Generate alternative queries
3. **Document Retrieval**: Extract relevant documents for each file
4. **Document Preparation**: Format documents for processing
5. **Parallel Individual Processing**: Process each file separately and simultaneously
   - All files are processed asynchronously in parallel (max 10 concurrent LLM calls)
   - Each file gets its own LLM response
   - Progress is tracked for the overall parallel processing
6. **Consolidation**: Combine all individual responses into final answer
   - Only the final consolidated response is streamed to the user
   - Individual responses are not streamed

## Progress Tracking

The system provides detailed progress updates during individual file processing:

- `individual_processing`: Processing files asynchronously in parallel
- `consolidating_individual_responses`: Combining responses

## Example

```python
import asyncio
from knowai.core import KnowAIAgent

async def example():
    agent = KnowAIAgent(
        vectorstore_path="./vectorstores/my_vectorstore",
        process_files_individually=True
    )
    
    def progress_callback(message, level, metadata):
        print(f"Progress: {message}")
    
    result = await agent.process_turn(
        user_question="What are the vegetation management strategies?",
        selected_files=["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"],
        progress_cb=progress_callback
    )
    
    print(result["generation"])

asyncio.run(example())
```

## Testing

Use the test script to verify both processing modes:

```bash
python scripts/test_individual_processing.py ./vectorstores/my_vectorstore
```

This will test both traditional batch processing and individual file processing with the same question and files.

## Performance Considerations

- **Individual processing** now processes all files in parallel (max 10 concurrent), significantly reducing total processing time
- **Concurrency limit** prevents overwhelming the LLM service and ensures stable performance
- **Traditional processing** is still generally faster for small numbers of files
- **Individual processing** provides better results for files with distinct topics
- Choose based on your specific use case and requirements

## Configuration

The feature can be configured at multiple levels:

1. **Agent-level**: Set during agent initialization
2. **Request-level**: Override per request
3. **Default**: Traditional batch processing (backward compatible)

## Backward Compatibility

The new feature is fully backward compatible. Existing code will continue to work with traditional batch processing by default.

## Concurrency Control

The individual file processing uses a concurrency limit to ensure stable performance:

- **Maximum 10 concurrent LLM calls**: Prevents overwhelming the LLM service
- **Automatic queuing**: Files beyond the limit are queued and processed as slots become available
- **Resource management**: Ensures consistent performance regardless of the number of files
- **Error isolation**: Individual file failures don't affect other files in the queue

This limit is optimized for most LLM services and can be adjusted if needed for specific use cases. 