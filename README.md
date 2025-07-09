[![test](https://github.com/crvernon/knowai/actions/workflows/build.yml/badge.svg)](https://github.com/crvernon/knowai/actions/workflows/build.yml)
[![DOI](https://zenodo.org/badge/976158351.svg)](https://doi.org/10.5281/zenodo.15460377)


## knowai
#### An agentic AI pipeline for multiple, large PDF reports interrogation

### Set up
- Clone this repostiory into a local directory of your choosing
- Build a virtual environment 
- Install `knowai` by running:  `pip install .` from the root directory of your clone (OR) install using `pip install knowai` from PyPI.
- Configure a `.env` file with the following:
    - `AZURE_OPENAI_API_KEY` - Your API key
    - `AZURE_OPENAI_ENDPOINT` - Your Azure endpoint
    - `AZURE_OPENAI_DEPLOYMENT` - Your LLM deployment name (e.g., "gpt-4o")
    - `AZURE_EMBEDDINGS_DEPLOYMENT` - Your embeddings model deployment name (e.g., "text-embedding-3-large")
    - `AZURE_OPENAI_API_VERSION` - Your Azure LLM deployment version (e.g., "2024-02-01")

### Building the vectorstore
From the root directory of this repository, run the following from a the terminal (ensuring that your virtual environment is active) to build the vectorstore:

`python scripts/build_vectorstore.py <directory_containing_your_input_pdf_files> --vectorstore_path <directory_name_for_vectorstore>`

By default, this will create a vectorstore using FAISS named "test_faiss_store" in the root directory of your repository.  

### Running the knowai in a simple chatbot example via streamlit
From the root directory, run the following in a terminal after you have your virtual environment active:  

`streamlit run app_chat_simple.py`

This will open the app in your default browser.

### Using knowai

Once your vector store is built, you can use **knowai** either programmatically or through the provided Streamlit interface.

#### Python quick‑start

The package ships with the `KnowAIAgent` class for fully programmatic access
inside notebooks or scripts:

```python
from knowai.core import KnowAIAgent

# Path that you supplied with --vectorstore_path when building
VSTORE_PATH = "test_faiss_store"

agent = KnowAIAgent(vectorstore_path=VSTORE_PATH)

# A single conversational turn
response = await agent.process_turn(
    user_question="Summarize the key findings in the 2025 maritime report",
    selected_files=["my_report.pdf"],
)

print(response["generation"])
```

#### Streaming Responses

For a more responsive user experience, you can enable streaming responses:

```python
def stream_callback(token: str):
    """Called for each token as it's generated."""
    print(token, end='', flush=True)

response = await agent.process_turn(
    user_question="Summarize the key findings in the 2025 maritime report",
    selected_files=["my_report.pdf"],
    streaming_callback=stream_callback  # Enable streaming
)
```

The response will be streamed in real-time via the callback, while still being available in the returned dictionary.

The returned dictionary contains:

| Key                           | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `generation`                  | Final answer synthesised from the selected documents.        |
| `individual_answers`          | Per‑file answers (when *bypass_individual_gen=False*).       |
| `documents_by_file`           | Retrieved document chunks keyed by filename.                 |
| `raw_documents_for_synthesis` | Raw text block used when bypassing individual generation.    |
| `bypass_individual_generation`| Whether the bypass mode was used for this turn.              |

#### Token Counting Configuration

KnowAI supports two methods for token counting to manage context window limits:

**Accurate Token Counting (Default)**
- Uses `tiktoken` library for precise token estimation
- More accurate batch sizing and context management
- Automatically falls back to heuristic method if `tiktoken` unavailable

```python
# Default behavior (accurate token counting)
agent = KnowAIAgent(vectorstore_path=VSTORE_PATH)

# Explicit accurate token counting
agent = KnowAIAgent(
    vectorstore_path=VSTORE_PATH,
    
)
```

**Heuristic Token Counting**
- Uses character-based estimation (4 characters ≈ 1 token)
- Faster performance, suitable when approximate estimation is sufficient
- Always available as fallback

```python
# Use heuristic token counting
agent = KnowAIAgent(
    vectorstore_path=VSTORE_PATH,
    
)
```

**CLI Configuration**
```bash
curl -X POST http://127.0.0.1:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "vectorstore_s3_uri": "/path/to/vectorstore",
    
  }'
```

**Benefits of Accurate Token Counting:**
- More precise token limits and batch sizing
- Reduced risk of context overflow
- Better resource utilization
- Improved reliability with large document sets

**When to Use Heuristic Counting:**
- `tiktoken` not available in environment
- Performance is critical
- Approximate estimation is sufficient
- Debugging token counting issues

#### Streamlit chat app

If you prefer a ready‑made UI, launch the demo:

```bash
streamlit run app_chat_simple.py
```

Upload or select PDF files, ask questions in the sidebar, and inspect per‑file
answers or the combined response in the main panel.

---

For advanced configuration options (e.g., conversation history length,
retriever *k* values, or combine thresholds) see the docstrings in
`knowai/core.py` and `knowai/agent.py`.

## Containerization

To build and run both the knowai service and the Svelte UI using Docker Compose:

1. Ensure Docker and Docker Compose are installed on your machine.
2. From the directory containing this README (the repo root), navigate to the Svelte example folder:
   ```bash
   cd example_apps/svelte
   ```
2a. Compile the Svelte app and package the build as `svelte-example`:
   ```bash
   npm install
   npm run build
   mv dist svelte-example
   ```
3. Start the services and build images:
   ```bash
   docker compose up --build
   ```
   This will:
   - Build the `knowai` service (listening on port 8000).
   - Build the `ui` service (Svelte app, listening on port 5173).
4. Open your browser and visit:
   - FastAPI docs: http://localhost:8000/docs
   - Svelte UI:      http://localhost:5173
5. To stop and remove containers, press `CTRL+C` and then run:
   ```bash
   docker compose down
   ```

## Running the knowai CLI Locally

You can start the FastAPI micro-service locally without Docker and point it to either a local vectorstore or one hosted on S3.

### Using a Local Vectorstore

1. Ensure you have a built FAISS vectorstore on disk (e.g., `test_faiss_store`).
2. Start the service:
   ```bash
   python -m knowai.cli
   ```
3. In another terminal, initialize the session:
   ```bash
   curl -X POST http://127.0.0.1:8000/initialize \
     -H "Content-Type: application/json" \
     -d '{"vectorstore_s3_uri":"/absolute/path/to/your/vectorstore"}'
   ```
4. Ask a question:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{
       "session_id":"<session_id>",
       "question":"Your question here",
       "selected_files":["file1.pdf","file2.pdf"]
     }'
   ```

### Streaming API

For real-time streaming responses, use the `/ask-stream` endpoint:

```bash
curl -X POST http://127.0.0.1:8000/ask-stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"<session_id>",
    "question":"Your question here",
    "selected_files":["file1.pdf","file2.pdf"]
  }' \
  --no-buffer
```

This will stream the response in real-time using Server-Sent Events (SSE). Each token will be sent as it's generated by the LLM.

For more details on streaming functionality, see [docs/STREAMING.md](docs/STREAMING.md).

### Using an S3-Hosted Vectorstore

1. Start the service:
   ```bash
   python -m knowai.cli
   ```
2. Initialize the session against your S3 bucket:
   ```bash
   curl -X POST http://127.0.0.1:8000/initialize \
     -H "Content-Type: application/json" \
     -d '{"vectorstore_s3_uri":"s3://your-bucket/path"}'
   ```
3. Ask a question in a similar way:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{
       "session_id":"<session_id>",
       "question":"Another question example",
       "selected_files":[]
     }'
   ```

## Enhanced User Feedback

KnowAI provides comprehensive feedback when the search process doesn't find relevant information in your documents.

### No-Chunks Feedback

When no text chunks are extracted for a query in a file, KnowAI ensures users are clearly informed:

- **Individual File Level**: Each file that has no matching content receives a specific message explaining that "The search did not retrieve any document chunks that match your query."
- **Synthesis Level**: The final response clearly states which files had no relevant content, helping users understand the scope of the search results.
- **Progress Tracking**: Files with no matching content are tracked separately from files with errors, providing clear distinction in the response.

### Example Response

When asking about "climate change impacts" across multiple reports:

```
I found information about climate change impacts in the following reports:

From report1.pdf (Page 15):
"Global temperatures have increased by 1.1°C since pre-industrial times..."

From report2.pdf (Page 8):
"Sea level rise is accelerating at a rate of 3.3mm per year..."

No matching content found in: report3.pdf (no matching content).
```

This helps users understand:
- Which files contained relevant information
- Which files were searched but had no matching content
- The specific nature of missing information

### Error Handling

KnowAI distinguishes between different types of issues:
- **No matching content**: Files that were searched but had no relevant chunks
- **Content policy violations**: Issues with AI provider content filters
- **Processing errors**: Technical issues during document processing

Each type is handled appropriately and communicated clearly to the user.

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest

# Test specific functionality
python -m pytest tests/test_prompts.py -v
python -m pytest tests/test_agent.py -v

# Test no-chunks feedback improvements
python scripts/test_no_chunks_feedback.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

#### Individual File Processing

KnowAI supports two processing modes for handling multiple files:

**Traditional Batch Processing (Default)**
- All documents from all files are combined and processed together
- Faster processing, good for related content across files

**Individual File Processing**
- Each file is processed separately by the LLM (in parallel, max 10 concurrent), then responses are consolidated
- Ensures each file gets equal attention, better for distinct topics
- Significantly faster than sequential processing for multiple files
- Concurrency limit prevents overwhelming the LLM service

```python
# Enable individual file processing
agent = KnowAIAgent(
    vectorstore_path=VSTORE_PATH,
    process_files_individually=True  # Enable individual processing
)

# Or enable per-request
response = await agent.process_turn(
    user_question="What are the main strategies?",
    selected_files=["file1.pdf", "file2.pdf"],
    process_files_individually=True  # Override for this request
)
```

**When to Use Individual File Processing:**
- Files contain distinct topics that should be analyzed separately
- You want to ensure each file gets equal attention from the LLM
- You want to see how each file contributes to the final answer
- Dealing with large files that might benefit from focused analysis

For more details, see [Individual File Processing Documentation](docs/INDIVIDUAL_FILE_PROCESSING.md).

#### Token Counting Configuration
