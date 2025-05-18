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

The returned dictionary contains:

| Key                           | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `generation`                  | Final answer synthesised from the selected documents.        |
| `individual_answers`          | Per‑file answers (when *bypass_individual_gen=False*).       |
| `documents_by_file`           | Retrieved document chunks keyed by filename.                 |
| `raw_documents_for_synthesis` | Raw text block used when bypassing individual generation.    |
| `bypass_individual_generation`| Whether the bypass mode was used for this turn.              |

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
