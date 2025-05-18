[![test](https://github.com/crvernon/knowai/actions/workflows/build.yml/badge.svg)](https://github.com/crvernon/knowai/actions/workflows/build.yml)

## knowai
#### An agentic AI pipeline for multiple, large PDF reports interrogation

### Set up
- Clone this repostiory into a local directory of your choosing
- Build a virtual environment 
- Install `knowai` by running:  `pip install knowai` from the root directory of your clone
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
