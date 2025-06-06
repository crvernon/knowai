import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# --- Configuration ---
# Make sure you have a .env file in the same directory with your OpenAI API Key
load_dotenv("/Users/d3y010/repos/crvernon/knowai/.env", override=True)


PDF_DIRECTORY = "/Users/d3y010/projects/gdo_wildfire/data/reports"
FAISS_INDEX_PATH = "faiss_openai_large_20250606"
PROGRESS_FILE = "processing_progress.json"



def get_text_chunks(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits documents into smaller chunks for processing.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A list of smaller Document chunks.
    """
    # print("  - Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Rename the 'source' metadata key to 'file_name'
    for chunk in chunks:
        if 'source' in chunk.metadata:
            chunk.metadata['file_name'] = os.path.basename(chunk.metadata.pop('source'))

    # print(f"  - Created {len(chunks)} chunks.")
    return chunks

def load_progress() -> set:
    """Loads the set of already processed file names from the progress file."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError):
        return set()

def save_progress(processed_files: set):
    """Saves the set of processed file names to the progress file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_files), f, indent=4)

def main(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Main function to orchestrate PDF processing and vector store creation.
    The process is resumable and saves progress after each file.
    """
    # 1. Initialize Embeddings Model
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION")
        )
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly in the .env file.")
        return

    # 2. Load progress to see which files have already been processed
    processed_files = load_progress()
    print(f"Found {len(processed_files)} previously processed files.")

    db = None

    # 3. Load existing FAISS index if it exists
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
        try:
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. A new index will be created if needed.")
            db = None
    
    # 4. Get the list of PDF files to process
    if not os.path.exists(PDF_DIRECTORY):
        print(f"Error: PDF directory '{PDF_DIRECTORY}' not found.")
        os.makedirs(PDF_DIRECTORY)
        print("Directory created. Please add PDFs and run again.")
        return

    all_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("All PDF files have already been processed.")
    else:
        print(f"Processing {len(files_to_process)} new PDF files...")

    # 5. Process each file one-by-one
    for filename in tqdm(files_to_process):
        try:
            # print(f"\n--- Processing file: {filename} ---")
            file_path = os.path.join(PDF_DIRECTORY, filename)

            # Load and chunk the document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = get_text_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if not chunks:
                # print("  - No chunks were created. Skipping file.")
                processed_files.add(filename)
                save_progress(processed_files)
                continue

            # Add chunks to the vector store
            if db is None:
                # Create the store with the first batch of chunks
                # print("  - Creating new FAISS index.")
                db = FAISS.from_documents(chunks, embeddings)
            else:
                # Add to the existing store
                # print("  - Adding chunks to existing FAISS index.")
                db.add_documents(chunks)
            
            # Persist the updated vector store to disk
            # print(f"  - Saving FAISS index to '{FAISS_INDEX_PATH}'...")
            db.save_local(FAISS_INDEX_PATH)

            # Mark this file as processed and save progress
            processed_files.add(filename)
            save_progress(processed_files)
            # print(f"  - Successfully processed and saved progress for {filename}.")

        except Exception as e:
            print(f"\n!!! An error occurred while processing {filename}: {e} !!!")
            print("The script will stop. Progress up to the previous file has been saved.")
            print("You can run the script again to resume processing from this point.")
            return # Stop execution

    # 6. Final summary
    print("\n--- All processing complete ---")
    print(f"Vector store is up-to-date with {len(processed_files)} files in '{FAISS_INDEX_PATH}'.")


if __name__ == "__main__":
    main(chunk_size=2000, chunk_overlap=300)
