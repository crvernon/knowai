import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# --- Configuration ---
load_dotenv("/Users/d3y010/repos/crvernon/knowai/.env", override=True)
FAISS_INDEX_PATH = "/Users/d3y010/repos/crvernon/knowai/vectorstores/faiss_openai_large_20250606"
METADATA_PARQUET_PATH = "/Users/d3y010/repos/crvernon/knowai/vectorstores/faiss_openai_large_20250606_metadata.parquet"

def export_metadata_to_parquet():
    """
    Loads a FAISS vector store, extracts all metadata and FAISS index IDs,
    and saves them to a Parquet file for efficient filtering.
    """
    print("--- Starting Metadata Export ---")

    # 1. Load the existing vector store
    print(f"Loading vector store from: {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Error: FAISS index path not found.")
        return
        
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION")
        )
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # 2. Extract metadata and index IDs
    print("Extracting metadata from the docstore...")
    
    # The 'index_to_docstore_id' is a dictionary mapping the
    # sequential numerical ID from the FAISS index (e.g., 0, 1, 2...)
    # to the unique UUID string of the document in the docstore.
    faiss_ids = list(vectorstore.index_to_docstore_id.keys())
    
    all_metadata = []
    
    for faiss_id in tqdm(faiss_ids, desc="Processing documents"):
        # Get the document's UUID from the FAISS ID
        doc_id = vectorstore.index_to_docstore_id[faiss_id]
        
        # Get the document from the docstore using its UUID
        doc = vectorstore.docstore.search(doc_id)
        
        if doc and hasattr(doc, 'metadata'):
            # Create a record with the crucial FAISS index ID and the metadata
            record = {
                "faiss_id": faiss_id,
                "file_name": doc.metadata.get("file_name"),
                "page": doc.metadata.get("page")
            }
            all_metadata.append(record)

    if not all_metadata:
        print("No metadata could be extracted. Exiting.")
        return

    # 3. Convert to a Pandas DataFrame and save as Parquet
    print("Converting to DataFrame and saving to Parquet...")
    df = pd.DataFrame(all_metadata)
    
    try:
        df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
        print(f"Successfully saved metadata for {len(df)} chunks to '{OUTPUT_PARQUET_PATH}'")
    except Exception as e:
        print(f"Error saving Parquet file: {e}")

    print("\n--- Metadata Export Complete ---")


if __name__ == "__main__":
    export_metadata_to_parquet()
