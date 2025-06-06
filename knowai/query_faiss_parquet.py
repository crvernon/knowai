import os
import pandas as pd
import numpy as np
import faiss  # We need the core faiss library now
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
load_dotenv("/Users/d3y010/repos/crvernon/knowai/.env", override=True)
FAISS_INDEX_PATH = "/Users/d3y010/repos/crvernon/knowai/vectorstores/faiss_openai_large_20250606"
METADATA_PARQUET_PATH = "/Users/d3y010/repos/crvernon/knowai/faiss_openai_large_20250606_metadata.parquet"


def query_with_pre_filter(query: str, file_name_filter: str, k: int = 2):
    """
    Performs a "filter-then-search" query.
    1. Filters metadata from a Parquet file.
    2. Uses the resulting IDs to conduct a targeted FAISS search.
    """
    print("--- Starting Query with Pre-filter ---")

    # 1. Load the vector store and embeddings model
    print("Loading vector store and embeddings...")
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
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # 2. Load metadata from Parquet and apply the filter
    print(f"Loading metadata and filtering for file: '{file_name_filter}'")
    try:
        metadata_df = pd.read_parquet(METADATA_PARQUET_PATH)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{METADATA_PARQUET_PATH}'.")
        print("Please run the export script first.")
        return

    # This is the "filter first" step
    filtered_ids = metadata_df[metadata_df['file_name'] == file_name_filter]['faiss_id'].tolist()
    
    if not filtered_ids:
        print("No documents found for the specified file name in the metadata.")
        return []

    print(f"Found {len(filtered_ids)} chunks belonging to the specified file.")
    
    # 3. Perform the targeted similarity search using the core faiss library
    print("Performing targeted search on the filtered IDs...")
    
    # Get the raw query vector
    query_vector = embeddings.embed_query(query)
    query_vector_np = np.array([query_vector], dtype=np.float32)

    # Get the raw FAISS index from the LangChain wrapper
    index = vectorstore.index

    # Create an ID selector to restrict the search space
    id_selector = faiss.IDSelectorArray(np.array(filtered_ids, dtype=np.int64))

    # Instantiate faiss.SearchParameters and then set the selector attribute.
    search_params = faiss.SearchParameters()
    search_params.selector = id_selector

    # The actual search call on the raw index, now with the correct params object.
    distances, result_faiss_ids = index.search(
        query_vector_np, 
        k=k, 
        params=search_params
    )

    # 4. Retrieve and VALIDATE the full documents for the results
    print("Retrieving full documents for search results...")
    
    final_docs = []
    if result_faiss_ids.size > 0:
        for faiss_id in result_faiss_ids[0]:
            if faiss_id != -1:  # faiss returns -1 for no result
                doc_id = vectorstore.index_to_docstore_id[faiss_id]
                doc = vectorstore.docstore.search(doc_id)
                
                # --- VALIDATION STEP ---
                # Check if the retrieved document's metadata matches the filter.
                # If it doesn't, it indicates the metadata file is out of sync.
                if doc and doc.metadata.get("file_name") == file_name_filter:
                    final_docs.append(doc)
                elif doc:
                    # This case should not happen if data is perfectly synced.
                    print("\n!!! WARNING: Data Integrity Issue Detected !!!")
                    print(f"  - Searched for file: '{file_name_filter}'")
                    print(f"  - FAISS ID {faiss_id} was in the filtered set for this file.")
                    print(f"  - However, this ID resolved to a document from a DIFFERENT file: '{doc.metadata.get('file_name')}'")
                    print("  - This almost always means the 'metadata.parquet' file is out of sync with the FAISS index.")
                    print("  - Please re-run the metadata export script to resolve this.\n")

    return final_docs


if __name__ == "__main__":
    query = "vegetation management"
    file_filter = "Hawaiian_Electric_Company_2024.pdf"
    
    results = query_with_pre_filter(query, file_filter, k=2)
    
    print("\n--- Query Results ---")
    if results:
        for doc in results:
            print(f"File: {doc.metadata.get('file_name')}, Page: {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:400]}...\n")
    else:
        print("No results found.")
