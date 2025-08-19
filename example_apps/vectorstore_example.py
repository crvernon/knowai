#!/usr/bin/env python3
"""
Example script demonstrating KnowAI vectorstore functionality.

This script shows how to use the vectorstore module to build and manage FAISS vector stores
from PDF documents.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai import (
    get_retriever_from_directory,
    load_vectorstore,
    show_vectorstore_schema,
    list_vectorstore_files
)


def main():
    """Example usage of KnowAI vectorstore functionality."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example paths - adjust these to your actual paths
    pdf_directory = "path/to/your/pdfs"
    metadata_path = "path/to/your/metadata.parquet"
    vectorstore_path = "example_faiss_store"
    
    # Check if paths exist
    if not Path(pdf_directory).exists():
        logger.error(f"PDF directory '{pdf_directory}' does not exist. Please update the path.")
        return 1
        
    if not Path(metadata_path).exists():
        logger.error(f"Metadata file '{metadata_path}' does not exist. Please update the path.")
        return 1
    
    # Example 1: Build a vectorstore from PDFs
    logger.info("Building vectorstore from PDFs...")
    retriever = get_retriever_from_directory(
        directory_path=pdf_directory,
        persist_directory=vectorstore_path,
        persist=True,
        metadata_parquet_path=metadata_path,
        k=10,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    if retriever is None:
        logger.error("Failed to build vectorstore.")
        return 1
    
    logger.info("Vectorstore built successfully!")
    
    # Example 2: Load an existing vectorstore
    logger.info("Loading existing vectorstore...")
    loaded_retriever = load_vectorstore(vectorstore_path, k=10)
    
    if loaded_retriever is None:
        logger.error("Failed to load vectorstore.")
        return 1
    
    logger.info("Vectorstore loaded successfully!")
    
    # Example 3: Inspect vectorstore schema
    logger.info("Inspecting vectorstore schema...")
    schema = show_vectorstore_schema(loaded_retriever.vectorstore)
    logger.info("Vectorstore schema:")
    for key, value in schema.items():
        logger.info(f"  {key}: {value}")
    
    # Example 4: List files in vectorstore
    logger.info("Listing files in vectorstore...")
    files = list_vectorstore_files(loaded_retriever.vectorstore)
    logger.info(f"Files in vectorstore: {files}")
    
    # Example 5: Use the retriever for similarity search
    logger.info("Testing similarity search...")
    try:
        results = loaded_retriever.get_relevant_documents("example query")
        logger.info(f"Retrieved {len(results)} documents")
        for i, doc in enumerate(results[:3]):  # Show first 3 results
            logger.info(f"  Result {i+1}: {doc.metadata.get('file_name', 'Unknown')} - Page {doc.metadata.get('page', 'Unknown')}")
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")
    
    logger.info("Vectorstore example completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
