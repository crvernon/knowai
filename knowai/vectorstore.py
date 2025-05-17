import logging
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# instantiate global logger
logger = logging.getLogger(__name__)


def load_vectorstore(
        vectorstore_directory: str, 
        embeddings: AzureOpenAIEmbeddings,
        allow_dangerous_deserialization: bool = True

) -> Optional[object]:
    """
    Load a persisted FAISS vector store from disk and return a retriever.
    """
    if not os.path.exists(vectorstore_directory):
        logger.error(f"Vectorstore directory '{vectorstore_directory}' does not exist.")
        return None
    
    try:
        vectorstore = FAISS.load_local(
            vectorstore_directory,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )

        logger.info(f"Loaded FAISS vector store from {vectorstore_directory}")
        return vectorstore # vectorstore.as_retriever(search_kwargs={"k": k})
    
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from {vectorstore_directory}: {e}")
        return None


def show_vectorstore_schema(vectorstore):
    """
    Display key information about the FAISS vectorstore:
    - Total number of vectors
    - Embedding dimension (if available)
    - Metadata fields present in the stored documents
    Returns a dict with these details.
    """
    if vectorstore is None:
        logger.error("Cannot show schema: vectorstore is None")
        return {}
    
    # FAISS index info
    try:
        total_vectors = vectorstore.index.ntotal
    except Exception:
        total_vectors = None

    try:
        dimension = vectorstore.index.d
    except Exception:
        dimension = None

    # Collect metadata keys
    metadata_keys = set()
    for _, doc in vectorstore.docstore._dict.items():
        if isinstance(doc.metadata, dict):
            metadata_keys.update(doc.metadata.keys())

    schema = {
        "total_vectors": total_vectors,
        "dimension": dimension,
        "metadata_fields": sorted(metadata_keys),
    }
    return schema


def list_vectorstore_assets(
            
):
    try:
        files_in_store = list_vectorstore_files(vectorstore)
        logging.info(f"Found {len(files_in_store)} files in vector store index: {files_in_store}")
    except Exception as e:
            logging.error(f"Error calling list_vectorstore_files: {e}")
            st.error(f"Error listing files from vector store: {e}")
            files_in_store = []
