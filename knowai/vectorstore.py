"""
Vectorstore utilities for KnowAI package.

This module provides functionality for building, loading, and managing FAISS vector stores
from PDF documents and metadata.

Environment Variables Required:
- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
- AZURE_EMBEDDINGS_DEPLOYMENT: Your embeddings model deployment name (defaults to "text-embedding-3-large")
- AZURE_OPENAI_EMBEDDINGS_API_VERSION: Your embeddings API version (defaults to "2024-02-01")
"""

import os
import fitz  # PyMuPDF
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from .utils import get_azure_credentials

logger = logging.getLogger(__name__)


def process_pdfs_to_documents(
    directory_path: str,
    metadata_map: dict,
    existing_files: set,
    text_splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Process PDF files in a directory, split into chunks, and return a list of Document objects.
    Skips files not in metadata_map or already in existing_files.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing PDF files
    metadata_map : dict
        Mapping of filename to metadata dictionary
    existing_files : set
        Set of filenames already processed
    text_splitter : RecursiveCharacterTextSplitter
        Text splitter instance for chunking
        
    Returns
    -------
    List[Document]
        List of Document objects with chunked text and metadata
    """
    new_docs: List[Document] = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        if filename not in metadata_map:
            logger.warning(f"Skipping {filename}: not found in metadata parquet.")
            continue
        if filename in existing_files:
            logger.info(f"Skipping {filename}: already in vector store.")
            continue
            
        file_path = os.path.join(directory_path, filename)
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Error opening PDF {filename}: {e}")
            continue
            
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text()
            except Exception as e:
                logger.error(f"Error reading page {page_num+1} of {filename}: {e}")
                continue
                
            if not text:
                continue
                
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                meta = dict(metadata_map[filename])
                meta["page"] = page_num + 1
                new_docs.append(Document(
                    page_content=chunk,
                    metadata=meta
                ))
        doc.close()
        
    return new_docs


def get_retriever_from_docs(
    docs: list,
    persist_directory: str = "faiss_store",
    persist: bool = True,
    k: int = 10,
    embeddings: Optional[AzureOpenAIEmbeddings] = None,
) -> Optional[object]:
    """
    Given a list of Document objects, creates or updates a FAISS vector store of all chunks,
    and returns a retriever.
    
    Parameters
    ----------
    docs : list
        List of Document objects to add to vector store
    persist_directory : str, default "faiss_store"
        Directory to persist the FAISS index
    persist : bool, default True
        Whether to persist the vector store to disk
    k : int, default 10
        Number of top results to return from retriever
    embeddings : Optional[AzureOpenAIEmbeddings], default None
        Embeddings instance to use. If None, will create from Azure credentials
        
    Returns
    -------
    Optional[object]
        FAISS retriever object or None if error
    """
    # Get Azure credentials
    credentials = get_azure_credentials()
    if not credentials:
        logger.error("Azure credentials not available for vector store building.")
        return None

    # Initialize embeddings if not provided
    if embeddings is None:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=credentials["embeddings_deployment"],
            azure_endpoint=credentials["azure_endpoint"],
            api_key=credentials["api_key"],
            api_version=credentials["embeddings_api_version"]
        )

    # Load or initialize vectorstore
    if persist and os.path.exists(persist_directory):
        try:
            vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS store from {persist_directory}")
        except Exception as e:
            logger.error(f"Error loading existing FAISS store: {e}")
            vectorstore = None
    else:
        vectorstore = None

    # Determine which files are already present
    existing_files = set()
    if vectorstore:
        for _, doc in vectorstore.docstore._dict.items():
            file_name = doc.metadata.get("file_name")
            if file_name:
                existing_files.add(file_name)

    # Filter out docs that are already present (by file_name)
    new_docs = []
    for doc in docs:
        file_name = doc.metadata.get("file_name")
        if file_name and file_name in existing_files:
            logger.info(f"Skipping {file_name}: already in vector store.")
            continue
        new_docs.append(doc)

    if not new_docs and not vectorstore:
        logger.error(f"No valid chunks to add to vectorstore.")
        return None

    # Build or update vectorstore
    if vectorstore:
        vectorstore.add_documents(new_docs)
        logger.info(f"Added {len(new_docs)} new chunks to FAISS store")
    else:
        vectorstore = FAISS.from_documents(documents=new_docs, embedding=embeddings)
        logger.info(f"Created new FAISS store with {len(new_docs)} chunks")

    # Persist if required
    if persist:
        vectorstore.save_local(persist_directory)
        logger.info(f"FAISS vector store saved to {persist_directory}")

    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_retriever_from_directory(
    directory_path: str,
    persist_directory: str = "faiss_store",
    persist: bool = True,
    metadata_parquet_path: str = "metadata.parquet",
    k: int = 10,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Optional[object]:
    """
    Build a FAISS vector store from PDF files in a directory with metadata from a parquet file.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing PDF files
    persist_directory : str, default "faiss_store"
        Directory to persist the FAISS index
    persist : bool, default True
        Whether to persist the vector store to disk
    metadata_parquet_path : str, default "metadata.parquet"
        Path to parquet file containing metadata
    k : int, default 10
        Number of top results to return from retriever
    chunk_size : int, default 1000
        Size of text chunks
    chunk_overlap : int, default 200
        Overlap between text chunks
        
    Returns
    -------
    Optional[object]
        FAISS retriever object or None if error
    """
    # Load metadata
    if not os.path.exists(metadata_parquet_path):
        logger.error(f"Metadata file {metadata_parquet_path} not found.")
        return None
        
    try:
        metadata_df = pd.read_parquet(metadata_parquet_path)
        metadata_map = metadata_df.set_index('file_name').to_dict('index')
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_parquet_path}: {e}")
        return None

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Get existing files from vectorstore if it exists
    existing_files = set()
    if persist and os.path.exists(persist_directory):
        try:
            credentials = get_azure_credentials()
            if credentials:
                embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=credentials["embeddings_deployment"],
                    azure_endpoint=credentials["azure_endpoint"],
                    api_key=credentials["api_key"],
                    api_version=credentials["embeddings_api_version"]
                )
                vectorstore = FAISS.load_local(
                    persist_directory,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                for _, doc in vectorstore.docstore._dict.items():
                    file_name = doc.metadata.get("file_name")
                    if file_name:
                        existing_files.add(file_name)
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore to check files: {e}")

    # Process PDFs to documents
    docs = process_pdfs_to_documents(
        directory_path=directory_path,
        metadata_map=metadata_map,
        existing_files=existing_files,
        text_splitter=text_splitter,
    )

    # Build vectorstore
    return get_retriever_from_docs(
        docs=docs,
        persist_directory=persist_directory,
        persist=persist,
        k=k
    )


def show_vectorstore_schema(vectorstore) -> Dict[str, Any]:
    """
    Display key information about the FAISS vectorstore.
    
    Parameters
    ----------
    vectorstore
        FAISS vectorstore instance or retriever object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing schema information
    """
    if vectorstore is None:
        logger.error("Cannot show schema: vectorstore is None")
        return {}
    
    # Handle both vectorstore objects and retriever objects
    actual_vectorstore = vectorstore
    if hasattr(vectorstore, 'vectorstore'):
        # This is a retriever, get the underlying vectorstore
        actual_vectorstore = vectorstore.vectorstore
        
    # FAISS index info
    try:
        total_vectors = actual_vectorstore.index.ntotal
    except Exception:
        total_vectors = None

    try:
        dimension = actual_vectorstore.index.d
    except Exception:
        dimension = None

    # Collect metadata keys
    metadata_keys = set()
    try:
        for _, doc in actual_vectorstore.docstore._dict.items():
            if isinstance(doc.metadata, dict):
                metadata_keys.update(doc.metadata.keys())
    except Exception as e:
        logger.warning(f"Could not access docstore metadata: {e}")
        metadata_keys = set()

    schema = {
        "total_vectors": total_vectors,
        "dimension": dimension,
        "metadata_fields": sorted(metadata_keys),
    }
    return schema


def load_vectorstore(persist_directory: str, k: int = 10) -> Optional[object]:
    """
    Load a persisted FAISS vector store from disk and return a retriever.
    
    Parameters
    ----------
    persist_directory : str
        Directory containing the persisted FAISS store
    k : int, default 10
        Number of top results to return from retriever
        
    Returns
    -------
    Optional[object]
        FAISS retriever object or None if error
    """
    if not os.path.exists(persist_directory):
        logger.error(f"Persist directory '{persist_directory}' does not exist.")
        return None
        
    try:
        credentials = get_azure_credentials()
        if not credentials:
            logger.error("Azure credentials not available.")
            return None
            
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=credentials["embeddings_deployment"],
            azure_endpoint=credentials["azure_endpoint"],
            api_key=credentials["api_key"],
            api_version=credentials["embeddings_api_version"]
        )
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS vector store from {persist_directory}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
    
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from {persist_directory}: {e}")
        return None


def list_vectorstore_files(vectorstore) -> List[str]:
    """
    Return a sorted list of unique PDF filenames stored in the FAISS vectorstore metadata.
    
    Parameters
    ----------
    vectorstore
        FAISS vectorstore instance or retriever object
        
    Returns
    -------
    List[str]
        List of filenames in the vectorstore
    """
    if vectorstore is None:
        logger.error("Cannot list files: vectorstore is None")
        return []
    
    # Handle both vectorstore objects and retriever objects
    actual_vectorstore = vectorstore
    if hasattr(vectorstore, 'vectorstore'):
        # This is a retriever, get the underlying vectorstore
        actual_vectorstore = vectorstore.vectorstore
        
    files = set()
    try:
        # Access the underlying docstore dictionary
        for _, doc in actual_vectorstore.docstore._dict.items():
            # Try different possible field names for filename
            filename = doc.metadata.get("file") or doc.metadata.get("file_name") or doc.metadata.get("filename")
            if filename:
                files.add(filename)
    except Exception as e:
        logger.warning(f"Could not access docstore metadata: {e}")
        return []
            
    file_list = sorted(files)
    logger.info(f"Files in vectorstore: {file_list}")
    return file_list
