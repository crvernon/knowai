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
from collections import defaultdict

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
    min_chunk_chars: int = 20,
) -> List[Document]:
    """
    Process PDF files in a directory, split into chunks, and return a list of Document objects.
    Processes all PDFs. If a file is missing from metadata_map, minimal metadata will be used. Each chunk will include 'file_name' and 'page'. Very short chunks (length < min_chunk_chars) are skipped.
    
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
    min_chunk_chars : int, default 20
        Minimum number of characters required for a chunk to be added.
        
    Returns
    -------
    List[Document]
        List of Document objects with chunked text and metadata
    """
    new_docs: List[Document] = []
    pdf_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")])
    
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        if filename not in metadata_map:
            logger.warning(f"{filename} not found in metadata parquet. Proceeding with minimal metadata.")
        if filename in existing_files:
            logger.info(f"{filename} appears in existing vector store; continuing to process pages (dedup handled later).")
            
        file_path = os.path.join(directory_path, filename)
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Error opening PDF {filename}: {e}")
            continue
            
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                # Try standard text extraction; fall back to block-based if empty
                text = (page.get_text("text") or "").strip()
                if not text:
                    try:
                        blocks = page.get_text("blocks") or []
                        # blocks: list of tuples (x0, y0, x1, y1, text, block_no, block_type)
                        # sort by top-left corner (y0, x0) to approximate reading order
                        blocks_sorted = sorted([b for b in blocks if len(b) >= 5 and b[4]], key=lambda b: (b[1], b[0]))
                        text = "\n".join(b[4] for b in blocks_sorted).strip()
                    except Exception:
                        text = ""
            except Exception as e:
                logger.error(f"Error reading page {page_num+1} of {filename}: {e}")
                continue
                
            if not text:
                continue
                
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks, start=1):
                c = (chunk or "").strip()
                if len(c) < min_chunk_chars:
                    continue
                base_meta = dict(metadata_map.get(filename, {}))
                base_meta.update({
                    "file_name": filename,
                    "source_path": os.path.join(directory_path, filename),
                    "page": page_num + 1,
                    "chunk_index": i,
                })
                new_docs.append(Document(page_content=c, metadata=base_meta))
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

    # Determine which files and pages are already present
    existing_files = set()
    existing_pages_by_file = defaultdict(set)
    if vectorstore:
        for _, doc in vectorstore.docstore._dict.items():
            file_name = doc.metadata.get("file_name") or doc.metadata.get("file") or doc.metadata.get("filename")
            page_no = doc.metadata.get("page")
            if file_name:
                existing_files.add(file_name)
                if isinstance(page_no, int):
                    existing_pages_by_file[file_name].add(page_no)

    # Filter out docs that are already present (by file_name + page)
    new_docs = []
    for doc in docs:
        file_name = doc.metadata.get("file_name") or doc.metadata.get("file") or doc.metadata.get("filename")
        page_no = doc.metadata.get("page")
        if file_name and isinstance(page_no, int) and page_no in existing_pages_by_file.get(file_name, set()):
            logger.info(f"Skipping {file_name} page {page_no}: already in vector store.")
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
    chunk_size: int = 1400,
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


def analyze_vectorstore_chunking(vectorstore) -> Dict[str, Any]:
    """
    Analyze chunk size and overlap from an existing vectorstore by examining stored documents.
    
    This function samples documents from the vectorstore and analyzes their characteristics
    to estimate the chunk size and overlap parameters used during creation.
    
    Parameters
    ----------
    vectorstore
        FAISS vectorstore instance or retriever object
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results including estimated chunk size, overlap, and statistics
    """
    if vectorstore is None:
        logger.error("Cannot analyze chunking: vectorstore is None")
        return {}
    
    # Handle both vectorstore objects and retriever objects
    actual_vectorstore = vectorstore
    if hasattr(vectorstore, 'vectorstore'):
        # This is a retriever, get the underlying vectorstore
        actual_vectorstore = vectorstore.vectorstore
    
    try:
        # Sample documents for analysis
        sample_docs = []
        doc_items = list(actual_vectorstore.docstore._dict.items())
        
        # Take a sample of documents (up to 1000 for analysis)
        sample_size = min(1000, len(doc_items))
        import random
        random.seed(42)  # For reproducible results
        sampled_items = random.sample(doc_items, sample_size)
        
        for _, doc in sampled_items:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                sample_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
        
        if not sample_docs:
            logger.warning("No documents found in vectorstore for analysis")
            return {}
        
        # Analyze chunk sizes
        chunk_lengths = [len(doc['content']) for doc in sample_docs]
        avg_chunk_size = sum(chunk_lengths) / len(chunk_lengths)
        median_chunk_size = sorted(chunk_lengths)[len(chunk_lengths) // 2]
        min_chunk_size = min(chunk_lengths)
        max_chunk_size = max(chunk_lengths)
        
        # Analyze overlap by looking at consecutive chunks from the same page
        overlap_estimates = []
        pages_by_file = {}
        
        # Group documents by file and page
        for doc in sample_docs:
            file_name = doc['metadata'].get('file_name') or doc['metadata'].get('file') or doc['metadata'].get('filename')
            page = doc['metadata'].get('page')
            if file_name and page:
                key = (file_name, page)
                if key not in pages_by_file:
                    pages_by_file[key] = []
                pages_by_file[key].append(doc)
        
        # Sort chunks within each page by chunk_index if available
        for key, docs in pages_by_file.items():
            if len(docs) > 1:
                # Sort by chunk_index if available, otherwise by content length (rough approximation)
                docs.sort(key=lambda x: x['metadata'].get('chunk_index', len(x['content'])))
                
                # Analyze consecutive chunks for overlap
                for i in range(len(docs) - 1):
                    chunk1 = docs[i]['content']
                    chunk2 = docs[i + 1]['content']
                    
                    # Find common text at the end of chunk1 and beginning of chunk2
                    overlap = 0
                    for j in range(min(50, len(chunk1), len(chunk2))):  # Check up to 50 characters
                        if chunk1[-(j+1):] == chunk2[:j+1]:
                            overlap = j + 1
                    
                    if overlap > 0:
                        overlap_estimates.append(overlap)
        
        avg_overlap = sum(overlap_estimates) / len(overlap_estimates) if overlap_estimates else 0
        median_overlap = sorted(overlap_estimates)[len(overlap_estimates) // 2] if overlap_estimates else 0
        
        # Analyze chunk distribution
        chunk_size_distribution = {
            '0-500': len([l for l in chunk_lengths if l <= 500]),
            '500-1000': len([l for l in chunk_lengths if 500 < l <= 1000]),
            '1000-1500': len([l for l in chunk_lengths if 1000 < l <= 1500]),
            '1500-2000': len([l for l in chunk_lengths if 1500 < l <= 2000]),
            '2000+': len([l for l in chunk_lengths if l > 2000])
        }
        
        analysis = {
            'total_documents_analyzed': len(sample_docs),
            'estimated_chunk_size': {
                'average': round(avg_chunk_size, 1),
                'median': median_chunk_size,
                'min': min_chunk_size,
                'max': max_chunk_size,
                'distribution': chunk_size_distribution
            },
            'estimated_overlap': {
                'average': round(avg_overlap, 1),
                'median': median_overlap,
                'samples_analyzed': len(overlap_estimates)
            },
            'recommended_settings': {
                'chunk_size': round(avg_chunk_size),
                'chunk_overlap': round(avg_overlap)
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing vectorstore chunking: {e}")
        return {}
