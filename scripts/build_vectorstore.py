import os
import fitz  # PyMuPDF
import logging
from dotenv import load_dotenv
from typing import List, Optional
import argparse
from tqdm import tqdm
import pandas as pd


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Load credentials; make sure dotenv overwrites any system variable settings
load_dotenv(".env", override=True)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai_embeddings_api_version = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION")

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
        embeddings: "AzureOpenAIEmbeddings" = None,
) -> Optional[object]:
    """
    Given a list of Document objects, creates or updates a FAISS vector store of all chunks,
    and returns a retriever. By default, the FAISS index is written to disk in `persist_directory`;
    set `persist=False` to skip saving and keep the index in memory only. If the store already exists,
    new docs are only added if not already present, based on filename metadata.
    """
    if not api_key or not azure_endpoint:
        logger.error("Azure credentials not available for vector store building.")
        return None

    # Initialize embeddings if not provided
    if embeddings is None:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_embeddings_api_version
        )

    # Load or initialize vectorstore
    if persist and os.path.exists(persist_directory):
        try:
            vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS store from {persist_directory} with dangerous deserialization allowed")
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


# Utility function: show_vectorstore_schema
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


def load_vectorstore(persist_directory: str, k: int = 10) -> Optional[object]:
    """
    Load a persisted FAISS vector store from disk and return a retriever.
    """
    if not os.path.exists(persist_directory):
        logger.error(f"Persist directory '{persist_directory}' does not exist.")
        return None
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embeddings_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version
        )
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS vector store from {persist_directory}")
        return vectorstore # vectorstore.as_retriever(search_kwargs={"k": k})
    
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from {persist_directory}: {e}")
        return None


def list_vectorstore_files(vectorstore) -> List[str]:
    """
    Return a sorted list of unique PDF filenames stored in the FAISS vectorstore metadata.
    """
    if vectorstore is None:
        logger.error("Cannot list files: vectorstore is None")
        return []
    files = set()
    # Access the underlying docstore dictionary
    for _, doc in vectorstore.docstore._dict.items():
        filename = doc.metadata.get("file")
        if filename:
            files.add(filename)
    file_list = sorted(files)
    logger.info(f"Files in vectorstore: {file_list}")
    return file_list


if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Build and load a FAISS vector store from PDFs.")
    parser.add_argument("pdf_directory", help="Path to the directory containing PDF files")
    parser.add_argument("--vectorstore_path", default="test_faiss_store", help="Path to save/load the FAISS store")
    parser.add_argument("--metadata_parquet_path", default="metadata.parquet", help="Path to the metadata parquet file")
    args = parser.parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO)

    # build or update the vector store
    get_retriever_from_directory(
        directory_path=args.pdf_directory,
        persist_directory=args.vectorstore_path,
        persist=True,
        metadata_parquet_path=args.metadata_parquet_path,
    )

    # load and inspect the store
    vs = load_vectorstore(args.vectorstore_path)
    schema = show_vectorstore_schema(vs)
    logger.info(f"Vectorstore schema: {schema}")
    files = list_vectorstore_files(vs)
    logger.info(f"Stored files: {files}")
