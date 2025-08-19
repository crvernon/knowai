#!/usr/bin/env python3
"""
Command-line interface for KnowAI vectorstore operations.

This module provides CLI functionality for building, loading, and managing FAISS vector stores
from PDF documents and metadata.
"""

import argparse
import logging
import sys
from pathlib import Path

from .vectorstore import (
    get_retriever_from_directory,
    load_vectorstore,
    show_vectorstore_schema,
    list_vectorstore_files,
    analyze_vectorstore_chunking
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def build_vectorstore(args):
    """Build a vectorstore from PDF directory."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    pdf_directory = Path(args.pdf_directory)
    if not pdf_directory.exists():
        logger.error(f"PDF directory '{pdf_directory}' does not exist.")
        return 1
        
    metadata_path = Path(args.metadata_parquet_path)
    if not metadata_path.exists():
        logger.error(f"Metadata file '{metadata_path}' does not exist.")
        return 1
    
    # Build vectorstore
    logger.info(f"Building vectorstore from {pdf_directory}")
    retriever = get_retriever_from_directory(
        directory_path=str(pdf_directory),
        persist_directory=args.vectorstore_path,
        persist=True,
        metadata_parquet_path=str(metadata_path),
        k=args.k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if retriever is None:
        logger.error("Failed to build vectorstore.")
        return 1
        
    logger.info(f"Vectorstore built successfully at {args.vectorstore_path}")
    return 0


def inspect_vectorstore(args):
    """Inspect an existing vectorstore."""
    logger = logging.getLogger(__name__)
    
    # Load vectorstore
    vectorstore = load_vectorstore(args.vectorstore_path, k=args.k)
    if vectorstore is None:
        logger.error(f"Failed to load vectorstore from {args.vectorstore_path}")
        return 1
    
    # Show schema
    schema = show_vectorstore_schema(vectorstore)
    logger.info("Vectorstore schema:")
    for key, value in schema.items():
        logger.info(f"  {key}: {value}")
    
    # List files
    files = list_vectorstore_files(vectorstore)
    logger.info(f"Files in vectorstore: {files}")
    
    # Analyze chunking if requested
    if args.analyze_chunking:
        logger.info("Analyzing chunking parameters...")
        analysis = analyze_vectorstore_chunking(vectorstore)
        if analysis:
            logger.info("Chunking analysis:")
            logger.info(f"  Documents analyzed: {analysis['total_documents_analyzed']}")
            logger.info(f"  Estimated chunk size: {analysis['estimated_chunk_size']['average']} chars (avg), {analysis['estimated_chunk_size']['median']} chars (median)")
            logger.info(f"  Estimated overlap: {analysis['estimated_overlap']['average']} chars (avg), {analysis['estimated_overlap']['median']} chars (median)")
            logger.info(f"  Recommended settings: chunk_size={analysis['recommended_settings']['chunk_size']}, chunk_overlap={analysis['recommended_settings']['chunk_overlap']}")
            logger.info("  Chunk size distribution:")
            for range_name, count in analysis['estimated_chunk_size']['distribution'].items():
                logger.info(f"    {range_name}: {count} chunks")
        else:
            logger.warning("Could not analyze chunking parameters")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KnowAI Vectorstore CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build a vectorstore from PDFs
  python -m knowai.cli_vectorstore build /path/to/pdfs --metadata metadata.parquet
  
  # Inspect an existing vectorstore
  python -m knowai.cli_vectorstore inspect /path/to/vectorstore
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build a vectorstore from PDFs')
    build_parser.add_argument('pdf_directory', help='Path to directory containing PDF files')
    build_parser.add_argument('--vectorstore_path', default='faiss_store', 
                             help='Path to save/load the FAISS store')
    build_parser.add_argument('--metadata_parquet_path', default='metadata.parquet',
                             help='Path to the metadata parquet file')
    build_parser.add_argument('--k', type=int, default=10,
                             help='Number of top results to return from retriever')
    build_parser.add_argument('--chunk_size', type=int, default=1000,
                             help='Size of text chunks')
    build_parser.add_argument('--chunk_overlap', type=int, default=200,
                             help='Overlap between text chunks')
    build_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose logging')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect an existing vectorstore')
    inspect_parser.add_argument('vectorstore_path', help='Path to the FAISS store')
    inspect_parser.add_argument('--k', type=int, default=10,
                               help='Number of top results to return from retriever')
    inspect_parser.add_argument('--analyze-chunking', action='store_true',
                               help='Analyze chunk size and overlap parameters used in the vectorstore')
    inspect_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'build':
        return build_vectorstore(args)
    elif args.command == 'inspect':
        return inspect_vectorstore(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
