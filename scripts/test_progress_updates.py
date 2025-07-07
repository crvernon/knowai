#!/usr/bin/env python3
"""
Test script to verify progress updates work correctly with the new counter system.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai import KnowAIAgent
from knowai.agent import _update_progress_callback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s')

def test_progress_callback(message: str, level: str, data: dict):
    """Test progress callback function."""
    print(f"[{level.upper()}] {message}")
    if data:
        print(f"  Data: {data}")

async def test_progress_updates():
    """Test the progress update system."""
    print("Testing progress updates...")
    
    # Create a mock state with progress callback
    mock_state = {
        "__progress_cb__": test_progress_callback
    }
    
    # Test individual file processing progress
    print("\n--- Testing Individual File Processing Progress ---")
    _update_progress_callback(
        mock_state, 
        "process_individual_files_node", 
        "individual_processing",
        {"completed": 0, "total": 10}
    )
    
    _update_progress_callback(
        mock_state, 
        "process_individual_files_node", 
        "individual_processing",
        {"completed": 5, "total": 10}
    )
    
    _update_progress_callback(
        mock_state, 
        "process_individual_files_node", 
        "individual_processing",
        {"completed": 10, "total": 10}
    )
    
    # Test batch processing progress
    print("\n--- Testing Batch Processing Progress ---")
    _update_progress_callback(
        mock_state, 
        "process_batches_node", 
        "processing_batch_1_3",
        {"completed": 0, "total": 3, "current_batch": 1}
    )
    
    _update_progress_callback(
        mock_state, 
        "process_batches_node", 
        "processing_batch_2_3",
        {"completed": 1, "total": 3, "current_batch": 2}
    )
    
    _update_progress_callback(
        mock_state, 
        "process_batches_node", 
        "processing_batch_3_3",
        {"completed": 2, "total": 3, "current_batch": 3}
    )
    
    # Test regular progress updates
    print("\n--- Testing Regular Progress Updates ---")
    _update_progress_callback(mock_state, "instantiate_embeddings_node", "initialization")
    _update_progress_callback(mock_state, "generate_multi_queries_node", "query_generation")
    _update_progress_callback(mock_state, "extract_documents_node", "document_retrieval")
    
    print("\n--- Progress Update Tests Completed ---")

if __name__ == "__main__":
    asyncio.run(test_progress_updates()) 