#!/usr/bin/env python3
"""
Test script to verify progress callback is working in the app context.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai import KnowAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(message)s')

def test_progress_callback(message: str, level: str, data: dict):
    """Test progress callback function."""
    print(f"[PROGRESS] {message}")
    if data:
        print(f"  Data: {data}")

async def test_progress_in_app():
    """Test the progress callback in the app context."""
    print("Testing progress callback in app context...")
    
    # Create a simple agent with a dummy vectorstore
    dummy_vectorstore = "/tmp/test_vectorstore"
    os.makedirs(dummy_vectorstore, exist_ok=True)
    
    # Create dummy files
    with open(os.path.join(dummy_vectorstore, "index.faiss"), "w") as f:
        f.write("dummy")
    with open(os.path.join(dummy_vectorstore, "index.pkl"), "w") as f:
        f.write("dummy")
    
    try:
        agent = KnowAIAgent(
            vectorstore_path=dummy_vectorstore,
            process_files_individually=True
        )
        
        print("Agent created successfully")
        
        # Test a simple process_turn call
        result = await agent.process_turn(
            user_question="Test question",
            selected_files=["test_file.txt"],
            progress_cb=test_progress_callback
        )
        
        print(f"Process turn completed: {result}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import shutil
        if os.path.exists(dummy_vectorstore):
            shutil.rmtree(dummy_vectorstore)

if __name__ == "__main__":
    asyncio.run(test_progress_in_app()) 