#!/usr/bin/env python3
"""
Test script for the new individual file processing functionality.

This script tests both traditional batch processing and individual file processing
modes to ensure they work correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import KnowAIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_processing_modes(vectorstore_path: str):
    """Test both processing modes with the same question and files."""
    
    question = "What are the main vegetation management strategies mentioned?"
    selected_files = ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
    
    print("=" * 80)
    print("TESTING KNOWAI INDIVIDUAL FILE PROCESSING")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Files: {selected_files}")
    print(f"Vectorstore: {vectorstore_path}")
    print()
    
    # Test 1: Traditional batch processing
    print("🧪 TEST 1: Traditional Batch Processing")
    print("-" * 50)
    
    agent_batch = KnowAIAgent(
        vectorstore_path=vectorstore_path,
        process_files_individually=False,
        log_graph=False
    )
    
    try:
        start_time = asyncio.get_event_loop().time()
        result_batch = await agent_batch.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=False  # Use small model for faster testing
        )
        batch_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Batch processing completed in {batch_time:.2f} seconds")
        print(f"   Response length: {len(result_batch.get('generation', ''))} characters")
        print(f"   Files processed: {len(result_batch.get('documents_by_file', {}))}")
        
        # Show preview
        generation = result_batch.get('generation', '')
        if generation:
            preview = generation[:200] + "..." if len(generation) > 200 else generation
            print(f"   Preview: {preview}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 2: Individual file processing (asynchronous)
    print("🧪 TEST 2: Individual File Processing (Asynchronous)")
    print("-" * 50)
    
    agent_individual = KnowAIAgent(
        vectorstore_path=vectorstore_path,
        process_files_individually=True,
        log_graph=False
    )
    
    try:
        start_time = asyncio.get_event_loop().time()
        result_individual = await agent_individual.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=False  # Use small model for faster testing
        )
        individual_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Individual processing completed in {individual_time:.2f} seconds")
        print(f"   Response length: {len(result_individual.get('generation', ''))} characters")
        print(f"   Files processed: {len(result_individual.get('documents_by_file', {}))}")
        
        # Show preview
        generation = result_individual.get('generation', '')
        if generation:
            preview = generation[:200] + "..." if len(generation) > 200 else generation
            print(f"   Preview: {preview}")
        
    except Exception as e:
        print(f"❌ Individual processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)

def main():
    """Main function to run the tests."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_individual_processing.py <vectorstore_path>")
        print("Example: python scripts/test_individual_processing.py ./vectorstores/my_vectorstore")
        sys.exit(1)
    
    vectorstore_path = sys.argv[1]
    
    if not Path(vectorstore_path).exists():
        print(f"Error: Vectorstore path does not exist: {vectorstore_path}")
        sys.exit(1)
    
    asyncio.run(test_processing_modes(vectorstore_path))

if __name__ == "__main__":
    main() 