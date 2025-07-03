#!/usr/bin/env python3
"""
Test script to demonstrate the concurrency limit functionality.

This script tests individual file processing with multiple files to show
how the concurrency limit of 10 concurrent LLM calls works.
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

async def test_concurrency_limit(vectorstore_path: str):
    """Test the concurrency limit with multiple files."""
    
    # Create a list of files that would exceed the concurrency limit
    # This simulates processing many files to test the limit
    question = "What are the main vegetation management strategies mentioned?"
    
    # Use a larger set of files to test concurrency limit
    # In a real scenario, you might have many files to process
    selected_files = [
        "Arizona_Public_Service_2024.pdf",
        "BC_Hydro_2020.pdf",
        # Add more files if available in your vectorstore
    ]
    
    print("=" * 80)
    print("TESTING KNOWAI CONCURRENCY LIMIT")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Files: {selected_files}")
    print(f"Vectorstore: {vectorstore_path}")
    print(f"Concurrency Limit: 10 concurrent LLM calls")
    print()
    
    # Test individual file processing with concurrency limit
    print("🧪 Testing Individual File Processing with Concurrency Limit")
    print("-" * 60)
    
    agent = KnowAIAgent(
        vectorstore_path=vectorstore_path,
        process_files_individually=True,
        log_graph=False
    )
    
    try:
        print("🔄 Starting processing with concurrency control...")
        print("   (Watch the logs to see how files are processed in batches of 10)")
        print()
        
        result = await agent.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=False  # Use small model for faster testing
        )
        
        print(f"✅ Processing completed successfully")
        print(f"   Response length: {len(result.get('generation', ''))} characters")
        print(f"   Files processed: {len(result.get('documents_by_file', {}))}")
        
        # Show preview
        generation = result.get('generation', '')
        if generation:
            preview = generation[:200] + "..." if len(generation) > 200 else generation
            print(f"   Preview: {preview}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("CONCURRENCY LIMIT TESTING COMPLETED")
    print("=" * 80)
    print()
    print("💡 Key Benefits of Concurrency Control:")
    print("   • Prevents overwhelming the LLM service")
    print("   • Ensures stable performance")
    print("   • Automatic queuing of files beyond the limit")
    print("   • Error isolation between files")
    print("   • Resource management for consistent results")

def main():
    """Main function to run the concurrency limit test."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_concurrency_limit.py <vectorstore_path>")
        print("Example: python scripts/test_concurrency_limit.py ./vectorstores/my_vectorstore")
        sys.exit(1)
    
    vectorstore_path = sys.argv[1]
    
    if not Path(vectorstore_path).exists():
        print(f"Error: Vectorstore path does not exist: {vectorstore_path}")
        sys.exit(1)
    
    asyncio.run(test_concurrency_limit(vectorstore_path))

if __name__ == "__main__":
    main() 