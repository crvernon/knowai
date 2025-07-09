#!/usr/bin/env python3
"""
Simple debug script to test hierarchical consolidation node directly.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import hierarchical_consolidation_node, MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION


def create_test_state():
    """Create a test state for hierarchical consolidation."""
    # Create more files than the threshold to trigger hierarchical consolidation
    num_files = MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION + 5  # 15 files
    
    test_files = [f"test_file_{i+1}.pdf" for i in range(num_files)]
    test_responses = {}
    
    # Create responses with varying lengths - make them much larger to trigger multiple batches
    for i, filename in enumerate(test_files):
        # Create a much larger response to trigger token-based batching
        response = f"This is a comprehensive response for {filename}. "
        response += f"It contains detailed information about topic {i+1}. "
        response += "The document discusses various aspects and provides insights. " * 1000  # Much larger
        response += f"Additional analysis for {filename} includes multiple sections. " * 500
        response += f"Conclusion for {filename} summarizes the key findings. " * 200
        test_responses[filename] = response
    
    return {
        "question": "What are the key findings across all documents?",
        "allowed_files": test_files,
        "individual_file_responses": test_responses,
        "process_files_individually": True,
        "conversation_history": [],
        "detailed_response_desired": True,
        "llm_large": None,  # Will be set by the node
        "llm_small": None,  # Will be set by the node
        "streaming_callback": None
    }


async def test_hierarchical_consolidation():
    """Test the hierarchical consolidation node."""
    print("🧪 Testing Hierarchical Consolidation Node")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create test state
    test_state = create_test_state()
    
    print(f"Created test state with {len(test_state['allowed_files'])} files")
    print(f"Files: {test_state['allowed_files'][:5]}... (showing first 5)")
    print(f"Responses: {len(test_state['individual_file_responses'])} responses")
    
    # Mock LLM instances (we won't actually call them)
    class MockLLM:
        def __init__(self, name):
            self.name = name
        
        async def ainvoke(self, *args, **kwargs):
            return f"Mock response from {self.name}"
        
        async def astream(self, *args, **kwargs):
            async def mock_stream():
                yield f"Mock streaming response from {self.name}"
            return mock_stream()
    
    test_state["llm_large"] = MockLLM("large")
    test_state["llm_small"] = MockLLM("small")
    
    print("\nRunning hierarchical consolidation node...")
    start_time = asyncio.get_event_loop().time()
    
    try:
        result_state = await hierarchical_consolidation_node(test_state)
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        print(f"\n✅ Hierarchical consolidation completed in {duration:.2f} seconds")
        
        # Check results
        hierarchical_results = result_state.get("hierarchical_consolidation_results")
        
        if hierarchical_results:
            print(f"✅ Hierarchical consolidation successful!")
            print(f"   Created {len(hierarchical_results)} batch summaries")
            
            for i, result in enumerate(hierarchical_results):
                print(f"   Batch {i+1}: {len(result)} characters")
                print(f"   Preview: {result[:100]}...")
        else:
            print("❌ Hierarchical consolidation failed - no results")
            
    except Exception as e:
        print(f"❌ Error during hierarchical consolidation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_hierarchical_consolidation()) 