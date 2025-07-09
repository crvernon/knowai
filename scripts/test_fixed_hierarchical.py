#!/usr/bin/env python3
"""
Test script to verify the fixes for hierarchical consolidation issues.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import hierarchical_consolidation_node, MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION, MAX_TOKENS_PER_HIERARCHICAL_BATCH


def create_test_state():
    """Create a test state for hierarchical consolidation with realistic response sizes."""
    # Create more files than the threshold to trigger hierarchical consolidation
    num_files = MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION + 5  # 15 files
    
    test_files = [f"test_file_{i+1}.pdf" for i in range(num_files)]
    test_responses = {}
    
    # Create responses with realistic sizes (not massive)
    for i, filename in enumerate(test_files):
        # Create a realistic response size (around 1000-3000 chars)
        response = f"Analysis of {filename}:\n\n"
        response += f"Key findings from {filename} include:\n"
        response += f"• Important point 1 from {filename} (Page 5)\n"
        response += f"• Important point 2 from {filename} (Page 12)\n"
        response += f"• Important point 3 from {filename} (Page 18)\n\n"
        response += f"Summary: {filename} contains relevant information about topic {i+1}. "
        response += "The document provides insights into various aspects of the subject matter. "
        response += "Key recommendations include focusing on efficiency and sustainability. "
        response += "Additional analysis shows promising results for future implementation.\n\n"
        response += f"Conclusion: {filename} offers valuable insights that should be considered in the overall analysis."
        
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
    """Test the hierarchical consolidation node with fixes."""
    print("🧪 Testing Fixed Hierarchical Consolidation Node")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create test state
    test_state = create_test_state()
    
    print(f"Created test state with {len(test_state['allowed_files'])} files")
    print(f"MAX_TOKENS_PER_HIERARCHICAL_BATCH: {MAX_TOKENS_PER_HIERARCHICAL_BATCH:,}")
    
    # Calculate total response size
    total_chars = sum(len(response) for response in test_state['individual_file_responses'].values())
    print(f"Total individual response size: {total_chars:,} characters")
    
    # Mock LLM instances
    class MockLLM:
        def __init__(self, name):
            self.name = name
        
        async def ainvoke(self, *args, **kwargs):
            return f"Mock consolidated response from {self.name} - this is a reasonable length summary that captures the key points from the individual file responses."
        
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
                if len(result) > 200:
                    print(f"   Preview: {result[:200]}...")
                else:
                    print(f"   Content: {result}")
                    
                # Check if response length is reasonable
                if len(result) > 15000:
                    print(f"   ⚠️  WARNING: Batch {i+1} response is very long ({len(result):,} chars)")
                elif len(result) > 10000:
                    print(f"   ⚠️  WARNING: Batch {i+1} response is long ({len(result):,} chars)")
                else:
                    print(f"   ✅ Batch {i+1} response length is reasonable ({len(result):,} chars)")
        else:
            print("❌ Hierarchical consolidation failed - no results")
            
    except Exception as e:
        print(f"❌ Error during hierarchical consolidation: {e}")
        import traceback
        traceback.print_exc()


def test_batch_creation():
    """Test batch creation with the new token limit."""
    print("\n" + "="*60)
    print("Testing Batch Creation with New Token Limit")
    print("="*60)
    
    from knowai.agent import create_individual_file_response_batches
    
    # Create test responses
    test_responses = {}
    for i in range(15):
        filename = f"test_file_{i+1}.pdf"
        # Create responses that should trigger multiple batches with 50k token limit
        response = f"Analysis of {filename}:\n\n" * 100  # Make it large enough to trigger batching
        test_responses[filename] = response
    
    test_files = list(test_responses.keys())
    
    batches = create_individual_file_response_batches(
        individual_file_responses=test_responses,
        allowed_files=test_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question="What are the key findings?",
        conversation_history="No previous conversation."
    )
    
    print(f"Created {len(batches)} batches with {MAX_TOKENS_PER_HIERARCHICAL_BATCH:,} token limit")
    for i, (batch_files, batch_tokens) in enumerate(batches):
        print(f"   Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")
        total_chars = sum(len(test_responses[f]) for f in batch_files)
        print(f"     Total chars: {total_chars:,}")


if __name__ == "__main__":
    asyncio.run(test_hierarchical_consolidation())
    test_batch_creation() 