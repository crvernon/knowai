#!/usr/bin/env python3
"""
Test script to verify token limit enforcement is working correctly.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import (
    _stream_final_generation,
    MAX_COMPLETION_TOKENS,
    get_synthesis_prompt_template
)
from langchain_core.prompts import PromptTemplate


def create_mock_llm():
    """Create a mock LLM that generates responses exceeding the token limit."""
    class MockLLM:
        def __init__(self):
            self.name = "mock"
        
        async def ainvoke(self, *args, **kwargs):
            # Generate a response that exceeds MAX_COMPLETION_TOKENS
            # 4+ million characters = ~1+ million tokens
            long_response = "This is a very long response. " * 200000  # ~4M chars
            return long_response
        
        async def astream(self, *args, **kwargs):
            async def mock_stream():
                # Generate a response that exceeds MAX_COMPLETION_TOKENS
                long_response = "This is a very long streaming response. " * 200000  # ~4M chars
                for i in range(0, len(long_response), 1000):
                    yield long_response[i:i+1000]
            return mock_stream()
    
    return MockLLM()


async def test_token_limit_enforcement():
    """Test that token limit enforcement is working."""
    print("🧪 Testing Token Limit Enforcement")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create mock LLM
    mock_llm = create_mock_llm()
    
    # Create a simple prompt
    prompt = PromptTemplate(
        template="Answer this question: {question}",
        input_variables=["question"]
    )
    
    print(f"MAX_COMPLETION_TOKENS: {MAX_COMPLETION_TOKENS:,}")
    print(f"Expected max response length: ~{MAX_COMPLETION_TOKENS * 4:,} characters")
    
    # Test regular invocation
    print("\n1. Testing regular invocation...")
    try:
        result = await _stream_final_generation(
            question="What is the answer?",
            content_llm="Some content",
            llm_instance=mock_llm,
            combo_prompt=prompt,
            conversation_history_str="No history",
            no_info_list=[],
            error_list=[],
            streaming_callback=None
        )
        
        print(f"   Result length: {len(result):,} characters")
        estimated_tokens = len(result) // 4
        print(f"   Estimated tokens: {estimated_tokens:,}")
        
        if len(result) > MAX_COMPLETION_TOKENS * 4:
            print("   ❌ FAILED: Response exceeds token limit")
        else:
            print("   ✅ PASSED: Response within token limit")
            
        # Check if truncation message is present
        if "[Response truncated due to token limit]" in result:
            print("   ✅ PASSED: Truncation message found")
        else:
            print("   ❌ FAILED: No truncation message found")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test streaming invocation
    print("\n2. Testing streaming invocation...")
    try:
        chunks = []
        def mock_callback(chunk):
            chunks.append(chunk)
        
        result = await _stream_final_generation(
            question="What is the answer?",
            content_llm="Some content",
            llm_instance=mock_llm,
            combo_prompt=prompt,
            conversation_history_str="No history",
            no_info_list=[],
            error_list=[],
            streaming_callback=mock_callback
        )
        
        print(f"   Result length: {len(result):,} characters")
        estimated_tokens = len(result) // 4
        print(f"   Estimated tokens: {estimated_tokens:,}")
        print(f"   Number of chunks: {len(chunks)}")
        
        if len(result) > MAX_COMPLETION_TOKENS * 4:
            print("   ❌ FAILED: Response exceeds token limit")
        else:
            print("   ✅ PASSED: Response within token limit")
            
        # Check if truncation message is present
        if "[Response truncated due to token limit]" in result:
            print("   ✅ PASSED: Truncation message found")
        else:
            print("   ❌ FAILED: No truncation message found")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")


def test_individual_file_processing():
    """Test individual file processing token limit enforcement."""
    print("\n3. Testing individual file processing...")
    
    from knowai.agent import process_individual_files_node
    
    # Create a test state with a mock LLM that generates long responses
    test_state = {
        "question": "What are the findings?",
        "allowed_files": ["test_file.pdf"],
        "documents_by_file": {
            "test_file.pdf": [{"page_content": "Some content", "metadata": {"page": 1}}]
        },
        "process_files_individually": True,
        "conversation_history": [],
        "detailed_response_desired": True,
        "llm_large": create_mock_llm(),
        "llm_small": create_mock_llm(),
        "streaming_callback": None
    }
    
    try:
        # This would normally run the full node, but we'll just test the token enforcement
        # by calling the LLM directly
        from knowai.agent import get_synthesis_prompt_template
        
        prompt = get_synthesis_prompt_template()
        
        result = asyncio.run(_stream_final_generation(
            question="What are the findings?",
            content_llm="Some content",
            llm_instance=test_state["llm_large"],
            combo_prompt=prompt,
            conversation_history_str="No history",
            no_info_list=[],
            error_list=[],
            streaming_callback=None
        ))
        
        print(f"   Individual file response length: {len(result):,} characters")
        estimated_tokens = len(result) // 4
        print(f"   Estimated tokens: {estimated_tokens:,}")
        
        if len(result) > MAX_COMPLETION_TOKENS * 4:
            print("   ❌ FAILED: Response exceeds token limit")
        else:
            print("   ✅ PASSED: Response within token limit")
            
        # Check if truncation message is present
        if "[Response truncated due to token limit]" in result:
            print("   ✅ PASSED: Truncation message found")
        else:
            print("   ❌ FAILED: No truncation message found")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(test_token_limit_enforcement())
    test_individual_file_processing() 