#!/usr/bin/env python3
"""
Test script for token management functionality in KnowAI agent.

This script tests:
1. Token estimation with both accurate (tiktoken) and heuristic methods
2. Batch creation with different token limits
3. Document truncation when exceeding limits
4. Overall token management workflow
"""

import asyncio
import logging
import sys
import os
from typing import List

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import (
    estimate_tokens,
    estimate_synthesis_tokens,
    create_content_batches,
    GraphState,
    USE_ACCURATE_TOKEN_COUNTING,
    TIKTOKEN_AVAILABLE
)
from langchain_core.documents import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_token_estimation():
    """Test token estimation with both methods."""
    print("\n=== Testing Token Estimation ===")
    
    test_texts = [
        "Hello world",
        "This is a longer text with more words and punctuation!",
        "A" * 1000,  # 1000 characters
        "Hello world " * 100,  # 1200 characters
        "",  # Empty string
        "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "Unicode: 🚀🌟🎉中文日本語한국어",
    ]
    
    for text in test_texts:
        print(f"\nText: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Test accurate counting (if available)
        if TIKTOKEN_AVAILABLE:
            accurate_tokens = estimate_tokens(text, use_accurate=True)
            print(f"  Accurate tokens: {accurate_tokens}")
        
        # Test heuristic counting
        heuristic_tokens = estimate_tokens(text, use_accurate=False)
        print(f"  Heuristic tokens: {heuristic_tokens}")
        
        # Compare if both available
        if TIKTOKEN_AVAILABLE:
            diff = abs(accurate_tokens - heuristic_tokens)
            diff_percent = (diff / accurate_tokens * 100) if accurate_tokens > 0 else 0
            print(f"  Difference: {diff} ({diff_percent:.1f}%)")


def test_synthesis_token_estimation():
    """Test synthesis token estimation."""
    print("\n=== Testing Synthesis Token Estimation ===")
    
    question = "What is the main topic of the documents?"
    content = "This is a sample document content with some information."
    conversation_history = "User: Previous question\nAssistant: Previous answer"
    files_no_info = "file1.pdf, file2.pdf"
    files_errors = "file3.pdf"
    
    # Test with accurate counting
    if TIKTOKEN_AVAILABLE:
        accurate_total = estimate_synthesis_tokens(
            question, content, conversation_history, files_no_info, files_errors,
            use_accurate=True
        )
        print(f"Accurate synthesis tokens: {accurate_total:,}")
    
    # Test with heuristic counting
    heuristic_total = estimate_synthesis_tokens(
        question, content, conversation_history, files_no_info, files_errors,
        use_accurate=False
    )
    print(f"Heuristic synthesis tokens: {heuristic_total:,}")


def test_batch_creation():
    """Test batch creation with different token limits."""
    print("\n=== Testing Batch Creation ===")
    
    # Create test documents
    documents = []
    for i in range(5):
        doc = Document(
            page_content=f"This is document {i+1} with some content. " * 50,  # ~2000 chars
            metadata={"file_name": f"test_file_{i+1}.pdf", "page": i+1}
        )
        documents.append(doc)
    
    question = "What is the main topic?"
    conversation_history = "No previous conversation."
    files_no_info = "None"
    files_errors = "None"
    
    # Test with different token limits
    token_limits = [5000, 10000, 20000]
    
    for limit in token_limits:
        print(f"\nToken limit: {limit:,}")
        
        # Test with accurate counting
        if TIKTOKEN_AVAILABLE:
            accurate_batches = create_content_batches(
                documents, limit, question, conversation_history,
                files_no_info, files_errors, use_accurate=True
            )
            print(f"  Accurate batches: {len(accurate_batches)}")
            for i, (batch_docs, batch_tokens) in enumerate(accurate_batches):
                print(f"    Batch {i+1}: {len(batch_docs)} docs, {batch_tokens:,} tokens")
        
        # Test with heuristic counting
        heuristic_batches = create_content_batches(
            documents, limit, question, conversation_history,
            files_no_info, files_errors, use_accurate=False
        )
        print(f"  Heuristic batches: {len(heuristic_batches)}")
        for i, (batch_docs, batch_tokens) in enumerate(heuristic_batches):
            print(f"    Batch {i+1}: {len(batch_docs)} docs, {batch_tokens:,} tokens")


def test_large_document_truncation():
    """Test truncation of documents that exceed token limits."""
    print("\n=== Testing Large Document Truncation ===")
    
    # Create a very large document
    large_content = "This is a very large document. " * 10000  # ~280,000 chars
    large_doc = Document(
        page_content=large_content,
        metadata={"file_name": "large_file.pdf", "page": 1}
    )
    
    question = "What is the main topic?"
    conversation_history = "No previous conversation."
    files_no_info = "None"
    files_errors = "None"
    token_limit = 10000  # Small limit to force truncation
    
    print(f"Large document size: {len(large_content):,} characters")
    
    # Test with accurate counting
    if TIKTOKEN_AVAILABLE:
        accurate_batches = create_content_batches(
            [large_doc], token_limit, question, conversation_history,
            files_no_info, files_errors, use_accurate=True
        )
        print(f"Accurate batches: {len(accurate_batches)}")
        for i, (batch_docs, batch_tokens) in enumerate(accurate_batches):
            print(f"  Batch {i+1}: {len(batch_docs)} docs, {batch_tokens:,} tokens")
            if batch_docs:
                content = batch_docs[0].page_content
                print(f"    Content length: {len(content):,} chars")
                print(f"    Truncated: {'[Content truncated' in content}")
    
    # Test with heuristic counting
    heuristic_batches = create_content_batches(
        [large_doc], token_limit, question, conversation_history,
        files_no_info, files_errors, use_accurate=False
    )
    print(f"Heuristic batches: {len(heuristic_batches)}")
    for i, (batch_docs, batch_tokens) in enumerate(heuristic_batches):
        print(f"  Batch {i+1}: {len(batch_docs)} docs, {batch_tokens:,} tokens")
        if batch_docs:
            content = batch_docs[0].page_content
            print(f"    Content length: {len(content):,} chars")
            print(f"    Truncated: {'[Content truncated' in content}")


def test_token_counting_configuration():
    """Test the token counting configuration options."""
    print("\n=== Testing Token Counting Configuration ===")
    
    test_text = "This is a test text for token counting configuration."
    
    print(f"Default USE_ACCURATE_TOKEN_COUNTING: {USE_ACCURATE_TOKEN_COUNTING}")
    print(f"TIKTOKEN_AVAILABLE: {TIKTOKEN_AVAILABLE}")
    
    # Test default behavior
    default_tokens = estimate_tokens(test_text)
    print(f"Default estimation: {default_tokens} tokens")
    
    # Test explicit accurate
    if TIKTOKEN_AVAILABLE:
        accurate_tokens = estimate_tokens(test_text, use_accurate=True)
        print(f"Explicit accurate: {accurate_tokens} tokens")
    
    # Test explicit heuristic
    heuristic_tokens = estimate_tokens(test_text, use_accurate=False)
    print(f"Explicit heuristic: {heuristic_tokens} tokens")


async def test_agent_integration():
    """Test token management integration with the agent workflow."""
    print("\n=== Testing Agent Integration ===")
    
    # Create a mock state for testing
    state: GraphState = {
        "question": "What is the main topic?",
        "allowed_files": ["test1.pdf", "test2.pdf"],
        "conversation_history": [],
        "combined_documents": [
            Document(
                page_content="This is a test document with some content. " * 100,
                metadata={"file_name": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="This is another test document. " * 100,
                metadata={"file_name": "test2.pdf", "page": 1}
            )
        ],
        "max_tokens_per_batch": 5000,
        "use_accurate_token_counting": True,
        "documents_by_file": {
            "test1.pdf": [Document(page_content="content", metadata={"file_name": "test1.pdf", "page": 1})],
            "test2.pdf": [Document(page_content="content", metadata={"file_name": "test2.pdf", "page": 1})]
        }
    }
    
    # Test token estimation in state context
    question = state["question"]
    conversation_history = "No previous conversation."
    combined_docs = state["combined_documents"]
    
    all_content = "\n\n".join([
        f"--- File: {doc.metadata.get('file_name', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')} ---\n{doc.page_content}"
        for doc in combined_docs
    ])
    
    total_tokens = estimate_synthesis_tokens(
        question=question,
        content=all_content,
        conversation_history=conversation_history,
        files_no_info="None",
        files_errors="None",
        use_accurate=state.get("use_accurate_token_counting", USE_ACCURATE_TOKEN_COUNTING)
    )
    
    print(f"Total estimated tokens: {total_tokens:,}")
    print(f"Max tokens per batch: {state['max_tokens_per_batch']:,}")
    print(f"Needs batching: {total_tokens > state['max_tokens_per_batch']}")
    
    if total_tokens > state['max_tokens_per_batch']:
        batches = create_content_batches(
            documents=combined_docs,
            max_tokens_per_batch=state['max_tokens_per_batch'],
            question=question,
            conversation_history=conversation_history,
            files_no_info="None",
            files_errors="None",
            use_accurate=state.get("use_accurate_token_counting", USE_ACCURATE_TOKEN_COUNTING)
        )
        print(f"Created {len(batches)} batches")


async def main():
    """Run all tests."""
    print("KnowAI Token Management Test Suite")
    print("=" * 50)
    
    try:
        # Test basic token estimation
        test_token_estimation()
        
        # Test synthesis token estimation
        test_synthesis_token_estimation()
        
        # Test batch creation
        test_batch_creation()
        
        # Test large document truncation
        test_large_document_truncation()
        
        # Test configuration options
        test_token_counting_configuration()
        
        # Test agent integration
        await test_agent_integration()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 