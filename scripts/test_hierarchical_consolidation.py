#!/usr/bin/env python3
"""
Test script to debug hierarchical consolidation performance issues.

This script helps identify why the hierarchical consolidation node is taking
a long time to run and why progress updates aren't showing correctly.
"""

import asyncio
import logging
import sys
import os
import time

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import (
    create_individual_file_response_batches,
    estimate_tokens,
    estimate_synthesis_tokens,
    MAX_TOKENS_PER_HIERARCHICAL_BATCH,
    MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION
)


def create_mock_responses(num_files: int, avg_chars_per_file: int = 5000) -> dict:
    """Create mock individual file responses for testing."""
    import random
    
    responses = {}
    for i in range(num_files):
        filename = f"test_file_{i+1}.pdf"
        # Create a response with random length around the average
        response_length = int(avg_chars_per_file * random.uniform(0.5, 1.5))
        response = f"This is a mock response for {filename}. " * (response_length // 50)
        responses[filename] = response
    
    return responses


def test_batch_creation():
    """Test the batch creation function with different scenarios."""
    print("Testing batch creation scenarios...")
    
    # Test scenario 1: Small number of files (should skip hierarchical consolidation)
    print("\n1. Testing with small number of files:")
    small_responses = create_mock_responses(5, 3000)
    small_files = list(small_responses.keys())
    
    batches = create_individual_file_response_batches(
        individual_file_responses=small_responses,
        allowed_files=small_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question="What is the main topic?",
        conversation_history="No previous conversation."
    )
    
    print(f"   Files: {len(small_files)}")
    print(f"   Batches created: {len(batches)}")
    for i, (batch_files, batch_tokens) in enumerate(batches):
        print(f"   Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")
    
    # Test scenario 2: Large number of files (should trigger hierarchical consolidation)
    print("\n2. Testing with large number of files:")
    large_responses = create_mock_responses(15, 8000)
    large_files = list(large_responses.keys())
    
    batches = create_individual_file_response_batches(
        individual_file_responses=large_responses,
        allowed_files=large_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question="What is the main topic?",
        conversation_history="No previous conversation."
    )
    
    print(f"   Files: {len(large_files)}")
    print(f"   Batches created: {len(batches)}")
    for i, (batch_files, batch_tokens) in enumerate(batches):
        print(f"   Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")
        total_chars = sum(len(large_responses[f]) for f in batch_files)
        print(f"     Total chars: {total_chars:,}")
    
    # Test scenario 3: Very large responses (should create single-file batches)
    print("\n3. Testing with very large responses:")
    huge_responses = create_mock_responses(8, 50000)  # 50k chars per file
    huge_files = list(huge_responses.keys())
    
    batches = create_individual_file_response_batches(
        individual_file_responses=huge_responses,
        allowed_files=huge_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question="What is the main topic?",
        conversation_history="No previous conversation."
    )
    
    print(f"   Files: {len(huge_files)}")
    print(f"   Batches created: {len(batches)}")
    for i, (batch_files, batch_tokens) in enumerate(batches):
        print(f"   Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")
        total_chars = sum(len(huge_responses[f]) for f in batch_files)
        print(f"     Total chars: {total_chars:,}")


def test_token_estimation():
    """Test token estimation functions."""
    print("\nTesting token estimation:")
    
    # Test overhead calculation
    overhead_tokens = estimate_synthesis_tokens(
        question="What is the main topic?",
        content="",
        conversation_history="No previous conversation.",
        files_no_info="",
        files_errors=""
    )
    print(f"   Overhead tokens: {overhead_tokens:,}")
    
    # Test individual response token estimation
    sample_response = "This is a sample response with some content. " * 100
    response_tokens = estimate_tokens(sample_response)
    print(f"   Sample response ({len(sample_response):,} chars): {response_tokens:,} tokens")
    
    # Test formatted response token estimation
    formatted_response = f"--- File: test.pdf ---\n{sample_response}"
    formatted_tokens = estimate_tokens(formatted_response)
    print(f"   Formatted response ({len(formatted_response):,} chars): {formatted_tokens:,} tokens")


def test_thresholds():
    """Test the thresholds that determine when hierarchical consolidation is used."""
    print("\nTesting hierarchical consolidation thresholds:")
    
    print(f"   MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION: {MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION}")
    print(f"   MAX_TOKENS_PER_HIERARCHICAL_BATCH: {MAX_TOKENS_PER_HIERARCHICAL_BATCH:,}")
    
    # Test with exactly the threshold number of files
    threshold_responses = create_mock_responses(MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION, 5000)
    threshold_files = list(threshold_responses.keys())
    
    print(f"\n   Testing with exactly {MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION} files:")
    batches = create_individual_file_response_batches(
        individual_file_responses=threshold_responses,
        allowed_files=threshold_files,
        max_tokens_per_batch=MAX_TOKENS_PER_HIERARCHICAL_BATCH,
        question="What is the main topic?",
        conversation_history="No previous conversation."
    )
    
    print(f"   Batches created: {len(batches)}")
    for i, (batch_files, batch_tokens) in enumerate(batches):
        print(f"   Batch {i+1}: {len(batch_files)} files, {batch_tokens:,} tokens")


def main():
    """Run all tests."""
    print("🔍 Testing Hierarchical Consolidation Performance")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        test_token_estimation()
        test_thresholds()
        test_batch_creation()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Recommendations:")
        print("1. Check if the number of files exceeds MAX_FILES_FOR_HIERARCHICAL_CONSOLIDATION")
        print("2. Monitor token counts to ensure they're within reasonable limits")
        print("3. Verify that progress callbacks are being called with correct parameters")
        print("4. Check if individual file responses are very large (causing single-file batches)")
        
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 