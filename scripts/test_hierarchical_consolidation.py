#!/usr/bin/env python3
"""
Test script for hierarchical consolidation functionality.

This script tests the new hierarchical consolidation feature that processes
individual file responses in batches of 10 to ensure all information is preserved.
"""

import asyncio
import logging
import tempfile
import os
from knowai.core import KnowAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_hierarchical_consolidation():
    """Test the hierarchical consolidation functionality."""
    
    # Create a temporary directory for the vectorstore
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Create a simple test agent
        agent = KnowAIAgent(
            vectorstore_path=temp_dir,
            process_files_individually=True,  # Enable individual file processing
            use_accurate_token_counting=False  # Disable for testing
        )
        
        # Simulate individual file responses (more than 10 to trigger hierarchical consolidation)
        test_files = [f"test_file_{i}.txt" for i in range(1, 26)]  # 25 files
        test_responses = {}
        
        for i, filename in enumerate(test_files):
            test_responses[filename] = f"This is the response for {filename}. It contains important information about topic {i+1}."
        
        # Set up the session state with test data
        agent.session_state.update({
            "question": "What are the key findings across all documents?",
            "allowed_files": test_files,
            "individual_file_responses": test_responses,
            "process_files_individually": True,
            "conversation_history": []
        })
        
        logger.info(f"Set up test with {len(test_files)} files")
        logger.info(f"Files: {test_files[:5]}... (showing first 5)")
        
        # Test the hierarchical consolidation node directly
        from knowai.agent import hierarchical_consolidation_node
        
        logger.info("Running hierarchical consolidation node...")
        result_state = await hierarchical_consolidation_node(agent.session_state)
        
        # Check the results
        hierarchical_results = result_state.get("hierarchical_consolidation_results")
        
        if hierarchical_results:
            logger.info(f"✅ Hierarchical consolidation successful!")
            logger.info(f"   Created {len(hierarchical_results)} batch summaries")
            
            # Should have 3 batches for 25 files (10, 10, 5)
            expected_batches = (len(test_files) + 9) // 10  # Ceiling division
            if len(hierarchical_results) == expected_batches:
                logger.info(f"✅ Correct number of batches: {len(hierarchical_results)} (expected {expected_batches})")
            else:
                logger.error(f"❌ Wrong number of batches: {len(hierarchical_results)} (expected {expected_batches})")
            
            # Show sample of first batch summary
            if hierarchical_results:
                first_summary = hierarchical_results[0]
                logger.info(f"   First batch summary length: {len(first_summary)} characters")
                logger.info(f"   First batch preview: {first_summary[:200]}...")
        else:
            logger.error("❌ Hierarchical consolidation failed - no results")
        
        # Test the combine_answers_node with hierarchical results
        logger.info("\nTesting combine_answers_node with hierarchical results...")
        
        # Set up the session state with hierarchical results
        agent.session_state.update({
            "question": "What are the key findings across all documents?",
            "allowed_files": test_files,
            "individual_file_responses": test_responses,
            "process_files_individually": True,
            "conversation_history": [],
            "hierarchical_consolidation_results": hierarchical_results
        })
        
        # Test the combine_answers_node directly
        from knowai.agent import combine_answers_node
        
        logger.info("Running combine_answers_node with hierarchical results...")
        final_state = await combine_answers_node(agent.session_state)
        
        # Check the final generation
        final_generation = final_state.get("generation")
        
        if final_generation:
            logger.info("✅ Final combination successful!")
            logger.info(f"   Final response length: {len(final_generation)} characters")
            logger.info(f"   Final response preview: {final_generation[:200]}...")
        else:
            logger.error("❌ Final combination failed - no generation result")


async def test_small_file_count():
    """Test that hierarchical consolidation is skipped for ≤10 files."""
    
    logger.info("\n" + "="*50)
    logger.info("Testing small file count (≤10 files)")
    logger.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test agent
        agent = KnowAIAgent(
            vectorstore_path=temp_dir,
            process_files_individually=True,
            use_accurate_token_counting=False
        )
        
        # Test with 8 files (should skip hierarchical consolidation)
        test_files = [f"small_file_{i}.txt" for i in range(1, 9)]  # 8 files
        test_responses = {}
        
        for i, filename in enumerate(test_files):
            test_responses[filename] = f"Response for {filename} with topic {i+1}."
        
        agent.session_state.update({
            "question": "What are the findings?",
            "allowed_files": test_files,
            "individual_file_responses": test_responses,
            "process_files_individually": True,
            "conversation_history": []
        })
        
        logger.info(f"Testing with {len(test_files)} files (should skip hierarchical consolidation)")
        
        # Test the hierarchical consolidation node
        from knowai.agent import hierarchical_consolidation_node
        
        result_state = await hierarchical_consolidation_node(agent.session_state)
        hierarchical_results = result_state.get("hierarchical_consolidation_results")
        
        if hierarchical_results is None:
            logger.info("✅ Correctly skipped hierarchical consolidation for ≤10 files")
        else:
            logger.error(f"❌ Should have skipped hierarchical consolidation, but got {len(hierarchical_results)} results")


if __name__ == "__main__":
    asyncio.run(test_hierarchical_consolidation())
    asyncio.run(test_small_file_count()) 