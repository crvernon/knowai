#!/usr/bin/env python3
"""
Test script for citation functionality in prompts.

This script tests that the prompts include proper citation instructions
and that the LLM will include citations when referencing document content.
"""

import asyncio
import logging
import tempfile
import os
from knowai.core import KnowAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_citation_prompts():
    """Test that the prompts include proper citation instructions."""
    
    logger.info("Testing citation instructions in prompts...")
    
    # Import the prompt functions
    from knowai.prompts import (
        get_hierarchical_consolidation_prompt_template,
        get_batch_combination_prompt_template,
        get_consolidation_prompt_template,
        get_synthesis_prompt_template
    )
    
    # Test each prompt template
    prompts_to_test = [
        ("Hierarchical Consolidation", get_hierarchical_consolidation_prompt_template()),
        ("Batch Combination", get_batch_combination_prompt_template()),
        ("Individual File Consolidation", get_consolidation_prompt_template()),
        ("Raw Documents Synthesis", get_synthesis_prompt_template())
    ]
    
    citation_keywords = [
        "citation", "cite", "(filename.pdf", "Page X", "filename.pdf"
    ]
    
    for prompt_name, prompt_template in prompts_to_test:
        logger.info(f"\nTesting {prompt_name} prompt:")
        
        # Get the prompt template string
        template_str = prompt_template.template
        
        # Check for citation keywords
        found_citations = []
        for keyword in citation_keywords:
            if keyword.lower() in template_str.lower():
                found_citations.append(keyword)
        
        if found_citations:
            logger.info(f"  ✅ Found citation instructions: {found_citations}")
        else:
            logger.warning(f"  ⚠️  No citation instructions found in {prompt_name}")
        
        # Show a preview of the prompt
        preview = template_str[:200] + "..." if len(template_str) > 200 else template_str
        logger.info(f"  Preview: {preview}")


async def test_citation_in_workflow():
    """Test that citations are included in a real workflow."""
    
    logger.info("\n" + "="*50)
    logger.info("Testing citations in workflow")
    logger.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test agent
        agent = KnowAIAgent(
            vectorstore_path=temp_dir,
            process_files_individually=True,
    
        )
        
        # Create test individual file responses with citations
        test_files = [f"document_{i}.pdf" for i in range(1, 6)]  # 5 files
        test_responses = {}
        
        for i, filename in enumerate(test_files):
            # Create responses that include citations
            test_responses[filename] = (
                f"According to {filename} (Page {i+1}), the key finding is that "
                f"topic {i+1} is important. This is supported by evidence from "
                f"{filename} (Page {i+2}) which states that this topic has "
                f"significant implications."
            )
        
        logger.info(f"Created test responses for {len(test_files)} files")
        logger.info(f"Sample response: {test_responses[test_files[0]][:100]}...")
        
        # Test the hierarchical consolidation node
        from knowai.agent import hierarchical_consolidation_node
        
        agent.session_state.update({
            "question": "What are the key findings across all documents?",
            "allowed_files": test_files,
            "individual_file_responses": test_responses,
            "process_files_individually": True,
            "conversation_history": []
        })
        
        logger.info("Running hierarchical consolidation node...")
        result_state = await hierarchical_consolidation_node(agent.session_state)
        
        hierarchical_results = result_state.get("hierarchical_consolidation_results")
        
        if hierarchical_results:
            logger.info(f"✅ Hierarchical consolidation created {len(hierarchical_results)} batch summaries")
            
            # Check if the results contain citations
            for i, result in enumerate(hierarchical_results):
                citation_count = result.count("(document_") + result.count("Page ")
                logger.info(f"  Batch {i+1}: {citation_count} citation references found")
                logger.info(f"  Preview: {result[:150]}...")
        else:
            logger.info("ℹ️  No hierarchical consolidation needed (≤10 files)")
            
            # Test the regular consolidation
            from knowai.agent import combine_answers_node
            
            logger.info("Testing regular consolidation...")
            final_state = await combine_answers_node(agent.session_state)
            final_generation = final_state.get("generation")
            
            if final_generation:
                citation_count = final_generation.count("(document_") + final_generation.count("Page ")
                logger.info(f"✅ Final consolidation: {citation_count} citation references found")
                logger.info(f"Preview: {final_generation[:200]}...")


if __name__ == "__main__":
    asyncio.run(test_citation_prompts())
    asyncio.run(test_citation_in_workflow()) 