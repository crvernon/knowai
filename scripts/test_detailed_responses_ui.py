#!/usr/bin/env python3
"""
Test script for detailed responses UI functionality.

This script tests that detailed responses are being generated and returned
correctly for display in the Streamlit UI.
"""

import asyncio
import logging
import tempfile
import os
from knowai.core import KnowAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_detailed_responses_ui():
    """Test that detailed responses are generated and returned correctly."""
    
    logger.info("Testing detailed responses UI functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test agent
        agent = KnowAIAgent(
            vectorstore_path=temp_dir,
            process_files_individually=True
        )
        
        # Test 1: With detailed responses enabled
        logger.info("Test 1: With detailed responses enabled")
        try:
            response_data = await agent.process_turn(
                user_question="Test question",
                selected_files=["test1.pdf", "test2.pdf"],
                show_detailed_individual_responses=True
            )
            
            print(f"✅ Response data keys: {list(response_data.keys())}")
            print(f"✅ Generation length: {len(response_data.get('generation', ''))}")
            print(f"✅ Detailed responses: {response_data.get('detailed_responses', 'None')}")
            
            # Debug: Check session state directly
            print(f"✅ Session state keys: {list(agent.session_state.keys())}")
            print(f"✅ detailed_responses_for_ui in session: {agent.session_state.get('detailed_responses_for_ui', 'None')}")
            print(f"✅ show_detailed_individual_responses in session: {agent.session_state.get('show_detailed_individual_responses', 'None')}")
            
            if response_data.get('detailed_responses'):
                print(f"✅ Detailed responses length: {len(response_data['detailed_responses'])}")
                print(f"✅ Detailed responses preview: {response_data['detailed_responses'][:200]}...")
            else:
                print("❌ No detailed responses returned")
                
        except Exception as e:
            print(f"❌ Error in test 1: {e}")
        
        # Test 2: With detailed responses disabled
        logger.info("Test 2: With detailed responses disabled")
        try:
            response_data = await agent.process_turn(
                user_question="Test question",
                selected_files=["test1.pdf", "test2.pdf"],
                show_detailed_individual_responses=False
            )
            
            print(f"✅ Response data keys: {list(response_data.keys())}")
            print(f"✅ Generation length: {len(response_data.get('generation', ''))}")
            print(f"✅ Detailed responses: {response_data.get('detailed_responses', 'None')}")
            
            if response_data.get('detailed_responses'):
                print("❌ Detailed responses returned when disabled")
            else:
                print("✅ No detailed responses returned (as expected)")
                
        except Exception as e:
            print(f"❌ Error in test 2: {e}")


if __name__ == "__main__":
    asyncio.run(test_detailed_responses_ui()) 