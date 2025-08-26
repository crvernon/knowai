#!/usr/bin/env python3
"""
Debug script to identify what's causing the app to freeze.

This script runs KnowAI with comprehensive logging and error capture
to help diagnose freezing issues.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import KnowAIAgent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_freeze.log')  # Also save to file
    ]
)

async def debug_freeze_issue(vectorstore_path: str):
    """Debug the freezing issue with comprehensive logging."""
    
    question = "What are the main vegetation management strategies mentioned?"
    selected_files = ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
    
    print("=" * 80)
    print("DEBUGGING KNOWAI FREEZE ISSUE")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Files: {selected_files}")
    print(f"Vectorstore: {vectorstore_path}")
    print(f"Logging Level: DEBUG")
    print(f"Log File: debug_freeze.log")
    print()
    
    try:
        print("🔄 Step 1: Creating KnowAI Agent...")
        start_time = time.perf_counter()
        
        agent = KnowAIAgent(
            vectorstore_path=vectorstore_path,
            process_files_individually=True,
            log_graph=False
        )
        
        agent_time = time.perf_counter() - start_time
        print(f"✅ Agent created successfully in {agent_time:.2f} seconds")
        print()
        
        print("🔄 Step 2: Starting process_turn...")
        turn_start_time = time.perf_counter()
        
        # Add a timeout to prevent infinite hanging
        try:
            result = await asyncio.wait_for(
                agent.process_turn(
                    user_question=question,
                    selected_files=selected_files,
                    detailed_response_desired=False  # Use small model for faster testing
                ),
                timeout=300  # 5 minute timeout
            )
            
            turn_time = time.perf_counter() - turn_start_time
            print(f"✅ process_turn completed successfully in {turn_time:.2f} seconds")
            print(f"   Response length: {len(result.get('generation', ''))} characters")
            print(f"   Files processed: {len(result.get('documents_by_file', {}))}")
            
            # Show preview
            generation = result.get('generation', '')
            if generation:
                preview = generation[:200] + "..." if len(generation) > 200 else generation
                print(f"   Preview: {preview}")
            
        except asyncio.TimeoutError:
            print("❌ process_turn timed out after 5 minutes")
            print("   This indicates the app is hanging/freezing")
            print("   Check debug_freeze.log for detailed error information")
            
        except Exception as e:
            print(f"❌ process_turn failed with exception: {e}")
            print(f"   Exception type: {type(e).__name__}")
            print("   Check debug_freeze.log for detailed error information")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        print(f"   Exception type: {type(e).__name__}")
        print("   Check debug_freeze.log for detailed error information")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("DEBUGGING COMPLETED")
    print("=" * 80)
    print()
    print("📋 Next Steps:")
    print("   1. Check debug_freeze.log for detailed error information")
    print("   2. Look for OpenAI API errors or configuration issues")
    print("   3. Check for timeout or connection issues")
    print("   4. Verify environment variables are set correctly")
    print()
    print("🔍 Common Issues to Check:")
    print("   • Azure OpenAI API key and endpoint configuration")
    print("   • Network connectivity to Azure OpenAI")
    print("   • Rate limiting or quota issues")
    print("   • Model deployment availability")
    print("   • Vectorstore accessibility")

def main():
    """Main function to run the debugging script."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_freeze.py <vectorstore_path>")
        print("Example: python scripts/debug_freeze.py ./vectorstores/my_vectorstore")
        sys.exit(1)
    
    vectorstore_path = sys.argv[1]
    
    if not Path(vectorstore_path).exists():
        print(f"Error: Vectorstore path does not exist: {vectorstore_path}")
        sys.exit(1)
    
    print("🚀 Starting debug session...")
    print("   Detailed logs will be saved to debug_freeze.log")
    print()
    
    asyncio.run(debug_freeze_issue(vectorstore_path))

if __name__ == "__main__":
    main() 