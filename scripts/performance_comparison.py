#!/usr/bin/env python3
"""
Performance comparison script for KnowAI processing modes.

This script compares the performance of traditional batch processing vs
asynchronous individual file processing to demonstrate the benefits.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import KnowAIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def compare_processing_performance(vectorstore_path: str):
    """Compare performance between batch and individual processing modes."""
    
    question = "What are the main vegetation management strategies mentioned?"
    selected_files = ["Arizona_Public_Service_2024.pdf", "BC_Hydro_2020.pdf"]
    
    print("=" * 80)
    print("KNOWAI PROCESSING MODE PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Files: {selected_files}")
    print(f"Vectorstore: {vectorstore_path}")
    print()
    
    results = {}
    
    # Test 1: Traditional batch processing
    print("🔄 Testing Traditional Batch Processing...")
    print("-" * 50)
    
    agent_batch = KnowAIAgent(
        vectorstore_path=vectorstore_path,
        process_files_individually=False,
        log_graph=False
    )
    
    try:
        start_time = time.perf_counter()
        result_batch = await agent_batch.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=False  # Use small model for faster testing
        )
        batch_time = time.perf_counter() - start_time
        
        results['batch'] = {
            'time': batch_time,
            'response_length': len(result_batch.get('generation', '')),
            'files_processed': len(result_batch.get('documents_by_file', {}))
        }
        
        print(f"✅ Batch processing completed in {batch_time:.2f} seconds")
        print(f"   Response length: {results['batch']['response_length']} characters")
        print(f"   Files processed: {results['batch']['files_processed']}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        results['batch'] = {'time': float('inf'), 'error': str(e)}
    
    print()
    
    # Test 2: Individual file processing (asynchronous)
    print("🔄 Testing Individual File Processing (Asynchronous)...")
    print("-" * 50)
    
    agent_individual = KnowAIAgent(
        vectorstore_path=vectorstore_path,
        process_files_individually=True,
        log_graph=False
    )
    
    try:
        start_time = time.perf_counter()
        result_individual = await agent_individual.process_turn(
            user_question=question,
            selected_files=selected_files,
            detailed_response_desired=False  # Use small model for faster testing
        )
        individual_time = time.perf_counter() - start_time
        
        results['individual'] = {
            'time': individual_time,
            'response_length': len(result_individual.get('generation', '')),
            'files_processed': len(result_individual.get('documents_by_file', {}))
        }
        
        print(f"✅ Individual processing completed in {individual_time:.2f} seconds")
        print(f"   Response length: {results['individual']['response_length']} characters")
        print(f"   Files processed: {results['individual']['files_processed']}")
        
    except Exception as e:
        print(f"❌ Individual processing failed: {e}")
        results['individual'] = {'time': float('inf'), 'error': str(e)}
    
    print()
    print("=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)
    
    if 'batch' in results and 'individual' in results:
        batch_time = results['batch']['time']
        individual_time = results['individual']['time']
        
        if batch_time != float('inf') and individual_time != float('inf'):
            speedup = batch_time / individual_time if individual_time > 0 else float('inf')
            
            print(f"📊 Performance Summary:")
            print(f"   Batch Processing:     {batch_time:.2f} seconds")
            print(f"   Individual Processing: {individual_time:.2f} seconds")
            print(f"   Speedup:              {speedup:.2f}x")
            
            if speedup > 1:
                print(f"   🚀 Individual processing is {speedup:.2f}x faster!")
            elif speedup < 1:
                print(f"   ⚠️  Batch processing is {1/speedup:.2f}x faster")
            else:
                print(f"   ⚖️  Both modes have similar performance")
            
            print()
            print(f"📈 Response Quality:")
            print(f"   Batch Response Length:     {results['batch']['response_length']} characters")
            print(f"   Individual Response Length: {results['individual']['response_length']} characters")
            
            length_diff = results['individual']['response_length'] - results['batch']['response_length']
            if length_diff > 0:
                print(f"   📝 Individual processing generated {length_diff} more characters")
            elif length_diff < 0:
                print(f"   📝 Batch processing generated {abs(length_diff)} more characters")
            else:
                print(f"   📝 Both modes generated similar response lengths")
        
        else:
            print("❌ One or both processing modes failed")
            if 'error' in results['batch']:
                print(f"   Batch processing error: {results['batch']['error']}")
            if 'error' in results['individual']:
                print(f"   Individual processing error: {results['individual']['error']}")
    
    print()
    print("💡 Key Benefits of Asynchronous Individual Processing:")
    print("   • All files processed simultaneously")
    print("   • Better handling of files with distinct topics")
    print("   • More detailed analysis per file")
    print("   • Scalable performance with multiple files")
    print()
    print("=" * 80)

def main():
    """Main function to run the performance comparison."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/performance_comparison.py <vectorstore_path>")
        print("Example: python scripts/performance_comparison.py ./vectorstores/my_vectorstore")
        sys.exit(1)
    
    vectorstore_path = sys.argv[1]
    
    if not Path(vectorstore_path).exists():
        print(f"Error: Vectorstore path does not exist: {vectorstore_path}")
        sys.exit(1)
    
    asyncio.run(compare_processing_performance(vectorstore_path))

if __name__ == "__main__":
    main() 