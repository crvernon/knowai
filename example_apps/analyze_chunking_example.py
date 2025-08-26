#!/usr/bin/env python3
"""
Example script demonstrating how to analyze chunking parameters from an existing vectorstore.

This script shows how to determine the chunk size and overlap that was used to build
an existing FAISS vectorstore by analyzing the stored documents.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai import load_vectorstore, analyze_vectorstore_chunking


def main():
    """Example usage of chunking analysis functionality."""
    # Example vectorstore path - update this to your actual path
    vectorstore_path = "/Users/d3y010/repos/crvernon/knowai/wildfire_knowai/vectorstores/faiss_openai_large_20250606"
    
    print("Analyzing chunking parameters from existing vectorstore...")
    print(f"Vectorstore path: {vectorstore_path}")
    print("-" * 60)
    
    # Load the vectorstore
    vectorstore = load_vectorstore(vectorstore_path)
    if vectorstore is None:
        print("❌ Failed to load vectorstore")
        return 1
    
    print("✅ Successfully loaded vectorstore")
    
    # Analyze chunking parameters
    analysis = analyze_vectorstore_chunking(vectorstore)
    if not analysis:
        print("❌ Failed to analyze chunking parameters")
        return 1
    
    # Display results
    print("\n📊 Chunking Analysis Results:")
    print(f"   Documents analyzed: {analysis['total_documents_analyzed']}")
    
    print(f"\n📏 Chunk Size Analysis:")
    chunk_size = analysis['estimated_chunk_size']
    print(f"   Average: {chunk_size['average']} characters")
    print(f"   Median: {chunk_size['median']} characters")
    print(f"   Range: {chunk_size['min']} - {chunk_size['max']} characters")
    
    print(f"\n📈 Chunk Size Distribution:")
    for range_name, count in chunk_size['distribution'].items():
        percentage = (count / analysis['total_documents_analyzed']) * 100
        print(f"   {range_name}: {count} chunks ({percentage:.1f}%)")
    
    print(f"\n🔄 Overlap Analysis:")
    overlap = analysis['estimated_overlap']
    print(f"   Average: {overlap['average']} characters")
    print(f"   Median: {overlap['median']} characters")
    print(f"   Samples analyzed: {overlap['samples_analyzed']}")
    
    print(f"\n💡 Recommended Settings:")
    recommended = analysis['recommended_settings']
    print(f"   chunk_size: {recommended['chunk_size']}")
    print(f"   chunk_overlap: {recommended['chunk_overlap']}")
    
    print(f"\n🎯 Interpretation:")
    if recommended['chunk_size'] > 0:
        print(f"   Your vectorstore was likely built with:")
        print(f"   - chunk_size ≈ {recommended['chunk_size']} characters")
        print(f"   - chunk_overlap ≈ {recommended['chunk_overlap']} characters")
        
        if recommended['chunk_overlap'] == 0:
            print(f"   - No overlap was detected between chunks")
        else:
            print(f"   - Overlap of {recommended['chunk_overlap']} characters between chunks")
    else:
        print("   Could not determine chunking parameters")
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
