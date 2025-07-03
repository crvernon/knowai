#!/usr/bin/env python3
"""
Example demonstrating different token counting methods in KnowAI.

This example shows how to:
1. Use accurate token counting with tiktoken (default)
2. Use heuristic token counting as fallback
3. Compare the differences between the two methods
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowai.core import KnowAIAgent
from knowai.agent import estimate_tokens, TIKTOKEN_AVAILABLE


async def demonstrate_token_counting():
    """Demonstrate different token counting methods."""
    
    print("KnowAI Token Counting Methods Demo")
    print("=" * 50)
    
    # Test texts with different characteristics
    test_texts = [
        "Simple English text",
        "Text with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "Unicode text: 🚀🌟🎉中文日本語한국어",
        "Repeated text " * 100,  # 1500 characters
        "A" * 1000,  # 1000 identical characters
    ]
    
    print(f"tiktoken available: {TIKTOKEN_AVAILABLE}")
    print()
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Length: {len(text)} characters")
        
        if TIKTOKEN_AVAILABLE:
            accurate_tokens = estimate_tokens(text, use_accurate=True)
            print(f"  Accurate tokens: {accurate_tokens}")
        
        heuristic_tokens = estimate_tokens(text, use_accurate=False)
        print(f"  Heuristic tokens: {heuristic_tokens}")
        
        if TIKTOKEN_AVAILABLE:
            diff = abs(accurate_tokens - heuristic_tokens)
            diff_percent = (diff / accurate_tokens * 100) if accurate_tokens > 0 else 0
            print(f"  Difference: {diff} ({diff_percent:.1f}%)")
        
        print()
    
    # Demonstrate agent configuration
    print("Agent Configuration Examples:")
    print("-" * 30)
    
    # Example 1: Default (accurate token counting)
    print("1. Default configuration (accurate token counting):")
    print("   agent = KnowAIAgent(vectorstore_path='path/to/vectorstore')")
    print("   # Uses tiktoken when available, falls back to heuristic")
    print()
    
    # Example 2: Explicit accurate
    print("2. Explicit accurate token counting:")
    print("   agent = KnowAIAgent(")
    print("       vectorstore_path='path/to/vectorstore',")
    print("       use_accurate_token_counting=True")
    print("   )")
    print("   # Forces use of tiktoken when available")
    print()
    
    # Example 3: Explicit heuristic
    print("3. Explicit heuristic token counting:")
    print("   agent = KnowAIAgent(")
    print("       vectorstore_path='path/to/vectorstore',")
    print("       use_accurate_token_counting=False")
    print("   )")
    print("   # Uses character-based estimation only")
    print()
    
    # Example 4: CLI configuration
    print("4. CLI configuration:")
    print("   POST /initialize")
    print("   {")
    print('       "vectorstore_s3_uri": "s3://bucket/path",')
    print('       "use_accurate_token_counting": true')
    print("   }")
    print()
    
    print("Benefits of accurate token counting:")
    print("- More precise token limits")
    print("- Better batch sizing")
    print("- Reduced risk of context overflow")
    print("- More efficient resource usage")
    print()
    
    print("When to use heuristic counting:")
    print("- tiktoken not available")
    print("- Performance is critical")
    print("- Approximate estimation is sufficient")
    print("- Debugging token counting issues")


def main():
    """Run the demonstration."""
    try:
        asyncio.run(demonstrate_token_counting())
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main()) 