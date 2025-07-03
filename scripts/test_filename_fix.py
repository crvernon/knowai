#!/usr/bin/env python3
"""
Test script to demonstrate the filename fix for synthesis prompts.

This script shows how the prompts have been updated to prevent
the model from using generic "file.pdf" instead of actual filenames.
"""

from knowai.prompts import get_synthesis_prompt_template

def test_synthesis_prompt():
    """Test that synthesis prompt includes filename instructions."""
    print("=== Testing Synthesis Prompt ===")
    
    # Test synthesis prompt
    template = get_synthesis_prompt_template()
    result = template.format(
        question="What is the main topic?",
        formatted_answers_or_raw_docs="Test content",
        files_no_info="None",
        files_errors="None",
        conversation_history="No previous conversation."
    )
    
    print("Synthesis Prompt:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Check for filename instructions
    assert "IMPORTANT: Always use the exact filename" in result
    assert "never use generic terms like \"file.pdf\"" in result
    
    print("✅ Synthesis prompt includes filename instructions")


if __name__ == "__main__":
    test_synthesis_prompt()
    print("\n🎉 All tests passed! The filename fix is working correctly.") 