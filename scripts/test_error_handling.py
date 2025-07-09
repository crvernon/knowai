#!/usr/bin/env python3
"""
Test script to verify rate limit and token limit error handling.

This script tests the error handling functionality for:
1. Rate limit errors (429)
2. Token limit errors (400)
3. Content policy errors
4. Generic errors
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowai.agent import (
    _is_rate_limit_error,
    _is_token_limit_error,
    _is_content_policy_error,
    _handle_llm_error,
    RateLimitError,
    TokenLimitError
)


class MockException(Exception):
    """Mock exception for testing."""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
    
    def __str__(self):
        return self.message


def test_error_detection():
    """Test the error detection functions."""
    print("Testing error detection functions...")
    
    # Test rate limit error detection
    rate_limit_errors = [
        MockException("Rate limit exceeded", 429),
        MockException("Too many requests"),
        MockException("429 error occurred"),
        MockException("Quota exceeded"),
        MockException("Rate exceeded")
    ]
    
    for error in rate_limit_errors:
        result = _is_rate_limit_error(error)
        print(f"Rate limit error '{error}': {result}")
        assert result, f"Expected rate limit error to be detected for: {error}"
    
    # Test token limit error detection
    token_limit_errors = [
        MockException("Token limit exceeded", 400),
        MockException("Context length exceeded"),
        MockException("Maximum context length"),
        MockException("Input too long"),
        MockException("Prompt too long"),
        MockException("Context window exceeded")
    ]
    
    for error in token_limit_errors:
        result = _is_token_limit_error(error)
        print(f"Token limit error '{error}': {result}")
        assert result, f"Expected token limit error to be detected for: {error}"
    
    # Test content policy error detection
    content_policy_errors = [
        MockException("Content filter triggered"),
        MockException("Content management policy violation"),
        MockException("Responsible AI policy"),
        MockException("Safety policy violation"),
        MockException("Prompt blocked")
    ]
    
    for error in content_policy_errors:
        result = _is_content_policy_error(error)
        print(f"Content policy error '{error}': {result}")
        assert result, f"Expected content policy error to be detected for: {error}"
    
    # Test non-matching errors
    generic_errors = [
        MockException("Network timeout"),
        MockException("Connection failed"),
        MockException("Unknown error")
    ]
    
    for error in generic_errors:
        rate_result = _is_rate_limit_error(error)
        token_result = _is_token_limit_error(error)
        content_result = _is_content_policy_error(error)
        print(f"Generic error '{error}': rate={rate_result}, token={token_result}, content={content_result}")
        assert not rate_result, f"Expected generic error to not be detected as rate limit: {error}"
        assert not token_result, f"Expected generic error to not be detected as token limit: {error}"
        assert not content_result, f"Expected generic error to not be detected as content policy: {error}"
    
    print("✅ All error detection tests passed!")


def test_error_handling():
    """Test the error handling function."""
    print("\nTesting error handling function...")
    
    # Mock streaming callback
    callback_messages = []
    
    def mock_callback(message: str):
        callback_messages.append(message)
    
    # Test rate limit error handling
    rate_limit_error = MockException("Rate limit exceeded", 429)
    try:
        result = _handle_llm_error(rate_limit_error, mock_callback, "test_node")
        assert False, "Expected RateLimitError to be raised"
    except RateLimitError as e:
        print(f"✅ Rate limit error correctly raised: {e}")
        assert len(callback_messages) == 1, "Expected callback to be called"
        assert "rate limit" in callback_messages[0].lower(), "Expected rate limit message"
    
    # Clear callback messages
    callback_messages.clear()
    
    # Test token limit error handling
    token_limit_error = MockException("Token limit exceeded", 400)
    try:
        result = _handle_llm_error(token_limit_error, mock_callback, "test_node")
        assert False, "Expected TokenLimitError to be raised"
    except TokenLimitError as e:
        print(f"✅ Token limit error correctly raised: {e}")
        assert len(callback_messages) == 1, "Expected callback to be called"
        assert "token limit" in callback_messages[0].lower(), "Expected token limit message"
    
    # Clear callback messages
    callback_messages.clear()
    
    # Test content policy error handling
    content_policy_error = MockException("Content filter triggered")
    result = _handle_llm_error(content_policy_error, mock_callback, "test_node")
    print(f"✅ Content policy error correctly handled: {result}")
    assert "content management policy" in result.lower(), "Expected content policy message"
    
    # Clear callback messages
    callback_messages.clear()
    
    # Test generic error handling
    generic_error = MockException("Network timeout")
    result = _handle_llm_error(generic_error, mock_callback, "test_node")
    print(f"✅ Generic error correctly handled: {result}")
    assert "error occurred" in result.lower(), "Expected generic error message"
    assert len(callback_messages) == 1, "Expected callback to be called"
    
    print("✅ All error handling tests passed!")


def main():
    """Run all tests."""
    print("🧪 Testing KnowAI Error Handling")
    print("=" * 50)
    
    try:
        test_error_detection()
        test_error_handling()
        print("\n🎉 All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 