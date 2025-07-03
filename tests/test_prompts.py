"""
Tests for the centralized prompts module.
"""

import pytest
from knowai.prompts import (
    get_synthesis_prompt_template,
    CONTENT_POLICY_MESSAGE,
    get_progress_message,
    PROGRESS_MESSAGES
)


def test_synthesis_prompt_template():
    """Test that synthesis prompt template works correctly."""
    template = get_synthesis_prompt_template()
    assert template is not None
    assert "question" in template.input_variables
    assert "formatted_answers_or_raw_docs" in template.input_variables
    assert "files_no_info" in template.input_variables
    assert "files_errors" in template.input_variables
    assert "conversation_history" in template.input_variables
    
    # Test template formatting
    result = template.format(
        question="Test question",
        formatted_answers_or_raw_docs="Test content",
        files_no_info="None",
        files_errors="None",
        conversation_history="No previous conversation."
    )
    assert "Test question" in result
    assert "Test content" in result


def test_content_policy_message():
    """Test that content policy message is defined."""
    assert CONTENT_POLICY_MESSAGE is not None
    assert isinstance(CONTENT_POLICY_MESSAGE, str)
    assert len(CONTENT_POLICY_MESSAGE) > 0


def test_progress_messages():
    """Test that progress messages are properly structured."""
    assert PROGRESS_MESSAGES is not None
    assert isinstance(PROGRESS_MESSAGES, dict)
    
    # Check that all expected stages exist
    expected_stages = [
        "initialization", "query_generation", "document_retrieval",
        "document_preparation", "synthesis"
    ]
    for stage in expected_stages:
        assert stage in PROGRESS_MESSAGES
        assert isinstance(PROGRESS_MESSAGES[stage], dict)


def test_get_progress_message():
    """Test the get_progress_message function."""
    # Test known node mappings
    assert "Setting up AI models" in get_progress_message("initialization", "instantiate_embeddings_node")
    assert "Generating search queries" in get_progress_message("query_generation", "generate_multi_queries_node")
    assert "Searching documents" in get_progress_message("document_retrieval", "extract_documents_node")
    assert "Preparing documents" in get_progress_message("document_preparation", "format_raw_documents_for_synthesis_node")
    assert "Synthesizing final response" in get_progress_message("synthesis", "combine_answers_node")
    
    # Test fallback for unknown node
    fallback = get_progress_message("unknown_stage", "unknown_node")
    assert "Processing unknown_node" in fallback 