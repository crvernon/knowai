"""
Tests for the centralized prompts module.
"""

import pytest
from knowai.prompts import (
    INDIVIDUAL_ANSWER_TEMPLATE,
    get_synthesis_prompt_template,
    CONTENT_POLICY_MESSAGE,
    get_progress_message,
    PROGRESS_MESSAGES
)


def test_individual_answer_template():
    """Test that the individual answer template has the correct structure."""
    assert INDIVIDUAL_ANSWER_TEMPLATE is not None
    assert "context" in INDIVIDUAL_ANSWER_TEMPLATE.input_variables
    assert "question" in INDIVIDUAL_ANSWER_TEMPLATE.input_variables
    assert "filename" in INDIVIDUAL_ANSWER_TEMPLATE.input_variables
    
    # Test template formatting
    result = INDIVIDUAL_ANSWER_TEMPLATE.format(
        context="Test context",
        question="Test question",
        filename="test.pdf"
    )
    assert "Test context" in result
    assert "Test question" in result
    assert "test.pdf" in result


def test_synthesis_prompt_templates():
    """Test that synthesis prompt templates work correctly."""
    # Test processed answers template
    processed_template = get_synthesis_prompt_template(bypass_individual_generation=False)
    assert processed_template is not None
    assert "question" in processed_template.input_variables
    assert "formatted_answers_or_raw_docs" in processed_template.input_variables
    assert "files_no_info" in processed_template.input_variables
    assert "files_errors" in processed_template.input_variables
    assert "conversation_history" in processed_template.input_variables
    
    # Test raw documents template
    raw_template = get_synthesis_prompt_template(bypass_individual_generation=True)
    assert raw_template is not None
    assert "question" in raw_template.input_variables
    assert "formatted_answers_or_raw_docs" in raw_template.input_variables
    assert "files_no_info" in raw_template.input_variables
    assert "files_errors" in raw_template.input_variables
    assert "conversation_history" in raw_template.input_variables
    
    # Test that templates are different
    processed_text = processed_template.template
    raw_text = raw_template.template
    assert processed_text != raw_text
    assert "PRE-PROCESSED" in processed_text
    assert "RAW text chunks" in raw_text


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
        "answer_generation", "document_preparation", "synthesis"
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
    assert "Generating answers" in get_progress_message("answer_generation", "generate_answers_node")
    assert "Preparing documents" in get_progress_message("document_preparation", "format_raw_documents_for_synthesis_node")
    assert "Synthesizing final response" in get_progress_message("synthesis", "combine_answers_node")
    
    # Test fallback for unknown node
    fallback = get_progress_message("unknown_stage", "unknown_node")
    assert "Processing unknown_node" in fallback 