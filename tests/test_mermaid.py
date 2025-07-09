"""
Tests for the Mermaid diagram generation functionality.
"""

import pytest
from knowai.core import get_workflow_mermaid_diagram, KnowAIAgent


def test_get_workflow_mermaid_diagram():
    """Test that the standalone function returns a valid Mermaid diagram."""
    diagram = get_workflow_mermaid_diagram()
    
    # Check that it's a string
    assert isinstance(diagram, str)
    
    # Check that it contains expected Mermaid elements
    assert "graph TD" in diagram
    assert "instantiate_embeddings_node" in diagram
    assert "combine_answers_node" in diagram
    assert "__start__" in diagram
    assert "__end__" in diagram


def test_get_workflow_mermaid_diagram_save_to_file(tmp_path):
    """Test saving the diagram to a file."""
    output_file = tmp_path / "test_diagram.md"
    diagram = get_workflow_mermaid_diagram(save_to_file=str(output_file))
    
    # Check that the file was created
    assert output_file.exists()
    
    # Check that the file contains the diagram
    content = output_file.read_text()
    assert "```mermaid" in content
    assert "graph TD" in content
    assert "```" in content
    
    # Check that the function still returns the diagram
    assert isinstance(diagram, str)
    assert "graph TD" in diagram


def test_knowai_agent_get_graph_mermaid():
    """Test the KnowAIAgent's get_graph_mermaid method."""
    agent = KnowAIAgent(vectorstore_path="tests/fixtures/vectorstore")
    diagram = agent.get_graph_mermaid()
    
    # Check that it's a string
    assert isinstance(diagram, str)
    
    # Check that it contains expected Mermaid elements
    assert "graph TD" in diagram
    assert "instantiate_embeddings_node" in diagram
    assert "combine_answers_node" in diagram


def test_knowai_agent_get_graph_mermaid_save_to_file(tmp_path):
    """Test saving the diagram from KnowAIAgent to a file."""
    agent = KnowAIAgent(vectorstore_path="tests/fixtures/vectorstore")
    output_file = tmp_path / "agent_diagram.md"
    diagram = agent.get_graph_mermaid(save_to_file=str(output_file))
    
    # Check that the file was created
    assert output_file.exists()
    
    # Check that the file contains the diagram
    content = output_file.read_text()
    assert "```mermaid" in content
    assert "graph TD" in content
    assert "```" in content
    
    # Check that the function still returns the diagram
    assert isinstance(diagram, str)
    assert "graph TD" in diagram 