"""
Tests for KnowAI vectorstore functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import pandas as pd

from knowai.vectorstore import (
    get_azure_credentials,
    show_vectorstore_schema,
    list_vectorstore_files
)


class TestVectorstoreUtils:
    """Test vectorstore utility functions."""
    
    def test_get_azure_credentials_success(self):
        """Test successful credential retrieval."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_API_KEY': 'test_key',
            'AZURE_OPENAI_ENDPOINT': 'test_endpoint',
            'AZURE_EMBEDDINGS_DEPLOYMENT': 'test_deployment',
            'AZURE_OPENAI_EMBEDDINGS_API_VERSION': 'test_version'
        }):
            credentials = get_azure_credentials()
            assert credentials is not None
            assert credentials['api_key'] == 'test_key'
            assert credentials['azure_endpoint'] == 'test_endpoint'
            assert credentials['embeddings_deployment'] == 'test_deployment'
            assert credentials['embeddings_api_version'] == 'test_version'
    
    def test_get_azure_credentials_missing(self):
        """Test credential retrieval when environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            credentials = get_azure_credentials()
            assert credentials is None
    
    def test_show_vectorstore_schema_none(self):
        """Test schema display with None vectorstore."""
        schema = show_vectorstore_schema(None)
        assert schema == {}
    
    def test_show_vectorstore_schema_mock(self):
        """Test schema display with mock vectorstore."""
        # Create a mock vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 100
        mock_vectorstore.index.d = 1536
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file_name': 'test1.pdf', 'page': 1}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file_name': 'test2.pdf', 'page': 2}
        
        mock_vectorstore.docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2
        }
        
        schema = show_vectorstore_schema(mock_vectorstore)
        assert schema['total_vectors'] == 100
        assert schema['dimension'] == 1536
        assert 'file_name' in schema['metadata_fields']
        assert 'page' in schema['metadata_fields']
    
    def test_list_vectorstore_files_none(self):
        """Test file listing with None vectorstore."""
        files = list_vectorstore_files(None)
        assert files == []
    
    def test_list_vectorstore_files_mock(self):
        """Test file listing with mock vectorstore."""
        # Create a mock vectorstore
        mock_vectorstore = MagicMock()
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file': 'test1.pdf'}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file': 'test2.pdf'}
        mock_doc3 = MagicMock()
        mock_doc3.metadata = {'file': 'test1.pdf'}  # Duplicate
        
        mock_vectorstore.docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2,
            'doc3': mock_doc3
        }
        
        files = list_vectorstore_files(mock_vectorstore)
        assert len(files) == 2
        assert 'test1.pdf' in files
        assert 'test2.pdf' in files
        assert files == sorted(files)  # Should be sorted
