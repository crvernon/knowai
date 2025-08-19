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
    list_vectorstore_files,
    analyze_vectorstore_chunking
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
        # Create a mock vectorstore with proper structure
        mock_vectorstore = MagicMock()
        
        # Ensure it doesn't have the 'vectorstore' attribute (so it's treated as a vectorstore, not retriever)
        del mock_vectorstore.vectorstore
        
        # Configure the index attributes
        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_index.d = 1536
        mock_vectorstore.index = mock_index
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file_name': 'test1.pdf', 'page': 1}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file_name': 'test2.pdf', 'page': 2}
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2
        }
        mock_vectorstore.docstore = mock_docstore
        
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
        # Create a mock vectorstore with proper structure
        mock_vectorstore = MagicMock()
        
        # Ensure it doesn't have the 'vectorstore' attribute (so it's treated as a vectorstore, not retriever)
        del mock_vectorstore.vectorstore
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file': 'test1.pdf'}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file': 'test2.pdf'}
        mock_doc3 = MagicMock()
        mock_doc3.metadata = {'file': 'test1.pdf'}  # Duplicate
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2,
            'doc3': mock_doc3
        }
        mock_vectorstore.docstore = mock_docstore
        
        files = list_vectorstore_files(mock_vectorstore)
        assert len(files) == 2
        assert 'test1.pdf' in files
        assert 'test2.pdf' in files
        assert files == sorted(files)  # Should be sorted
    
    def test_analyze_vectorstore_chunking_mock(self):
        """Test chunking analysis with mock vectorstore."""
        # Create a mock vectorstore with sample documents
        mock_vectorstore = MagicMock()
        
        # Ensure it doesn't have the 'vectorstore' attribute (so it's treated as a vectorstore, not retriever)
        del mock_vectorstore.vectorstore
        
        # Mock document store with sample chunks
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "This is a sample chunk with some content. " * 50  # ~1400 chars
        mock_doc1.metadata = {'file_name': 'test1.pdf', 'page': 1, 'chunk_index': 1}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "This is another sample chunk with different content. " * 45  # ~1300 chars
        mock_doc2.metadata = {'file_name': 'test1.pdf', 'page': 1, 'chunk_index': 2}
        
        mock_doc3 = MagicMock()
        mock_doc3.page_content = "This is a third chunk with more content. " * 55  # ~1500 chars
        mock_doc3.metadata = {'file_name': 'test2.pdf', 'page': 1, 'chunk_index': 1}
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2,
            'doc3': mock_doc3
        }
        mock_vectorstore.docstore = mock_docstore
        
        analysis = analyze_vectorstore_chunking(mock_vectorstore)
        
        # Check that analysis was performed
        assert analysis is not None
        assert 'total_documents_analyzed' in analysis
        assert 'estimated_chunk_size' in analysis
        assert 'estimated_overlap' in analysis
        assert 'recommended_settings' in analysis
        
        # Check that we analyzed the expected number of documents
        assert analysis['total_documents_analyzed'] == 3
        
        # Check that chunk size analysis is reasonable
        chunk_size = analysis['estimated_chunk_size']
        assert chunk_size['average'] > 0
        assert chunk_size['median'] > 0
        assert chunk_size['min'] > 0
        assert chunk_size['max'] > 0
        
        # Check that recommended settings are provided
        recommended = analysis['recommended_settings']
        assert recommended['chunk_size'] > 0
        assert 'chunk_overlap' in recommended
    
    def test_analyze_vectorstore_chunking_none(self):
        """Test chunking analysis with None vectorstore."""
        analysis = analyze_vectorstore_chunking(None)
        assert analysis == {}
    
    def test_show_vectorstore_schema_retriever(self):
        """Test schema display with retriever object."""
        # Create a mock retriever (has vectorstore attribute)
        mock_retriever = MagicMock()
        
        # Create the underlying vectorstore
        mock_vectorstore = MagicMock()
        
        # Configure the index attributes
        mock_index = MagicMock()
        mock_index.ntotal = 200
        mock_index.d = 1536
        mock_vectorstore.index = mock_index
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file_name': 'test1.pdf', 'page': 1}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file_name': 'test2.pdf', 'page': 2}
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2
        }
        mock_vectorstore.docstore = mock_docstore
        
        # Set the vectorstore attribute on the retriever
        mock_retriever.vectorstore = mock_vectorstore
        
        schema = show_vectorstore_schema(mock_retriever)
        assert schema['total_vectors'] == 200
        assert schema['dimension'] == 1536
        assert 'file_name' in schema['metadata_fields']
        assert 'page' in schema['metadata_fields']
    
    def test_list_vectorstore_files_retriever(self):
        """Test file listing with retriever object."""
        # Create a mock retriever (has vectorstore attribute)
        mock_retriever = MagicMock()
        
        # Create the underlying vectorstore
        mock_vectorstore = MagicMock()
        
        # Mock document store
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {'file': 'test1.pdf'}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {'file': 'test2.pdf'}
        
        mock_docstore = MagicMock()
        mock_docstore._dict = {
            'doc1': mock_doc1,
            'doc2': mock_doc2
        }
        mock_vectorstore.docstore = mock_docstore
        
        # Set the vectorstore attribute on the retriever
        mock_retriever.vectorstore = mock_vectorstore
        
        files = list_vectorstore_files(mock_retriever)
        assert len(files) == 2
        assert 'test1.pdf' in files
        assert 'test2.pdf' in files
        assert files == sorted(files)  # Should be sorted
