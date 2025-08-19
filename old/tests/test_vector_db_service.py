"""
Unit tests for vector database service
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from app.services.vector_db_service import VectorDBService
from app.models.document import DocumentChunk


class TestVectorDBService:
    """Test cases for VectorDBService"""
    
    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary directory for test database"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def vector_service(self, temp_db_path):
        """Create VectorDBService instance for testing"""
        service = VectorDBService()
        
        # Mock settings to use temp directory
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = temp_db_path
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            await service.initialize()
            yield service
            await service.close()
    
    @pytest.fixture
    def sample_chunks(self) -> List[DocumentChunk]:
        """Create sample document chunks for testing"""
        return [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                content="This is the first chunk of text content.",
                chunk_index=0,
                metadata={"section": "introduction"}
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_1",
                content="This is the second chunk with different content.",
                chunk_index=1,
                metadata={"section": "body"}
            ),
            DocumentChunk(
                id="chunk_3",
                document_id="doc_2",
                content="This chunk belongs to a different document.",
                chunk_index=0,
                metadata={"section": "conclusion"}
            )
        ]
    
    async def test_initialization(self, temp_db_path):
        """Test vector database service initialization"""
        service = VectorDBService()
        
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = temp_db_path
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            await service.initialize()
            
            assert service._client is not None
            assert service._collection is not None
            assert service._embedding_model is not None
            
            await service.close()
    
    async def test_initialization_failure(self):
        """Test handling of initialization failures"""
        service = VectorDBService()
        
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = "/invalid/path"
            mock_settings.embedding_model = "invalid-model"
            
            with pytest.raises(Exception):
                await service.initialize()
    
    async def test_ensure_initialized_raises_error(self):
        """Test that operations fail when service is not initialized"""
        service = VectorDBService()
        
        with pytest.raises(RuntimeError, match="VectorDBService not initialized"):
            service._ensure_initialized()
    
    async def test_store_document_chunks_success(self, vector_service, sample_chunks):
        """Test successful storage of document chunks"""
        result = await vector_service.store_document_chunks(sample_chunks)
        
        assert result is True
        
        # Verify chunks were stored by checking count
        count_doc1 = await vector_service.get_document_chunk_count("doc_1")
        count_doc2 = await vector_service.get_document_chunk_count("doc_2")
        
        assert count_doc1 == 2
        assert count_doc2 == 1
    
    async def test_store_empty_chunks(self, vector_service):
        """Test storing empty list of chunks"""
        result = await vector_service.store_document_chunks([])
        assert result is True
    
    async def test_store_chunks_with_embeddings(self, vector_service):
        """Test storing chunks that already have embeddings"""
        chunk_with_embedding = DocumentChunk(
            id="chunk_with_emb",
            document_id="doc_emb",
            content="Content with pre-computed embedding",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 77  # 385 dimensions for all-MiniLM-L6-v2
        )
        
        result = await vector_service.store_document_chunks([chunk_with_embedding])
        assert result is True
    
    async def test_search_similar_chunks(self, vector_service, sample_chunks):
        """Test semantic search for similar chunks"""
        # First store the chunks
        await vector_service.store_document_chunks(sample_chunks)
        
        # Search for similar content
        results = await vector_service.search_similar_chunks(
            query="first chunk text",
            limit=5
        )
        
        assert len(results) > 0
        assert all("chunk_id" in result for result in results)
        assert all("content" in result for result in results)
        assert all("similarity_score" in result for result in results)
        assert all("document_id" in result for result in results)
        
        # Results should be sorted by similarity (highest first)
        if len(results) > 1:
            assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    
    async def test_search_with_document_filter(self, vector_service, sample_chunks):
        """Test search with document ID filtering"""
        await vector_service.store_document_chunks(sample_chunks)
        
        # Search only in doc_1
        results = await vector_service.search_similar_chunks(
            query="chunk content",
            document_ids=["doc_1"],
            limit=5
        )
        
        # All results should be from doc_1
        assert all(result["document_id"] == "doc_1" for result in results)
    
    async def test_search_with_similarity_threshold(self, vector_service, sample_chunks):
        """Test search with minimum similarity threshold"""
        await vector_service.store_document_chunks(sample_chunks)
        
        # Search with high similarity threshold
        results = await vector_service.search_similar_chunks(
            query="completely unrelated query about quantum physics",
            min_similarity=0.8,
            limit=5
        )
        
        # Should return fewer or no results due to high threshold
        assert all(result["similarity_score"] >= 0.8 for result in results)
    
    async def test_search_no_results(self, vector_service):
        """Test search when no chunks are stored"""
        results = await vector_service.search_similar_chunks(
            query="any query",
            limit=5
        )
        
        assert results == []
    
    async def test_delete_document_chunks(self, vector_service, sample_chunks):
        """Test deletion of document chunks"""
        # Store chunks first
        await vector_service.store_document_chunks(sample_chunks)
        
        # Verify chunks exist
        count_before = await vector_service.get_document_chunk_count("doc_1")
        assert count_before == 2
        
        # Delete chunks for doc_1
        result = await vector_service.delete_document_chunks("doc_1")
        assert result is True
        
        # Verify chunks are deleted
        count_after = await vector_service.get_document_chunk_count("doc_1")
        assert count_after == 0
        
        # Verify other document chunks still exist
        count_doc2 = await vector_service.get_document_chunk_count("doc_2")
        assert count_doc2 == 1
    
    async def test_delete_nonexistent_document(self, vector_service):
        """Test deletion of chunks for non-existent document"""
        result = await vector_service.delete_document_chunks("nonexistent_doc")
        assert result is True  # Should succeed even if no chunks found
    
    async def test_get_document_chunk_count(self, vector_service, sample_chunks):
        """Test getting chunk count for documents"""
        # Initially no chunks
        count = await vector_service.get_document_chunk_count("doc_1")
        assert count == 0
        
        # Store chunks
        await vector_service.store_document_chunks(sample_chunks)
        
        # Check counts
        count_doc1 = await vector_service.get_document_chunk_count("doc_1")
        count_doc2 = await vector_service.get_document_chunk_count("doc_2")
        count_nonexistent = await vector_service.get_document_chunk_count("nonexistent")
        
        assert count_doc1 == 2
        assert count_doc2 == 1
        assert count_nonexistent == 0
    
    async def test_health_check_healthy(self, vector_service, sample_chunks):
        """Test health check when service is healthy"""
        await vector_service.store_document_chunks(sample_chunks)
        
        health = await vector_service.health_check()
        
        assert health["status"] == "healthy"
        assert "collection_name" in health
        assert "total_chunks" in health
        assert "embedding_model" in health
        assert health["total_chunks"] >= 3
    
    async def test_health_check_unhealthy(self):
        """Test health check when service is not initialized"""
        service = VectorDBService()
        
        health = await service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    async def test_reset_database(self, vector_service, sample_chunks):
        """Test database reset functionality"""
        # Store some chunks
        await vector_service.store_document_chunks(sample_chunks)
        
        # Verify chunks exist
        health_before = await vector_service.health_check()
        assert health_before["total_chunks"] > 0
        
        # Reset database
        result = await vector_service.reset_database()
        assert result is True
        
        # Verify database is empty
        health_after = await vector_service.health_check()
        assert health_after["total_chunks"] == 0
    
    async def test_concurrent_operations(self, vector_service, sample_chunks):
        """Test concurrent database operations"""
        import asyncio
        
        # Prepare multiple chunk sets
        chunk_sets = [
            [DocumentChunk(
                id=f"chunk_{i}_{j}",
                document_id=f"doc_{i}",
                content=f"Content for document {i}, chunk {j}",
                chunk_index=j
            ) for j in range(2)]
            for i in range(3)
        ]
        
        # Store chunks concurrently
        tasks = [
            vector_service.store_document_chunks(chunks)
            for chunks in chunk_sets
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(results)
        
        # Verify all chunks were stored
        total_count = 0
        for i in range(3):
            count = await vector_service.get_document_chunk_count(f"doc_{i}")
            total_count += count
        
        assert total_count == 6  # 3 documents * 2 chunks each


class TestVectorDBServiceErrorHandling:
    """Test error handling in VectorDBService"""
    
    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary directory for test database"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    async def test_store_chunks_chromadb_error(self, temp_db_path):
        """Test handling of ChromaDB errors during storage"""
        service = VectorDBService()
        
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = temp_db_path
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            await service.initialize()
            
            # Mock ChromaDB collection to raise an error
            with patch.object(service._collection, 'add', side_effect=Exception("ChromaDB error")):
                chunk = DocumentChunk(
                    id="test_chunk",
                    document_id="test_doc",
                    content="Test content",
                    chunk_index=0
                )
                
                result = await service.store_document_chunks([chunk])
                assert result is False
            
            await service.close()
    
    async def test_search_chromadb_error(self, temp_db_path):
        """Test handling of ChromaDB errors during search"""
        service = VectorDBService()
        
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = temp_db_path
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            await service.initialize()
            
            # Mock ChromaDB collection to raise an error
            with patch.object(service._collection, 'query', side_effect=Exception("ChromaDB error")):
                results = await service.search_similar_chunks("test query")
                assert results == []
            
            await service.close()
    
    async def test_delete_chunks_chromadb_error(self, temp_db_path):
        """Test handling of ChromaDB errors during deletion"""
        service = VectorDBService()
        
        with patch('app.services.vector_db_service.settings') as mock_settings:
            mock_settings.vector_db_path = temp_db_path
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            await service.initialize()
            
            # Mock ChromaDB collection to raise an error
            with patch.object(service._collection, 'get', side_effect=Exception("ChromaDB error")):
                result = await service.delete_document_chunks("test_doc")
                assert result is False
            
            await service.close()