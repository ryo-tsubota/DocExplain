"""
Unit tests for DocumentService
"""
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi import UploadFile
from io import BytesIO

from app.services.document_service import DocumentService
from app.models.document import ProcessingStatus, SupportedFileType, DocumentChunk


class TestDocumentService:
    """Test cases for DocumentService"""
    
    @pytest.fixture
    def service(self):
        """Create a DocumentService instance for testing"""
        with patch('app.services.document_service.settings') as mock_settings:
            mock_settings.upload_dir = "test_uploads"
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            service = DocumentService()
            # Mock the embedding model to avoid downloading during tests
            service.embedding_model = Mock()
            # Create mock arrays that have tolist method
            import numpy as np
            mock_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
            service.embedding_model.encode = Mock(return_value=mock_embeddings)
            service.embedding_model.get_sentence_embedding_dimension = Mock(return_value=384)
            return service
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a mock PDF file for testing"""
        content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        file = UploadFile(
            filename="test.pdf",
            file=BytesIO(content),
            size=len(content),
            headers={"content-type": "application/pdf"}
        )
        return file
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a mock TXT file for testing"""
        content = b"This is a test document with some sample text content."
        file = UploadFile(
            filename="test.txt",
            file=BytesIO(content),
            size=len(content),
            headers={"content-type": "text/plain"}
        )
        return file
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, service, sample_txt_file):
        """Test successful document upload"""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()
            
            response = await service.upload_document(sample_txt_file)
            
            assert response.document.filename == "test.txt"
            assert response.document.file_type == SupportedFileType.TXT
            assert response.document.processing_status == ProcessingStatus.PENDING
            assert "uploaded successfully" in response.message
    
    @pytest.mark.asyncio
    async def test_upload_document_invalid_type(self, service):
        """Test upload with invalid file type"""
        invalid_file = UploadFile(
            filename="test.exe",
            file=BytesIO(b"invalid content"),
            size=100
        )
        
        with pytest.raises(Exception):
            await service.upload_document(invalid_file)
    
    @pytest.mark.asyncio
    async def test_get_documents(self, service, sample_txt_file):
        """Test getting document list"""
        with patch('builtins.open', create=True):
            await service.upload_document(sample_txt_file)
            
            documents = await service.get_documents()
            
            assert len(documents) == 1
            assert documents[0].filename == "test.txt"
    
    @pytest.mark.asyncio
    async def test_delete_document(self, service, sample_txt_file):
        """Test document deletion"""
        with patch('builtins.open', create=True):
            response = await service.upload_document(sample_txt_file)
            document_id = response.document.id
            
            # Mock file existence and deletion
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.unlink') as mock_unlink:
                
                result = await service.delete_document(document_id)
                
                assert result is True
                mock_unlink.assert_called_once()
                
                # Verify document is removed
                documents = await service.get_documents()
                assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_extract_txt_text(self, service):
        """Test text extraction from TXT file"""
        test_content = "This is a test document with multiple lines.\nSecond line here."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            f.flush()
            temp_path = Path(f.name)
            
        try:
            extracted_text = await service._extract_text(temp_path)
            assert test_content in extracted_text
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_extract_markdown_text(self, service):
        """Test text extraction from Markdown file"""
        test_content = "# Test Document\n\nThis is a **test** document with *markdown* formatting."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            f.flush()
            temp_path = Path(f.name)
            
        try:
            extracted_text = await service._extract_text(temp_path)
            assert "Test Document" in extracted_text
            assert "test" in extracted_text
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_create_chunks(self, service):
        """Test document chunking"""
        document_id = "test_doc_123"
        text_content = "This is a long document. " * 100  # Create long text
        
        chunks = await service._create_chunks(document_id, text_content)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(chunk.document_id == document_id for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)  # All chunks should have content
        assert all(chunk.chunk_index >= 0 for chunk in chunks)  # Valid indices
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, service):
        """Test embedding generation for chunks"""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                content="This is the first chunk",
                chunk_index=0
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_1",
                content="This is the second chunk",
                chunk_index=1
            )
        ]
        
        await service._generate_embeddings(chunks)
        
        # Check that embeddings were generated
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(len(chunk.embedding) > 0 for chunk in chunks)
        assert all("embedding_model" in chunk.metadata for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, service):
        """Test semantic search functionality"""
        # Setup test chunks with embeddings
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_1",
                content="Python programming language",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            ),
            DocumentChunk(
                id="chunk_2",
                document_id="doc_1",
                content="JavaScript web development",
                chunk_index=1,
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        service._document_chunks["doc_1"] = chunks
        
        # Mock the embedding model for query
        service.embedding_model.encode = Mock(return_value=[[0.1, 0.2, 0.3]])
        
        results = await service.search_similar_chunks("Python coding", top_k=1)
        
        assert len(results) == 1
        assert "Python" in results[0].content
    
    @pytest.mark.asyncio
    async def test_get_document_chunks(self, service):
        """Test retrieving chunks for a document"""
        document_id = "test_doc"
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id=document_id,
                content="First chunk",
                chunk_index=0
            )
        ]
        
        service._document_chunks[document_id] = chunks
        
        retrieved_chunks = await service.get_document_chunks(document_id)
        
        assert len(retrieved_chunks) == 1
        assert retrieved_chunks[0].content == "First chunk"
    
    @pytest.mark.asyncio
    async def test_get_processing_status(self, service, sample_txt_file):
        """Test getting document processing status"""
        with patch('builtins.open', create=True):
            response = await service.upload_document(sample_txt_file)
            document_id = response.document.id
            
            status = await service.get_processing_status(document_id)
            
            assert status == ProcessingStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_process_document_async_success(self, service):
        """Test successful asynchronous document processing"""
        document_id = "test_doc"
        test_content = "This is test content for processing."
        
        # Create a test document
        from app.models.document import Document, generate_content_hash
        document = Document(
            id=document_id,
            filename="test.txt",
            file_type=SupportedFileType.TXT,
            content_hash=generate_content_hash(test_content.encode()),
            file_size=len(test_content)
        )
        service._documents[document_id] = document
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            f.flush()
            temp_path = Path(f.name)
            
        try:
            await service._process_document_async(document_id, temp_path)
            
            # Check that processing completed
            assert service._documents[document_id].processing_status == ProcessingStatus.COMPLETED
            assert service._documents[document_id].chunk_count > 0
            assert document_id in service._document_chunks
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_document_async_failure(self, service):
        """Test document processing failure handling"""
        document_id = "test_doc"
        
        # Create a test document
        from app.models.document import Document, generate_content_hash
        document = Document(
            id=document_id,
            filename="test.txt",
            file_type=SupportedFileType.TXT,
            content_hash=generate_content_hash(b"test content"),
            file_size=100
        )
        service._documents[document_id] = document
        
        # Use non-existent file path to trigger failure
        non_existent_path = Path("non_existent_file.txt")
        
        await service._process_document_async(document_id, non_existent_path)
        
        # Check that processing failed
        assert service._documents[document_id].processing_status == ProcessingStatus.FAILED
    
    def test_get_file_type(self, service):
        """Test file type determination"""
        assert service._get_file_type("test.pdf") == SupportedFileType.PDF
        assert service._get_file_type("test.docx") == SupportedFileType.DOCX
        assert service._get_file_type("test.txt") == SupportedFileType.TXT
        assert service._get_file_type("test.md") == SupportedFileType.MD
    
    @pytest.mark.asyncio
    async def test_validate_file_success(self, service, sample_txt_file):
        """Test successful file validation"""
        # Should not raise any exception
        await service._validate_file(sample_txt_file)
    
    @pytest.mark.asyncio
    async def test_validate_file_no_filename(self, service):
        """Test file validation with no filename"""
        file = UploadFile(filename="", file=BytesIO(b"content"))
        
        with pytest.raises(Exception):
            await service._validate_file(file)
    
    @pytest.mark.asyncio
    async def test_validate_file_invalid_extension(self, service):
        """Test file validation with invalid extension"""
        file = UploadFile(filename="test.exe", file=BytesIO(b"content"))
        
        with pytest.raises(Exception):
            await service._validate_file(file)


class TestDocumentServiceIntegration:
    """Integration tests for DocumentService"""
    
    @pytest.fixture
    def service(self):
        """Create a DocumentService instance for integration testing"""
        with patch('app.services.document_service.settings') as mock_settings:
            mock_settings.upload_dir = "test_uploads"
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            service = DocumentService()
            # Use a mock embedding model to avoid downloading
            service.embedding_model = Mock()
            # Create mock function that returns the right number of embeddings
            import numpy as np
            def mock_encode(texts, convert_to_tensor=False):
                return [np.array([0.1, 0.2, 0.3]) for _ in texts]
            service.embedding_model.encode = mock_encode
            service.embedding_model.get_sentence_embedding_dimension = Mock(return_value=384)
            return service
    
    @pytest.mark.asyncio
    async def test_full_document_processing_pipeline(self, service):
        """Test the complete document processing pipeline"""
        # Create a test document
        test_content = "This is a comprehensive test document. " * 50
        document_id = "test_integration_doc"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            f.flush()
            temp_path = Path(f.name)
            
        try:
            # Create a test document directly in the service
            from app.models.document import Document, generate_content_hash
            document = Document(
                id=document_id,
                filename="integration_test.txt",
                file_type=SupportedFileType.TXT,
                content_hash=generate_content_hash(test_content.encode()),
                file_size=len(test_content)
            )
            service._documents[document_id] = document
            
            # Process document manually using the temp file directly
            await service._process_document_async(document_id, temp_path)
            
            # Verify processing results
            document = await service.get_document(document_id)
            assert document.processing_status == ProcessingStatus.COMPLETED
            assert document.chunk_count > 0
            
            # Verify chunks were created
            chunks = await service.get_document_chunks(document_id)
            assert len(chunks) > 0
            assert all(chunk.embedding is not None for chunk in chunks)
            
            # Test semantic search
            results = await service.search_similar_chunks("test document", [document_id])
            assert len(results) > 0
            
        finally:
            if temp_path.exists():
                temp_path.unlink()