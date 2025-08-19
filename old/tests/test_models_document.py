"""
Unit tests for document data models
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.document import (
    Document,
    DocumentChunk,
    DocumentSummary,
    DocumentResponse,
    DocumentUploadRequest,
    ProcessingStatus,
    SupportedFileType,
    generate_content_hash,
)


class TestDocument:
    """Test cases for Document model"""
    
    def test_valid_document_creation(self):
        """Test creating a valid document"""
        doc = Document(
            id="doc-123",
            filename="test.pdf",
            file_type=SupportedFileType.PDF,
            content_hash="a" * 64,  # Valid SHA-256 hash
            file_size=1024
        )
        
        assert doc.id == "doc-123"
        assert doc.filename == "test.pdf"
        assert doc.file_type == SupportedFileType.PDF
        assert doc.processing_status == ProcessingStatus.PENDING
        assert doc.chunk_count == 0
        assert doc.file_size == 1024
        assert isinstance(doc.upload_timestamp, datetime)
    
    def test_document_filename_validation(self):
        """Test filename validation"""
        # Empty filename
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="",
                file_type=SupportedFileType.PDF,
                content_hash="a" * 64,
                file_size=1024
            )
        assert "Filename cannot be empty" in str(exc_info.value)
        
        # Too long filename
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="a" * 256,
                file_type=SupportedFileType.PDF,
                content_hash="a" * 64,
                file_size=1024
            )
        assert "Filename too long" in str(exc_info.value)
        
        # Whitespace handling
        doc = Document(
            id="doc-123",
            filename="  test.pdf  ",
            file_type=SupportedFileType.PDF,
            content_hash="a" * 64,
            file_size=1024
        )
        assert doc.filename == "test.pdf"
    
    def test_document_file_size_validation(self):
        """Test file size validation"""
        # Zero file size
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="test.pdf",
                file_type=SupportedFileType.PDF,
                content_hash="a" * 64,
                file_size=0
            )
        assert "File size must be positive" in str(exc_info.value)
        
        # Negative file size
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="test.pdf",
                file_type=SupportedFileType.PDF,
                content_hash="a" * 64,
                file_size=-1
            )
        assert "File size must be positive" in str(exc_info.value)
        
        # File too large (over 50MB)
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="test.pdf",
                file_type=SupportedFileType.PDF,
                content_hash="a" * 64,
                file_size=51 * 1024 * 1024
            )
        assert "File size exceeds 50MB limit" in str(exc_info.value)
    
    def test_document_content_hash_validation(self):
        """Test content hash validation"""
        # Invalid hash length
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="test.pdf",
                file_type=SupportedFileType.PDF,
                content_hash="invalid",
                file_size=1024
            )
        assert "Content hash must be a valid SHA-256 hash" in str(exc_info.value)
        
        # Empty hash
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="doc-123",
                filename="test.pdf",
                file_type=SupportedFileType.PDF,
                content_hash="",
                file_size=1024
            )
        assert "Content hash must be a valid SHA-256 hash" in str(exc_info.value)


class TestDocumentChunk:
    """Test cases for DocumentChunk model"""
    
    def test_valid_chunk_creation(self):
        """Test creating a valid document chunk"""
        chunk = DocumentChunk(
            id="chunk-123",
            document_id="doc-123",
            content="This is test content",
            chunk_index=0
        )
        
        assert chunk.id == "chunk-123"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is test content"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
        assert chunk.metadata == {}
    
    def test_chunk_content_validation(self):
        """Test chunk content validation"""
        # Empty content
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                id="chunk-123",
                document_id="doc-123",
                content="",
                chunk_index=0
            )
        assert "Chunk content cannot be empty" in str(exc_info.value)
        
        # Whitespace handling
        chunk = DocumentChunk(
            id="chunk-123",
            document_id="doc-123",
            content="  test content  ",
            chunk_index=0
        )
        assert chunk.content == "test content"
    
    def test_chunk_index_validation(self):
        """Test chunk index validation"""
        # Negative index
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                id="chunk-123",
                document_id="doc-123",
                content="test content",
                chunk_index=-1
            )
        assert "Chunk index must be non-negative" in str(exc_info.value)


class TestDocumentUploadRequest:
    """Test cases for DocumentUploadRequest model"""
    
    def test_valid_upload_request(self):
        """Test creating a valid upload request"""
        request = DocumentUploadRequest(
            filename="test.pdf",
            content_type="application/pdf",
            file_size=1024
        )
        
        assert request.filename == "test.pdf"
        assert request.content_type == "application/pdf"
        assert request.file_size == 1024
    
    def test_filename_validation(self):
        """Test filename validation in upload request"""
        # Invalid extension
        with pytest.raises(ValidationError) as exc_info:
            DocumentUploadRequest(
                filename="test.exe",
                content_type="application/pdf",
                file_size=1024
            )
        assert "Unsupported file type" in str(exc_info.value)
        
        # Valid extensions
        valid_files = ["test.pdf", "test.docx", "test.txt", "test.md"]
        for filename in valid_files:
            request = DocumentUploadRequest(
                filename=filename,
                content_type="application/pdf",
                file_size=1024
            )
            assert request.filename == filename
    
    def test_content_type_validation(self):
        """Test content type validation"""
        # Invalid content type
        with pytest.raises(ValidationError) as exc_info:
            DocumentUploadRequest(
                filename="test.pdf",
                content_type="application/exe",
                file_size=1024
            )
        assert "Unsupported content type" in str(exc_info.value)
        
        # Valid content types
        valid_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown"
        ]
        for content_type in valid_types:
            request = DocumentUploadRequest(
                filename="test.pdf",
                content_type=content_type,
                file_size=1024
            )
            assert request.content_type == content_type
    
    def test_file_size_validation_in_upload(self):
        """Test file size validation in upload request"""
        # File too large
        with pytest.raises(ValidationError) as exc_info:
            DocumentUploadRequest(
                filename="test.pdf",
                content_type="application/pdf",
                file_size=51 * 1024 * 1024
            )
        assert "File size exceeds 50MB limit" in str(exc_info.value)


class TestDocumentSummary:
    """Test cases for DocumentSummary model"""
    
    def test_valid_summary_creation(self):
        """Test creating a valid document summary"""
        summary = DocumentSummary(
            id="doc-123",
            filename="test.pdf",
            file_type=SupportedFileType.PDF,
            upload_timestamp=datetime.utcnow(),
            processing_status=ProcessingStatus.COMPLETED,
            chunk_count=5,
            file_size=1024
        )
        
        assert summary.id == "doc-123"
        assert summary.filename == "test.pdf"
        assert summary.file_type == SupportedFileType.PDF
        assert summary.processing_status == ProcessingStatus.COMPLETED
        assert summary.chunk_count == 5
        assert summary.file_size == 1024


class TestDocumentResponse:
    """Test cases for DocumentResponse model"""
    
    def test_valid_response_creation(self):
        """Test creating a valid document response"""
        doc = Document(
            id="doc-123",
            filename="test.pdf",
            file_type=SupportedFileType.PDF,
            content_hash="a" * 64,
            file_size=1024
        )
        
        response = DocumentResponse(document=doc)
        
        assert response.document == doc
        assert response.message == "Document processed successfully"
        
        # Custom message
        response_custom = DocumentResponse(
            document=doc,
            message="Custom message"
        )
        assert response_custom.message == "Custom message"


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_generate_content_hash(self):
        """Test content hash generation"""
        content = b"test content"
        hash_value = generate_content_hash(content)
        
        assert len(hash_value) == 64
        assert isinstance(hash_value, str)
        
        # Same content should produce same hash
        hash_value2 = generate_content_hash(content)
        assert hash_value == hash_value2
        
        # Different content should produce different hash
        different_content = b"different content"
        different_hash = generate_content_hash(different_content)
        assert hash_value != different_hash


class TestEnums:
    """Test cases for enum values"""
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enum values"""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
    
    def test_supported_file_type_enum(self):
        """Test SupportedFileType enum values"""
        assert SupportedFileType.PDF == "pdf"
        assert SupportedFileType.DOCX == "docx"
        assert SupportedFileType.TXT == "txt"
        assert SupportedFileType.MD == "md"