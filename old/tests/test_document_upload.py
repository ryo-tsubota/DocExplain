"""
Tests for document upload functionality
"""
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_service import DocumentService
from app.models.document import ProcessingStatus


client = TestClient(app)


class TestDocumentUploadEndpoint:
    """Test cases for document upload API endpoint"""
    
    def test_upload_valid_pdf_file(self):
        """Test uploading a valid PDF file"""
        # Create a mock PDF file
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "document" in data
        assert data["document"]["filename"] == "test.pdf"
        assert data["document"]["file_type"] == "pdf"
        assert data["document"]["processing_status"] == "pending"
        assert data["message"] == "Document uploaded successfully and queued for processing"
    
    def test_upload_valid_txt_file(self):
        """Test uploading a valid text file"""
        txt_content = b"This is a test document with some content for testing purposes."
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document"]["filename"] == "test.txt"
        assert data["document"]["file_type"] == "txt"
        assert data["document"]["file_size"] == len(txt_content)
    
    def test_upload_valid_markdown_file(self):
        """Test uploading a valid markdown file"""
        md_content = b"# Test Document\n\nThis is a **test** markdown document.\n\n## Section 1\n\nSome content here."
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.md", io.BytesIO(md_content), "text/markdown")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document"]["filename"] == "test.md"
        assert data["document"]["file_type"] == "md"
    
    def test_upload_unsupported_file_type(self):
        """Test uploading an unsupported file type"""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.jpg", io.BytesIO(b"fake image content"), "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_file_too_large(self):
        """Test uploading a file that exceeds size limit"""
        # Create a large file (51MB)
        large_content = b"x" * (51 * 1024 * 1024)
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File size exceeds maximum" in response.json()["detail"]
    
    def test_upload_empty_filename(self):
        """Test uploading a file with empty filename"""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("", io.BytesIO(b"content"), "text/plain")}
        )
        
        assert response.status_code == 422  # FastAPI returns 422 for validation errors
        # FastAPI returns a list of validation errors
        detail = response.json()["detail"]
        assert isinstance(detail, list)
        assert len(detail) > 0
    
    def test_upload_mismatched_content_type(self):
        """Test uploading a file with mismatched content type and extension"""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(b"plain text"), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Content type" in response.json()["detail"]
        assert "does not match file extension" in response.json()["detail"]


class TestDocumentService:
    """Test cases for DocumentService class"""
    
    @pytest.fixture
    def service(self):
        """Create a DocumentService instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('app.services.document_service.settings.upload_dir', temp_dir):
                service = DocumentService()
                yield service
    
    @pytest.fixture
    def sample_upload_file(self):
        """Create a sample UploadFile for testing"""
        content = b"This is test content for document processing."
        file = UploadFile(
            filename="test.txt",
            file=io.BytesIO(content),
            size=len(content),
            headers={"content-type": "text/plain"}
        )
        return file
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, service, sample_upload_file):
        """Test successful document upload"""
        response = await service.upload_document(sample_upload_file)
        
        assert response.document.filename == "test.txt"
        assert response.document.file_type == "txt"
        assert response.document.processing_status == ProcessingStatus.PENDING
        assert response.document.file_size == 45  # Length of test content (updated)
        assert len(response.document.content_hash) == 64  # SHA-256 hash length
        assert response.message == "Document uploaded successfully and queued for processing"
    
    @pytest.mark.asyncio
    async def test_get_documents_empty(self, service):
        """Test getting documents when none are uploaded"""
        documents = await service.get_documents()
        assert documents == []
    
    @pytest.mark.asyncio
    async def test_get_documents_with_uploads(self, service, sample_upload_file):
        """Test getting documents after uploading"""
        # Upload a document
        response = await service.upload_document(sample_upload_file)
        document_id = response.document.id
        
        # Get documents list
        documents = await service.get_documents()
        assert len(documents) == 1
        assert documents[0].id == document_id
        assert documents[0].filename == "test.txt"
    
    @pytest.mark.asyncio
    async def test_get_document_by_id(self, service, sample_upload_file):
        """Test getting a specific document by ID"""
        # Upload a document
        response = await service.upload_document(sample_upload_file)
        document_id = response.document.id
        
        # Get document by ID
        document = await service.get_document(document_id)
        assert document is not None
        assert document.id == document_id
        assert document.filename == "test.txt"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, service):
        """Test getting a document that doesn't exist"""
        document = await service.get_document("nonexistent-id")
        assert document is None
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self, service, sample_upload_file):
        """Test successful document deletion"""
        # Upload a document
        response = await service.upload_document(sample_upload_file)
        document_id = response.document.id
        
        # Verify document exists
        document = await service.get_document(document_id)
        assert document is not None
        
        # Delete document
        success = await service.delete_document(document_id)
        assert success is True
        
        # Verify document is deleted
        document = await service.get_document(document_id)
        assert document is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, service):
        """Test deleting a document that doesn't exist"""
        success = await service.delete_document("nonexistent-id")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_processing_status(self, service, sample_upload_file):
        """Test getting document processing status"""
        # Upload a document
        response = await service.upload_document(sample_upload_file)
        document_id = response.document.id
        
        # Get processing status
        status = await service.get_processing_status(document_id)
        assert status == ProcessingStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_processing_status_nonexistent(self, service):
        """Test getting processing status for nonexistent document"""
        status = await service.get_processing_status("nonexistent-id")
        assert status is None


class TestDocumentManagementEndpoints:
    """Test cases for document management API endpoints"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear any existing documents
        from app.services.document_service import document_service
        document_service._documents.clear()
    
    def test_get_documents_empty(self):
        """Test getting documents when none exist"""
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_get_documents_with_data(self):
        """Test getting documents after uploading some"""
        # Upload a document first
        txt_content = b"Test document content"
        upload_response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
        )
        assert upload_response.status_code == 200
        
        # Get documents list
        response = client.get("/api/documents")
        assert response.status_code == 200
        documents = response.json()
        assert len(documents) == 1
        assert documents[0]["filename"] == "test.txt"
    
    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Upload a document first
        txt_content = b"Test document content"
        upload_response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
        )
        document_id = upload_response.json()["document"]["id"]
        
        # Delete the document
        response = client.delete(f"/api/documents/{document_id}")
        assert response.status_code == 200
        assert response.json()["message"] == "Document deleted successfully"
    
    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist"""
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"
    
    def test_get_document_status_success(self):
        """Test getting document status"""
        # Upload a document first
        txt_content = b"Test document content"
        upload_response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
        )
        document_id = upload_response.json()["document"]["id"]
        
        # Get document status
        response = client.get(f"/api/documents/{document_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == document_id
        # Status could be pending or completed depending on processing speed
        assert data["processing_status"] in ["pending", "completed"]
        assert f"Document is {data['processing_status']}" in data["message"]
    
    def test_get_status_nonexistent_document(self):
        """Test getting status for nonexistent document"""
        response = client.get("/api/documents/nonexistent-id/status")
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"