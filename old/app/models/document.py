"""
Document data models
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
import hashlib


class ProcessingStatus(str, Enum):
    """Document processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SupportedFileType(str, Enum):
    """Supported file types for document upload"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class Document(BaseModel):
    """Core document model"""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: SupportedFileType = Field(..., description="File type")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    content_hash: str = Field(..., description="SHA-256 hash of file content")
    chunk_count: int = Field(default=0, description="Number of chunks created from document")
    file_size: int = Field(..., description="File size in bytes")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Filename cannot be empty")
        if len(v) > 255:
            raise ValueError("Filename too long (max 255 characters)")
        return v.strip()
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError("File size must be positive")
        # 50MB limit
        if v > 50 * 1024 * 1024:
            raise ValueError("File size exceeds 50MB limit")
        return v
    
    @validator('content_hash')
    def validate_content_hash(cls, v):
        if not v or len(v) != 64:
            raise ValueError("Content hash must be a valid SHA-256 hash")
        return v


class DocumentChunk(BaseModel):
    """Document chunk model for vector storage"""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunk_index: int = Field(..., description="Index of chunk within document")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Chunk content cannot be empty")
        return v.strip()
    
    @validator('chunk_index')
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v


class DocumentSummary(BaseModel):
    """Summary model for document listing"""
    id: str
    filename: str
    file_type: SupportedFileType
    upload_timestamp: datetime
    processing_status: ProcessingStatus
    chunk_count: int
    file_size: int


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    document: Document
    message: str = Field(default="Document processed successfully")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload validation"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Filename cannot be empty")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}")
        
        return v.strip()
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown'
        }
        if v not in allowed_types:
            raise ValueError(f"Unsupported content type: {v}")
        return v
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError("File size must be positive")
        if v > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("File size exceeds 50MB limit")
        return v


def generate_content_hash(content: bytes) -> str:
    """Generate SHA-256 hash for file content"""
    return hashlib.sha256(content).hexdigest()