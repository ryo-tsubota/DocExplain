"""
Common data models shared across the application
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Error code enumeration"""
    # File upload errors
    INVALID_FILE_TYPE = "invalid_file_type"
    FILE_TOO_LARGE = "file_too_large"
    FILE_CORRUPTED = "file_corrupted"
    UPLOAD_FAILED = "upload_failed"
    
    # Processing errors
    PROCESSING_FAILED = "processing_failed"
    TEXT_EXTRACTION_FAILED = "text_extraction_failed"
    EMBEDDING_GENERATION_FAILED = "embedding_generation_failed"
    
    # Query errors
    INVALID_QUESTION = "invalid_question"
    NO_DOCUMENTS_FOUND = "no_documents_found"
    NO_RELEVANT_CONTENT = "no_relevant_content"
    
    # System errors
    DATABASE_ERROR = "database_error"
    LLM_API_ERROR = "llm_api_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    
    # Authentication/Authorization errors
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    
    # Validation errors
    VALIDATION_ERROR = "validation_error"
    MISSING_REQUIRED_FIELD = "missing_required_field"


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: ErrorCode = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseModel):
    """Standard success response model"""
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(default="1.0.0", description="Application version")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Dependency status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }