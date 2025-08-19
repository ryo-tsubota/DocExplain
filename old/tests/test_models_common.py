"""
Unit tests for common data models
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.common import (
    ErrorCode,
    ErrorResponse,
    SuccessResponse,
    HealthCheckResponse,
)


class TestErrorCode:
    """Test cases for ErrorCode enum"""
    
    def test_error_code_values(self):
        """Test all error code enum values"""
        # File upload errors
        assert ErrorCode.INVALID_FILE_TYPE == "invalid_file_type"
        assert ErrorCode.FILE_TOO_LARGE == "file_too_large"
        assert ErrorCode.FILE_CORRUPTED == "file_corrupted"
        assert ErrorCode.UPLOAD_FAILED == "upload_failed"
        
        # Processing errors
        assert ErrorCode.PROCESSING_FAILED == "processing_failed"
        assert ErrorCode.TEXT_EXTRACTION_FAILED == "text_extraction_failed"
        assert ErrorCode.EMBEDDING_GENERATION_FAILED == "embedding_generation_failed"
        
        # Query errors
        assert ErrorCode.INVALID_QUESTION == "invalid_question"
        assert ErrorCode.NO_DOCUMENTS_FOUND == "no_documents_found"
        assert ErrorCode.NO_RELEVANT_CONTENT == "no_relevant_content"
        
        # System errors
        assert ErrorCode.DATABASE_ERROR == "database_error"
        assert ErrorCode.LLM_API_ERROR == "llm_api_error"
        assert ErrorCode.INTERNAL_SERVER_ERROR == "internal_server_error"
        
        # Authentication/Authorization errors
        assert ErrorCode.UNAUTHORIZED == "unauthorized"
        assert ErrorCode.FORBIDDEN == "forbidden"
        
        # Validation errors
        assert ErrorCode.VALIDATION_ERROR == "validation_error"
        assert ErrorCode.MISSING_REQUIRED_FIELD == "missing_required_field"


class TestErrorResponse:
    """Test cases for ErrorResponse model"""
    
    def test_valid_error_response_creation(self):
        """Test creating a valid error response"""
        error = ErrorResponse(
            error_code=ErrorCode.INVALID_FILE_TYPE,
            message="The uploaded file type is not supported"
        )
        
        assert error.error_code == ErrorCode.INVALID_FILE_TYPE
        assert error.message == "The uploaded file type is not supported"
        assert error.details is None
        assert isinstance(error.timestamp, datetime)
        assert error.request_id is None
    
    def test_error_response_with_details(self):
        """Test error response with additional details"""
        details = {
            "supported_types": ["pdf", "docx", "txt", "md"],
            "received_type": "exe"
        }
        
        error = ErrorResponse(
            error_code=ErrorCode.INVALID_FILE_TYPE,
            message="The uploaded file type is not supported",
            details=details,
            request_id="req-123"
        )
        
        assert error.details == details
        assert error.request_id == "req-123"
    
    def test_error_response_json_encoding(self):
        """Test error response JSON encoding"""
        error = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed"
        )
        
        # Should be able to convert to dict
        error_dict = error.dict()
        assert "timestamp" in error_dict
        assert isinstance(error_dict["timestamp"], datetime)
        
        # JSON encoding should work with custom encoder
        import json
        json_str = error.json()
        assert isinstance(json_str, str)
        
        # Parse back to verify timestamp is ISO format
        parsed = json.loads(json_str)
        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], str)


class TestSuccessResponse:
    """Test cases for SuccessResponse model"""
    
    def test_valid_success_response_creation(self):
        """Test creating a valid success response"""
        response = SuccessResponse(message="Operation completed successfully")
        
        assert response.message == "Operation completed successfully"
        assert response.data is None
        assert isinstance(response.timestamp, datetime)
    
    def test_success_response_with_data(self):
        """Test success response with data"""
        data = {
            "document_id": "doc-123",
            "status": "processed"
        }
        
        response = SuccessResponse(
            message="Document processed successfully",
            data=data
        )
        
        assert response.message == "Document processed successfully"
        assert response.data == data
    
    def test_success_response_json_encoding(self):
        """Test success response JSON encoding"""
        response = SuccessResponse(message="Success")
        
        # Should be able to convert to dict
        response_dict = response.dict()
        assert "timestamp" in response_dict
        assert isinstance(response_dict["timestamp"], datetime)
        
        # JSON encoding should work
        json_str = response.json()
        assert isinstance(json_str, str)
        
        # Parse back to verify timestamp is ISO format
        import json
        parsed = json.loads(json_str)
        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], str)


class TestHealthCheckResponse:
    """Test cases for HealthCheckResponse model"""
    
    def test_valid_health_check_creation(self):
        """Test creating a valid health check response"""
        health = HealthCheckResponse()
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.dependencies == {}
        assert isinstance(health.timestamp, datetime)
    
    def test_health_check_with_dependencies(self):
        """Test health check with dependency status"""
        dependencies = {
            "database": "healthy",
            "llm_api": "healthy",
            "vector_db": "degraded"
        }
        
        health = HealthCheckResponse(
            status="degraded",
            version="1.2.0",
            dependencies=dependencies
        )
        
        assert health.status == "degraded"
        assert health.version == "1.2.0"
        assert health.dependencies == dependencies
    
    def test_health_check_json_encoding(self):
        """Test health check JSON encoding"""
        health = HealthCheckResponse()
        
        # Should be able to convert to dict
        health_dict = health.dict()
        assert "timestamp" in health_dict
        assert isinstance(health_dict["timestamp"], datetime)
        
        # JSON encoding should work
        json_str = health.json()
        assert isinstance(json_str, str)
        
        # Parse back to verify timestamp is ISO format
        import json
        parsed = json.loads(json_str)
        assert "timestamp" in parsed
        assert isinstance(parsed["timestamp"], str)


class TestModelIntegration:
    """Integration tests for common models"""
    
    def test_error_response_with_all_error_codes(self):
        """Test error response works with all error codes"""
        for error_code in ErrorCode:
            error = ErrorResponse(
                error_code=error_code,
                message=f"Test error for {error_code.value}"
            )
            assert error.error_code == error_code
            assert error.message == f"Test error for {error_code.value}"
    
    def test_model_serialization_consistency(self):
        """Test that all models serialize consistently"""
        # Error response
        error = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Test error"
        )
        error_json = error.json()
        assert isinstance(error_json, str)
        
        # Success response
        success = SuccessResponse(message="Test success")
        success_json = success.json()
        assert isinstance(success_json, str)
        
        # Health check response
        health = HealthCheckResponse()
        health_json = health.json()
        assert isinstance(health_json, str)
        
        # All should have timestamp in ISO format
        import json
        for json_str in [error_json, success_json, health_json]:
            parsed = json.loads(json_str)
            assert "timestamp" in parsed
            # Should be able to parse as datetime
            datetime.fromisoformat(parsed["timestamp"].replace('Z', '+00:00'))