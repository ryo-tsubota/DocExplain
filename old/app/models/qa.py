"""
Question-Answer data models
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
import uuid


class SourceChunk(BaseModel):
    """Source chunk reference in QA response"""
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Source document filename")
    content: str = Field(..., description="Relevant chunk content")
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    chunk_index: int = Field(..., description="Chunk index within document")
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")
        return v
    
    @validator('chunk_index')
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v


class QARequest(BaseModel):
    """Question-Answer request model"""
    question: str = Field(..., description="User question")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    max_chunks: int = Field(default=5, description="Maximum number of source chunks to return")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Question cannot be empty")
        if len(v.strip()) > 1000:
            raise ValueError("Question too long (max 1000 characters)")
        return v.strip()
    
    @validator('max_chunks')
    def validate_max_chunks(cls, v):
        if v < 1 or v > 20:
            raise ValueError("max_chunks must be between 1 and 20")
        return v
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v is not None and (not v or len(v.strip()) == 0):
            raise ValueError("Session ID cannot be empty if provided")
        return v


class QAResponse(BaseModel):
    """Question-Answer response model"""
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence score (0.0-1.0)")
    source_chunks: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    session_id: str = Field(..., description="Conversation session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v
    
    @validator('answer')
    def validate_answer(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Answer cannot be empty")
        return v.strip()


class QAExchange(BaseModel):
    """Single question-answer exchange in conversation"""
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="System answer")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Exchange timestamp")
    source_chunks: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    confidence_score: float = Field(..., description="Answer confidence score")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v


class ConversationHistory(BaseModel):
    """Conversation history model"""
    session_id: str = Field(..., description="Session identifier")
    exchanges: List[QAExchange] = Field(default_factory=list, description="Q&A exchanges")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Session ID cannot be empty")
        return v


class ConversationRequest(BaseModel):
    """Request to create new conversation"""
    initial_question: Optional[str] = Field(None, description="Optional initial question")
    document_ids: Optional[List[str]] = Field(None, description="Documents to include in conversation")
    
    @validator('initial_question')
    def validate_initial_question(cls, v):
        if v is not None and (not v or len(v.strip()) == 0):
            raise ValueError("Initial question cannot be empty if provided")
        return v


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())