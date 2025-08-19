"""
Unit tests for QA data models
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.qa import (
    SourceChunk,
    QARequest,
    QAResponse,
    QAExchange,
    ConversationHistory,
    ConversationRequest,
    generate_session_id,
)


class TestSourceChunk:
    """Test cases for SourceChunk model"""
    
    def test_valid_source_chunk_creation(self):
        """Test creating a valid source chunk"""
        chunk = SourceChunk(
            document_id="doc-123",
            document_name="test.pdf",
            content="This is relevant content",
            relevance_score=0.85,
            chunk_index=0
        )
        
        assert chunk.document_id == "doc-123"
        assert chunk.document_name == "test.pdf"
        assert chunk.content == "This is relevant content"
        assert chunk.relevance_score == 0.85
        assert chunk.chunk_index == 0
    
    def test_relevance_score_validation(self):
        """Test relevance score validation"""
        # Score too low
        with pytest.raises(ValidationError) as exc_info:
            SourceChunk(
                document_id="doc-123",
                document_name="test.pdf",
                content="content",
                relevance_score=-0.1,
                chunk_index=0
            )
        assert "Relevance score must be between 0.0 and 1.0" in str(exc_info.value)
        
        # Score too high
        with pytest.raises(ValidationError) as exc_info:
            SourceChunk(
                document_id="doc-123",
                document_name="test.pdf",
                content="content",
                relevance_score=1.1,
                chunk_index=0
            )
        assert "Relevance score must be between 0.0 and 1.0" in str(exc_info.value)
        
        # Valid boundary values
        chunk_min = SourceChunk(
            document_id="doc-123",
            document_name="test.pdf",
            content="content",
            relevance_score=0.0,
            chunk_index=0
        )
        assert chunk_min.relevance_score == 0.0
        
        chunk_max = SourceChunk(
            document_id="doc-123",
            document_name="test.pdf",
            content="content",
            relevance_score=1.0,
            chunk_index=0
        )
        assert chunk_max.relevance_score == 1.0
    
    def test_chunk_index_validation(self):
        """Test chunk index validation"""
        # Negative index
        with pytest.raises(ValidationError) as exc_info:
            SourceChunk(
                document_id="doc-123",
                document_name="test.pdf",
                content="content",
                relevance_score=0.5,
                chunk_index=-1
            )
        assert "Chunk index must be non-negative" in str(exc_info.value)


class TestQARequest:
    """Test cases for QARequest model"""
    
    def test_valid_qa_request_creation(self):
        """Test creating a valid QA request"""
        request = QARequest(question="What is the system architecture?")
        
        assert request.question == "What is the system architecture?"
        assert request.document_ids is None
        assert request.session_id is None
        assert request.max_chunks == 5
    
    def test_qa_request_with_optional_fields(self):
        """Test QA request with optional fields"""
        request = QARequest(
            question="What is the system architecture?",
            document_ids=["doc-1", "doc-2"],
            session_id="session-123",
            max_chunks=10
        )
        
        assert request.question == "What is the system architecture?"
        assert request.document_ids == ["doc-1", "doc-2"]
        assert request.session_id == "session-123"
        assert request.max_chunks == 10
    
    def test_question_validation(self):
        """Test question validation"""
        # Empty question
        with pytest.raises(ValidationError) as exc_info:
            QARequest(question="")
        assert "Question cannot be empty" in str(exc_info.value)
        
        # Question too long
        with pytest.raises(ValidationError) as exc_info:
            QARequest(question="a" * 1001)
        assert "Question too long" in str(exc_info.value)
        
        # Whitespace handling
        request = QARequest(question="  What is this?  ")
        assert request.question == "What is this?"
    
    def test_max_chunks_validation(self):
        """Test max_chunks validation"""
        # Too few chunks
        with pytest.raises(ValidationError) as exc_info:
            QARequest(question="test", max_chunks=0)
        assert "max_chunks must be between 1 and 20" in str(exc_info.value)
        
        # Too many chunks
        with pytest.raises(ValidationError) as exc_info:
            QARequest(question="test", max_chunks=21)
        assert "max_chunks must be between 1 and 20" in str(exc_info.value)
        
        # Valid boundary values
        request_min = QARequest(question="test", max_chunks=1)
        assert request_min.max_chunks == 1
        
        request_max = QARequest(question="test", max_chunks=20)
        assert request_max.max_chunks == 20
    
    def test_session_id_validation(self):
        """Test session ID validation"""
        # Empty session ID (should fail)
        with pytest.raises(ValidationError) as exc_info:
            QARequest(question="test", session_id="")
        assert "Session ID cannot be empty if provided" in str(exc_info.value)
        
        # None session ID (should pass)
        request = QARequest(question="test", session_id=None)
        assert request.session_id is None


class TestQAResponse:
    """Test cases for QAResponse model"""
    
    def test_valid_qa_response_creation(self):
        """Test creating a valid QA response"""
        source_chunk = SourceChunk(
            document_id="doc-123",
            document_name="test.pdf",
            content="relevant content",
            relevance_score=0.8,
            chunk_index=0
        )
        
        response = QAResponse(
            answer="The system uses microservices architecture.",
            confidence_score=0.9,
            source_chunks=[source_chunk],
            session_id="session-123"
        )
        
        assert response.answer == "The system uses microservices architecture."
        assert response.confidence_score == 0.9
        assert len(response.source_chunks) == 1
        assert response.session_id == "session-123"
        assert isinstance(response.timestamp, datetime)
        assert response.processing_time_ms is None
    
    def test_confidence_score_validation(self):
        """Test confidence score validation"""
        # Score too low
        with pytest.raises(ValidationError) as exc_info:
            QAResponse(
                answer="test answer",
                confidence_score=-0.1,
                session_id="session-123"
            )
        assert "Confidence score must be between 0.0 and 1.0" in str(exc_info.value)
        
        # Score too high
        with pytest.raises(ValidationError) as exc_info:
            QAResponse(
                answer="test answer",
                confidence_score=1.1,
                session_id="session-123"
            )
        assert "Confidence score must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_answer_validation(self):
        """Test answer validation"""
        # Empty answer
        with pytest.raises(ValidationError) as exc_info:
            QAResponse(
                answer="",
                confidence_score=0.8,
                session_id="session-123"
            )
        assert "Answer cannot be empty" in str(exc_info.value)
        
        # Whitespace handling
        response = QAResponse(
            answer="  test answer  ",
            confidence_score=0.8,
            session_id="session-123"
        )
        assert response.answer == "test answer"


class TestQAExchange:
    """Test cases for QAExchange model"""
    
    def test_valid_qa_exchange_creation(self):
        """Test creating a valid QA exchange"""
        source_chunk = SourceChunk(
            document_id="doc-123",
            document_name="test.pdf",
            content="relevant content",
            relevance_score=0.8,
            chunk_index=0
        )
        
        exchange = QAExchange(
            question="What is the architecture?",
            answer="Microservices architecture",
            confidence_score=0.9,
            source_chunks=[source_chunk]
        )
        
        assert exchange.question == "What is the architecture?"
        assert exchange.answer == "Microservices architecture"
        assert exchange.confidence_score == 0.9
        assert len(exchange.source_chunks) == 1
        assert isinstance(exchange.timestamp, datetime)
    
    def test_confidence_score_validation_in_exchange(self):
        """Test confidence score validation in exchange"""
        with pytest.raises(ValidationError) as exc_info:
            QAExchange(
                question="test",
                answer="test",
                confidence_score=1.5
            )
        assert "Confidence score must be between 0.0 and 1.0" in str(exc_info.value)


class TestConversationHistory:
    """Test cases for ConversationHistory model"""
    
    def test_valid_conversation_creation(self):
        """Test creating a valid conversation history"""
        conversation = ConversationHistory(session_id="session-123")
        
        assert conversation.session_id == "session-123"
        assert conversation.exchanges == []
        assert isinstance(conversation.created_at, datetime)
        assert isinstance(conversation.last_updated, datetime)
    
    def test_conversation_with_exchanges(self):
        """Test conversation with exchanges"""
        exchange = QAExchange(
            question="What is this?",
            answer="This is a test",
            confidence_score=0.8
        )
        
        conversation = ConversationHistory(
            session_id="session-123",
            exchanges=[exchange]
        )
        
        assert len(conversation.exchanges) == 1
        assert conversation.exchanges[0] == exchange
    
    def test_session_id_validation_in_conversation(self):
        """Test session ID validation in conversation"""
        # Empty session ID
        with pytest.raises(ValidationError) as exc_info:
            ConversationHistory(session_id="")
        assert "Session ID cannot be empty" in str(exc_info.value)


class TestConversationRequest:
    """Test cases for ConversationRequest model"""
    
    def test_valid_conversation_request(self):
        """Test creating a valid conversation request"""
        request = ConversationRequest()
        
        assert request.initial_question is None
        assert request.document_ids is None
    
    def test_conversation_request_with_fields(self):
        """Test conversation request with optional fields"""
        request = ConversationRequest(
            initial_question="What is this system?",
            document_ids=["doc-1", "doc-2"]
        )
        
        assert request.initial_question == "What is this system?"
        assert request.document_ids == ["doc-1", "doc-2"]
    
    def test_initial_question_validation(self):
        """Test initial question validation"""
        # Empty initial question (should fail)
        with pytest.raises(ValidationError) as exc_info:
            ConversationRequest(initial_question="")
        assert "Initial question cannot be empty if provided" in str(exc_info.value)
        
        # None initial question (should pass)
        request = ConversationRequest(initial_question=None)
        assert request.initial_question is None


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_generate_session_id(self):
        """Test session ID generation"""
        session_id = generate_session_id()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Should generate unique IDs
        session_id2 = generate_session_id()
        assert session_id != session_id2
        
        # Should be valid UUID format
        import uuid
        try:
            uuid.UUID(session_id)
            uuid.UUID(session_id2)
        except ValueError:
            pytest.fail("Generated session IDs are not valid UUIDs")