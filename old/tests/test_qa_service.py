"""
Unit tests for QA service
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Dict, Any

from app.services.qa_service import QAService
from app.models.qa import (
    QARequest, 
    QAResponse, 
    SourceChunk, 
    ConversationHistory,
    generate_session_id
)


class TestQAService:
    """Test cases for QA service"""
    
    @pytest.fixture
    def qa_service(self):
        """Create QA service instance for testing"""
        return QAService(use_langgraph=False)  # Use legacy mode for existing tests
    
    @pytest.fixture
    def sample_qa_request(self):
        """Sample QA request for testing"""
        return QARequest(
            question="システムのアーキテクチャについて教えてください",
            document_ids=["doc1", "doc2"],
            max_chunks=5
        )
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results from vector database"""
        return [
            {
                "chunk_id": "doc1_chunk_0",
                "content": "システムはマイクロサービスアーキテクチャを採用しています。",
                "metadata": {"document_id": "doc1", "chunk_index": 0},
                "similarity_score": 0.85,
                "document_id": "doc1",
                "chunk_index": 0
            },
            {
                "chunk_id": "doc1_chunk_1", 
                "content": "各サービスは独立してデプロイ可能です。",
                "metadata": {"document_id": "doc1", "chunk_index": 1},
                "similarity_score": 0.72,
                "document_id": "doc1",
                "chunk_index": 1
            }
        ]
    
    @pytest.fixture
    def mock_document(self):
        """Mock document object"""
        mock_doc = Mock()
        mock_doc.filename = "test_document.pdf"
        return mock_doc
    
    @pytest.mark.asyncio
    async def test_ask_question_with_llm_success(self, qa_service, sample_qa_request, sample_search_results, mock_document):
        """Test successful question processing with LLM"""
        # Mock dependencies
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db, \
             patch('app.services.qa_service.document_service') as mock_doc_service, \
             patch.object(qa_service, '_llm') as mock_llm:
            
            # Setup mocks with AsyncMock for async methods
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=sample_search_results)
            mock_doc_service.get_document = AsyncMock(return_value=mock_document)
            
            mock_response = Mock()
            mock_response.content = "システムはマイクロサービスアーキテクチャを採用しており、各サービスは独立してデプロイできます。"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            
            # Execute
            response = await qa_service.ask_question(sample_qa_request)
            
            # Verify
            assert isinstance(response, QAResponse)
            assert response.answer == "システムはマイクロサービスアーキテクチャを採用しており、各サービスは独立してデプロイできます。"
            assert 0.0 <= response.confidence_score <= 1.0
            assert len(response.source_chunks) == 2
            assert response.session_id is not None
            assert response.processing_time_ms is not None
            
            # Verify source chunks
            source_chunk = response.source_chunks[0]
            assert source_chunk.document_name == "test_document.pdf"
            assert source_chunk.relevance_score == 0.85
            assert source_chunk.chunk_index == 0
    
    @pytest.mark.asyncio
    async def test_ask_question_no_relevant_chunks(self, qa_service, sample_qa_request):
        """Test question processing when no relevant chunks are found"""
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db:
            # Setup mock to return empty results
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=[])
            
            # Execute
            response = await qa_service.ask_question(sample_qa_request)
            
            # Verify
            assert isinstance(response, QAResponse)
            assert "関連する情報を見つけることができませんでした" in response.answer
            assert response.confidence_score == 0.0
            assert len(response.source_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_ask_question_without_llm_fallback(self, qa_service, sample_qa_request, sample_search_results, mock_document):
        """Test question processing fallback when LLM is not available"""
        # Disable LLM
        qa_service._llm = None
        
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db, \
             patch('app.services.qa_service.document_service') as mock_doc_service:
            
            # Setup mocks
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=sample_search_results)
            mock_doc_service.get_document = AsyncMock(return_value=mock_document)
            
            # Execute
            response = await qa_service.ask_question(sample_qa_request)
            
            # Verify
            assert isinstance(response, QAResponse)
            assert "システムはマイクロサービスアーキテクチャを採用しています" in response.answer
            assert "自動検索結果です" in response.answer
            assert response.confidence_score == 0.85  # Should match best chunk score
            assert len(response.source_chunks) == 2
    
    @pytest.mark.asyncio
    async def test_search_relevant_chunks(self, qa_service, sample_search_results):
        """Test semantic search functionality"""
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db:
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=sample_search_results)
            
            # Execute
            results = await qa_service._search_relevant_chunks(
                query="アーキテクチャ",
                document_ids=["doc1"],
                max_chunks=5
            )
            
            # Verify
            assert len(results) == 2
            assert results[0]["similarity_score"] == 0.85
            assert "マイクロサービス" in results[0]["content"]
            
            # Verify vector DB service was called correctly
            mock_vector_db.search_similar_chunks.assert_called_once_with(
                query="アーキテクチャ",
                document_ids=["doc1"],
                limit=5,
                min_similarity=0.3
            )
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_llm(self, qa_service, sample_search_results):
        """Test answer generation with LLM"""
        with patch.object(qa_service, '_llm') as mock_llm, \
             patch.object(qa_service, '_get_conversation_context', return_value=""):
            
            mock_response = Mock()
            mock_response.content = "テストレスポンス"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            
            # Execute
            answer, confidence = await qa_service._generate_answer(
                question="テスト質問",
                relevant_chunks=sample_search_results,
                session_id="test_session"
            )
            
            # Verify
            assert answer == "テストレスポンス"
            assert 0.0 <= confidence <= 1.0
            mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_fallback_answer(self, qa_service, sample_search_results):
        """Test fallback answer generation"""
        # Execute
        answer, confidence = await qa_service._generate_fallback_answer(
            question="テスト質問",
            relevant_chunks=sample_search_results
        )
        
        # Verify
        assert "システムはマイクロサービスアーキテクチャを採用しています" in answer
        assert "自動検索結果です" in answer
        assert confidence == 0.85  # Should match best chunk score
    
    @pytest.mark.asyncio
    async def test_generate_fallback_answer_no_chunks(self, qa_service):
        """Test fallback answer when no chunks available"""
        # Execute
        answer, confidence = await qa_service._generate_fallback_answer(
            question="テスト質問",
            relevant_chunks=[]
        )
        
        # Verify
        assert answer == "関連する情報が見つかりませんでした。"
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_convert_to_source_chunks(self, qa_service, sample_search_results, mock_document):
        """Test conversion of search results to SourceChunk objects"""
        with patch('app.services.qa_service.document_service') as mock_doc_service:
            mock_doc_service.get_document = AsyncMock(return_value=mock_document)
            
            # Execute
            source_chunks = await qa_service._convert_to_source_chunks(sample_search_results)
            
            # Verify
            assert len(source_chunks) == 2
            
            chunk = source_chunks[0]
            assert isinstance(chunk, SourceChunk)
            assert chunk.document_id == "doc1"
            assert chunk.document_name == "test_document.pdf"
            assert chunk.relevance_score == 0.85
            assert chunk.chunk_index == 0
            assert "マイクロサービス" in chunk.content
    
    @pytest.mark.asyncio
    async def test_conversation_management(self, qa_service):
        """Test conversation creation and management"""
        # Create conversation
        session_id = await qa_service.create_conversation()
        assert session_id is not None
        
        # Verify conversation exists
        conversation = await qa_service.get_conversation_history(session_id)
        assert conversation is not None
        assert conversation.session_id == session_id
        assert len(conversation.exchanges) == 0
        
        # Delete conversation
        deleted = await qa_service.delete_conversation(session_id)
        assert deleted is True
        
        # Verify conversation is deleted
        conversation = await qa_service.get_conversation_history(session_id)
        assert conversation is None
    
    @pytest.mark.asyncio
    async def test_conversation_with_initial_question(self, qa_service, sample_search_results, mock_document):
        """Test conversation creation with initial question"""
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db, \
             patch('app.services.qa_service.document_service') as mock_doc_service:
            
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=sample_search_results)
            mock_doc_service.get_document = AsyncMock(return_value=mock_document)
            
            # Create conversation with initial question
            session_id = await qa_service.create_conversation(
                initial_question="初期質問"
            )
            
            # Verify conversation has the exchange
            conversation = await qa_service.get_conversation_history(session_id)
            assert len(conversation.exchanges) == 1
            assert conversation.exchanges[0].question == "初期質問"
    
    @pytest.mark.asyncio
    async def test_update_conversation_history(self, qa_service):
        """Test conversation history updates"""
        session_id = "test_session"
        
        # Create mock response
        response = QAResponse(
            answer="テスト回答",
            confidence_score=0.8,
            source_chunks=[],
            session_id=session_id
        )
        
        # Update conversation history
        await qa_service._update_conversation_history(
            session_id=session_id,
            question="テスト質問",
            response=response
        )
        
        # Verify conversation was created and updated
        conversation = await qa_service.get_conversation_history(session_id)
        assert conversation is not None
        assert len(conversation.exchanges) == 1
        
        exchange = conversation.exchanges[0]
        assert exchange.question == "テスト質問"
        assert exchange.answer == "テスト回答"
        assert exchange.confidence_score == 0.8
    
    @pytest.mark.asyncio
    async def test_conversation_history_limit(self, qa_service):
        """Test conversation history is limited to prevent memory issues"""
        session_id = "test_session"
        
        # Add many exchanges
        for i in range(15):
            response = QAResponse(
                answer=f"回答{i}",
                confidence_score=0.8,
                source_chunks=[],
                session_id=session_id
            )
            
            await qa_service._update_conversation_history(
                session_id=session_id,
                question=f"質問{i}",
                response=response
            )
        
        # Verify only last 10 exchanges are kept
        conversation = await qa_service.get_conversation_history(session_id)
        assert len(conversation.exchanges) == 10
        assert conversation.exchanges[0].question == "質問5"  # Should start from 5th question
        assert conversation.exchanges[-1].question == "質問14"  # Should end with 14th question
    
    @pytest.mark.asyncio
    async def test_get_conversation_context(self, qa_service):
        """Test conversation context retrieval"""
        session_id = "test_session"
        
        # Add some exchanges
        for i in range(5):
            response = QAResponse(
                answer=f"回答{i}",
                confidence_score=0.8,
                source_chunks=[],
                session_id=session_id
            )
            
            await qa_service._update_conversation_history(
                session_id=session_id,
                question=f"質問{i}",
                response=response
            )
        
        # Get conversation context
        context = await qa_service._get_conversation_context(session_id)
        
        # Verify context contains last 3 exchanges
        assert "質問2" in context
        assert "質問3" in context
        assert "質問4" in context
        assert "回答2" in context
        assert "回答3" in context
        assert "回答4" in context
        
        # Should not contain earlier exchanges
        assert "質問0" not in context
        assert "質問1" not in context
    
    @pytest.mark.asyncio
    async def test_health_check(self, qa_service):
        """Test QA service health check"""
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db:
            mock_vector_db.health_check = AsyncMock(return_value={"status": "healthy"})
            
            # Execute
            health = await qa_service.health_check()
            
            # Verify
            assert health["status"] == "healthy"
            assert "llm_available" in health
            assert "active_conversations" in health
            assert "vector_db_status" in health
    
    @pytest.mark.asyncio
    async def test_error_handling_in_ask_question(self, qa_service, sample_qa_request):
        """Test error handling in ask_question method"""
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db:
            # Make vector DB service raise an exception
            mock_vector_db.search_similar_chunks = AsyncMock(side_effect=Exception("Database error"))
            
            # Execute
            response = await qa_service.ask_question(sample_qa_request)
            
            # Verify error response - when search fails, it returns empty results which triggers "no relevant chunks" message
            assert isinstance(response, QAResponse)
            assert "関連する情報を見つけることができませんでした" in response.answer
            assert response.confidence_score == 0.0
            assert len(response.source_chunks) == 0
    
    def test_generate_session_id(self):
        """Test session ID generation"""
        session_id = generate_session_id()
        assert session_id is not None
        assert len(session_id) > 0
        
        # Generate another and verify they're different
        session_id2 = generate_session_id()
        assert session_id != session_id2


class TestQAServiceIntegration:
    """Integration tests for QA service"""
    
    @pytest.mark.asyncio
    async def test_full_qa_pipeline_without_llm(self):
        """Test complete QA pipeline without LLM dependency"""
        qa_service = QAService(use_langgraph=False)  # Use legacy mode for this test
        qa_service._llm = None  # Disable LLM for testing
        
        # Mock the vector database and document service
        with patch('app.services.qa_service.vector_db_service') as mock_vector_db, \
             patch('app.services.qa_service.document_service') as mock_doc_service:
            
            # Setup mock data
            mock_search_results = [
                {
                    "chunk_id": "doc1_chunk_0",
                    "content": "FastAPIは高性能なWebフレームワークです。",
                    "metadata": {"document_id": "doc1", "chunk_index": 0},
                    "similarity_score": 0.9,
                    "document_id": "doc1",
                    "chunk_index": 0
                }
            ]
            
            mock_document = Mock()
            mock_document.filename = "api_guide.pdf"
            
            mock_vector_db.search_similar_chunks = AsyncMock(return_value=mock_search_results)
            mock_doc_service.get_document = AsyncMock(return_value=mock_document)
            
            # Create request
            request = QARequest(
                question="FastAPIについて教えてください",
                max_chunks=3
            )
            
            # Execute
            response = await qa_service.ask_question(request)
            
            # Verify complete response
            assert isinstance(response, QAResponse)
            assert "FastAPI" in response.answer
            assert response.confidence_score > 0
            assert len(response.source_chunks) == 1
            assert response.source_chunks[0].document_name == "api_guide.pdf"
            assert response.session_id is not None
            assert response.processing_time_ms is not None
            
            # Verify conversation was stored
            conversation = await qa_service.get_conversation_history(response.session_id)
            assert conversation is not None
            assert len(conversation.exchanges) == 1