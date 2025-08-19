"""
Tests for LangGraph workflow implementation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List

from app.services.langgraph_workflow import LangGraphQAWorkflow, WorkflowState
from app.models.qa import QARequest, QAResponse, SourceChunk
from app.core.config import settings


class TestLangGraphQAWorkflow:
    """Test cases for LangGraph QA workflow"""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance for testing"""
        return LangGraphQAWorkflow()
    
    @pytest.fixture
    def sample_request(self):
        """Sample QA request for testing"""
        return QARequest(
            question="システムのアーキテクチャについて教えてください",
            document_ids=["doc1", "doc2"],
            session_id="test-session",
            max_chunks=5
        )
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            {
                "document_id": "doc1",
                "content": "システムはマイクロサービスアーキテクチャを採用しています。",
                "similarity_score": 0.85,
                "chunk_index": 0
            },
            {
                "document_id": "doc2", 
                "content": "APIゲートウェイを通じて各サービスにアクセスします。",
                "similarity_score": 0.72,
                "chunk_index": 1
            }
        ]
    
    @pytest.fixture
    def sample_state(self, sample_search_results):
        """Sample workflow state for testing"""
        return WorkflowState(
            question="システムのアーキテクチャについて教えてください",
            document_ids=["doc1", "doc2"],
            session_id="test-session",
            max_chunks=5,
            conversation_history=None,
            analyzed_query=None,
            search_results=sample_search_results,
            context=None,
            answer=None,
            confidence_score=0.0,
            source_chunks=[],
            error_message=None,
            processing_steps=[],
            start_time=datetime.utcnow().timestamp(),
            messages=[]
        )

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization"""
        assert workflow is not None
        assert workflow._workflow is not None
        # LLM may or may not be initialized depending on API key availability
    
    @pytest.mark.asyncio
    async def test_process_question_success(self, workflow, sample_request):
        """Test successful question processing"""
        # Mock the workflow execution
        mock_final_state = {
            "answer": "システムはマイクロサービスアーキテクチャを採用しています。",
            "confidence_score": 0.85,
            "source_chunks": [
                SourceChunk(
                    document_id="doc1",
                    document_name="architecture.md",
                    content="システムはマイクロサービスアーキテクチャを採用しています。",
                    relevance_score=0.85,
                    chunk_index=0
                )
            ],
            "session_id": "test-session",
            "error_message": None
        }
        
        with patch.object(workflow._workflow, 'ainvoke', return_value=mock_final_state):
            response = await workflow.process_question(sample_request)
            
            assert isinstance(response, QAResponse)
            assert response.answer == "システムはマイクロサービスアーキテクチャを採用しています。"
            assert response.confidence_score == 0.85
            assert len(response.source_chunks) == 1
            assert response.session_id == "test-session"
            assert response.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_process_question_workflow_error(self, workflow, sample_request):
        """Test question processing when workflow fails"""
        with patch.object(workflow._workflow, 'ainvoke', side_effect=Exception("Workflow error")):
            response = await workflow.process_question(sample_request)
            
            assert isinstance(response, QAResponse)
            assert "処理中にエラーが発生しました" in response.answer
            assert response.confidence_score == 0.0
            assert len(response.source_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_process_question_no_workflow(self, sample_request):
        """Test question processing when workflow is not initialized"""
        workflow = LangGraphQAWorkflow()
        workflow._workflow = None
        
        response = await workflow.process_question(sample_request)
        
        assert isinstance(response, QAResponse)
        assert "Workflow not available" in response.answer
        assert response.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_query_node(self, workflow, sample_state):
        """Test query analysis node"""
        result_state = await workflow._analyze_query_node(sample_state)
        
        assert "query_analysis" in result_state["processing_steps"]
        assert result_state["analyzed_query"] is not None
        assert "original_question" in result_state["analyzed_query"]
        assert "question_type" in result_state["analyzed_query"]
        assert "key_terms" in result_state["analyzed_query"]
    
    @pytest.mark.asyncio
    async def test_search_documents_node(self, workflow, sample_state, sample_search_results):
        """Test document search node"""
        with patch('app.services.langgraph_workflow.vector_db_service.search_similar_chunks', 
                  return_value=sample_search_results):
            result_state = await workflow._search_documents_node(sample_state)
            
            assert "document_search" in result_state["processing_steps"]
            assert len(result_state["search_results"]) == 2
            assert result_state["search_results"][0]["similarity_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_search_documents_node_error(self, workflow, sample_state):
        """Test document search node with error"""
        with patch('app.services.langgraph_workflow.vector_db_service.search_similar_chunks', 
                  side_effect=Exception("Search error")):
            result_state = await workflow._search_documents_node(sample_state)
            
            assert result_state["error_message"] is not None
            assert "文書検索エラー" in result_state["error_message"]
    
    @pytest.mark.asyncio
    async def test_assemble_context_node(self, workflow, sample_state):
        """Test context assembly node"""
        # Set up state with search results
        sample_state["conversation_history"] = "前回の質問: APIについて"
        
        result_state = await workflow._assemble_context_node(sample_state)
        
        assert "context_assembly" in result_state["processing_steps"]
        assert result_state["context"] is not None
        assert "会話履歴" in result_state["context"]
        assert "関連文書" in result_state["context"]
    
    @pytest.mark.asyncio
    async def test_generate_answer_node_no_results(self, workflow, sample_state):
        """Test answer generation with no search results"""
        sample_state["search_results"] = []
        
        result_state = await workflow._generate_answer_node(sample_state)
        
        assert "answer_generation" in result_state["processing_steps"]
        assert "関連する情報を見つけることができませんでした" in result_state["answer"]
        assert result_state["confidence_score"] == 0.0
        assert len(result_state["source_chunks"]) == 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_node_with_results(self, workflow, sample_state):
        """Test answer generation with search results"""
        # Mock document service
        mock_document = MagicMock()
        mock_document.filename = "test.md"
        
        with patch('app.services.langgraph_workflow.document_service.get_document', 
                  return_value=mock_document):
            result_state = await workflow._generate_answer_node(sample_state)
            
            assert "answer_generation" in result_state["processing_steps"]
            assert result_state["answer"] is not None
            assert result_state["confidence_score"] > 0
            assert len(result_state["source_chunks"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_node_with_llm(self, workflow, sample_state):
        """Test answer generation with LLM"""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "これはLLMからの回答です。"
        mock_llm.ainvoke.return_value = mock_response
        workflow._llm = mock_llm
        
        sample_state["context"] = "テストコンテキスト"
        
        # Mock document service
        mock_document = MagicMock()
        mock_document.filename = "test.md"
        
        with patch('app.services.langgraph_workflow.document_service.get_document', 
                  return_value=mock_document):
            result_state = await workflow._generate_answer_node(sample_state)
            
            assert result_state["answer"] == "これはLLMからの回答です。"
            assert result_state["confidence_score"] > 0
    
    @pytest.mark.asyncio
    async def test_validate_response_node(self, workflow, sample_state):
        """Test response validation node"""
        sample_state["answer"] = "これは十分な長さの回答です。システムについて詳しく説明しています。"
        sample_state["confidence_score"] = 0.8
        sample_state["source_chunks"] = [
            SourceChunk(
                document_id="doc1",
                document_name="test.md",
                content="テスト内容",
                relevance_score=0.9,
                chunk_index=0
            )
        ]
        
        result_state = await workflow._validate_response_node(sample_state)
        
        assert "response_validation" in result_state["processing_steps"]
        assert result_state["confidence_score"] <= 1.0
        assert result_state["confidence_score"] > 0
    
    @pytest.mark.asyncio
    async def test_handle_error_node(self, workflow, sample_state):
        """Test error handling node"""
        sample_state["error_message"] = "テストエラー"
        
        result_state = await workflow._handle_error_node(sample_state)
        
        assert "error_handling" in result_state["processing_steps"]
        assert "処理中にエラーが発生しました" in result_state["answer"]
        assert result_state["confidence_score"] == 0.0
        assert len(result_state["source_chunks"]) == 0
    
    def test_classify_question_type(self, workflow):
        """Test question type classification"""
        assert workflow._classify_question_type("何ですか？") == "what"
        assert workflow._classify_question_type("どうやって実装しますか？") == "how"
        assert workflow._classify_question_type("なぜこの設計にしたのですか？") == "why"
        assert workflow._classify_question_type("いつリリースしますか？") == "when"
        assert workflow._classify_question_type("どこに配置しますか？") == "where"
        assert workflow._classify_question_type("システムについて教えて") == "general"
    
    def test_extract_key_terms(self, workflow):
        """Test key term extraction"""
        question = "システムのアーキテクチャについて教えてください"
        terms = workflow._extract_key_terms(question)
        
        assert isinstance(terms, list)
        assert len(terms) > 0
        # Check if any meaningful terms are extracted (more flexible check)
        meaningful_found = any(
            term for term in terms 
            if len(term) > 2 and term not in ["教えて", "ください", "について"]
        )
        assert meaningful_found, f"No meaningful terms found in: {terms}"
    
    def test_should_continue_after_analysis(self, workflow, sample_state):
        """Test conditional edge after analysis"""
        # Normal case
        assert workflow._should_continue_after_analysis(sample_state) == "search"
        
        # Error case
        sample_state["error_message"] = "エラー"
        assert workflow._should_continue_after_analysis(sample_state) == "error"
    
    def test_should_continue_after_search(self, workflow, sample_state):
        """Test conditional edge after search"""
        # Normal case with results
        assert workflow._should_continue_after_search(sample_state) == "assemble"
        
        # No results case
        sample_state["search_results"] = []
        assert workflow._should_continue_after_search(sample_state) == "no_results"
        
        # Error case
        sample_state["error_message"] = "エラー"
        assert workflow._should_continue_after_search(sample_state) == "error"
    
    def test_should_validate_response(self, workflow, sample_state):
        """Test conditional edge for response validation"""
        # High confidence case
        sample_state["answer"] = "良い回答"
        sample_state["confidence_score"] = 0.8
        assert workflow._should_validate_response(sample_state) == "validate"
        
        # Low confidence case
        sample_state["confidence_score"] = 0.3
        assert workflow._should_validate_response(sample_state) == "end"
        
        # Error case
        sample_state["error_message"] = "エラー"
        assert workflow._should_validate_response(sample_state) == "error"
    
    def test_fallback_generate_answer_no_results(self, workflow):
        """Test fallback answer generation with no results"""
        answer, confidence = workflow._fallback_generate_answer([])
        
        assert answer == "関連する情報が見つかりませんでした。"
        assert confidence == 0.0
    
    def test_fallback_generate_answer_with_results(self, workflow, sample_search_results):
        """Test fallback answer generation with results"""
        answer, confidence = workflow._fallback_generate_answer(sample_search_results)
        
        assert "以下の文書から関連する情報を見つけました" in answer
        assert "システムはマイクロサービスアーキテクチャ" in answer
        assert confidence == 0.85  # Best chunk's similarity score
    
    @pytest.mark.asyncio
    async def test_convert_to_source_chunks(self, workflow, sample_search_results):
        """Test conversion of search results to source chunks"""
        # Mock document service
        mock_document = MagicMock()
        mock_document.filename = "test.md"
        
        with patch('app.services.langgraph_workflow.document_service.get_document', 
                  return_value=mock_document):
            source_chunks = await workflow._convert_to_source_chunks(sample_search_results)
            
            assert len(source_chunks) == 2
            assert isinstance(source_chunks[0], SourceChunk)
            assert source_chunks[0].document_id == "doc1"
            assert source_chunks[0].document_name == "test.md"
            assert source_chunks[0].relevance_score == 0.85
    
    @pytest.mark.asyncio
    async def test_convert_to_source_chunks_error(self, workflow, sample_search_results):
        """Test conversion with document service error"""
        with patch('app.services.langgraph_workflow.document_service.get_document', 
                  side_effect=Exception("Document not found")):
            source_chunks = await workflow._convert_to_source_chunks(sample_search_results)
            
            # Should handle errors gracefully and return empty list
            assert len(source_chunks) == 0
    
    def test_create_error_response(self, workflow):
        """Test error response creation"""
        response = workflow._create_error_response(
            session_id="test-session",
            error_message="テストエラー",
            processing_time=100
        )
        
        assert isinstance(response, QAResponse)
        assert "テストエラー" in response.answer
        assert response.confidence_score == 0.0
        assert len(response.source_chunks) == 0
        assert response.session_id == "test-session"
        assert response.processing_time_ms == 100
    
    @pytest.mark.asyncio
    async def test_llm_analyze_query_success(self, workflow):
        """Test LLM query analysis success"""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "分析結果"
        mock_llm.ainvoke.return_value = mock_response
        workflow._llm = mock_llm
        
        result = await workflow._llm_analyze_query("テスト質問")
        
        assert isinstance(result, dict)
        assert "enhanced_query" in result
        assert "intent" in result
        assert "llm_analysis" in result
    
    @pytest.mark.asyncio
    async def test_llm_analyze_query_error(self, workflow):
        """Test LLM query analysis error"""
        # Mock LLM with error
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        workflow._llm = mock_llm
        
        result = await workflow._llm_analyze_query("テスト質問")
        
        assert isinstance(result, dict)
        assert len(result) == 0  # Empty dict on error
    
    @pytest.mark.asyncio
    async def test_llm_generate_answer_success(self, workflow):
        """Test LLM answer generation success"""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "LLMからの回答"
        mock_llm.ainvoke.return_value = mock_response
        workflow._llm = mock_llm
        
        answer, confidence = await workflow._llm_generate_answer(
            "質問", "コンテキスト"
        )
        
        assert answer == "LLMからの回答"
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_llm_generate_answer_error(self, workflow):
        """Test LLM answer generation error"""
        # Mock LLM with error
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        workflow._llm = mock_llm
        
        answer, confidence = await workflow._llm_generate_answer(
            "質問", "コンテキスト"
        )
        
        assert "LLMによる回答生成に失敗しました" in answer
        assert confidence == 0.3


class TestWorkflowIntegration:
    """Integration tests for the complete workflow"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete workflow integration"""
        workflow = LangGraphQAWorkflow()
        
        # Mock all external dependencies
        mock_search_results = [
            {
                "document_id": "doc1",
                "content": "統合テスト用のコンテンツです。",
                "similarity_score": 0.8,
                "chunk_index": 0
            }
        ]
        
        mock_document = MagicMock()
        mock_document.filename = "integration_test.md"
        
        with patch('app.services.langgraph_workflow.vector_db_service.search_similar_chunks', 
                  return_value=mock_search_results), \
             patch('app.services.langgraph_workflow.document_service.get_document', 
                  return_value=mock_document):
            
            request = QARequest(
                question="統合テストについて教えてください",
                session_id="integration-test"
            )
            
            response = await workflow.process_question(request)
            
            assert isinstance(response, QAResponse)
            assert response.session_id == "integration-test"
            assert response.processing_time_ms is not None
            # Answer should be generated (either by LLM or fallback)
            assert len(response.answer) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_conversation_history(self):
        """Test workflow with conversation history"""
        workflow = LangGraphQAWorkflow()
        
        mock_search_results = [
            {
                "document_id": "doc1",
                "content": "会話履歴テスト用のコンテンツです。",
                "similarity_score": 0.7,
                "chunk_index": 0
            }
        ]
        
        mock_document = MagicMock()
        mock_document.filename = "conversation_test.md"
        
        with patch('app.services.langgraph_workflow.vector_db_service.search_similar_chunks', 
                  return_value=mock_search_results), \
             patch('app.services.langgraph_workflow.document_service.get_document', 
                  return_value=mock_document):
            
            request = QARequest(
                question="前回の話の続きを教えてください",
                session_id="conversation-test"
            )
            
            conversation_history = "前回の質問: システムについて\n前回の回答: システムの概要を説明しました。"
            
            response = await workflow.process_question(
                request, 
                conversation_history=conversation_history
            )
            
            assert isinstance(response, QAResponse)
            assert len(response.answer) > 0