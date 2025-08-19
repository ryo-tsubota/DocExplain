"""
Question-Answer processing service
"""
import logging
import time
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from app.core.config import settings
from app.models.qa import (
    QARequest, 
    QAResponse, 
    QAExchange, 
    ConversationHistory, 
    SourceChunk,
    generate_session_id
)
from app.services.vector_db_service import vector_db_service
from app.services.document_service import document_service
from app.services.langgraph_workflow import langgraph_workflow

logger = logging.getLogger(__name__)


class QAService:
    """Service for handling question-answer processing"""
    
    def __init__(self, use_langgraph: bool = True):
        self._llm: Optional[ChatOpenAI] = None
        self._conversations: Dict[str, ConversationHistory] = {}
        self._use_langgraph = use_langgraph
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the LLM client"""
        try:
            if settings.openai_api_key:
                self._llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=settings.llm_model,
                    temperature=0.1,  # Low temperature for consistent answers
                    max_tokens=1000
                )
                logger.info(f"Initialized LLM: {settings.llm_model}")
            else:
                logger.warning("OpenAI API key not provided, LLM functionality will be limited")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._llm = None
    
    async def ask_question(self, request: QARequest) -> QAResponse:
        """
        Process a question and generate an answer using LangGraph workflow or legacy method
        
        Args:
            request: QA request containing question and optional parameters
            
        Returns:
            QAResponse with generated answer and source information
        """
        # Use LangGraph workflow if enabled, otherwise use legacy method
        if self._use_langgraph:
            try:
                # Generate session ID if not provided
                session_id = request.session_id or generate_session_id()
                
                # Get conversation context for the session
                conversation_context = await self._get_conversation_context(session_id)
                
                # Process question through LangGraph workflow
                response = await langgraph_workflow.process_question(
                    request=request,
                    conversation_history=conversation_context
                )
                
                # Ensure session ID is set
                if not response.session_id or response.session_id == "unknown":
                    response.session_id = session_id
                
                # Store in conversation history
                await self._update_conversation_history(
                    session_id=response.session_id,
                    question=request.question,
                    response=response
                )
                
                logger.info(f"Processed question via LangGraph in {response.processing_time_ms}ms with {len(response.source_chunks)} sources")
                return response
                
            except Exception as e:
                logger.error(f"Failed to process question via LangGraph: {e}")
                # Fallback to legacy processing
                logger.info("Falling back to legacy QA processing")
                try:
                    return await self._legacy_ask_question(request)
                except Exception as fallback_error:
                    logger.error(f"Legacy fallback also failed: {fallback_error}")
                    # Return a basic error response
                    return QAResponse(
                        answer=f"申し訳ございませんが、質問の処理中にエラーが発生しました: {str(e)}",
                        confidence_score=0.0,
                        source_chunks=[],
                        session_id=request.session_id or generate_session_id(),
                        processing_time_ms=0
                    )
        else:
            # Use legacy method directly
            return await self._legacy_ask_question(request)
    
    async def _legacy_ask_question(self, request: QARequest) -> QAResponse:
        """
        Legacy question processing method (fallback when LangGraph fails)
        
        Args:
            request: QA request containing question and optional parameters
            
        Returns:
            QAResponse with generated answer and source information
        """
        start_time = time.time()
        
        try:
            # Generate session ID if not provided
            session_id = request.session_id or generate_session_id()
            
            # Perform semantic search to find relevant chunks
            relevant_chunks = await self._search_relevant_chunks(
                query=request.question,
                document_ids=request.document_ids,
                max_chunks=request.max_chunks
            )
            
            if not relevant_chunks:
                # No relevant content found
                answer = "申し訳ございませんが、アップロードされた文書からご質問に関連する情報を見つけることができませんでした。質問を言い換えるか、関連する文書がアップロードされているかご確認ください。"
                confidence_score = 0.0
                source_chunks = []
            else:
                # Generate answer using LLM
                answer, confidence_score = await self._generate_answer(
                    question=request.question,
                    relevant_chunks=relevant_chunks,
                    session_id=session_id
                )
                
                # Convert search results to SourceChunk objects
                source_chunks = await self._convert_to_source_chunks(relevant_chunks)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create response
            response = QAResponse(
                answer=answer,
                confidence_score=confidence_score,
                source_chunks=source_chunks,
                session_id=session_id,
                processing_time_ms=processing_time
            )
            
            # Store in conversation history
            await self._update_conversation_history(
                session_id=session_id,
                question=request.question,
                response=response
            )
            
            logger.info(f"Processed question (legacy) in {processing_time}ms with {len(source_chunks)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process question (legacy): {e}")
            # Return error response
            return QAResponse(
                answer=f"申し訳ございませんが、質問の処理中にエラーが発生しました: {str(e)}",
                confidence_score=0.0,
                source_chunks=[],
                session_id=request.session_id or generate_session_id(),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    async def _search_relevant_chunks(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using semantic similarity
        
        Args:
            query: Search query
            document_ids: Optional list of document IDs to filter by
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of relevant chunk data
        """
        try:
            # Use vector database service for semantic search
            search_results = await vector_db_service.search_similar_chunks(
                query=query,
                document_ids=document_ids,
                limit=max_chunks,
                min_similarity=0.3  # Minimum similarity threshold
            )
            
            logger.info(f"Found {len(search_results)} relevant chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]],
        session_id: str
    ) -> tuple[str, float]:
        """
        Generate an answer using LLM based on relevant chunks
        
        Args:
            question: User question
            relevant_chunks: List of relevant document chunks
            session_id: Conversation session ID
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not self._llm:
            # Fallback to simple text matching if LLM not available
            return await self._generate_fallback_answer(question, relevant_chunks)
        
        try:
            # Prepare context from relevant chunks
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(f"[文書 {i+1}]\n{chunk['content']}\n")
            
            context = "\n".join(context_parts)
            
            # Get conversation history for context
            conversation_context = await self._get_conversation_context(session_id)
            
            # Create prompt template
            system_prompt = """あなたは技術文書の内容について質問に答えるAIアシスタントです。
以下のルールに従って回答してください：

1. 提供された文書の内容のみに基づいて回答する
2. 文書に記載されていない情報については推測しない
3. 回答は正確で具体的にする
4. 日本語で回答する
5. 文書から引用する場合は、どの文書からの情報かを明示する

文書の内容:
{context}

{conversation_history}"""

            human_prompt = "質問: {question}"
            
            # Format conversation history
            history_text = ""
            if conversation_context:
                history_text = "\n過去の会話:\n" + conversation_context
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt.format(
                    context=context,
                    conversation_history=history_text
                )),
                HumanMessage(content=human_prompt.format(question=question))
            ]
            
            # Generate response
            response = await self._llm.ainvoke(messages)
            answer = response.content.strip()
            
            # Calculate confidence based on chunk relevance scores
            avg_relevance = sum(chunk.get('similarity_score', 0) for chunk in relevant_chunks) / len(relevant_chunks)
            confidence_score = min(avg_relevance * 1.2, 1.0)  # Scale up slightly but cap at 1.0
            
            return answer, confidence_score
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return await self._generate_fallback_answer(question, relevant_chunks)
    
    async def _generate_fallback_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]]
    ) -> tuple[str, float]:
        """
        Generate a fallback answer when LLM is not available
        
        Args:
            question: User question
            relevant_chunks: List of relevant document chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not relevant_chunks:
            return "関連する情報が見つかりませんでした。", 0.0
        
        # Simple fallback: return the most relevant chunk with explanation
        best_chunk = max(relevant_chunks, key=lambda x: x.get('similarity_score', 0))
        
        answer = f"""以下の文書から関連する情報を見つけました：

{best_chunk['content']}

注意: この回答は自動検索結果です。より詳細な分析が必要な場合は、LLM機能を有効にしてください。"""
        
        confidence_score = best_chunk.get('similarity_score', 0.5)
        
        return answer, confidence_score
    
    async def _convert_to_source_chunks(
        self, 
        search_results: List[Dict[str, Any]]
    ) -> List[SourceChunk]:
        """
        Convert search results to SourceChunk objects
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of SourceChunk objects
        """
        source_chunks = []
        
        for result in search_results:
            try:
                # Get document information
                document_id = result['document_id']
                document = await document_service.get_document(document_id)
                document_name = document.filename if document else "Unknown Document"
                
                source_chunk = SourceChunk(
                    document_id=document_id,
                    document_name=document_name,
                    content=result['content'],
                    relevance_score=result.get('similarity_score', 0.0),
                    chunk_index=result.get('chunk_index', 0)
                )
                source_chunks.append(source_chunk)
                
            except Exception as e:
                logger.error(f"Failed to convert search result to SourceChunk: {e}")
                continue
        
        return source_chunks
    
    async def _update_conversation_history(
        self, 
        session_id: str, 
        question: str, 
        response: QAResponse
    ) -> None:
        """
        Update conversation history with new Q&A exchange
        
        Args:
            session_id: Session identifier
            question: User question
            response: QA response
        """
        try:
            # Get or create conversation
            if session_id not in self._conversations:
                self._conversations[session_id] = ConversationHistory(
                    session_id=session_id,
                    exchanges=[]
                )
            
            # Create new exchange
            exchange = QAExchange(
                question=question,
                answer=response.answer,
                source_chunks=response.source_chunks,
                confidence_score=response.confidence_score
            )
            
            # Add to conversation
            conversation = self._conversations[session_id]
            conversation.exchanges.append(exchange)
            conversation.last_updated = datetime.utcnow()
            
            # Limit conversation history (keep last 10 exchanges)
            if len(conversation.exchanges) > 10:
                conversation.exchanges = conversation.exchanges[-10:]
            
        except Exception as e:
            logger.error(f"Failed to update conversation history: {e}")
    
    async def _get_conversation_context(self, session_id: str) -> str:
        """
        Get conversation context for the session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation context string
        """
        if session_id not in self._conversations:
            return ""
        
        conversation = self._conversations[session_id]
        if not conversation.exchanges:
            return ""
        
        # Format last few exchanges for context
        context_parts = []
        for exchange in conversation.exchanges[-3:]:  # Last 3 exchanges
            context_parts.append(f"Q: {exchange.question}")
            context_parts.append(f"A: {exchange.answer}")
        
        return "\n".join(context_parts)
    
    async def get_conversation_history(self, session_id: str) -> Optional[ConversationHistory]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationHistory object or None if not found
        """
        return self._conversations.get(session_id)
    
    async def create_conversation(self, initial_question: Optional[str] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            initial_question: Optional initial question
            
        Returns:
            New session ID
        """
        session_id = generate_session_id()
        
        conversation = ConversationHistory(
            session_id=session_id,
            exchanges=[]
        )
        
        self._conversations[session_id] = conversation
        
        # Process initial question if provided
        if initial_question:
            request = QARequest(
                question=initial_question,
                session_id=session_id
            )
            await self.ask_question(request)
        
        return session_id
    
    async def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on QA service
        
        Returns:
            Dict containing health status information
        """
        try:
            status = {
                "status": "healthy",
                "llm_available": self._llm is not None,
                "llm_model": settings.llm_model if self._llm else None,
                "active_conversations": len(self._conversations)
            }
            
            # Test vector database connection
            vector_health = await vector_db_service.health_check()
            status["vector_db_status"] = vector_health.get("status", "unknown")
            
            return status
            
        except Exception as e:
            logger.error(f"QA service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global service instance
qa_service = QAService(use_langgraph=True)