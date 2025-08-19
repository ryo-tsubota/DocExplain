"""
LangGraph workflow for AI QA processing
"""
import logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.models.qa import QARequest, QAResponse, SourceChunk
from app.services.vector_db_service import vector_db_service
from app.services.document_service import document_service

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State for the LangGraph workflow"""
    # Input
    question: str
    document_ids: Optional[List[str]]
    session_id: str
    max_chunks: int
    conversation_history: Optional[str]
    
    # Processing state
    analyzed_query: Optional[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    context: Optional[str]
    
    # Output
    answer: Optional[str]
    confidence_score: float
    source_chunks: List[SourceChunk]
    error_message: Optional[str]
    
    # Metadata
    processing_steps: List[str]
    start_time: float
    
    # Messages for LLM interaction
    messages: Annotated[List[BaseMessage], add_messages]


class LangGraphQAWorkflow:
    """LangGraph workflow for question-answer processing"""
    
    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._workflow = None
        self._initialize_llm()
        self._build_workflow()
    
    def _initialize_llm(self) -> None:
        """Initialize the LLM client"""
        try:
            if settings.openai_api_key:
                self._llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=settings.llm_model,
                    temperature=0.1,
                    max_tokens=1000
                )
                logger.info(f"Initialized LLM for workflow: {settings.llm_model}")
            else:
                logger.warning("OpenAI API key not provided, workflow will use fallback methods")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for workflow: {e}")
            self._llm = None
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow"""
        try:
            # Create the workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add nodes
            workflow.add_node("analyze_query", self._analyze_query_node)
            workflow.add_node("search_documents", self._search_documents_node)
            workflow.add_node("assemble_context", self._assemble_context_node)
            workflow.add_node("generate_answer", self._generate_answer_node)
            workflow.add_node("validate_response", self._validate_response_node)
            workflow.add_node("handle_error", self._handle_error_node)
            
            # Set entry point
            workflow.set_entry_point("analyze_query")
            
            # Add edges
            workflow.add_conditional_edges(
                "analyze_query",
                self._should_continue_after_analysis,
                {
                    "search": "search_documents",
                    "error": "handle_error"
                }
            )
            
            workflow.add_conditional_edges(
                "search_documents",
                self._should_continue_after_search,
                {
                    "assemble": "assemble_context",
                    "no_results": "generate_answer",
                    "error": "handle_error"
                }
            )
            
            workflow.add_edge("assemble_context", "generate_answer")
            
            workflow.add_conditional_edges(
                "generate_answer",
                self._should_validate_response,
                {
                    "validate": "validate_response",
                    "end": END,
                    "error": "handle_error"
                }
            )
            
            workflow.add_edge("validate_response", END)
            workflow.add_edge("handle_error", END)
            
            # Compile the workflow
            self._workflow = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            self._workflow = None
    
    async def process_question(
        self, 
        request: QARequest, 
        conversation_history: Optional[str] = None
    ) -> QAResponse:
        """
        Process a question through the LangGraph workflow
        
        Args:
            request: QA request
            conversation_history: Optional conversation context
            
        Returns:
            QA response
        """
        if not self._workflow:
            logger.error("Workflow not initialized")
            return self._create_error_response(
                request.session_id or "unknown",
                "Workflow not available"
            )
        
        start_time = datetime.utcnow().timestamp()
        
        try:
            # Initialize workflow state
            initial_state = WorkflowState(
                question=request.question,
                document_ids=request.document_ids,
                session_id=request.session_id or "unknown",
                max_chunks=request.max_chunks,
                conversation_history=conversation_history,
                analyzed_query=None,
                search_results=[],
                context=None,
                answer=None,
                confidence_score=0.0,
                source_chunks=[],
                error_message=None,
                processing_steps=[],
                start_time=start_time,
                messages=[]
            )
            
            # Run the workflow
            final_state = await self._workflow.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time = int((datetime.utcnow().timestamp() - start_time) * 1000)
            
            # Create response
            if final_state.get("error_message"):
                return self._create_error_response(
                    final_state["session_id"],
                    final_state["error_message"],
                    processing_time
                )
            
            return QAResponse(
                answer=final_state["answer"] or "回答を生成できませんでした。",
                confidence_score=final_state["confidence_score"],
                source_chunks=final_state["source_chunks"],
                session_id=final_state["session_id"],
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            processing_time = int((datetime.utcnow().timestamp() - start_time) * 1000)
            return self._create_error_response(
                request.session_id or "unknown",
                f"処理中にエラーが発生しました: {str(e)}",
                processing_time
            )
    
    async def _analyze_query_node(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze the user query to understand intent and extract key information
        """
        try:
            state["processing_steps"].append("query_analysis")
            logger.info(f"Analyzing query: {state['question'][:100]}...")
            
            # Basic query analysis
            query_analysis = {
                "original_question": state["question"],
                "question_type": self._classify_question_type(state["question"]),
                "key_terms": self._extract_key_terms(state["question"]),
                "requires_context": len(state["question"]) > 20,  # Simple heuristic
                "language": "ja"  # Assuming Japanese for this system
            }
            
            # Enhanced analysis with LLM if available
            if self._llm:
                try:
                    enhanced_analysis = await self._llm_analyze_query(state["question"])
                    query_analysis.update(enhanced_analysis)
                except Exception as e:
                    logger.warning(f"LLM query analysis failed, using basic analysis: {e}")
            
            state["analyzed_query"] = query_analysis
            logger.info(f"Query analysis completed: {query_analysis['question_type']}")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["error_message"] = f"クエリ分析エラー: {str(e)}"
        
        return state
    
    async def _search_documents_node(self, state: WorkflowState) -> WorkflowState:
        """
        Search for relevant document chunks using semantic similarity
        """
        try:
            state["processing_steps"].append("document_search")
            logger.info("Searching for relevant documents...")
            
            # Prepare search query
            search_query = state["question"]
            if state["analyzed_query"] and state["analyzed_query"].get("enhanced_query"):
                search_query = state["analyzed_query"]["enhanced_query"]
            
            # Perform semantic search
            search_results = await vector_db_service.search_similar_chunks(
                query=search_query,
                document_ids=state["document_ids"],
                limit=state["max_chunks"],
                min_similarity=0.3
            )
            
            state["search_results"] = search_results
            logger.info(f"Found {len(search_results)} relevant chunks")
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            state["error_message"] = f"文書検索エラー: {str(e)}"
        
        return state
    
    async def _assemble_context_node(self, state: WorkflowState) -> WorkflowState:
        """
        Assemble context from search results and conversation history
        """
        try:
            state["processing_steps"].append("context_assembly")
            logger.info("Assembling context from search results...")
            
            context_parts = []
            
            # Add conversation history if available
            if state["conversation_history"]:
                context_parts.append(f"会話履歴:\n{state['conversation_history']}\n")
            
            # Add document chunks
            if state["search_results"]:
                context_parts.append("関連文書:")
                for i, chunk in enumerate(state["search_results"]):
                    context_parts.append(f"\n[文書 {i+1}] (関連度: {chunk.get('similarity_score', 0):.2f})")
                    context_parts.append(f"{chunk['content']}\n")
            
            state["context"] = "\n".join(context_parts)
            logger.info(f"Context assembled: {len(state['context'])} characters")
            
        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            state["error_message"] = f"コンテキスト組み立てエラー: {str(e)}"
        
        return state
    
    async def _generate_answer_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate answer using LLM or fallback method
        """
        try:
            state["processing_steps"].append("answer_generation")
            logger.info("Generating answer...")
            
            if not state["search_results"]:
                # No relevant documents found
                state["answer"] = "申し訳ございませんが、アップロードされた文書からご質問に関連する情報を見つけることができませんでした。質問を言い換えるか、関連する文書がアップロードされているかご確認ください。"
                state["confidence_score"] = 0.0
                state["source_chunks"] = []
                return state
            
            # Generate answer with LLM if available
            if self._llm and state["context"]:
                answer, confidence = await self._llm_generate_answer(
                    state["question"],
                    state["context"]
                )
                state["answer"] = answer
                state["confidence_score"] = confidence
            else:
                # Fallback method
                answer, confidence = self._fallback_generate_answer(state["search_results"])
                state["answer"] = answer
                state["confidence_score"] = confidence
            
            # Convert search results to source chunks
            state["source_chunks"] = await self._convert_to_source_chunks(state["search_results"])
            
            logger.info(f"Answer generated with confidence: {state['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            state["error_message"] = f"回答生成エラー: {str(e)}"
        
        return state
    
    async def _validate_response_node(self, state: WorkflowState) -> WorkflowState:
        """
        Validate the generated response for quality and accuracy
        """
        try:
            state["processing_steps"].append("response_validation")
            logger.info("Validating response...")
            
            # Basic validation checks
            validation_score = 1.0
            
            # Check answer length
            if not state["answer"] or len(state["answer"].strip()) < 10:
                validation_score *= 0.5
                logger.warning("Answer too short")
            
            # Check if answer is relevant to question
            if state["answer"] and "申し訳" in state["answer"]:
                validation_score *= 0.3  # Apology responses get lower confidence
            
            # Check source chunk relevance
            if state["source_chunks"]:
                avg_relevance = sum(chunk.relevance_score for chunk in state["source_chunks"]) / len(state["source_chunks"])
                validation_score *= avg_relevance
            
            # Adjust confidence score based on validation
            state["confidence_score"] = min(state["confidence_score"] * validation_score, 1.0)
            
            logger.info(f"Response validated with final confidence: {state['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            # Don't fail the entire workflow for validation errors
            logger.warning("Continuing with unvalidated response")
        
        return state
    
    async def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """
        Handle errors that occur during workflow execution
        """
        state["processing_steps"].append("error_handling")
        
        if not state.get("error_message"):
            state["error_message"] = "不明なエラーが発生しました"
        
        logger.error(f"Workflow error: {state['error_message']}")
        
        # Set default error response
        state["answer"] = f"申し訳ございませんが、処理中にエラーが発生しました: {state['error_message']}"
        state["confidence_score"] = 0.0
        state["source_chunks"] = []
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_analysis(self, state: WorkflowState) -> str:
        """Determine next step after query analysis"""
        if state.get("error_message"):
            return "error"
        return "search"
    
    def _should_continue_after_search(self, state: WorkflowState) -> str:
        """Determine next step after document search"""
        if state.get("error_message"):
            return "error"
        if not state["search_results"]:
            return "no_results"
        return "assemble"
    
    def _should_validate_response(self, state: WorkflowState) -> str:
        """Determine if response should be validated"""
        if state.get("error_message"):
            return "error"
        if state.get("answer") and state["confidence_score"] > 0.5:
            return "validate"
        return "end"
    
    # Helper methods
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "何", "なに"]):
            return "what"
        elif any(word in question_lower for word in ["how", "どう", "どのように"]):
            return "how"
        elif any(word in question_lower for word in ["why", "なぜ", "どうして"]):
            return "why"
        elif any(word in question_lower for word in ["when", "いつ"]):
            return "when"
        elif any(word in question_lower for word in ["where", "どこ"]):
            return "where"
        else:
            return "general"
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from the question"""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {"は", "が", "を", "に", "で", "と", "の", "から", "まで", "について", "に関して", "教えて", "ください"}
        
        # Split by various delimiters including Japanese particles
        # Use a more comprehensive regex for Japanese text
        terms = re.findall(r'[\w]+', question)
        
        # Filter out stop words and short terms
        key_terms = []
        for term in terms:
            if len(term) > 1 and term not in stop_words:
                key_terms.append(term)
        
        # If no good terms found, try splitting the original question differently
        if not key_terms:
            # Fallback: split by common particles and take meaningful parts
            parts = re.split(r'[のについて]', question)
            for part in parts:
                cleaned = part.strip()
                if len(cleaned) > 1:
                    key_terms.append(cleaned)
        
        return key_terms[:10]  # Limit to top 10 terms
    
    async def _llm_analyze_query(self, question: str) -> Dict[str, Any]:
        """Enhanced query analysis using LLM"""
        try:
            system_prompt = """あなたは質問分析の専門家です。与えられた質問を分析し、以下の情報を抽出してください：

1. 質問の意図（情報取得、説明要求、比較など）
2. 重要なキーワード
3. 検索に適した拡張クエリ

JSON形式で回答してください。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"質問: {question}")
            ]
            
            response = await self._llm.ainvoke(messages)
            
            # Parse LLM response (simplified - in production, use structured output)
            return {
                "enhanced_query": question,  # Fallback to original
                "intent": "information_retrieval",
                "llm_analysis": response.content
            }
            
        except Exception as e:
            logger.error(f"LLM query analysis failed: {e}")
            return {}
    
    async def _llm_generate_answer(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using LLM"""
        try:
            system_prompt = """あなたは技術文書の内容について質問に答えるAIアシスタントです。
以下のルールに従って回答してください：

1. 提供された文書の内容のみに基づいて回答する
2. 文書に記載されていない情報については推測しない
3. 回答は正確で具体的にする
4. 日本語で回答する
5. 文書から引用する場合は、どの文書からの情報かを明示する

文書の内容:
{context}"""

            messages = [
                SystemMessage(content=system_prompt.format(context=context)),
                HumanMessage(content=f"質問: {question}")
            ]
            
            response = await self._llm.ainvoke(messages)
            answer = response.content.strip()
            
            # Simple confidence calculation based on answer characteristics
            confidence = 0.8 if len(answer) > 50 else 0.6
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return "LLMによる回答生成に失敗しました。", 0.3
    
    def _fallback_generate_answer(self, search_results: List[Dict[str, Any]]) -> tuple[str, float]:
        """Generate fallback answer when LLM is not available"""
        if not search_results:
            return "関連する情報が見つかりませんでした。", 0.0
        
        # Return the most relevant chunk with explanation
        best_chunk = max(search_results, key=lambda x: x.get('similarity_score', 0))
        
        answer = f"""以下の文書から関連する情報を見つけました：

{best_chunk['content']}

注意: この回答は自動検索結果です。より詳細な分析が必要な場合は、LLM機能を有効にしてください。"""
        
        confidence = best_chunk.get('similarity_score', 0.5)
        
        return answer, confidence
    
    async def _convert_to_source_chunks(self, search_results: List[Dict[str, Any]]) -> List[SourceChunk]:
        """Convert search results to SourceChunk objects"""
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
    
    def _create_error_response(
        self, 
        session_id: str, 
        error_message: str, 
        processing_time: Optional[int] = None
    ) -> QAResponse:
        """Create an error response"""
        return QAResponse(
            answer=f"申し訳ございませんが、エラーが発生しました: {error_message}",
            confidence_score=0.0,
            source_chunks=[],
            session_id=session_id,
            processing_time_ms=processing_time
        )


# Global workflow instance
langgraph_workflow = LangGraphQAWorkflow()