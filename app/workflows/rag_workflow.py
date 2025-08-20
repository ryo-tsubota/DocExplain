from langgraph.graph import StateGraph, START, END
from typing import Dict, List, TypedDict, Optional
from ..usecases.rag_usecase import RAGUseCase
import os

from ..interfaces.vector_db_repository import VectorDBRepository
from ..interfaces.llm_service import LLMService

class RAGState(TypedDict):
    query: str
    selected_files: List[str]
    search_results: Dict
    answer: str
    sources: List[Dict]


class RAGWorkflow:

    def __init__(self, ai_key: str, db_repository: VectorDBRepository, llm_service: LLMService):
        self.rag_usecase = RAGUseCase(ai_key, db_repository, llm_service)
        self.llm_service = llm_service
        self.graph = self._build_graph()

    def _select_file(self, state: RAGState) -> RAGState:
        """ファイル選定ノード 質問内容に基づいて関連ファイルを選定する
        　　選定したファイルを where_filter に設定し、RAG検索に利用する
        
        """
        query = state["query"]
        
        # データフォルダのファイル一覧を取得
        try:
            if os.path.exists("data"):
                files = os.listdir("data")
            else:
                files = []
            # .mdと.pdfファイルのみを対象（文字化けファイルを除外）
            relevant_files = [f for f in files if f.endswith(('.md', '.pdf', '.txt'))]
            
            
            if not relevant_files:
                state["selected_files"] = []
                return state
                
            # LLMに最適なファイルを選んでもらう（複数選択対応）
            newline = "\n"
            file_list = newline.join([f"- {file}" for file in relevant_files])
            file_selection_prompt = f"""
            以下のファイル一覧から、ユーザーの質問「{query}」に関連する可能性の高いファイルを最大3つまで選んでください。

            ファイル一覧:
            {file_list}

            選択基準:
            - ファイル名から内容を推測し、質問に関連しそうなものを選ぶ
            - 「要件」「要求」に関する質問なら要件定義系のファイル
            - 「設計」に関する質問なら設計書系のファイル
            - 「技術」「AI」の一般的な質問なら技術資料
            - 関連度が高い順に選択してください

            回答は以下の形式で返してください（説明不要）：
            ファイル名1
            ファイル名2
            ファイル名3
            """

            
            # LLMに質問
            if self.llm_service:
                llm_response = self.llm_service.simple_query(file_selection_prompt)
                print(f"LLMの回答: {llm_response}")
                
                # 回答から複数のファイル名を抽出
                selected_files = []
                
                # LLMの回答を行ごとに分割
                response_lines = [line.strip() for line in llm_response.strip().split('\n') if line.strip()]
                
                # 各行でファイル名を検索
                for line in response_lines:
                    for file in relevant_files:
                        if file in line and file not in selected_files:
                            selected_files.append(file)
                            break
                
                # 直接的なファイル名マッチングも試行
                if not selected_files:
                    for file in relevant_files:
                        if file in llm_response and file not in selected_files:
                            selected_files.append(file)
                
                if selected_files:
                    print(f"LLMが選定したファイル: {selected_files}")
                    state["selected_files"] = selected_files
                else:
                    print(f"LLMの選定結果が無効: {llm_response}")
                    print(f"利用可能ファイル: {relevant_files}")
                    state["selected_files"] = []
            else:
                state["selected_files"] = []
                    
        except Exception as e:
            print(f"ファイル選定エラー: {e}")
            state["selected_files"] = []
            
        return state

    def _search_documents(self, state: RAGState) -> RAGState:
        """ドキュメント検索ノード"""
        query = state["query"]
        selected_files = state.get("selected_files", [])
        
        # where_filterを設定（複数ファイル対応）
        where_filter = None
        if selected_files:
            where_filter = {"filename": {"$in": selected_files}}
            print(f"選定されたファイルでフィルタリング: {selected_files}")
        
        results = self.rag_usecase.search_and_generate_with_filter(query, where_filter)
        print(f"検索結果: {results}")

        state["search_results"] = results
        state["answer"] = results["answer"]
        state["sources"] = results["sources"]

        return state

    def _build_graph(self) -> StateGraph:
        """LangGraphワークフローを構築"""
        graph = StateGraph(RAGState)

        # ノード追加
        graph.add_node("select_file", self._select_file)
        graph.add_node("search", self._search_documents)

        # エントリーポイント設定
        graph.add_edge(START, "select_file")
        
        # ファイル選定 → 検索
        graph.add_edge("select_file", "search")

        # 終了条件設定
        graph.add_edge("search", END)

        return graph.compile()

    def execute(self, query: str) -> Dict:
        """ワークフロー実行"""
        initial_state = {
            "query": query,
            "selected_files": [],
            "search_results": {},
            "answer": "",
            "sources": []
        }

        result = self.graph.invoke(initial_state)
        return result