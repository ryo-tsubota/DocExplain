from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import List, Dict, Any
import os
from ..interfaces.llm_service import LLMService as LLMServiceInterface

class LLMService(LLMServiceInterface):
    def __init__(self, model: str = "gemini-2.5-flash"):
        """key は環境変数から取得すること"""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
        )
        
        self.prompt = PromptTemplate.from_template(
            """
                以下のドキュメントを参考にして、質問に答えてください。

                コンテキスト:
                {context}

                質問: {query}

                回答:
                 ・ ソースファイルを明記すること
            """
        )
        
        
        self.chain = self.prompt | self.llm | StrOutputParser() 
        

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """コンテキストを使用して回答を生成"""
        try:
            # コンテキストを結合（ファイル名情報も付与）
            context_items = []
            for doc_info in context_docs: # 最初の3つだけ使用
                content = doc_info.get("content", "")
                metadata = doc_info.get("metadata", {})
                filename = metadata.get("filename", "不明")
                context_items.append(f"--- ソースファイル: {filename} ---" + "\n" + content)

            context_text = "\n\n".join(context_items)

            if not context_text.strip():
                return "関連するドキュメントが見つかりませんでした。"

            response = self.chain.invoke({
                "context": context_text,
                "query": query
            })

            return response
        except Exception as e:
            return f"Error generating response: {e}"
    
    def simple_query(self, prompt: str) -> str:
        """シンプルなプロンプト用のメソッド（コンテキスト不要）"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error in simple query: {e}"
