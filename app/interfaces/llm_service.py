from abc import ABC, abstractmethod
from typing import Dict, List, Any

class LLMService(ABC):
    
    @abstractmethod
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        コンテキストを基に回答を生成
        
        Args:
            query: ユーザーの質問
            context: 検索結果のコンテキスト情報のリスト
                    各要素は {"content": str, "metadata": Dict} の形式
            
        Returns:
            生成された回答テキスト
        """
        pass
    
    @abstractmethod
    def simple_query(self, prompt: str) -> str:
        """
        シンプルなプロンプトに対する回答を生成
        
        Args:
            prompt: 質問やプロンプト
            
        Returns:
            生成された回答テキスト
        """
        pass