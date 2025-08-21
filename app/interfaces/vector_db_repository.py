from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class VectorDBRepository(ABC):
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """
        ドキュメントをベクターDBに追加
        
        Args:
            documents: ドキュメントのテキストリスト
            metadatas: 各ドキュメントのメタデータリスト
            
        Returns:
            追加されたドキュメントのIDリスト
        """
        pass
    
    @abstractmethod
    def search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> Dict:
        """
        ベクター検索を実行
        
        Args:
            query: 検索クエリ
            n_results: 取得する結果数
            where_filter: フィルター条件
            
        Returns:
            検索結果（documents, metadatas, distances等を含む辞書）
        """
        pass
    
    @abstractmethod
    def clear_collection(self) -> None:
        """
        コレクション内のすべてのドキュメントを削除
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict:
        """
        コレクションの情報を取得
        
        Returns:
            コレクション情報（ドキュメント数等）
        """
        pass
    
    @abstractmethod
    def cleanup_database(self) -> None:
        """
        データベース全体をクリーンアップ（物理ファイル削除）
        """
        pass