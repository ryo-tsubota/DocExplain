import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from ..interfaces.vector_db_repository import VectorDBRepository

class GeminiEmbeddingFunction:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def name(self):
        return "google-generativeai-embeddings"

    def __call__(self, input):
        return self.embeddings.embed_documents(input)

class ChromaDBRepository(VectorDBRepository):
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv("GOOGLE_API_KEY"),
            )

        # クォータエラー回避のため、デフォルトembeddingを使用
        # self.langchain_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # self.embedding_function = GeminiEmbeddingFunction(self.langchain_embeddings)
        try:
            # 既存のコレクションを取得（embedding function設定済み）
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"既存のコレクション '{self.collection_name}' を使用します")
        except:
            # 新しいコレクション作成時のみembedding functionを指定
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            print(f"新しいコレクション '{self.collection_name}' を作成しました")

    def clear_collection(self):
        """コレクションを削除して再作成する"""
        print(f"コレクションをクリアします: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        # 再作成
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        print("コレクションを再作成しました。")
    
    def cleanup_database(self):
        """データベース全体をクリーンアップ（物理ファイル削除）"""
        import shutil
        print("データベースディレクトリを完全にクリーンアップします...")
        try:
            # クライアント接続を閉じる
            self.client = None
            # ディレクトリ全体を削除
            shutil.rmtree("./chroma_db")
            print("chroma_dbディレクトリを削除しました")
            # 新しいクライアントを作成
            self.client = chromadb.PersistentClient(path="./chroma_db")
            # コレクションを再作成
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            print("データベースを初期化しました")
        except Exception as e:
            print(f"クリーンアップエラー: {e}")
            # エラー時は通常の初期化
            self.client = chromadb.PersistentClient(path="./chroma_db")

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str,  Any]]):
        """ドキュメントをベクトルDBに追加"""
        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def search(self, query: str, n_results: int = 10, where_filter: Dict = None) -> Dict:
        """類似ドキュメントを検索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def get_collection_info(self) -> Dict:
        """コレクション情報を取得"""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "document_count": count
        }
