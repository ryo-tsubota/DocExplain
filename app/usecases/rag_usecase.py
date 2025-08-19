from ..interfaces.vector_db_repository import VectorDBRepository
from ..interfaces.llm_service import LLMService
import re
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class IndexingUseCase:
    def __init__(self, db_repository: VectorDBRepository):
        self.vector_db = db_repository
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def index_files_from_directory(self, directory_path: str = "data"):
        """dataディレクトリのファイルをインデックス化"""
        self.vector_db.clear_collection()
        documents = []
        metadatas = []

        # ファイル名をソートして、正しいファイル名を優先
        file_list = sorted(os.listdir(directory_path), key=lambda x: x.startswith('�'))
        
        for filename in file_list:
            file_path = os.path.join(directory_path, filename)
            
            if filename.endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                continue
            
            try:
                docs = loader.load()
                if not docs:
                    continue

                title = ""
                if docs[0].page_content.strip().startswith('#'):
                    # Corrected line with '\n'
                    title = docs[0].page_content.strip().split('\n', 1)[0] + "\n\n"
                
                chunks = self.text_splitter.split_documents(docs)

                for chunk in chunks:
                    chunk.page_content = title + chunk.page_content

                for i, chunk in enumerate(chunks):
                    documents.append(chunk.page_content)
                    metadatas.append({
                        "filename": filename,
                        "chunk_id": i,
                        "file_path": file_path
                    })
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        if documents:
            ids = self.vector_db.add_documents(documents, metadatas)
            print(f"インデックス化完了: {len(documents)}個のドキュメントチャンク")
            return len(documents)
        
        print("インデックス化するドキュメントがありませんでした。")
        return 0

class RAGUseCase:
    def __init__(self, model: str, db_repository: VectorDBRepository, llm_service: LLMService):
        self.vector_db = db_repository
        self.llm_service = llm_service
    
    
    def search_and_generate_with_filter(self, query: str, where_filter: Dict = None, n_results: int = 10) -> Dict:
        """フィルター指定付きで検索して回答を生成"""
        search_results = self.vector_db.search(query, n_results, where_filter=where_filter)

        context_docs = (search_results.get('documents') or [[]])[0]
        source_metadatas = (search_results.get('metadatas') or [[]])[0]

        source_info = []
        for i, (doc, metadata) in enumerate(zip(context_docs, source_metadatas)):
            source_info.append({
                "chunk_id": i,
                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "filename": metadata.get("filename", "unknown"),
                "file_path": metadata.get("file_path", "unknown"),
                "chunk_number": metadata.get("chunk_id", 0)
            })

        if self.llm_service and context_docs:
            llm_context = []
            for doc, meta in zip(context_docs, source_metadatas):
                llm_context.append({"content": doc, "metadata": meta})
            
            response = self.llm_service.generate_response(query, llm_context)
        else:
            response = "関連するドキュメントが見つからなかったか、LLMサービスが利用できません。"

        return {
            "answer": response,
            "sources": source_info,
        }