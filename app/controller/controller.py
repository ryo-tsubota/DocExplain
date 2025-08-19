from ..usecases.read_file_usecase import ReadFileUseCase
from ..interfaces.read_file import FileRepository
from ..usecases.rag_usecase import IndexingUseCase
from ..workflows.rag_workflow import RAGWorkflow
from ..infrastructure.vector_db import ChromaDBRepository
from ..infrastructure.llm import LLMService


class FileController:

    def __init__(self, file_path:str, repository: FileRepository):
        self.usecase = ReadFileUseCase(repository)
        self.file_path = file_path
    
    def print_file(self):
        doc = self.usecase.execute(self.file_path)

        if doc:
            print("filename:", doc.filename)
            print("content:", doc.content)
        else:
            print("File not found or could not be read.")

    def list_files(self, prefix: str = "") -> list:
        """
        ファイル一覧を表示する
        """
        files = self.usecase.list_files(prefix)
        
        if files:
            print(f"Files found (prefix: '{prefix}'):")
            for file in files:
                print(f"  - {file}")
        else:
            print(f"No files found with prefix: '{prefix}'")
        return files

    def download_file(self, cloud_path: str, local_folder: str = "data"):
        """
        ファイルをdataフォルダにダウンロード
        """
        import os

        # ローカルファイル名を生成（パスの最後の部分を使用）
        filename = os.path.basename(cloud_path)
        local_path = os.path.join(local_folder, filename)

        print(f"Downloading {cloud_path} to {local_path}...")

        success = self.usecase.download_file(cloud_path, local_path)

        if success:
            print(f"Successfully downloaded to {local_path}")
        else:
            print(f"Failed to download {cloud_path}")

        return success


class RAGController:
    def __init__(self, ai_key: str = None):
        db_repository = ChromaDBRepository()
        llm_service = LLMService()
        self.indexing_usecase = IndexingUseCase(db_repository)
        self.rag_workflow = RAGWorkflow(ai_key, db_repository, llm_service)

    def create_index(self):
        """dataフォルダのファイルをインデックス化"""
        count = self.indexing_usecase.index_files_from_directory()
        print(f"インデックス化完了: {count}個のドキュメントチャンク")
        return count

    def search(self, query: str):
        """RAG検索実行"""
        print(f"検索クエリ: {query}")
        result = self.rag_workflow.execute(query)
        
        # ソース情報を詳細表示
        if result.get('sources'):
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. ファイル: {source['filename']}")
                print(f"     パス: {source['file_path']}")
                print(f"     チャンク: {source['chunk_number']}")
                
                print()
        else:
            print("  参照ソースなし")

        return result