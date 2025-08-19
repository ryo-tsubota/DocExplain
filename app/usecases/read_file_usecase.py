from ..domain.document import Document
from ..interfaces.read_file import FileRepository
from typing import Optional, List

class ReadFileUseCase():
    def __init__(self, file_repository:FileRepository):
        self.file_repository = file_repository

    def execute(self, file_path: str) -> Optional[Document]:
        return self.file_repository.read_file(file_path)

    def list_files(self, prefix: str = "") -> List[str]:
        """
        ファイル一覧を取得するUseCase
        """
        return self.file_repository.list_files(prefix)
    
    def download_file(self, cloud_path: str, local_path: str) -> bool:
        """ファイルダウンロードのUseCase"""
        return self.file_repository.download_file(cloud_path, local_path)




        
