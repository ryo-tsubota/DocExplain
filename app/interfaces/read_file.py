from abc import ABC, abstractmethod
from ..domain.document import Document
from typing import Optional, List

class FileRepository(ABC):

    @abstractmethod
    def read_file(self, file_path: str) -> Optional[Document]:
        """
        ファイルを読み取り、Documentオブジェクトを返す
        """
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """
        ファイル一覧を取得する
        
        Args:
            prefix: ファイルパスのプレフィックス（フォルダパスなど）
            
        Returns:
            ファイルパス一覧
        """
        pass

    @abstractmethod
    def download_file(self, cloud_path: str, local_path: str) -> bool:
        """
        クラウドからローカルにファイルをダウンロード

        Args:
            cloud_path: クラウド上のファイルパス
            local_path: ローカル保存先パス

        Returns:
            ダウンロード成功時True、失敗時False
        """
        pass

    