from ..interfaces.read_file import FileRepository
from ..domain.document import Document
from typing import Optional, List
import os
import glob
from google.cloud import storage

class LocalFileRepository(FileRepository):
    """
    LocalFileRepository implements the FileRepository interface to read files from the local filesystem.
    """
    def read_file(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                return Document(filename=file_path, content=content)
        except FileNotFoundError as e:
            return None
        except Exception as e:            
            return None

    def list_files(self, prefix: str = "") -> List[str]:
        """
        ローカルファイルシステムからファイル一覧を取得
        """
        try:
            if prefix:
                # プレフィックスがある場合はそのパスから検索
                pattern = os.path.join(prefix, "*")
                return glob.glob(pattern)
            else:
                # プレフィックスがない場合はカレントディレクトリから検索
                return glob.glob("*")
        except Exception as e:
            return []
        
    def download_file(self, cloud_path: str, local_path: str) -> bool:
          """ローカルファイルリポジトリではコピー処理(未実装)"""
          pass  # ローカルではダウンロードは不要なので実装しない


class CloudStorageRepository(FileRepository):
    """
    Cloud Storageからファイルを読み取り、ファイル一覧を取得する実装
    """
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def read_file(self, file_path: str) -> Optional[Document]:
        """
        Cloud Storageからファイルを読み取り
        実際の実装では各クラウドサービスのSDKを使用
        """
        try: 
            blob = self.bucket.blob(file_path)
            content = blob.download_as_text()
            return Document(filename=file_path, content=content)
        except Exception as e:
            return None
        
        
    def list_files(self, prefix: str = "") -> List[str]:
        """
        Cloud Storageからファイル一覧を取得
        """
        # 
        try:
            print(f"Listing files with prefix: '{prefix}'")
            print(f"Bucket name: {self.bucket_name}")

            blobs = self.bucket.list_blobs(prefix=prefix)
            files = []
            for blob in blobs:
                # UTF-8デコードを明示的に行う
                try:
                    filename = blob.name.encode('latin-1').decode('utf-8')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    filename = blob.name
                files.append(filename)
            print(f"Found {len(files)} files")

            return files
        except Exception as e:
            return []
        
    def download_file(self, cloud_path: str, local_path: str) -> bool:
        """Cloud Storageからファイルをダウンロード"""
        try:
            import os

            # ローカルディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob = self.bucket.blob(cloud_path)
            # バイナリでダウンロードしてからUTF-8で保存
            content = blob.download_as_bytes()
            with open(local_path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Download error: {e}")
            return False