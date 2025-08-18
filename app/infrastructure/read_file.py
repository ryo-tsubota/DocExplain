from ..interfaces.read_file import FileRepository
from ..domain.document import Document
from typing import Optional

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
