from abc import ABC, abstractmethod
from ..domain.document import Document
from typing import Optional

class FileRepository(ABC):

    @abstractmethod
    def read_file(self, file_path: str) -> Optional[Document]:
        pass

    