from ..domain.document import Document
from ..interfaces.read_file import FileRepository
from typing import Optional

class ReadFileUseCase():
    def __init__(self, file_repository:FileRepository):
        self.file_repository = file_repository

    def execute(self, file_path: str) -> Optional[Document]:
        return self.file_repository.read_file(file_path)




        
