from ..usecases.read_file_usecase import ReadFileUseCase
from ..interfaces.read_file import FileRepository


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


