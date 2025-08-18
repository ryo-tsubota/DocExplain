from app.controller.controller import FileController
from app.infrastructure.read_file import LocalFileRepository

if __name__ == "__main__":

    file_path = "data/NYC_TRIPS.txt"

    controller = FileController(file_path, LocalFileRepository())
    controller.print_file()