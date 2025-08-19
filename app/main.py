from app.controller.controller import FileController, RAGController
from app.infrastructure.read_file import LocalFileRepository, CloudStorageRepository
from dotenv import load_dotenv

load_dotenv(override=True)


if __name__ == "__main__":

    file_path = "data/NYC_TRIPS.txt"
    bucket_path = "doc_explain"

    # # ローカルファイル操作
    # print("=== Local File Repository ===")
    # local_controller = FileController(file_path, LocalFileRepository())
    # local_controller.print_file()
    
    # print("\n=== Local File List ===")
    # local_controller.list_files("data")
    
    # Cloud Storage操作
    print("\n=== Cloud Storage Repository ===")
    cloud_controller = FileController(file_path, CloudStorageRepository(bucket_path))
    
    print("\n=== Cloud Storage File List ===")
    gcs_files = cloud_controller.list_files("")
    print(gcs_files)
    
    print("\n=== Downloading File from Cloud Storage ===")
    for gcs_file in gcs_files:
      cloud_controller.download_file(gcs_file, "data")
    

    # RAGコントローラーの初期化
    
    rag_controller = RAGController()
    # RAG操作
    # インデックス作成
    print("\n=== Creating Vector Index ===")
    rag_controller.create_index()

    # RAG検索テスト
    print("\n=== RAG Search Test ===")
    input_word = input("質問事項を入力してください。")
    result = rag_controller.search(input_word) # "要件について概要を教えて。また、関連するファイルも教えて。"
    print("Final Answer:", result.get("answer"))
    

    

