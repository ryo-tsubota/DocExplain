from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
import os
from dotenv import load_dotenv
import markdown
from markdown.extensions import codehilite, tables, fenced_code

# 環境変数を読み込み
load_dotenv()

# パスを追加してappモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.controller.controller import RAGController, FileController
from app.infrastructure.read_file import CloudStorageRepository

# RAGコントローラーの初期化
rag_controller = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global rag_controller
    # 起動時
    print("Creating vector index...")
    rag_controller = RAGController()
    rag_controller.create_index()
    print("Vector index created successfully!")
    
    yield
    
    # 終了時の処理（必要に応じて）
    print("Application shutdown")

app = FastAPI(
    title="DocExplain", 
    description="Document Explanation RAG System",
    lifespan=lifespan
)

# テンプレートとスタティックファイルの設定
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ホーム画面（質問入力フォーム）"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, question: str = Form(...)):
    try:
        print(f"Received question: {question}")
    except UnicodeEncodeError:
        print("Received question with special characters")

    
    """RAG検索を実行"""
    global rag_controller
    
    if not rag_controller:
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": "RAGコントローラーが初期化されていません"
        })
    
    try:
        # RAG検索実行
        try:
            print(f"Searching for: {question}")
        except UnicodeEncodeError:
            print("Searching for question with special characters")
        result = rag_controller.search(question)
        try:
            print(result)
        except UnicodeEncodeError:
            print("Search completed successfully")
        answer = result.get("answer", "回答が見つかりませんでした")
        
        # マークダウンをHTMLに変換
        md = markdown.Markdown(extensions=[
            'codehilite',
            'tables', 
            'fenced_code',
            'nl2br'
        ])
        answer_html = md.convert(answer)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "question": question,
            "answer": answer,
            "answer_html": answer_html
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"検索中にエラーが発生しました: {str(e)}"
        })

@app.get("/files", response_class=HTMLResponse)
async def files(request: Request):
    """参照ファイル一覧画面"""
    try:
        # dataフォルダのファイル一覧を取得
        import os
        data_dir = "data"
        files = []
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    # ファイル情報を取得
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    # ファイルサイズを読みやすい形式に変換
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.1f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.1f} MB"
                    
                    files.append({
                        "filename": filename,
                        "size": size_str,
                        "path": file_path
                    })
        
        return templates.TemplateResponse("files.html", {
            "request": request,
            "files": files
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"ファイル一覧の取得中にエラーが発生しました: {str(e)}"
        })

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy"}

@app.post("/download")
async def download_data(request: Request):
    """Cloud Storageからデータをダウンロードして更新"""
    global rag_controller
    
    try:
        # RAGコントローラーが初期化されているかチェック
        if not rag_controller:
            return templates.TemplateResponse("error.html", {
                "request": request, 
                "error": "RAGコントローラーが初期化されていません"
            })
        
        # Cloud Storageの設定
        bucket_name = os.getenv("GCS_BUCKET_NAME", "doc_explain")
        cloud_storage = CloudStorageRepository(bucket_name)
        file_controller = FileController("", cloud_storage)
        
        print(f"Cloud Storageからデータをダウンロード中... (Bucket: {bucket_name})")
        
        # Cloud Storageからファイル一覧を取得
        gcs_files = file_controller.list_files("")
        print(f"Cloud Storageに {len(gcs_files)} 個のファイルが見つかりました")
        
        # デバッグ: 取得したファイル一覧を表示
        print("=== 取得したファイル一覧 ===")
        for i, gcs_file in enumerate(gcs_files):
            print(f"{i+1}: '{gcs_file}' (type: {type(gcs_file)})")
        print("=========================")
        
        if not gcs_files:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": f"Cloud Storage (Bucket: {bucket_name}) にファイルが見つかりませんでした"
            })
        
        # dataディレクトリを作成（存在しない場合）
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # 各ファイルをダウンロード
        downloaded_count = 0
        for gcs_file in gcs_files:
            try:
                print(f"\n--- 処理中のファイル: '{gcs_file}' ---")
                filename = os.path.basename(gcs_file)
                print(f"抽出されたファイル名: '{filename}'")
                
                # ファイル名が空の場合（フォルダの場合）をチェック
                if not filename:
                    print(f"ファイル名が空です。これはフォルダかもしれません: {gcs_file}")
                    continue
                
                local_path = os.path.join(data_dir, filename)
                print(f"ローカルパス: '{local_path}'")
                
                success = file_controller.download_file(gcs_file, data_dir)
                if success:
                    downloaded_count += 1
                    print(f"✓ ダウンロード成功: {gcs_file} -> {local_path}")
                else:
                    print(f"✗ ダウンロード失敗: {gcs_file}")
                    
            except Exception as e:
                print(f"✗ ファイル {gcs_file} のダウンロードでエラー: {str(e)}")
        
        print(f"ダウンロード完了: {downloaded_count}/{len(gcs_files)} ファイル")
        
        # ベクターインデックスを再作成
        print("ベクターインデックスを再作成中...")
        rag_controller.create_index()
        print("ベクターインデックス更新完了!")
        
        # ファイル一覧を取得して表示
        files = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.1f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.1f} MB"
                    
                    files.append({
                        "filename": filename,
                        "size": size_str,
                        "path": file_path
                    })
        
        return templates.TemplateResponse("files.html", {
            "request": request,
            "files": files,
            "update_success": True,
            "download_count": downloaded_count,
            "total_count": len(gcs_files)
        })
        
    except Exception as e:
        print(f"ダウンロードエラー: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Cloud Storageからのダウンロード中にエラーが発生しました: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)