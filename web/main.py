from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
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
from app.controller.controller import RAGController

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)