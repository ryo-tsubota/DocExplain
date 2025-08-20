# DocExplain Dockerfile
FROM python:3.11

# 作業ディレクトリの設定
WORKDIR /app

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app/ ./app/
COPY web/ ./web/
COPY data/ ./data/

# ChromaDBのデータディレクトリを作成
RUN mkdir -p /app/chroma_db /app/data

# 環境変数の設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ポート8080を公開
EXPOSE 8080

# アプリケーション実行
CMD ["sh", "-c", "uvicorn web.main:app --host 0.0.0.0 --port ${PORT:-8080}"]