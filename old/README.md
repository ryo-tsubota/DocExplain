# AI QA System

AI-powered document Q&A system using FastAPI, LangChain, and LangGraph.

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` file with your configuration

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── routers/             # API route handlers
│   ├── documents.py     # Document management endpoints
│   └── qa.py           # Q&A endpoints
├── services/            # Business logic
│   ├── document_service.py
│   └── qa_service.py
├── models/              # Data models
│   ├── document.py
│   └── qa.py
└── core/                # Core functionality
    ├── config.py        # Configuration management
    └── dependencies.py  # Dependency injection
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```