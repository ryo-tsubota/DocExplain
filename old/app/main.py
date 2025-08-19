"""
AI QA System - Main FastAPI Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.vector_db import vector_db_manager
from app.routers import documents, qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await vector_db_manager.startup()
    yield
    # Shutdown
    await vector_db_manager.shutdown()


app = FastAPI(
    title="AI QA System",
    description="AI-powered document Q&A system using LangChain and LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/")
async def root():
    return {"message": "AI QA System is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint with vector database status"""
    from app.services.vector_db_service import vector_db_service
    
    try:
        vector_db_health = await vector_db_service.health_check()
        return {
            "status": "healthy",
            "vector_db": vector_db_health
        }
    except Exception as e:
        return {
            "status": "degraded",
            "vector_db": {"status": "unhealthy", "error": str(e)}
        }