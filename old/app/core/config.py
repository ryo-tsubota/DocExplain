"""
Configuration management for AI QA System
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "AI QA System"
    debug: bool = False
    
    # File upload settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = [".pdf", ".docx", ".txt", ".md"]
    upload_dir: str = "uploads"
    
    # Vector database settings
    vector_db_path: str = "./vector_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()