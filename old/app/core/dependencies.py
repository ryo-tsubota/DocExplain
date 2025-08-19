"""
Dependency injection for FastAPI
"""
from app.core.config import settings
from app.services.vector_db_service import vector_db_service


def get_settings():
    """Get application settings"""
    return settings


def get_vector_db_service():
    """Get vector database service instance"""
    return vector_db_service