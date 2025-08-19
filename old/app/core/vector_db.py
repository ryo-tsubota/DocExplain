"""
Vector database connection management
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.services.vector_db_service import vector_db_service

logger = logging.getLogger(__name__)


class VectorDBManager:
    """Manager for vector database connections and lifecycle"""
    
    def __init__(self):
        self._initialized = False
    
    async def startup(self) -> None:
        """Initialize vector database on application startup"""
        try:
            await vector_db_service.initialize()
            self._initialized = True
            logger.info("Vector database startup completed")
        except Exception as e:
            logger.error(f"Vector database startup failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Clean up vector database on application shutdown"""
        try:
            if self._initialized:
                await vector_db_service.close()
                self._initialized = False
                logger.info("Vector database shutdown completed")
        except Exception as e:
            logger.error(f"Vector database shutdown failed: {e}")
            # Still mark as not initialized even if shutdown fails
            self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector database is initialized"""
        return self._initialized


# Global manager instance
vector_db_manager = VectorDBManager()


@asynccontextmanager
async def get_vector_db() -> AsyncGenerator[None, None]:
    """
    Context manager for vector database operations
    Ensures the database is initialized before use
    """
    if not vector_db_manager.is_initialized:
        await vector_db_manager.startup()
    
    try:
        yield
    finally:
        # Context manager doesn't close the connection
        # as it's managed by the application lifecycle
        pass