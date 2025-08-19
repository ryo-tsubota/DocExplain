"""
Unit tests for vector database manager
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.core.vector_db import VectorDBManager, vector_db_manager, get_vector_db


class TestVectorDBManager:
    """Test cases for VectorDBManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh VectorDBManager instance for testing"""
        return VectorDBManager()
    
    async def test_startup_success(self, manager):
        """Test successful startup"""
        with patch('app.core.vector_db.vector_db_service') as mock_service:
            mock_service.initialize = AsyncMock()
            
            await manager.startup()
            
            assert manager.is_initialized is True
            mock_service.initialize.assert_called_once()
    
    async def test_startup_failure(self, manager):
        """Test startup failure handling"""
        with patch('app.core.vector_db.vector_db_service') as mock_service:
            mock_service.initialize = AsyncMock(side_effect=Exception("Initialization failed"))
            
            with pytest.raises(Exception, match="Initialization failed"):
                await manager.startup()
            
            assert manager.is_initialized is False
    
    async def test_shutdown_when_initialized(self, manager):
        """Test shutdown when manager is initialized"""
        with patch('app.core.vector_db.vector_db_service') as mock_service:
            mock_service.initialize = AsyncMock()
            mock_service.close = AsyncMock()
            
            # Initialize first
            await manager.startup()
            assert manager.is_initialized is True
            
            # Then shutdown
            await manager.shutdown()
            
            assert manager.is_initialized is False
            mock_service.close.assert_called_once()
    
    async def test_shutdown_when_not_initialized(self, manager):
        """Test shutdown when manager is not initialized"""
        with patch('app.core.vector_db.vector_db_service') as mock_service:
            mock_service.close = AsyncMock()
            
            await manager.shutdown()
            
            # Should not call close if not initialized
            mock_service.close.assert_not_called()
            assert manager.is_initialized is False
    
    async def test_shutdown_with_error(self, manager):
        """Test shutdown error handling"""
        with patch('app.core.vector_db.vector_db_service') as mock_service:
            mock_service.initialize = AsyncMock()
            mock_service.close = AsyncMock(side_effect=Exception("Shutdown failed"))
            
            # Initialize first
            await manager.startup()
            
            # Shutdown should not raise exception even if close fails
            await manager.shutdown()
            
            # Should still mark as not initialized
            assert manager.is_initialized is False
    
    def test_is_initialized_property(self, manager):
        """Test is_initialized property"""
        assert manager.is_initialized is False
        
        manager._initialized = True
        assert manager.is_initialized is True
        
        manager._initialized = False
        assert manager.is_initialized is False


class TestGetVectorDB:
    """Test cases for get_vector_db context manager"""
    
    async def test_context_manager_with_initialized_db(self):
        """Test context manager when database is already initialized"""
        with patch('app.core.vector_db.vector_db_manager') as mock_manager:
            mock_manager.is_initialized = True
            mock_manager.startup = AsyncMock()
            
            async with get_vector_db():
                pass
            
            # Should not call startup if already initialized
            mock_manager.startup.assert_not_called()
    
    async def test_context_manager_with_uninitialized_db(self):
        """Test context manager when database is not initialized"""
        with patch('app.core.vector_db.vector_db_manager') as mock_manager:
            mock_manager.is_initialized = False
            mock_manager.startup = AsyncMock()
            
            async with get_vector_db():
                pass
            
            # Should call startup if not initialized
            mock_manager.startup.assert_called_once()
    
    async def test_context_manager_exception_handling(self):
        """Test context manager handles exceptions properly"""
        with patch('app.core.vector_db.vector_db_manager') as mock_manager:
            mock_manager.is_initialized = False
            mock_manager.startup = AsyncMock()
            
            try:
                async with get_vector_db():
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Should still have called startup
            mock_manager.startup.assert_called_once()


class TestGlobalVectorDBManager:
    """Test cases for global vector_db_manager instance"""
    
    def test_global_instance_exists(self):
        """Test that global instance exists and is correct type"""
        assert vector_db_manager is not None
        assert isinstance(vector_db_manager, VectorDBManager)
    
    def test_global_instance_initial_state(self):
        """Test initial state of global instance"""
        # Note: This might be affected by other tests, so we just check the type
        assert hasattr(vector_db_manager, 'is_initialized')
        assert hasattr(vector_db_manager, 'startup')
        assert hasattr(vector_db_manager, 'shutdown')