"""
Tests for connection pooling implementation.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.connection_pool.database_pool import DatabasePoolManager
from src.connection_pool.redis_pool import RedisPoolManager
from src.connection_pool.config import DatabasePoolConfig, RedisPoolConfig


@pytest.mark.asyncio
async def test_database_pool_manager_creation():
    """Test database pool manager creation and basic functionality."""
    manager = DatabasePoolManager()
    
    # Test creating a pool
    with patch('asyncpg.create_pool') as mock_create_pool:
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        
        pool = await manager.create_pool(
            pool_id="test_db_pool",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password",
            min_size=2,
            max_size=5
        )
        
        # Verify pool was created
        assert pool == mock_pool
        mock_create_pool.assert_called_once_with(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password",
            min_size=2,
            max_size=5,
            command_timeout=60.0,
            server_settings={}
        )
        
        # Verify pool is stored
        stored_pool = await manager.get_pool("test_db_pool")
        assert stored_pool == mock_pool


@pytest.mark.asyncio
async def test_database_pool_manager_get_or_create():
    """Test get_or_create functionality for database pools."""
    manager = DatabasePoolManager()
    
    with patch('asyncpg.create_pool') as mock_create_pool:
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        
        # First call should create pool
        pool1 = await manager.get_or_create_pool(
            pool_id="test_db_pool",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password"
        )
        
        assert pool1 == mock_pool
        assert mock_create_pool.call_count == 1
        
        # Second call should return existing pool
        pool2 = await manager.get_or_create_pool(
            pool_id="test_db_pool",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password"
        )
        
        assert pool2 == mock_pool
        # Should not have called create_pool again
        assert mock_create_pool.call_count == 1


@pytest.mark.asyncio
async def test_database_pool_manager_connection_context():
    """Test database connection context manager."""
    manager = DatabasePoolManager()
    
    with patch('asyncpg.create_pool') as mock_create_pool:
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_create_pool.return_value = mock_pool
        
        await manager.create_pool(
            pool_id="test_db_pool",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password"
        )
        
        # Test connection context manager
        async with manager.get_connection("test_db_pool") as conn:
            assert conn == mock_connection


@pytest.mark.asyncio
async def test_database_pool_manager_health_check():
    """Test database pool health check."""
    manager = DatabasePoolManager()
    
    with patch('asyncpg.create_pool') as mock_create_pool:
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.fetchval.return_value = 1
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_create_pool.return_value = mock_pool
        
        await manager.create_pool(
            pool_id="test_db_pool",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password"
        )
        
        # Test health check
        result = await manager.health_check("test_db_pool")
        assert result is True


@pytest.mark.asyncio
async def test_redis_pool_manager_creation():
    """Test Redis pool manager creation and basic functionality."""
    manager = RedisPoolManager()
    
    # Test creating a pool
    with patch('redis.asyncio.ConnectionPool') as mock_connection_pool, \
         patch('redis.asyncio.Redis') as mock_redis:
        mock_pool = AsyncMock()
        mock_connection_pool.return_value = AsyncMock()
        mock_redis.return_value = mock_pool
        mock_pool.ping = AsyncMock()
        
        pool = await manager.create_pool(
            pool_id="test_redis_pool",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            max_connections=10
        )
        
        # Verify pool was created
        assert pool == mock_pool
        mock_redis.assert_called_once()
        mock_pool.ping.assert_called_once()


@pytest.mark.asyncio
async def test_redis_pool_manager_get_or_create():
    """Test get_or_create functionality for Redis pools."""
    manager = RedisPoolManager()
    
    with patch('redis.asyncio.ConnectionPool') as mock_connection_pool, \
         patch('redis.asyncio.Redis') as mock_redis:
        mock_pool = AsyncMock()
        mock_connection_pool.return_value = AsyncMock()
        mock_redis.return_value = mock_pool
        mock_pool.ping = AsyncMock()
        
        # First call should create pool
        pool1 = await manager.get_or_create_pool(
            pool_id="test_redis_pool",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            max_connections=10
        )
        
        assert pool1 == mock_pool
        assert mock_redis.call_count == 1
        
        # Second call should return existing pool
        pool2 = await manager.get_or_create_pool(
            pool_id="test_redis_pool",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            max_connections=10
        )
        
        assert pool2 == mock_pool
        # Should not have called Redis again
        assert mock_redis.call_count == 1


@pytest.mark.asyncio
async def test_redis_pool_manager_client_context():
    """Test Redis client context manager."""
    manager = RedisPoolManager()
    
    with patch('redis.asyncio.ConnectionPool') as mock_connection_pool, \
         patch('redis.asyncio.Redis') as mock_redis:
        mock_pool = AsyncMock()
        mock_connection_pool.return_value = AsyncMock()
        mock_redis.return_value = mock_pool
        mock_pool.ping = AsyncMock()
        
        await manager.create_pool(
            pool_id="test_redis_pool",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            max_connections=10
        )
        
        # Test client context manager
        async with manager.get_client("test_redis_pool") as client:
            assert client == mock_pool


@pytest.mark.asyncio
async def test_redis_pool_manager_health_check():
    """Test Redis pool health check."""
    manager = RedisPoolManager()
    
    with patch('redis.asyncio.ConnectionPool') as mock_connection_pool, \
         patch('redis.asyncio.Redis') as mock_redis:
        mock_pool = AsyncMock()
        mock_connection_pool.return_value = AsyncMock()
        mock_redis.return_value = mock_pool
        mock_pool.ping = AsyncMock()
        mock_pool.ping.return_value = True
        
        await manager.create_pool(
            pool_id="test_redis_pool",
            host="localhost",
            port=6379,
            db=0,
            password=None,
            max_connections=10
        )
        
        # Test health check
        result = await manager.health_check("test_redis_pool")
        assert result is True


@pytest.mark.asyncio
async def test_database_pool_config_validation():
    """Test database pool configuration validation."""
    # Valid config
    config = DatabasePoolConfig(
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_password",
        min_size=2,
        max_size=5
    )
    
    # Invalid port
    with pytest.raises(ValueError):
        DatabasePoolConfig(port=99999)
    
    # Invalid min_size
    with pytest.raises(ValueError):
        DatabasePoolConfig(min_size=0)
    
    # Invalid max_size < min_size
    with pytest.raises(ValueError):
        DatabasePoolConfig(min_size=5, max_size=3)


@pytest.mark.asyncio
async def test_redis_pool_config_validation():
    """Test Redis pool configuration validation."""
    # Valid config
    config = RedisPoolConfig(
        host="localhost",
        port=6379,
        db=0,
        max_connections=10
    )
    
    # Invalid port
    with pytest.raises(ValueError):
        RedisPoolConfig(port=99999)
    
    # Invalid db
    with pytest.raises(ValueError):
        RedisPoolConfig(db=20)
    
    # Invalid max_connections
    with pytest.raises(ValueError):
        RedisPoolConfig(max_connections=0)


if __name__ == "__main__":
    pytest.main([__file__])