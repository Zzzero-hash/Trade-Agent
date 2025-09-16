"""
Database connection pool manager for TimescaleDB connections.
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)


class DatabasePoolManager:
    """
    Manages database connection pools with lifecycle management.
    
    Provides centralized connection pool management with health checks,
    proper initialization, and cleanup procedures.
    """
    
    def __init__(self):
        """Initialize database pool manager."""
        self._pools: Dict[str, Pool] = {}
        self._pool_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_pool(
        self,
        pool_id: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 60.0,
        server_settings: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Pool:
        """
        Create a new database connection pool.
        
        Args:
            pool_id: Unique identifier for the pool
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            min_size: Minimum connection pool size
            max_size: Maximum connection pool size
            command_timeout: Command timeout in seconds
            server_settings: Additional server settings
            **kwargs: Additional arguments for pool creation
            
        Returns:
            Created connection pool
            
        Raises:
            ValueError: If pool_id already exists
            Exception: If pool creation fails
        """
        async with self._lock:
            if pool_id in self._pools:
                raise ValueError(f"Pool with ID '{pool_id}' already exists")
            
            try:
                pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=command_timeout,
                    server_settings=server_settings or {},
                    **kwargs
                )
                
                self._pools[pool_id] = pool
                self._pool_configs[pool_id] = {
                    'host': host,
                    'port': port,
                    'database': database,
                    'username': username,
                    'password': password,
                    'min_size': min_size,
                    'max_size': max_size,
                    'command_timeout': command_timeout,
                    'server_settings': server_settings or {}
                }
                
                logger.info(f"Created database connection pool '{pool_id}' with size {min_size}-{max_size}")
                return pool
                
            except Exception as e:
                logger.error(f"Failed to create database connection pool '{pool_id}': {e}")
                raise
    
    async def get_pool(self, pool_id: str) -> Optional[Pool]:
        """
        Get existing connection pool by ID.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Connection pool or None if not found
        """
        return self._pools.get(pool_id)
    
    async def get_or_create_pool(
        self,
        pool_id: str,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: float = 60.0,
        server_settings: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Pool:
        """
        Get existing pool or create a new one if it doesn't exist.
        
        Args:
            pool_id: Unique identifier for the pool
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            min_size: Minimum connection pool size
            max_size: Maximum connection pool size
            command_timeout: Command timeout in seconds
            server_settings: Additional server settings
            **kwargs: Additional arguments for pool creation
            
        Returns:
            Connection pool
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            pool = await self.create_pool(
                pool_id, host, port, database, username, password,
                min_size, max_size, command_timeout, server_settings, **kwargs
            )
        return pool
    
    @asynccontextmanager
    async def get_connection(self, pool_id: str):
        """
        Context manager for database connection.
        
        Args:
            pool_id: Pool identifier
            
        Yields:
            Database connection
            
        Raises:
            ValueError: If pool doesn't exist
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            raise ValueError(f"Pool '{pool_id}' not found")
        
        async with pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed in pool '{pool_id}': {e}")
                raise
    
    async def health_check(self, pool_id: str) -> bool:
        """
        Check health status of a connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            True if healthy, False otherwise
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            return False
        
        try:
            async with self.get_connection(pool_id) as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Health check failed for pool '{pool_id}': {e}")
            return False
    
    async def close_pool(self, pool_id: str) -> None:
        """
        Close a specific connection pool.
        
        Args:
            pool_id: Pool identifier
        """
        async with self._lock:
            if pool_id in self._pools:
                try:
                    await self._pools[pool_id].close()
                    logger.info(f"Closed database connection pool '{pool_id}'")
                except Exception as e:
                    logger.error(f"Error closing database connection pool '{pool_id}': {e}")
                finally:
                    del self._pools[pool_id]
                    if pool_id in self._pool_configs:
                        del self._pool_configs[pool_id]
    
    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        pool_ids = list(self._pools.keys())
        for pool_id in pool_ids:
            await self.close_pool(pool_id)
    
    async def get_pool_stats(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Pool statistics or None if pool doesn't exist
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            return None
        
        try:
            # Get pool statistics
            stats = {
                'pool_id': pool_id,
                'min_size': pool.get_min_size(),
                'max_size': pool.get_max_size(),
                'current_size': pool.get_size(),
                'idle_size': pool.get_idle_size(),
                'queue_length': len(pool._queue._waiters) if hasattr(pool._queue, '_waiters') else 0
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get pool stats for '{pool_id}': {e}")
            return None
    
    def get_pool_config(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Pool configuration or None if pool doesn't exist
        """
        return self._pool_configs.get(pool_id)


# Global instance
_database_pool_manager: Optional[DatabasePoolManager] = None


def get_database_pool_manager() -> DatabasePoolManager:
    """Get global database pool manager instance."""
    global _database_pool_manager
    if _database_pool_manager is None:
        _database_pool_manager = DatabasePoolManager()
    return _database_pool_manager


async def close_database_pool_manager() -> None:
    """Close global database pool manager and all pools."""
    global _database_pool_manager
    if _database_pool_manager:
        await _database_pool_manager.close_all_pools()
        _database_pool_manager = None