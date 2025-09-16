"""
Redis connection pool manager for Redis connections.
"""
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisPoolManager:
    """
    Manages Redis connection pools with lifecycle management.
    
    Provides centralized connection pool management with health checks,
    proper initialization, and cleanup procedures.
    """
    
    def __init__(self):
        """Initialize Redis pool manager."""
        self._pools: Dict[str, Redis] = {}
        self._pool_configs: Dict[str, Dict[str, Any]] = {}
        self._connection_pools: Dict[str, redis.ConnectionPool] = {}
        self._lock = asyncio.Lock()
    
    async def create_pool(
        self,
        pool_id: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        **kwargs
    ) -> Redis:
        """
        Create a new Redis connection pool.
        
        Args:
            pool_id: Unique identifier for the pool
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout
            retry_on_timeout: Retry on timeout errors
            health_check_interval: Health check interval in seconds
            **kwargs: Additional arguments for pool creation
            
        Returns:
            Created Redis client
            
        Raises:
            ValueError: If pool_id already exists
            Exception: If pool creation fails
        """
        async with self._lock:
            if pool_id in self._pools:
                raise ValueError(f"Pool with ID '{pool_id}' already exists")
            
            try:
                # Create connection pool
                connection_pool = redis.ConnectionPool(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    max_connections=max_connections,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    retry_on_timeout=retry_on_timeout,
                    health_check_interval=health_check_interval,
                    **kwargs
                )
                
                # Create Redis client
                redis_client = Redis(connection_pool=connection_pool)
                
                # Test connection
                await redis_client.ping()
                
                self._pools[pool_id] = redis_client
                self._connection_pools[pool_id] = connection_pool
                self._pool_configs[pool_id] = {
                    'host': host,
                    'port': port,
                    'db': db,
                    'password': password,
                    'max_connections': max_connections,
                    'socket_timeout': socket_timeout,
                    'socket_connect_timeout': socket_connect_timeout,
                    'retry_on_timeout': retry_on_timeout,
                    'health_check_interval': health_check_interval
                }
                
                logger.info(f"Created Redis connection pool '{pool_id}' with max {max_connections} connections")
                return redis_client
                
            except Exception as e:
                logger.error(f"Failed to create Redis connection pool '{pool_id}': {e}")
                raise
    
    async def get_pool(self, pool_id: str) -> Optional[Redis]:
        """
        Get existing Redis client by pool ID.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Redis client or None if not found
        """
        return self._pools.get(pool_id)
    
    async def get_or_create_pool(
        self,
        pool_id: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        **kwargs
    ) -> Redis:
        """
        Get existing pool or create a new one if it doesn't exist.
        
        Args:
            pool_id: Unique identifier for the pool
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout
            retry_on_timeout: Retry on timeout errors
            health_check_interval: Health check interval in seconds
            **kwargs: Additional arguments for pool creation
            
        Returns:
            Redis client
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            pool = await self.create_pool(
                pool_id, host, port, db, password,
                max_connections, socket_timeout, socket_connect_timeout,
                retry_on_timeout, health_check_interval, **kwargs
            )
        return pool
    
    @asynccontextmanager
    async def get_client(self, pool_id: str):
        """
        Context manager for Redis client.
        
        Args:
            pool_id: Pool identifier
            
        Yields:
            Redis client
            
        Raises:
            ValueError: If pool doesn't exist
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            raise ValueError(f"Pool '{pool_id}' not found")
        
        try:
            yield pool
        except Exception as e:
            logger.error(f"Redis operation failed in pool '{pool_id}': {e}")
            raise
    
    async def health_check(self, pool_id: str) -> bool:
        """
        Check health status of a Redis connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            True if healthy, False otherwise
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            return False
        
        try:
            await pool.ping()
            return True
        except Exception as e:
            logger.error(f"Health check failed for Redis pool '{pool_id}': {e}")
            return False
    
    async def close_pool(self, pool_id: str) -> None:
        """
        Close a specific Redis connection pool.
        
        Args:
            pool_id: Pool identifier
        """
        async with self._lock:
            if pool_id in self._pools:
                try:
                    await self._pools[pool_id].close()
                    logger.info(f"Closed Redis connection pool '{pool_id}'")
                except Exception as e:
                    logger.error(f"Error closing Redis connection pool '{pool_id}': {e}")
                finally:
                    del self._pools[pool_id]
                    if pool_id in self._connection_pools:
                        del self._connection_pools[pool_id]
                    if pool_id in self._pool_configs:
                        del self._pool_configs[pool_id]
    
    async def close_all_pools(self) -> None:
        """Close all Redis connection pools."""
        pool_ids = list(self._pools.keys())
        for pool_id in pool_ids:
            await self.close_pool(pool_id)
    
    async def get_pool_stats(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a Redis connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Pool statistics or None if pool doesn't exist
        """
        pool = await self.get_pool(pool_id)
        if pool is None:
            return None
        
        try:
            # Get Redis info
            info = await pool.info()
            
            stats = {
                'pool_id': pool_id,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get pool stats for Redis pool '{pool_id}': {e}")
            return None
    
    def get_pool_config(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a Redis connection pool.
        
        Args:
            pool_id: Pool identifier
            
        Returns:
            Pool configuration or None if pool doesn't exist
        """
        return self._pool_configs.get(pool_id)


# Global instance
_redis_pool_manager: Optional[RedisPoolManager] = None


def get_redis_pool_manager() -> RedisPoolManager:
    """Get global Redis pool manager instance."""
    global _redis_pool_manager
    if _redis_pool_manager is None:
        _redis_pool_manager = RedisPoolManager()
    return _redis_pool_manager


async def close_redis_pool_manager() -> None:
    """Close global Redis pool manager and all pools."""
    global _redis_pool_manager
    if _redis_pool_manager:
        await _redis_pool_manager.close_all_pools()
        _redis_pool_manager = None