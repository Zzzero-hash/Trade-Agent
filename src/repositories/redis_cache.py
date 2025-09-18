"""
Redis caching layer for real-time data and predictions.
"""
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union
import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError
from ..connection_pool.redis_pool import get_redis_pool_manager

from ..models.market_data import MarketData
from ..models.trading_signal import TradingSignal
from ..models.time_series import MarketDataPoint, PredictionPoint


logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based caching layer for real-time data and predictions.
    
    Provides high-performance caching with TTL support, pub/sub capabilities,
    and data consistency features.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout
            retry_on_timeout: Retry on timeout errors
            health_check_interval: Health check interval in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        
        # Use a unique pool ID for this cache instance
        self._pool_id = f"redis_{host}_{port}_{db}"
        self._pool_manager = get_redis_pool_manager()
        
        self._redis: Optional[Redis] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            # Get or create pool through the pool manager
            self._redis = await self._pool_manager.get_or_create_pool(
                self._pool_id,
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=self.health_check_interval
            )
            
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        try:
            await self._pool_manager.close_pool(self._pool_id)
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    @asynccontextmanager
    async def get_client(self):
        """Context manager for Redis client."""
        # Ensure pool is initialized
        if not self._redis:
            await self.connect()
        
        # Use pool manager to get client
        async with self._pool_manager.get_client(self._pool_id) as client:
            try:
                yield client
            except Exception as e:
                logger.error(f"Redis operation failed: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Check Redis health status."""
        try:
            return await self._pool_manager.health_check(self._pool_id)
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def _generate_key(self, prefix: str, *args: str) -> str:
        """Generate cache key with prefix and arguments."""
        key_parts = [prefix] + list(args)
        return ":".join(key_parts)
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        if isinstance(data, (dict, list, str, int, float, bool)):
            # Use JSON for simple types
            return json.dumps(data, default=str).encode('utf-8')
        else:
            # Use pickle for complex objects
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set cache value with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.get_client() as client:
                serialized_value = self._serialize_data(value)
                
                if ttl:
                    result = await client.setex(key, ttl, serialized_value)
                else:
                    result = await client.set(key, serialized_value)
                
                return bool(result)
                
        except RedisError as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting cache key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cache value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            async with self.get_client() as client:
                data = await client.get(key)
                
                if data is None:
                    return None
                
                return self._deserialize_data(data)
                
        except RedisError as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting cache key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cache key."""
        try:
            async with self.get_client() as client:
                result = await client.delete(key)
                return bool(result)
                
        except RedisError as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if cache key exists."""
        try:
            async with self.get_client() as client:
                result = await client.exists(key)
                return bool(result)
                
        except RedisError as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking cache key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing cache key."""
        try:
            async with self.get_client() as client:
                result = await client.expire(key, ttl)
                return bool(result)
                
        except RedisError as e:
            logger.error(f"Failed to set TTL for cache key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting TTL for cache key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for cache key."""
        try:
            async with self.get_client() as client:
                ttl = await client.ttl(key)
                return ttl if ttl >= 0 else None
                
        except RedisError as e:
            logger.error(f"Failed to get TTL for cache key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting TTL for cache key {key}: {e}")
            return None
    
    # Market Data Caching Methods
    
    async def cache_market_data(
        self, 
        market_data: MarketData, 
        ttl: int = 300
    ) -> bool:
        """
        Cache market data with symbol-based key.
        
        Args:
            market_data: Market data to cache
            ttl: Time to live in seconds (default: 5 minutes)
        """
        key = self._generate_key(
            "market_data", 
            market_data.exchange.value, 
            market_data.symbol
        )
        
        data = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume,
            'exchange': market_data.exchange.value
        }
        
        return await self.set(key, data, ttl)
    
    async def get_market_data(
        self, 
        symbol: str, 
        exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached market data for symbol."""
        key = self._generate_key("market_data", exchange, symbol)
        return await self.get(key)
    
    async def cache_trading_signal(
        self, 
        signal: TradingSignal, 
        ttl: int = 600
    ) -> bool:
        """
        Cache trading signal.
        
        Args:
            signal: Trading signal to cache
            ttl: Time to live in seconds (default: 10 minutes)
        """
        key = self._generate_key("trading_signal", signal.symbol)
        
        data = {
            'symbol': signal.symbol,
            'action': signal.action,
            'confidence': signal.confidence,
            'position_size': signal.position_size,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'timestamp': signal.timestamp.isoformat(),
            'model_version': signal.model_version
        }
        
        return await self.set(key, data, ttl)
    
    async def get_trading_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached trading signal for symbol."""
        key = self._generate_key("trading_signal", symbol)
        return await self.get(key)
    
    async def cache_prediction(
        self, 
        prediction: PredictionPoint, 
        ttl: int = 300
    ) -> bool:
        """
        Cache model prediction.
        
        Args:
            prediction: Prediction to cache
            ttl: Time to live in seconds (default: 5 minutes)
        """
        key = self._generate_key(
            "prediction", 
            prediction.symbol, 
            prediction.model_name
        )
        
        data = {
            'symbol': prediction.symbol,
            'timestamp': prediction.timestamp.isoformat(),
            'model_name': prediction.model_name,
            'model_version': prediction.model_version,
            'predicted_price': prediction.predicted_price,
            'predicted_direction': prediction.predicted_direction,
            'confidence_score': prediction.confidence_score,
            'uncertainty': prediction.uncertainty,
            'feature_importance': prediction.feature_importance
        }
        
        return await self.set(key, data, ttl)
    
    async def get_prediction(
        self, 
        symbol: str, 
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction for symbol and model."""
        key = self._generate_key("prediction", symbol, model_name)
        return await self.get(key)
    
    # Batch Operations
    
    async def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple cache values."""
        try:
            async with self.get_client() as client:
                data_list = await client.mget(keys)
                
                results = []
                for data in data_list:
                    if data is None:
                        results.append(None)
                    else:
                        results.append(self._deserialize_data(data))
                
                return results
                
        except RedisError as e:
            logger.error(f"Failed to get multiple cache keys: {e}")
            return [None] * len(keys)
        except Exception as e:
            logger.error(f"Unexpected error getting multiple cache keys: {e}")
            return [None] * len(keys)
    
    async def mset(
        self, 
        mapping: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple cache values."""
        try:
            async with self.get_client() as client:
                # Serialize all values
                serialized_mapping = {
                    key: self._serialize_data(value)
                    for key, value in mapping.items()
                }
                
                # Set all values
                result = await client.mset(serialized_mapping)
                
                # Set TTL if specified
                if ttl and result:
                    pipeline = client.pipeline()
                    for key in mapping.keys():
                        pipeline.expire(key, ttl)
                    await pipeline.execute()
                
                return bool(result)
                
        except RedisError as e:
            logger.error(f"Failed to set multiple cache keys: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting multiple cache keys: {e}")
            return False
    
    # Pattern-based Operations
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            async with self.get_client() as client:
                keys = []
                async for key in client.scan_iter(match=pattern):
                    keys.append(key)
                
                if keys:
                    return await client.delete(*keys)
                return 0
                
        except RedisError as e:
            logger.error(f"Failed to delete keys with pattern {pattern}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error deleting pattern {pattern}: {e}")
            return 0
    
    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get all keys matching pattern."""
        try:
            async with self.get_client() as client:
                keys = []
                async for key in client.scan_iter(match=pattern):
                    keys.append(key.decode('utf-8') if isinstance(key, bytes) else key)
                return keys
                
        except RedisError as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting pattern {pattern}: {e}")
            return []
    
    # Pub/Sub Operations
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to Redis channel."""
        try:
            async with self.get_client() as client:
                serialized_message = self._serialize_data(message)
                return await client.publish(channel, serialized_message)
                
        except RedisError as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error publishing to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, *channels: str):
        """Subscribe to Redis channels."""
        try:
            async with self.get_client() as client:
                pubsub = client.pubsub()
                await pubsub.subscribe(*channels)
                return pubsub
                
        except RedisError as e:
            logger.error(f"Failed to subscribe to channels {channels}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error subscribing to channels {channels}: {e}")
            raise
    
    # Cache Statistics
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            async with self.get_client() as client:
                info = await client.info()
                
                return {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'uptime_in_seconds': info.get('uptime_in_seconds', 0)
                }
                
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting cache stats: {e}")
            return {}