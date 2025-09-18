# Intelligent Caching System with TTLCache Design

## 1. Executive Summary

Design for intelligent caching system using TTLCache to optimize feature extraction and predictions in AI Trading Platform.

## 2. Current System Analysis

### 2.1 Existing Implementation

1. **CachedFeatureExtractor** in [`src/ml/feature_extraction/cached_extractor.py`](file:///d%3A/trade-agent/src/ml/feature_extraction/cached_extractor.py) with basic TTLCache
2. **FeatureExtractionConfig** with `cache_size: 1000`, `cache_ttl_seconds: 60`
3. Model serving cache in [`src/api/model_serving.py`](file:///d%3A/trade-agent/src/api/model_serving.py)
4. Redis cache in [`src/repositories/redis_cache.py`](file:///d%3A/trade-agent/src/repositories/redis_cache.py)

### 2.2 Identified Gaps

1. Limited cache scope (only feature extraction)
2. Single-level caching (no memory + distributed strategy)
3. Static TTL configuration
4. Limited cache invalidation
5. No cache warming
6. Limited monitoring

## 3. Architecture Design

### 3.1 Multi-Level Cache Manager

```python
class MultiLevelCacheManager:
    def __init__(self, config: CacheConfig):
        self.memory_cache = TTLCache(
            maxsize=config.memory_cache_size,
            ttl=config.memory_cache_ttl
        )
        self.distributed_cache = RedisCache(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db
        )
    
    async def get(self, key: str) -> Optional[Any]:
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check distributed cache
        distributed_result = await self.distributed_cache.get(key)
        if distributed_result is not None:
            # Promote to memory cache
            self.memory_cache[key] = distributed_result
            return distributed_result
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Set in both cache levels
        self.memory_cache[key] = value
        await self.distributed_cache.set(key, value, ttl)
```

## 4. Cache Configuration

### 4.1 TTL Strategies

| Data Type | Base TTL | Notes |
|-----------|----------|-------|
| Market Features | 60s | High frequency trading data |
| CNN+LSTM Features | 60s | Model-based features |
| Predictions | 30s | Trading signals |
| Models | 3600s | Infrequent updates |

### 4.2 Cache Sizes

```yaml
memory_cache:
  max_size: 10000
  ttl: 60

distributed_cache:
  host: "localhost"
  port: 6379
  db: 2
  ttl: 300
```

## 5. Integration Plan

### 5.1 Enhanced Feature Extractor

```python
class IntelligentCachedFeatureExtractor(FeatureExtractor):
    def __init__(self, extractor: FeatureExtractor, config: FeatureExtractionConfig):
        self.extractor = extractor
        self.config = config
        self.memory_cache = TTLCache(maxsize=config.cache_size, ttl=config.cache_ttl_seconds)
        
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        # Generate cache key
        cache_key = hashlib.md5(data.tobytes()).hexdigest()
        
        # Check cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Extract and cache
        result = self.extractor.extract_features(data)
        self.memory_cache[cache_key] = result
        return result
```

## 6. Cache Invalidation

### 6.1 Event-Driven Invalidation

```python
class EventDrivenCacheInvalidator:
    async def handle_market_data_update(self, symbol: str) -> None:
        # Invalidate feature cache for symbol
        pattern = f"features:*{symbol}*"
        await self.cache_manager.invalidate_pattern(pattern)
```

## 7. Implementation Roadmap

1. **Phase 1**: Enhance existing cache infrastructure
2. **Phase 2**: Integrate with model serving
3. **Phase 3**: Add monitoring and optimization
4. **Phase 4**: Production deployment

## 8. Conclusion

Intelligent caching with TTLCache will significantly reduce latency while maintaining data freshness.