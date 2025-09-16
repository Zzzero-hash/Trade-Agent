# Cache Integration Approach

## 1. Feature Extraction Pipeline Integration

### 1.1 Enhanced CachedFeatureExtractor

```python
# src/ml/feature_extraction/intelligent_cached_extractor.py
from cachetools import TTLCache
from typing import Dict, Any
import numpy as np
import time
import hashlib

class IntelligentCachedFeatureExtractor(CachedFeatureExtractor):
    """Enhanced feature extractor with intelligent caching strategies."""
    
    def __init__(self, extractor: FeatureExtractor, config: FeatureExtractionConfig):
        super().__init__(extractor, config.cache_size, config.cache_ttl_seconds)
        self.config = config
        self.adaptive_ttl_manager = AdaptiveTTLManager(config.cache_ttl_seconds)
        self.metrics_collector = CacheMetricsCollector()
        self.key_generator = IntelligentCacheKeyGenerator()
    
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features with intelligent caching."""
        start_time = time.time()
        cache_hit = False
        error_occurred = False
        
        try:
            # Generate intelligent cache key
            cache_key = self.key_generator.generate_feature_key(data, self.config)
            
            # Record access for adaptive TTL
            self.adaptive_ttl_manager.record_access(cache_key)
            
            # Check cache first
            if cache_key in self.cache:
                cache_hit = True
                self.metrics_collector.record_hit('memory')
                result = self.cache[cache_key]
            else:
                # Extract features and cache result
                result = self.extractor.extract_features(data)
                self.cache[cache_key] = result
                self.metrics_collector.record_miss()
                
            return result
            
        except Exception as e:
            error_occurred = True
            self.metrics_collector.record_error(str(e))
            raise FeatureExtractionError(f"Cached feature extraction failed: {e}") from e
            
        finally:
            execution_time = time.time() - start_time
            self.metrics_collector.record_execution_time(execution_time, cache_hit, error_occurred)
```

### 1.2 Feature Extractor Factory Integration

```python
# src/ml/feature_extraction/factory.py
class FeatureExtractorFactory:
    @staticmethod
    def create_extractor(hybrid_model, config: Optional[FeatureExtractionConfig] = None) -> FeatureExtractor:
        if config is None:
            config = FeatureExtractionConfig()
        
        # Create base CNN+LSTM extractor
        base_extractor = CNNLSTMExtractor(hybrid_model)
        
        # Add intelligent caching if enabled
        if config.enable_caching:
            base_extractor = IntelligentCachedFeatureExtractor(
                base_extractor,
                config
            )
        
        # Add fallback if enabled
        if config.enable_fallback:
            base_extractor = FallbackFeatureExtractor(base_extractor)
        
        return base_extractor
```

## 2. Model Serving Integration

### 2.1 Enhanced CNNLSTMPredictor

```python
# Enhanced Ray Serve deployment with intelligent caching
@serve.deployment(
    name="intelligent_cnn_lstm_predictor",
    autoscaling_config=CNN_LSTM_AUTOSCALING_CONFIG
)
class IntelligentCNNLSTMPredictor:
    """Enhanced CNN+LSTM predictor with intelligent caching."""
    
    def __init__(self, model_path: str = None):
        # Initialize model
        if model_path:
            self.model = CNNLSTMHybridModel.load_from_path(model_path)
        else:
            self.model = self._load_default_model()
        
        self.model.eval()
        
        # Initialize intelligent caching
        self.cache_manager = MultiLevelCacheManager(
            CacheConfig(
                memory_cache_size=5000,
                memory_cache_ttl=60,
                redis_host="localhost",
                redis_port=6379,
                redis_db=2
            )
        )
        
        self.cache_key_generator = IntelligentCacheKeyGenerator()
    
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def batch_predict(self, requests: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch prediction method with intelligent caching."""
        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Process each request with caching
        for request_data in requests:
            # Generate cache key
            cache_key = self.cache_key_generator.generate_prediction_key(
                request_data, self.model.version
            )
            
            # Try to get from cache
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                results.append(cached_result)
                cache_hits += 1
            else:
                # Perform inference
                input_tensor = torch.FloatTensor(request_data)
                with torch.no_grad():
                    predictions = self.model.forward(
                        input_tensor,
                        return_features=True,
                        use_ensemble=True
                    )
                
                # Process result
                result = self._process_model_output(predictions)
                results.append(result)
                
                # Cache the result
                await self.cache_manager.set(cache_key, result, ttl=30)
                cache_misses += 1
        
        return results
```

## 3. Data Pipeline Integration

### 3.1 Market Data Cache Integration

```python
# src/ml/data_cache.py
class MarketDataCache:
    """Enhanced market data cache with intelligent caching."""
    
    def __init__(self, data: pd.DataFrame, symbols: List[str]):
        self.data = data
        self.symbols = symbols
        self.feature_cache = TTLCache(maxsize=10000, ttl=60)
        self.price_cache = TTLCache(maxsize=5000, ttl=30)
        self._build_indices()
    
    def get_features_window(self, end_step: int, window_size: int, feature_columns: List[str]) -> np.ndarray:
        """Get feature window with intelligent caching."""
        cache_key = f"features_window:{end_step}:{window_size}:{hash(str(feature_columns))}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Compute features
        features = self._compute_features_window(end_step, window_size, feature_columns)
        self.feature_cache[cache_key] = features
        return features
```

## 4. Redis Cache Integration

### 4.1 Enhanced Redis Cache

```python
# src/repositories/redis_cache.py
class EnhancedRedisCache(RedisCache):
    """Enhanced Redis cache with intelligent caching strategies."""
    
    async def get_with_ttl(self, key: str) -> Tuple[Optional[Any], Optional[int]]:
        """Get cache value with remaining TTL."""
        try:
            async with self.get_client() as client:
                # Get value and TTL in a pipeline
                pipe = client.pipeline()
                pipe.get(key)
                pipe.ttl(key)
                results = await pipe.execute()
                
                data, ttl = results
                if data is None:
                    return None, None
                
                return self._deserialize_data(data), ttl if ttl >= 0 else None
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key} with TTL: {e}")
            return None, None
    
    async def set_with_adaptive_ttl(self, key: str, value: Any, base_ttl: int, 
                                  access_pattern: str = "normal") -> bool:
        """Set cache value with adaptive TTL based on access patterns."""
        # Adjust TTL based on access pattern
        if access_pattern == "frequent":
            ttl = max(30, base_ttl // 2)
        elif access_pattern == "infrequent":
            ttl = min(3600, base_ttl * 2)
        else:
            ttl = base_ttl
        
        return await self.set(key, value, ttl)
```

## 5. Integration Testing Strategy

### 5.1 Cache Hit Rate Testing

```python
# tests/test_cache_integration.py
class TestCacheIntegration:
    def test_feature_extraction_cache_hit_rate(self):
        """Test cache hit rate for feature extraction."""
        # Create feature extractor with caching
        config = FeatureExtractionConfig(
            cache_size=1000,
            cache_ttl_seconds=60,
            enable_caching=True
        )
        
        # Process same data multiple times
        data = np.random.rand(10, 50, 60)
        extractor = FeatureExtractorFactory.create_extractor(mock_model, config)
        
        # First call (cache miss)
        result1 = extractor.extract_features(data)
        
        # Second call (cache hit)
        result2 = extractor.extract_features(data)
        
        # Verify results are identical
        assert np.array_equal(result1['fused_features'], result2['fused_features'])
    
    def test_prediction_caching(self):
        """Test prediction caching in model serving."""
        # Test that repeated predictions use cache
        # Implementation would involve mocking the model and cache
        pass
```

## 6. Monitoring and Metrics Integration

### 6.1 Cache Metrics Collection

```python
# src/ml/feature_extraction/metrics.py
class CacheMetricsCollector:
    """Collects cache metrics for monitoring."""
    
    def __init__(self):
        self.hits = Counter()
        self.misses = Counter()
        self.errors = Counter()
    
    def record_hit(self, cache_layer: str = 'memory') -> None:
        """Record cache hit."""
        self.hits[cache_layer] += 1
    
    def record_miss(self, cache_layer: str = 'memory') -> None:
        """Record cache miss."""
        self.misses[cache_layer] += 1
    
    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_hits = sum(self.hits.values())
        total_requests = total_hits + sum(self.misses.values())
        return total_hits / total_requests if total_requests > 0 else 0.0
```

## 7. Deployment Integration

### 7.1 Kubernetes Configuration

```yaml
# k8s/cache-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
    spec:
      containers:
      - name: redis
        image: redis:6.2-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: REDIS_MAXMEMORY
          value: "256mb"
        - name: REDIS_MAXMEMORY_POLICY
          value: "allkeys-lru"
```

## 8. Rollout Strategy

### 8.1 Phased Deployment

1. **Phase 1**: Deploy enhanced caching to staging environment
2. **Phase 2**: Gradual rollout to production with 5% traffic
3. **Phase 3**: Increase traffic to 25%, then 50%, then 100%
4. **Phase 4**: Monitor performance and adjust configurations

### 8.2 Rollback Plan

1. **Health Checks**: Monitor cache hit rates, latency, and error rates
2. **Alerts**: Set up alerts for performance degradation
3. **Rollback**: Quick rollback to previous caching implementation if issues detected