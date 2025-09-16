# Cache Configuration Specifications

## 1. Multi-Level Cache Configuration

### 1.1 Memory Cache (TTLCache)

```yaml
memory_cache:
  enabled: true
  max_size: 10000
  default_ttl_seconds: 60
  eviction_policy: "ttl"
```

### 1.2 Distributed Cache (Redis)

```yaml
distributed_cache:
  enabled: true
  host: "localhost"
  port: 6379
  db: 2
  default_ttl_seconds: 300
  connection_pool_size: 20
  max_connections: 100
```

## 2. TTL Strategies by Data Type

### 2.1 Feature Extraction Cache TTL

| Data Type | Base TTL | Adaptive Range | Notes |
|-----------|----------|----------------|-------|
| Market Features | 60 seconds | 30-120 seconds | High frequency trading data |
| Technical Indicators | 120 seconds | 60-300 seconds | Calculated indicators change slower |
| CNN+LSTM Features | 60 seconds | 30-180 seconds | Model-based features |
| Fallback Features | 300 seconds | 120-600 seconds | Basic calculations, stable |

### 2.2 Prediction Cache TTL

| Prediction Type | Base TTL | Adaptive Range | Notes |
|----------------|----------|----------------|-------|
| Real-time Trading Signals | 30 seconds | 15-60 seconds | Critical for trading decisions |
| Portfolio Recommendations | 60 seconds | 30-120 seconds | Portfolio-level decisions |
| Backtesting Results | 300 seconds | 180-600 seconds | Historical analysis |
| Model Performance Metrics | 600 seconds | 300-1800 seconds | Infrequent updates |

### 2.3 Model Cache TTL

| Model Type | Base TTL | Adaptive Range | Notes |
|------------|----------|----------------|-------|
| CNN+LSTM Models | 3600 seconds | 1800-7200 seconds | Large models, infrequent updates |
| RL Models | 1800 seconds | 900-3600 seconds | Medium models |
| Configuration | 300 seconds | 120-600 seconds | Frequently updated parameters |

## 3. Cache Size Configuration

### 3.1 Memory Cache Sizes

```python
class CacheSizeConfig:
    FEATURE_EXTRACTION_CACHE_SIZE = 10000
    PREDICTION_CACHE_SIZE = 5000
    MODEL_CACHE_SIZE = 100
    METADATA_CACHE_SIZE = 1000
```

### 3.2 Distributed Cache Sizes

```python
class DistributedCacheSizeConfig:
    FEATURE_CACHE_SIZE = 100000
    PREDICTION_CACHE_SIZE = 50000
    MODEL_CACHE_SIZE = 1000
    METADATA_CACHE_SIZE = 10000
```

## 4. Adaptive TTL Configuration

### 4.1 Frequency-Based TTL Adjustment

```python
class AdaptiveTTLConfig:
    HIGH_FREQUENCY_THRESHOLD = 10  # requests per minute
    LOW_FREQUENCY_THRESHOLD = 1    # requests per minute
    
    HIGH_FREQUENCY_TTL_MULTIPLIER = 0.5
    LOW_FREQUENCY_TTL_MULTIPLIER = 2.0
```

### 4.2 Data Type TTL Multipliers

```python
class DataTypeTTLConfig:
    FEATURES_MULTIPLIER = 1.0
    PREDICTIONS_MULTIPLIER = 1.5
    MODELS_MULTIPLIER = 3.0
```

## 5. Cache Warming Configuration

```yaml
cache_warming:
  enabled: true
  warming_interval_seconds: 300
  prefetch_depth: 5
  warming_strategies:
    - "most_recent"
    - "most_frequent"
    - "predicted_next"
```

## 6. Cache Invalidation Configuration

```yaml
cache_invalidation:
  enabled: true
  invalidation_strategies:
    - "ttl_based"
    - "content_based"
    - "event_driven"
  event_sources:
    - "market_data_updates"
    - "model_updates"
    - "system_events"