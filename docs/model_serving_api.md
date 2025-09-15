# Model Serving API Documentation

The Model Serving API provides a robust, scalable infrastructure for serving machine learning models with caching, batch processing, and A/B testing capabilities.

## Features

- **FastAPI-based REST API** with automatic OpenAPI documentation
- **Model Caching** with TTL and LRU eviction policies
- **Batch Inference Optimization** for improved throughput
- **A/B Testing Framework** for model comparison
- **Real-time Metrics** and monitoring
- **Uncertainty Quantification** for prediction confidence
- **Ensemble Model Support** with dynamic weight adjustment

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m src.api.app

# Or use uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```

### 2. Load a Model

```bash
curl -X POST "http://localhost:8080/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn_lstm_hybrid",
    "version": "v1.0",
    "file_path": "/path/to/model.pth",
    "config": {
      "input_dim": 50,
      "sequence_length": 60,
      "num_classes": 3,
      "device": "cpu"
    }
  }'
```

### 3. Make Predictions

```bash
curl -X POST "http://localhost:8080/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn_lstm_hybrid",
    "model_version": "v1.0",
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "return_uncertainty": true,
    "use_ensemble": true
  }'
```

## API Endpoints

### Core Prediction Endpoints

#### POST `/api/v1/predict`
Make a single prediction.

**Request Body:**
```json
{
  "model_type": "cnn_lstm_hybrid",
  "model_version": "latest",
  "data": [[1.0, 2.0, 3.0]],
  "return_uncertainty": true,
  "use_ensemble": true
}
```

**Response:**
```json
{
  "request_id": "uuid-string",
  "model_type": "cnn_lstm_hybrid",
  "model_version": "v1.0",
  "predictions": {
    "classification_probs": [[0.2, 0.3, 0.5]],
    "regression_pred": [[1.5]]
  },
  "uncertainty": {
    "regression_uncertainty": [[0.1]],
    "ensemble_weights": [0.2, 0.3, 0.5]
  },
  "confidence_scores": [0.5],
  "processing_time_ms": 45.2,
  "timestamp": "2024-01-01T12:00:00Z",
  "ab_test_group": null
}
```

#### POST `/api/v1/predict/batch`
Make batch predictions for improved throughput.

**Request Body:**
```json
{
  "requests": [
    {
      "model_type": "cnn_lstm_hybrid",
      "data": [[1.0, 2.0, 3.0]]
    },
    {
      "model_type": "cnn_lstm_hybrid", 
      "data": [[4.0, 5.0, 6.0]]
    }
  ],
  "priority": 1
}
```

### Model Management Endpoints

#### POST `/api/v1/models/load`
Load a model into the serving cache.

#### GET `/api/v1/models/cache/stats`
Get model cache statistics.

#### GET `/api/v1/models/types`
Get available model types and their configurations.

#### POST `/api/v1/models/warmup`
Warm up models by preloading them into cache.

### A/B Testing Endpoints

#### POST `/api/v1/experiments/create`
Create an A/B testing experiment.

**Request Body:**
```json
{
  "experiment_id": "model_comparison_v1",
  "model_variants": {
    "control": {"model_type": "cnn_lstm_hybrid", "version": "v1.0"},
    "treatment": {"model_type": "cnn_lstm_hybrid", "version": "v1.1"}
  },
  "traffic_split": {
    "control": 0.5,
    "treatment": 0.5
  },
  "duration_hours": 24
}
```

#### GET `/api/v1/experiments/{experiment_id}/results`
Get results for an A/B testing experiment.

#### GET `/api/v1/experiments`
List all A/B testing experiments.

#### DELETE `/api/v1/experiments/{experiment_id}`
Stop an A/B testing experiment.

### Monitoring Endpoints

#### GET `/api/v1/health`
Health check endpoint.

#### GET `/api/v1/metrics`
Get serving metrics and performance statistics.

## Model Types

### CNN+LSTM Hybrid Model

The CNN+LSTM hybrid model combines convolutional neural networks for spatial feature extraction with LSTM networks for temporal processing.

**Features:**
- Multi-task learning (classification + regression)
- Uncertainty quantification with Monte Carlo dropout
- Ensemble capabilities with learnable weights
- Attention mechanisms for interpretability

**Input Format:** 3D tensor `(batch_size, channels, sequence_length)`
**Output Format:** Classification probabilities and regression predictions

### RL Ensemble Model

The RL ensemble combines multiple reinforcement learning agents for trading decisions.

**Features:**
- Multiple RL algorithms (PPO, SAC, TD3, DQN)
- Dynamic weight adjustment
- Confidence scoring
- Thompson sampling for exploration

**Input Format:** 2D tensor `(batch_size, features)`
**Output Format:** Action probabilities and values

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_DEBUG=true

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model Paths
MODEL_REGISTRY_PATH=models/
```

### Configuration File

Create `config/model_serving.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8080
  debug: true

redis:
  host: "localhost"
  port: 6379
  db: 0

model_cache:
  max_models: 10
  ttl_hours: 24

batch_processing:
  max_batch_size: 100
  batch_timeout_ms: 100
```

## Caching Strategy

### Model Cache

- **LRU Eviction:** Least recently used models are evicted when cache is full
- **TTL Expiration:** Models expire after configured time-to-live
- **Usage Tracking:** Track model usage for optimization
- **Metadata Storage:** Store model configuration and performance metrics

### Cache Configuration

```python
model_cache:
  max_models: 10        # Maximum models in cache
  ttl_hours: 24         # Time-to-live in hours
  enable_metrics: true  # Track cache performance
```

## Batch Processing

### Optimization Strategies

1. **Request Grouping:** Group requests by model type and version
2. **Batch Size Optimization:** Automatically determine optimal batch sizes
3. **Queue Management:** Priority-based request queuing
4. **Parallel Processing:** Concurrent batch processing

### Configuration

```python
batch_processing:
  max_batch_size: 100      # Maximum requests per batch
  batch_timeout_ms: 100    # Maximum wait time for batching
  max_queue_size: 1000     # Maximum queued requests
  worker_threads: 4        # Number of batch processing workers
```

## A/B Testing

### Creating Experiments

```python
experiment = {
    "experiment_id": "unique_experiment_id",
    "model_variants": {
        "control": {"model_type": "cnn_lstm_hybrid", "version": "v1.0"},
        "treatment": {"model_type": "cnn_lstm_hybrid", "version": "v1.1"}
    },
    "traffic_split": {"control": 0.5, "treatment": 0.5},
    "duration_hours": 24
}
```

### Traffic Splitting

- **Consistent Assignment:** Same request ID always gets same variant
- **Hash-based Distribution:** Uses MD5 hash for deterministic assignment
- **Configurable Split:** Support for any traffic split ratio

### Metrics Collection

- **Request Counts:** Total requests per variant
- **Error Rates:** Error percentage per variant
- **Latency Metrics:** Average, P95, P99 latency per variant
- **Statistical Significance:** Automatic significance testing

## Monitoring and Metrics

### Key Metrics

- `prediction_requests_total`: Total prediction requests
- `prediction_latency_seconds`: Request latency histogram
- `prediction_errors_total`: Total prediction errors
- `model_cache_hits_total`: Cache hit count
- `model_cache_misses_total`: Cache miss count
- `ab_test_requests_total`: A/B test request count

### Health Checks

- **Redis Connectivity:** Check Redis connection
- **Model Cache Status:** Verify cache functionality
- **Batch Processor Status:** Monitor background workers

### Alerting

Configure alerts for:
- High error rates (>5%)
- High latency (>1000ms P95)
- Cache miss rate (>50%)
- Failed health checks

## Performance Optimization

### Latency Optimization

1. **Model Caching:** Keep frequently used models in memory
2. **Batch Processing:** Group requests for better throughput
3. **Connection Pooling:** Reuse database/Redis connections
4. **Async Processing:** Use async/await for I/O operations

### Throughput Optimization

1. **Horizontal Scaling:** Deploy multiple API instances
2. **Load Balancing:** Distribute requests across instances
3. **GPU Utilization:** Use GPU acceleration for model inference
4. **Request Queuing:** Queue and batch requests efficiently

### Memory Management

1. **Model Eviction:** Remove unused models from cache
2. **Garbage Collection:** Proper cleanup of temporary objects
3. **Memory Monitoring:** Track memory usage and leaks
4. **Resource Limits:** Set memory limits for containers

## Security

### Authentication

- JWT token-based authentication
- API key authentication for service-to-service calls
- Rate limiting per user/API key

### Input Validation

- Request schema validation with Pydantic
- Input sanitization and bounds checking
- Maximum request size limits

### Network Security

- CORS configuration for web clients
- HTTPS enforcement in production
- Trusted host middleware

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY config/ config/

EXPOSE 8080
CMD ["python", "-m", "src.api.app"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving-api
  template:
    metadata:
      labels:
        app: model-serving-api
    spec:
      containers:
      - name: api
        image: trading-platform/model-serving:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Production Checklist

- [ ] Configure Redis for persistence and clustering
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Set up SSL/TLS certificates
- [ ] Configure rate limiting and DDoS protection
- [ ] Set up backup and disaster recovery
- [ ] Performance testing and capacity planning
- [ ] Security audit and penetration testing

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check file paths and permissions
   - Verify model format compatibility
   - Check available memory and disk space

2. **High Latency**
   - Check model cache hit rates
   - Monitor batch processing efficiency
   - Verify network connectivity to Redis

3. **Memory Issues**
   - Monitor model cache size
   - Check for memory leaks
   - Adjust cache TTL and size limits

4. **A/B Test Issues**
   - Verify traffic split configuration
   - Check Redis connectivity for persistence
   - Monitor experiment duration and status

### Debug Mode

Enable debug mode for detailed logging:

```bash
export API_DEBUG=true
python -m src.api.app
```

### Log Analysis

Key log patterns to monitor:
- `Model loaded successfully`: Model loading events
- `Prediction failed`: Prediction errors
- `Cache miss`: Cache performance issues
- `A/B test assignment`: Traffic splitting events

## API Reference

Complete API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI JSON: `http://localhost:8080/openapi.json`

## Examples

See `examples/model_serving_demo.py` for a complete demonstration of the API functionality.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API logs for error details
3. Verify configuration settings
4. Test with the demo script