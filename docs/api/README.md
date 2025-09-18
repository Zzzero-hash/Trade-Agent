# API Documentation

The AI Trading Platform provides a comprehensive REST API and WebSocket interface for accessing trading signals, managing portfolios, and monitoring system performance.

## Quick Start

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://api.ai-trading-platform.com`

### Authentication
All API endpoints require JWT authentication. Get your token from the `/auth/login` endpoint.

```bash
# Login to get token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "http://localhost:8000/api/v1/signals"
```

### Rate Limits
- **Free Tier**: 100 requests per hour
- **Pro Tier**: 1000 requests per hour
- **Enterprise**: Custom limits

## API Reference

### Authentication Endpoints

#### POST /auth/login
Authenticate user and receive JWT token.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "uuid",
  "tier": "free|pro|enterprise"
}
```

#### POST /auth/refresh
Refresh JWT token.

**Request:**
```json
{
  "refresh_token": "string"
}
```

#### POST /auth/logout
Logout and invalidate token.

### Trading Signal Endpoints

#### GET /api/v1/signals
Get latest trading signals.

**Parameters:**
- `symbol` (optional): Filter by symbol (e.g., "AAPL", "EUR/USD", "BTC-USD")
- `exchange` (optional): Filter by exchange ("robinhood", "oanda", "coinbase")
- `limit` (optional): Number of signals to return (default: 10, max: 100)

**Response:**
```json
{
  "signals": [
    {
      "id": "uuid",
      "symbol": "AAPL",
      "exchange": "robinhood",
      "signal_type": "buy|sell|hold",
      "confidence": 0.85,
      "price_target": 150.25,
      "stop_loss": 145.00,
      "position_size": 0.05,
      "reasoning": "Strong momentum with CNN confidence 0.87",
      "timestamp": "2024-01-01T12:00:00Z",
      "expires_at": "2024-01-01T16:00:00Z"
    }
  ],
  "total": 25,
  "page": 1,
  "has_more": true
}
```

#### GET /api/v1/signals/{signal_id}
Get specific signal details.

**Response:**
```json
{
  "id": "uuid",
  "symbol": "AAPL",
  "exchange": "robinhood",
  "signal_type": "buy",
  "confidence": 0.85,
  "price_target": 150.25,
  "stop_loss": 145.00,
  "position_size": 0.05,
  "reasoning": "Strong momentum with CNN confidence 0.87",
  "timestamp": "2024-01-01T12:00:00Z",
  "expires_at": "2024-01-01T16:00:00Z",
  "model_version": "cnn-lstm-v1.2.3",
  "features_used": ["rsi", "macd", "volume_profile"],
  "uncertainty": 0.12,
  "explanation": {
    "shap_values": {...},
    "attention_weights": {...}
  }
}
```

#### POST /api/v1/signals/generate
Generate new trading signal for a symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "exchange": "robinhood",
  "timeframe": "1h|4h|1d",
  "model_version": "latest|specific_version"
}
```

### Portfolio Endpoints

#### GET /api/v1/portfolio
Get current portfolio status.

**Response:**
```json
{
  "total_value": 100000.00,
  "cash": 25000.00,
  "positions": [
    {
      "symbol": "AAPL",
      "exchange": "robinhood",
      "quantity": 100,
      "avg_price": 145.50,
      "current_price": 150.25,
      "market_value": 15025.00,
      "unrealized_pnl": 475.00,
      "unrealized_pnl_pct": 3.26,
      "weight": 0.15
    }
  ],
  "performance": {
    "total_return": 0.05,
    "sharpe_ratio": 1.25,
    "max_drawdown": -0.08,
    "volatility": 0.15
  },
  "risk_metrics": {
    "var_95": -2500.00,
    "beta": 1.1,
    "correlation_spy": 0.85
  }
}
```

#### POST /api/v1/portfolio/rebalance
Trigger portfolio rebalancing.

**Request:**
```json
{
  "strategy": "equal_weight|risk_parity|momentum",
  "constraints": {
    "max_position_size": 0.20,
    "min_cash_ratio": 0.05
  }
}
```

### Risk Management Endpoints

#### GET /api/v1/risk/metrics
Get current risk metrics.

**Response:**
```json
{
  "portfolio_var": -2500.00,
  "portfolio_cvar": -3200.00,
  "max_drawdown": -0.08,
  "sharpe_ratio": 1.25,
  "sortino_ratio": 1.45,
  "calmar_ratio": 0.95,
  "beta": 1.1,
  "correlation_matrix": {...},
  "position_limits": {
    "max_single_position": 0.20,
    "max_sector_exposure": 0.30
  },
  "alerts": [
    {
      "type": "position_limit",
      "message": "AAPL position exceeds 15% limit",
      "severity": "warning",
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### POST /api/v1/risk/limits
Update risk limits.

**Request:**
```json
{
  "max_position_size": 0.15,
  "max_daily_loss": 0.02,
  "max_drawdown": 0.10,
  "var_limit": -5000.00
}
```

### Model Serving Endpoints

#### POST /api/v1/models/predict
Get model predictions.

**Request:**
```json
{
  "model_type": "cnn_lstm|rl_ensemble",
  "symbol": "AAPL",
  "features": {
    "price_data": [...],
    "volume_data": [...],
    "technical_indicators": {...}
  },
  "model_version": "latest"
}
```

**Response:**
```json
{
  "prediction": {
    "signal": "buy|sell|hold",
    "confidence": 0.85,
    "price_target": 150.25,
    "uncertainty": 0.12
  },
  "model_info": {
    "version": "v1.2.3",
    "trained_at": "2024-01-01T00:00:00Z",
    "performance_metrics": {...}
  },
  "explanation": {
    "feature_importance": {...},
    "shap_values": {...}
  }
}
```

#### GET /api/v1/models/status
Get model serving status.

**Response:**
```json
{
  "models": [
    {
      "name": "cnn_lstm_hybrid",
      "version": "v1.2.3",
      "status": "active|loading|error",
      "last_updated": "2024-01-01T00:00:00Z",
      "cache_hit_rate": 0.85,
      "avg_inference_time": 45.2
    }
  ],
  "system_status": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "gpu_usage": 0.45
  }
}
```

### Monitoring Endpoints

#### GET /api/v1/monitoring/health
System health check.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "model_serving": "healthy",
    "data_feeds": "degraded"
  },
  "metrics": {
    "uptime": 86400,
    "request_rate": 150.5,
    "error_rate": 0.001
  }
}
```

#### GET /api/v1/monitoring/metrics
Get system metrics.

**Response:**
```json
{
  "performance": {
    "avg_response_time": 125.5,
    "p95_response_time": 250.0,
    "requests_per_second": 150.5,
    "error_rate": 0.001
  },
  "model_metrics": {
    "prediction_accuracy": 0.78,
    "model_drift_score": 0.05,
    "feature_drift_scores": {...}
  },
  "system_metrics": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "disk_usage": 0.45
  }
}
```

### Usage Tracking Endpoints

#### GET /api/v1/usage/current
Get current usage statistics.

**Response:**
```json
{
  "current_period": {
    "requests_made": 45,
    "requests_limit": 100,
    "signals_generated": 12,
    "signals_limit": 50
  },
  "billing_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "total_requests": 1250,
    "total_cost": 25.50
  },
  "tier": "free|pro|enterprise",
  "upgrade_available": true
}
```

## WebSocket API

### Connection
Connect to WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Send authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your_jwt_token'
}));
```

### Message Types

#### Subscribe to Signals
```json
{
  "type": "subscribe",
  "channel": "signals",
  "symbols": ["AAPL", "EUR/USD", "BTC-USD"]
}
```

#### Real-time Signal Updates
```json
{
  "type": "signal",
  "data": {
    "symbol": "AAPL",
    "signal_type": "buy",
    "confidence": 0.85,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### Market Data Updates
```json
{
  "type": "market_data",
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1000000,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "status_code": 400,
  "request_id": "uuid",
  "timestamp": 1704067200.0
}
```

### Common Error Codes
- **400**: Bad Request - Invalid parameters
- **401**: Unauthorized - Invalid or missing token
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource not found
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server error

## SDKs and Libraries

### Python SDK
```python
from ai_trading_platform import TradingClient

client = TradingClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Get signals
signals = client.get_signals(symbol="AAPL")

# Generate prediction
prediction = client.predict(
    symbol="AAPL",
    model_type="cnn_lstm"
)
```

### JavaScript SDK
```javascript
import { TradingClient } from '@ai-trading-platform/sdk';

const client = new TradingClient({
  apiKey: 'your_api_key',
  baseUrl: 'http://localhost:8000'
});

// Get signals
const signals = await client.getSignals({ symbol: 'AAPL' });

// Subscribe to real-time updates
client.subscribe('signals', { symbols: ['AAPL'] }, (signal) => {
  console.log('New signal:', signal);
});
```

## Testing

### Sandbox Environment
Use the sandbox environment for testing:
- **Base URL**: `https://sandbox-api.ai-trading-platform.com`
- **Paper Trading**: All trades are simulated
- **Test Data**: Historical market data for backtesting

### Postman Collection
Import our Postman collection for easy API testing:
[Download Collection](./postman/ai-trading-platform.json)

## OpenAPI/Swagger Integration

### Interactive API Documentation

The platform provides interactive API documentation using Swagger UI:

- **Local Development**: `http://localhost:8000/docs`
- **Production**: `https://api.ai-trading-platform.com/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON Format**: `/openapi.json`
- **YAML Format**: [openapi-spec.yaml](./openapi-spec.yaml)

### Code Generation

Generate client SDKs using the OpenAPI specification:

```bash
# Generate Python client
openapi-generator generate -i openapi-spec.yaml -g python -o ./clients/python

# Generate JavaScript client
openapi-generator generate -i openapi-spec.yaml -g javascript -o ./clients/javascript

# Generate TypeScript client
openapi-generator generate -i openapi-spec.yaml -g typescript-axios -o ./clients/typescript
```

### Postman Integration

Import the OpenAPI specification directly into Postman:

1. Open Postman
2. Click "Import" → "Link"
3. Enter: `http://localhost:8000/openapi.json`
4. Click "Continue" and "Import"

## Support

- **Documentation**: [docs.ai-trading-platform.com](https://docs.ai-trading-platform.com)
- **API Status**: [status.ai-trading-platform.com](https://status.ai-trading-platform.com)
- **Support Email**: api-support@ai-trading-platform.com
- **Discord**: [Join our community](https://discord.gg/ai-trading-platform)