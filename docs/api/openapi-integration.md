# OpenAPI/Swagger Integration

The AI Trading Platform provides comprehensive API documentation through OpenAPI 3.0 specification with interactive Swagger UI.

## Accessing API Documentation

### Interactive Swagger UI
Once the platform is running, access the interactive API documentation at:
- **Local Development**: `http://localhost:8000/docs`
- **Production**: `https://your-domain.com/docs`

### OpenAPI Specification
The raw OpenAPI specification is available at:
- **JSON Format**: `http://localhost:8000/openapi.json`
- **YAML Format**: Available through the `/docs` endpoint

## API Overview

The platform exposes several API groups:

### Authentication Endpoints
- `POST /auth/login` - User authentication
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Trading Endpoints
- `GET /trading/signals` - Get trading signals
- `POST /trading/signals/generate` - Generate new signals
- `GET /trading/positions` - Get current positions
- `POST /trading/orders` - Place trading orders

### Portfolio Endpoints
- `GET /portfolio/overview` - Portfolio summary
- `GET /portfolio/performance` - Performance metrics
- `POST /portfolio/rebalance` - Trigger rebalancing

### Risk Management Endpoints
- `GET /risk/metrics` - Current risk metrics
- `GET /risk/limits` - Risk limit configuration
- `POST /risk/limits` - Update risk limits

### Model Serving Endpoints
- `POST /models/predict` - Get model predictions
- `GET /models/status` - Model health status
- `POST /models/explain` - Get model explanations

### Monitoring Endpoints
- `GET /monitoring/health` - System health check
- `GET /monitoring/metrics` - Performance metrics
- `GET /monitoring/alerts` - Active alerts

## Authentication

All API endpoints (except health checks) require JWT authentication:

```bash
# Get access token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Use token in subsequent requests
curl -X GET "http://localhost:8000/trading/signals" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Rate Limiting

API endpoints are rate-limited based on user tier:

- **Free Tier**: 100 requests per hour
- **Premium Tier**: 1000 requests per hour
- **Enterprise Tier**: Unlimited

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Window reset time

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "symbol",
      "issue": "Symbol 'INVALID' not supported"
    }
  }
}
```

Common error codes:
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

## WebSocket API

Real-time data is available through WebSocket connections:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market-data');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Market data update:', data);
};
```

WebSocket endpoints:
- `/ws/market-data` - Real-time market data
- `/ws/signals` - Trading signal updates
- `/ws/portfolio` - Portfolio updates
- `/ws/alerts` - Risk and system alerts

## SDK Integration

Official SDKs are available for popular languages:

### Python SDK
```python
from ai_trading_platform import TradingClient

client = TradingClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

signals = client.get_trading_signals()
```

### JavaScript/TypeScript SDK
```typescript
import { TradingClient } from '@ai-trading-platform/sdk';

const client = new TradingClient({
  apiKey: 'your_api_key',
  baseUrl: 'http://localhost:8000'
});

const signals = await client.getTradingSignals();
```

## API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Base URL: `http://localhost:8000/api/v1/`

Version compatibility:
- `v1`: Current stable version
- `v2`: Beta version (when available)

## Testing with Swagger UI

The interactive Swagger UI allows you to:

1. **Explore Endpoints**: Browse all available endpoints with descriptions
2. **Try It Out**: Execute API calls directly from the browser
3. **View Schemas**: Inspect request/response data models
4. **Authentication**: Set up JWT tokens for authenticated requests

### Setting up Authentication in Swagger UI

1. Click the "Authorize" button in Swagger UI
2. Enter your JWT token in the format: `Bearer YOUR_TOKEN`
3. Click "Authorize" to apply to all requests

## Custom Documentation

To extend the API documentation:

1. **Add Endpoint Descriptions**: Use FastAPI's `description` parameter
2. **Document Parameters**: Add detailed parameter descriptions
3. **Include Examples**: Provide request/response examples
4. **Tag Endpoints**: Group related endpoints with tags

Example endpoint documentation:

```python
@router.post(
    "/signals/generate",
    response_model=TradingSignalResponse,
    summary="Generate Trading Signals",
    description="Generate new trading signals using the latest market data and AI models",
    tags=["Trading"],
    responses={
        200: {"description": "Signals generated successfully"},
        400: {"description": "Invalid request parameters"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def generate_signals(
    request: GenerateSignalsRequest = Body(
        ...,
        example={
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "timeframe": "1h",
            "strategy": "hybrid_cnn_lstm"
        }
    )
):
    # Implementation
    pass
```

This ensures comprehensive, interactive API documentation that helps developers integrate with the platform effectively.