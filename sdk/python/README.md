# AI Trading Platform Python SDK

The official Python SDK for the AI Trading Platform, providing comprehensive access to trading signals, portfolio management, risk analysis, and real-time market data.

## Features

- **Async/Await Support**: Full async support for high-performance applications
- **Multiple Authentication Methods**: API keys, OAuth2, username/password
- **Real-time Data**: WebSocket streaming for live market data and signals
- **Plugin Architecture**: Extensible system for custom strategies and indicators
- **Webhook Integration**: Event-driven integrations with external systems
- **Type Safety**: Full type hints and Pydantic models
- **Error Handling**: Comprehensive exception hierarchy
- **Rate Limiting**: Built-in rate limit handling and retry logic

## Installation

```bash
pip install ai-trading-platform-sdk
```

### Development Installation

```bash
git clone https://github.com/ai-trading-platform/python-sdk.git
cd python-sdk
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from ai_trading_platform import TradingPlatformClient

async def main():
    async with TradingPlatformClient("https://api.tradingplatform.com") as client:
        # Authenticate
        await client.login_with_api_key("your-api-key")
        
        # Generate trading signal
        signal = await client.generate_signal("AAPL")
        print(f"Signal: {signal.action} {signal.symbol} (confidence: {signal.confidence})")
        
        # Get portfolio
        portfolio = await client.get_portfolio()
        print(f"Portfolio value: ${portfolio.total_value}")
        
        # Real-time signals
        async for signal in client.subscribe_to_signals():
            print(f"Live signal: {signal}")

asyncio.run(main())
```

## Authentication

### API Key
```python
await client.login_with_api_key("your-api-key")
```

### Username/Password
```python
await client.login("username", "password")
```

### OAuth2
```python
# Get authorization URL
auth_url = client.get_oauth_url("http://localhost:3000/callback")
print(f"Visit: {auth_url}")

# Exchange code for token
await client.login_with_oauth(authorization_code, "http://localhost:3000/callback")
```

## API Reference

### Trading Signals

```python
# Generate signal
signal = await client.generate_signal("AAPL")

# Get signal history
signals = await client.get_signal_history(
    symbol="AAPL",
    start_date=datetime(2024, 1, 1),
    limit=50
)

# Get performance metrics
performance = await client.get_signal_performance(symbol="AAPL", days=30)
```

### Portfolio Management

```python
# Get portfolio
portfolio = await client.get_portfolio()

# Rebalance portfolio
result = await client.rebalance_portfolio({
    "AAPL": 0.3,
    "GOOGL": 0.2,
    "MSFT": 0.2,
    "TSLA": 0.1,
    "CASH": 0.2
})

# Optimize allocation
optimization = await client.optimize_portfolio(
    optimization_method="mean_variance",
    risk_tolerance=0.7
)
```

### Risk Management

```python
# Get risk metrics
risk_metrics = await client.get_risk_metrics(portfolio)

# Check risk limits
alerts = await client.check_risk_limits(portfolio)
```

### Market Data

```python
# Get historical data
market_data = await client.get_market_data(
    symbol="AAPL",
    timeframe="1h",
    limit=100
)
```

## WebSocket Streaming

### Real-time Signals
```python
async for signal in client.subscribe_to_signals():
    print(f"New signal: {signal}")
```

### Real-time Market Data
```python
symbols = ["AAPL", "GOOGL", "MSFT"]
async for data in client.subscribe_to_market_data(symbols):
    print(f"Market update: {data}")
```

## Plugin System

### Custom Trading Strategy

```python
from ai_trading_platform.plugins import TradingStrategy

class MyStrategy(TradingStrategy):
    async def initialize(self):
        self.lookback_period = self.config.get("lookback_period", 20)
    
    async def generate_signal(self, market_data, portfolio):
        # Your strategy logic here
        pass
    
    async def validate_signal(self, signal):
        return signal.confidence > 0.5

# Load and use strategy
plugin_manager = PluginManager()
await plugin_manager.load_plugin(MyStrategy, {"lookback_period": 30})
signal = await plugin_manager.execute_strategy("MyStrategy", market_data, portfolio)
```

### Custom Technical Indicator

```python
from ai_trading_platform.plugins import TechnicalIndicator

class CustomRSI(TechnicalIndicator):
    async def calculate(self, data):
        # RSI calculation logic
        return rsi_values

await plugin_manager.load_plugin(CustomRSI, {"period": 14})
rsi = await plugin_manager.calculate_indicator("CustomRSI", market_data)
```

## Webhook Integration

```python
from ai_trading_platform.webhooks import WebhookManager, WebhookEndpoint

webhook_manager = WebhookManager()
await webhook_manager.start()

# Add webhook endpoint
endpoint = WebhookEndpoint(
    url="https://your-app.com/webhooks",
    secret="your-secret",
    events=["signal_generated", "portfolio_updated"]
)

webhook_manager.add_endpoint("my-app", endpoint)
```

## Error Handling

```python
from ai_trading_platform.exceptions import (
    AuthenticationError,
    RateLimitError,
    ValidationError
)

try:
    signal = await client.generate_signal("AAPL")
except AuthenticationError:
    print("Authentication failed")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation errors: {e.validation_errors}")
```

## Examples

See the [examples](examples/) directory for complete examples:

- [Trading Bot](examples/trading_bot.py)
- [Portfolio Rebalancer](examples/portfolio_rebalancer.py)
- [Risk Monitor](examples/risk_monitor.py)
- [Custom Strategy](examples/custom_strategy.py)
- [Webhook Handler](examples/webhook_handler.py)

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=ai_trading_platform --cov-report=html

# Run specific test
pytest tests/test_client.py::test_generate_signal
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://docs.tradingplatform.com/sdk/python
- GitHub Issues: https://github.com/ai-trading-platform/python-sdk/issues
- Discord: https://discord.gg/trading-platform
- Email: support@tradingplatform.com