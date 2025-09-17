# AI Trading Platform SDK Documentation

The AI Trading Platform SDK provides comprehensive access to the platform's APIs across multiple programming languages. This documentation covers installation, authentication, usage examples, and advanced features.

## Table of Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [WebSocket Streaming](#websocket-streaming)
- [Plugin System](#plugin-system)
- [Webhook Integration](#webhook-integration)
- [OAuth2 Integration](#oauth2-integration)
- [Code Generation](#code-generation)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Installation

### Python SDK

```bash
pip install ai-trading-platform-sdk
```

### JavaScript/TypeScript SDK

```bash
npm install @ai-trading-platform/sdk
```

### Generated SDKs

Use the code generation tools to create SDKs for other languages:

```bash
python sdk/codegen/generator.py openapi.yaml java output/java --package-name com.example.trading
```

## Authentication

The SDK supports multiple authentication methods:

### API Key Authentication

```python
from ai_trading_platform import TradingPlatformClient

client = TradingPlatformClient("https://api.tradingplatform.com")
await client.login_with_api_key("your-api-key")
```

```typescript
import { TradingPlatformClient } from '@ai-trading-platform/sdk';

const client = new TradingPlatformClient({
  baseUrl: 'https://api.tradingplatform.com',
  apiKey: 'your-api-key'
});
```

### Username/Password Authentication

```python
await client.login("username", "password")
```

```typescript
await client.login("username", "password");
```

### OAuth2 Authentication

```python
# Get authorization URL
auth_url = client.get_oauth_url("http://localhost:3000/callback")
print(f"Visit: {auth_url}")

# After user authorization, exchange code for token
await client.login_with_oauth(authorization_code, "http://localhost:3000/callback")
```

## Quick Start

### Generate Trading Signals

```python
import asyncio
from ai_trading_platform import TradingPlatformClient

async def main():
    async with TradingPlatformClient("https://api.tradingplatform.com") as client:
        await client.login_with_api_key("your-api-key")
        
        # Generate trading signal
        signal = await client.generate_signal("AAPL")
        print(f"Signal: {signal.action} {signal.symbol} (confidence: {signal.confidence})")
        
        # Get portfolio
        portfolio = await client.get_portfolio()
        print(f"Portfolio value: ${portfolio.total_value}")

asyncio.run(main())
```

```typescript
import { TradingPlatformClient } from '@ai-trading-platform/sdk';

async function main() {
  const client = new TradingPlatformClient({
    baseUrl: 'https://api.tradingplatform.com',
    apiKey: 'your-api-key'
  });
  
  // Generate trading signal
  const signal = await client.generateSignal('AAPL');
  console.log(`Signal: ${signal.action} ${signal.symbol} (confidence: ${signal.confidence})`);
  
  // Get portfolio
  const portfolio = await client.getPortfolio();
  console.log(`Portfolio value: $${portfolio.totalValue}`);
}

main().catch(console.error);
```

## API Reference

### Trading Signals

#### Generate Signal
```python
signal = await client.generate_signal("AAPL")
```

#### Get Signal History
```python
signals = await client.get_signal_history(
    symbol="AAPL",
    start_date=datetime(2024, 1, 1),
    limit=50
)
```

#### Get Signal Performance
```python
performance = await client.get_signal_performance(symbol="AAPL", days=30)
```

### Portfolio Management

#### Get Portfolio
```python
portfolio = await client.get_portfolio()
```

#### Rebalance Portfolio
```python
result = await client.rebalance_portfolio({
    "AAPL": 0.3,
    "GOOGL": 0.2,
    "MSFT": 0.2,
    "TSLA": 0.1,
    "CASH": 0.2
})
```

#### Optimize Portfolio
```python
optimization = await client.optimize_portfolio(
    optimization_method="mean_variance",
    risk_tolerance=0.7
)
```

### Risk Management

#### Get Risk Metrics
```python
portfolio = await client.get_portfolio()
risk_metrics = await client.get_risk_metrics(portfolio)
```

#### Check Risk Limits
```python
alerts = await client.check_risk_limits(portfolio)
```

### Market Data

#### Get Historical Data
```python
market_data = await client.get_market_data(
    symbol="AAPL",
    timeframe="1h",
    limit=100
)
```

### Monitoring

#### Get System Health
```python
health = await client.get_system_health()
```

#### Get Model Status
```python
status = await client.get_model_status("cnn_lstm_hybrid")
```

#### Get Alerts
```python
alerts = await client.get_alerts(hours=24, severity="high")
```

## WebSocket Streaming

### Real-time Trading Signals

```python
async def handle_signals():
    async with client.connect_websocket() as ws:
        async for signal in client.subscribe_to_signals():
            print(f"New signal: {signal}")

asyncio.create_task(handle_signals())
```

### Real-time Market Data

```python
async def handle_market_data():
    symbols = ["AAPL", "GOOGL", "MSFT"]
    async for data in client.subscribe_to_market_data(symbols):
        print(f"Market update: {data}")

asyncio.create_task(handle_market_data())
```

## Plugin System

### Creating Custom Trading Strategies

```python
from ai_trading_platform.plugins import TradingStrategy
import pandas as pd

class MyStrategy(TradingStrategy):
    async def initialize(self):
        self.lookback_period = self.config.get("lookback_period", 20)
    
    async def generate_signal(self, market_data, portfolio):
        # Implement your strategy logic
        df = pd.DataFrame(market_data)
        
        # Simple moving average example
        sma = df['close'].rolling(self.lookback_period).mean()
        current_price = df['close'].iloc[-1]
        
        if current_price > sma.iloc[-1] * 1.02:  # 2% above SMA
            return TradingSignal(
                symbol=df.attrs['symbol'],
                action="BUY",
                confidence=0.8,
                reasoning="Price above SMA threshold"
            )
        
        return None
    
    async def validate_signal(self, signal):
        return signal.confidence > 0.5

# Register and use the strategy
plugin_manager = PluginManager()
await plugin_manager.load_plugin(MyStrategy, {"lookback_period": 30})

# Execute strategy
signal = await plugin_manager.execute_strategy("MyStrategy", market_data, portfolio)
```

### Creating Custom Technical Indicators

```python
from ai_trading_platform.plugins import TechnicalIndicator

class CustomRSI(TechnicalIndicator):
    async def initialize(self):
        self.period = self.config.get("period", 14)
    
    async def calculate(self, data):
        df = pd.DataFrame(data)
        delta = df['close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

# Use the indicator
await plugin_manager.load_plugin(CustomRSI, {"period": 21})
rsi_values = await plugin_manager.calculate_indicator("CustomRSI", market_data)
```

## Webhook Integration

### Setting up Webhooks

```python
from ai_trading_platform.webhooks import WebhookManager, WebhookEndpoint, WebhookEventType

webhook_manager = WebhookManager()
await webhook_manager.start()

# Add webhook endpoint
endpoint = WebhookEndpoint(
    url="https://your-app.com/webhooks/trading",
    secret="your-webhook-secret",
    events=[WebhookEventType.SIGNAL_GENERATED, WebhookEventType.PORTFOLIO_UPDATED]
)

webhook_manager.add_endpoint("my-app", endpoint)

# Emit events
await webhook_manager.emit_event(WebhookEvent(
    id="signal_123",
    type=WebhookEventType.SIGNAL_GENERATED,
    timestamp=datetime.now(),
    data={"symbol": "AAPL", "action": "BUY", "confidence": 0.85}
))
```

### Receiving Webhooks

```python
from ai_trading_platform.webhooks import WebhookReceiver

receiver = WebhookReceiver(webhook_manager)

@receiver.add_handler("signal_generated")
async def handle_signal(data):
    print(f"Received signal: {data}")
    # Process the signal
    return {"status": "processed"}

# In your FastAPI app
from fastapi import FastAPI
app = FastAPI()

@app.post("/webhooks/trading")
async def receive_webhook(request: Request):
    return await receiver.process_webhook(request)
```

## OAuth2 Integration

### Setting up OAuth2 Server

```python
from ai_trading_platform.oauth import OAuthManager, OAuthClient, GrantType

oauth_manager = OAuthManager(jwt_secret="your-secret-key")

# Register OAuth2 client
client = OAuthClient(
    client_id="your-client-id",
    client_secret="your-client-secret",
    name="My Trading App",
    redirect_uris=["https://myapp.com/callback"],
    scopes=["read:signals", "read:portfolio"],
    grant_types=[GrantType.AUTHORIZATION_CODE]
)

oauth_manager.register_client(client)

# Create OAuth2 endpoints
from ai_trading_platform.oauth import create_oauth_router
oauth_app = create_oauth_router(oauth_manager)
```

### Using OAuth2 in Third-party Apps

```python
# Step 1: Redirect user to authorization URL
auth_url = f"https://api.tradingplatform.com/oauth/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=read:signals read:portfolio"

# Step 2: Exchange authorization code for access token
async with httpx.AsyncClient() as client:
    response = await client.post("https://api.tradingplatform.com/oauth/token", data={
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret
    })
    
    tokens = response.json()
    access_token = tokens["access_token"]

# Step 3: Use access token to make API calls
headers = {"Authorization": f"Bearer {access_token}"}
response = await client.get("https://api.tradingplatform.com/api/v1/trading/signals/history", headers=headers)
```

## Code Generation

### Generate SDK for Java

```bash
python sdk/codegen/generator.py \
  openapi.yaml \
  java \
  output/java-sdk \
  --package-name com.mycompany.trading \
  --config config.yaml
```

### Generate SDK for C#

```bash
python sdk/codegen/generator.py \
  openapi.yaml \
  csharp \
  output/csharp-sdk \
  --namespace MyCompany.Trading
```

### Custom Configuration

Create a `config.yaml` file:

```yaml
# Java configuration
package_name: "com.mycompany.trading"
group_id: "com.mycompany"
artifact_id: "trading-client"
version: "1.0.0"

# C# configuration
namespace: "MyCompany.Trading"
package_version: "1.0.0"

# Dependencies
dependencies:
  java:
    - "com.fasterxml.jackson.core:jackson-core:2.15.0"
    - "org.apache.httpcomponents:httpclient:4.5.14"
  csharp:
    - "Newtonsoft.Json:13.0.3"
    - "System.Net.Http:4.3.4"
```

## Examples

### Complete Trading Bot

```python
import asyncio
from datetime import datetime, timedelta
from ai_trading_platform import TradingPlatformClient
from ai_trading_platform.plugins import PluginManager, MovingAverageStrategy

class TradingBot:
    def __init__(self, api_key: str):
        self.client = TradingPlatformClient("https://api.tradingplatform.com")
        self.api_key = api_key
        self.plugin_manager = PluginManager()
        self.running = False
    
    async def initialize(self):
        await self.client.login_with_api_key(self.api_key)
        
        # Load trading strategy
        await self.plugin_manager.load_plugin(
            MovingAverageStrategy,
            {"short_window": 10, "long_window": 30}
        )
    
    async def run(self):
        self.running = True
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        while self.running:
            try:
                for symbol in symbols:
                    # Get market data
                    market_data = await self.client.get_market_data(
                        symbol=symbol,
                        timeframe="1h",
                        limit=50
                    )
                    
                    # Get current portfolio
                    portfolio = await self.client.get_portfolio()
                    
                    # Execute strategy
                    signal = await self.plugin_manager.execute_strategy(
                        "MovingAverageStrategy",
                        market_data['data'],
                        portfolio.model_dump()
                    )
                    
                    if signal:
                        print(f"Generated signal: {signal}")
                        
                        # Check risk limits
                        alerts = await self.client.check_risk_limits(portfolio)
                        if not alerts:
                            # Execute trade (this would integrate with your broker)
                            print(f"Would execute: {signal.action} {signal.symbol}")
                        else:
                            print(f"Risk alerts prevent trading: {alerts}")
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def stop(self):
        self.running = False

# Usage
async def main():
    bot = TradingBot("your-api-key")
    await bot.initialize()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()
        print("Trading bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
```

### Portfolio Rebalancing Service

```python
import asyncio
from datetime import datetime, time
from ai_trading_platform import TradingPlatformClient

class PortfolioRebalancer:
    def __init__(self, api_key: str, target_allocation: dict):
        self.client = TradingPlatformClient("https://api.tradingplatform.com")
        self.api_key = api_key
        self.target_allocation = target_allocation
        self.rebalance_time = time(9, 30)  # 9:30 AM
    
    async def initialize(self):
        await self.client.login_with_api_key(self.api_key)
    
    async def should_rebalance(self) -> bool:
        """Check if portfolio needs rebalancing"""
        portfolio = await self.client.get_portfolio()
        
        current_allocation = {}
        for symbol, position in portfolio.positions.items():
            current_allocation[symbol] = position.market_value / portfolio.total_value
        
        # Check if any allocation is off by more than 5%
        for symbol, target_weight in self.target_allocation.items():
            current_weight = current_allocation.get(symbol, 0)
            if abs(current_weight - target_weight) > 0.05:
                return True
        
        return False
    
    async def rebalance(self):
        """Perform portfolio rebalancing"""
        if await self.should_rebalance():
            print("Rebalancing portfolio...")
            
            result = await self.client.rebalance_portfolio(self.target_allocation)
            print(f"Rebalancing result: {result}")
            
            # Monitor rebalancing progress
            await self.monitor_rebalancing(result.get("rebalance_id"))
        else:
            print("Portfolio is already balanced")
    
    async def monitor_rebalancing(self, rebalance_id: str):
        """Monitor rebalancing progress"""
        # This would check the status of rebalancing trades
        # Implementation depends on your broker integration
        pass
    
    async def run_daily(self):
        """Run daily rebalancing check"""
        while True:
            now = datetime.now().time()
            
            if now >= self.rebalance_time:
                try:
                    await self.rebalance()
                except Exception as e:
                    print(f"Rebalancing error: {e}")
                
                # Wait until next day
                await asyncio.sleep(24 * 60 * 60)
            else:
                # Wait until rebalance time
                await asyncio.sleep(60)

# Usage
target_allocation = {
    "AAPL": 0.25,
    "GOOGL": 0.20,
    "MSFT": 0.20,
    "TSLA": 0.15,
    "SPY": 0.10,
    "CASH": 0.10
}

rebalancer = PortfolioRebalancer("your-api-key", target_allocation)
await rebalancer.initialize()
await rebalancer.run_daily()
```

## Error Handling

### Exception Types

```python
from ai_trading_platform.exceptions import (
    TradingPlatformError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError,
    APIError,
    NetworkError,
    TimeoutError,
    WebSocketError,
    PluginError
)

try:
    signal = await client.generate_signal("AAPL")
except AuthenticationError:
    print("Please check your API credentials")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ValidationError as e:
    print(f"Validation errors: {e.validation_errors}")
except NetworkError:
    print("Network connection issue")
except TradingPlatformError as e:
    print(f"API error: {e.message}")
```

### Retry Logic

```python
import asyncio
from ai_trading_platform.exceptions import RateLimitError, NetworkError

async def retry_request(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError as e:
            if e.retry_after:
                await asyncio.sleep(e.retry_after)
            else:
                await asyncio.sleep(base_delay * (2 ** attempt))
        except NetworkError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))

# Usage
signal = await retry_request(lambda: client.generate_signal("AAPL"))
```

## Best Practices

### 1. Use Async Context Managers

```python
async with TradingPlatformClient("https://api.tradingplatform.com") as client:
    await client.login_with_api_key("your-api-key")
    # Client will be automatically closed
```

### 2. Handle Rate Limits

```python
import asyncio
from ai_trading_platform.exceptions import RateLimitError

async def rate_limited_request(client, func, *args, **kwargs):
    while True:
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            wait_time = e.retry_after or 60
            print(f"Rate limited. Waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
```

### 3. Implement Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

### 4. Use Connection Pooling

```python
# The SDK automatically handles connection pooling
# But you can configure it:

client = TradingPlatformClient(
    "https://api.tradingplatform.com",
    timeout=30.0
)

# Reuse the same client instance across your application
```

### 5. Implement Proper Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    signal = await client.generate_signal("AAPL")
    logger.info(f"Generated signal: {signal}")
except Exception as e:
    logger.error(f"Failed to generate signal: {e}", exc_info=True)
```

### 6. Validate Data

```python
from pydantic import ValidationError as PydanticValidationError

try:
    portfolio = await client.get_portfolio()
    # Portfolio is automatically validated by Pydantic models
except PydanticValidationError as e:
    logger.error(f"Invalid portfolio data: {e}")
```

### 7. Use Environment Variables

```python
import os
from ai_trading_platform import TradingPlatformClient

client = TradingPlatformClient(
    base_url=os.getenv("TRADING_API_URL", "https://api.tradingplatform.com"),
    timeout=float(os.getenv("TRADING_API_TIMEOUT", "30.0"))
)

await client.login_with_api_key(os.getenv("TRADING_API_KEY"))
```

## Support

For support and questions:

- Documentation: https://docs.tradingplatform.com
- GitHub Issues: https://github.com/ai-trading-platform/sdk
- Discord: https://discord.gg/trading-platform
- Email: support@tradingplatform.com

## License

This SDK is licensed under the MIT License. See LICENSE file for details.