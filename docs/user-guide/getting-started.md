# Getting Started with AI Trading Platform

Welcome to the AI Trading Platform! This guide will help you get up and running quickly with our AI-powered trading system.

## What is AI Trading Platform?

The AI Trading Platform is a comprehensive solution that combines machine learning with algorithmic trading to provide intelligent trading signals and portfolio management across multiple asset classes including stocks, forex, and cryptocurrencies.

### Key Features

- **Multi-Exchange Support**: Trade across Robinhood (stocks/ETFs), OANDA (forex), and Coinbase (crypto)
- **AI-Powered Signals**: CNN+LSTM hybrid models with reinforcement learning
- **Real-Time Processing**: Live market data aggregation and feature extraction
- **Risk Management**: Automated position sizing and portfolio optimization
- **Interpretable AI**: Understand why the AI made specific recommendations

## Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-trading-platform.git
cd ai-trading-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (use any text editor)
nano .env
```

**Minimal .env configuration:**
```bash
# Database (use SQLite for quick start)
DATABASE_URL=sqlite:///./trading.db

# Redis (optional for quick start)
REDIS_URL=redis://localhost:6379/0

# API Configuration
SECRET_KEY=your-secret-key-here
DEBUG=true

# Paper Trading Mode (SAFE - no real money)
PAPER_TRADING=true
```

### 3. Initialize Database

```bash
# Create database tables
python -c "
from src.config.database import create_tables
create_tables()
print('Database initialized successfully!')
"
```

### 4. Start the Platform

```bash
# Start the API server
python src/main.py
```

The platform will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Get Your First Signal

```bash
# Get a trading signal (using demo data)
curl "http://localhost:8000/api/v1/signals?symbol=AAPL&limit=1"
```

**Expected Response:**
```json
{
  "signals": [
    {
      "id": "abc123-def456",
      "symbol": "AAPL",
      "signal_type": "buy",
      "confidence": 0.78,
      "price_target": 152.30,
      "reasoning": "Strong momentum with CNN confidence 0.82"
    }
  ]
}
```

🎉 **Congratulations!** You now have the AI Trading Platform running locally.

## Understanding Your First Signal

Let's break down what you just received:

- **Signal Type**: `buy` - The AI recommends buying this stock
- **Confidence**: `0.78` - The AI is 78% confident in this recommendation
- **Price Target**: `$152.30` - The AI's predicted target price
- **Reasoning**: Brief explanation of why this signal was generated

### Getting Signal Explanations

```bash
# Get detailed explanation for a signal
curl "http://localhost:8000/api/v1/signals/abc123-def456/explanation"
```

This will show you:
- Which features influenced the decision
- How confident the model is
- What patterns it detected

## Next Steps

### 1. Set Up Exchange Connections

To get real market data and signals, configure exchange connections:

#### Robinhood (Stocks/ETFs)
```bash
# Add to your .env file
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password
ROBINHOOD_PAPER_TRADING=true  # Keep this true for safety
```

#### OANDA (Forex)
```bash
# Add to your .env file
OANDA_API_KEY=your_api_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # Use practice environment
```

#### Coinbase (Crypto)
```bash
# Add to your .env file
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase
COINBASE_SANDBOX=true  # Use sandbox for testing
```

### 2. Explore the Web Dashboard

Start the frontend dashboard:

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000 to access:
- Real-time trading signals
- Portfolio overview
- Risk monitoring
- AI explanations

### 3. Train Your Own Models

```bash
# Start model training (this may take several hours)
python scripts/train_models.py --symbols AAPL,GOOGL,MSFT --days 365
```

### 4. Set Up Monitoring

```bash
# Enable system monitoring
python scripts/setup_monitoring.py
```

## Core Concepts

### Trading Signals

Signals are AI-generated recommendations with:
- **Direction**: Buy, Sell, or Hold
- **Confidence**: How sure the AI is (0-100%)
- **Price Target**: Expected price movement
- **Risk Assessment**: Potential downside
- **Expiration**: When the signal expires

### Portfolio Management

The platform automatically:
- Calculates optimal position sizes
- Manages risk exposure
- Rebalances portfolios
- Monitors performance

### Risk Management

Built-in safety features:
- **Paper Trading**: Test without real money
- **Position Limits**: Maximum position sizes
- **Stop Losses**: Automatic loss protection
- **Drawdown Limits**: Portfolio protection

## Safety First

### Paper Trading Mode

**Always start in paper trading mode:**

```bash
# Ensure paper trading is enabled
PAPER_TRADING=true
ROBINHOOD_PAPER_TRADING=true
OANDA_ENVIRONMENT=practice
COINBASE_SANDBOX=true
```

### Risk Limits

Set conservative risk limits:

```yaml
# In config/settings.yaml
risk_management:
  max_position_size: 0.05  # 5% max per position
  max_daily_loss: 0.02     # 2% max daily loss
  max_drawdown: 0.10       # 10% max drawdown
```

### Gradual Scaling

1. **Week 1**: Paper trading only
2. **Week 2**: Small real positions (1% of portfolio)
3. **Month 1**: Gradually increase if performance is good
4. **Never**: Risk more than you can afford to lose

## Common Use Cases

### 1. Signal Generation

```python
from src.api.client import TradingClient

client = TradingClient(api_key="your_key")

# Get signals for specific symbols
signals = client.get_signals(symbols=["AAPL", "GOOGL", "TSLA"])

for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type} "
          f"(confidence: {signal.confidence:.1%})")
```

### 2. Portfolio Monitoring

```python
# Get portfolio status
portfolio = client.get_portfolio()

print(f"Total Value: ${portfolio.total_value:,.2f}")
print(f"Total Return: {portfolio.performance.total_return:.1%}")
print(f"Sharpe Ratio: {portfolio.performance.sharpe_ratio:.2f}")
```

### 3. Risk Assessment

```python
# Get risk metrics
risk_metrics = client.get_risk_metrics()

print(f"Portfolio VaR: ${risk_metrics.portfolio_var:,.2f}")
print(f"Max Drawdown: {risk_metrics.max_drawdown:.1%}")

# Check for alerts
if risk_metrics.alerts:
    for alert in risk_metrics.alerts:
        print(f"⚠️ {alert.message}")
```

### 4. Model Explanations

```python
# Understand why AI made a decision
explanation = client.explain_signal("signal_id")

print(f"Top factors supporting {explanation.signal_type}:")
for factor in explanation.positive_factors[:3]:
    print(f"  • {factor.name}: {factor.impact:+.3f}")
```

## Configuration Guide

### Environment Variables

Key configuration options:

```bash
# Application Settings
DEBUG=false                    # Enable debug mode
LOG_LEVEL=INFO                # Logging level
API_PORT=8000                 # API server port

# Database
DATABASE_URL=postgresql://user:pass@localhost/db
DATABASE_POOL_SIZE=10         # Connection pool size

# Redis Cache
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20      # Max connections

# Machine Learning
ML_DEVICE=auto                # auto, cpu, cuda
ML_BATCH_SIZE=32             # Training batch size
ML_MODEL_CACHE_SIZE=1000     # Model cache size

# Risk Management
MAX_POSITION_SIZE=0.20        # 20% max position
MAX_DAILY_LOSS=0.05          # 5% max daily loss
STOP_LOSS_PCT=0.10           # 10% stop loss

# Paper Trading (IMPORTANT!)
PAPER_TRADING=true           # Enable paper trading
```

### Configuration Files

Create environment-specific configs:

**config/development.yaml**
```yaml
api:
  debug: true
  cors_origins: ["http://localhost:3000"]

database:
  echo: true  # Log SQL queries

exchanges:
  robinhood:
    paper_trading: true
  oanda:
    environment: "practice"
  coinbase:
    sandbox: true
```

**config/production.yaml**
```yaml
api:
  debug: false
  cors_origins: ["https://yourdomain.com"]

database:
  echo: false
  pool_size: 20

logging:
  level: "INFO"
  format: "json"
```

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError"
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

#### Issue: "Database connection failed"
```bash
# Solution: Check database configuration
echo $DATABASE_URL
# For SQLite (simplest):
export DATABASE_URL="sqlite:///./trading.db"
```

#### Issue: "No trading signals generated"
```bash
# Solution: Check if exchanges are configured
python -c "
from src.exchanges import get_available_exchanges
print('Available exchanges:', get_available_exchanges())
"
```

#### Issue: "High memory usage"
```bash
# Solution: Reduce batch sizes and cache sizes
export ML_BATCH_SIZE=16
export ML_MODEL_CACHE_SIZE=500
```

### Getting Help

1. **Check Logs**: Look in `logs/` directory
2. **Health Check**: Visit http://localhost:8000/health
3. **Documentation**: Visit http://localhost:8000/docs
4. **Community**: Join our [Discord](https://discord.gg/ai-trading-platform)
5. **Issues**: Report bugs on [GitHub](https://github.com/your-org/ai-trading-platform/issues)

## Learning Resources

### Tutorials

1. **[Model Training Guide](../ml/training-guide.md)**: Learn to train your own models
2. **[API Documentation](../api/README.md)**: Complete API reference
3. **[Interpretability Guide](./interpretability.md)**: Understand AI decisions
4. **[Risk Management](./risk-management.md)**: Manage trading risks

### Example Code

Check the `examples/` directory for:
- Basic signal generation
- Portfolio management
- Model training
- Risk assessment
- Custom strategies

### Video Tutorials

- **Getting Started** (10 min): Basic setup and first signal
- **Model Training** (30 min): Train your own AI models
- **Risk Management** (15 min): Set up safety controls
- **Advanced Features** (45 min): Custom strategies and integrations

## Best Practices

### 1. Start Small

- Begin with paper trading
- Use small position sizes
- Test thoroughly before scaling

### 2. Diversify

- Don't rely on a single signal
- Use multiple timeframes
- Diversify across asset classes

### 3. Monitor Performance

- Track key metrics daily
- Set up alerts for issues
- Review and adjust regularly

### 4. Stay Informed

- Understand market conditions
- Keep models updated
- Follow platform updates

### 5. Risk Management

- Never risk more than you can afford to lose
- Use stop losses
- Maintain emergency funds

## What's Next?

Now that you have the basics working:

1. **Explore the Dashboard**: Visual interface for all features
2. **Train Custom Models**: Adapt AI to your trading style
3. **Set Up Monitoring**: Get alerts for important events
4. **Join the Community**: Connect with other users
5. **Contribute**: Help improve the platform

### Advanced Features to Explore

- **Custom Strategies**: Build your own trading algorithms
- **Backtesting**: Test strategies on historical data
- **A/B Testing**: Compare different models
- **API Integration**: Connect to external tools
- **Webhooks**: Automate responses to signals

Welcome to the future of algorithmic trading! 🚀

For detailed guides on specific topics, see:
- [Local Development Setup](../setup/local-development.md)
- [Cloud Deployment](../deployment/cloud-deployment.md)
- [Troubleshooting Guide](../setup/troubleshooting.md)