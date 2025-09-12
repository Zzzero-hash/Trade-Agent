# AI Trading Platform

An AI-powered trading platform that combines CNN+LSTM models with reinforcement learning for intelligent trading decisions across multiple asset classes.

## Project Structure

```
ai-trading-platform/
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py           # Settings and environment configuration
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   ├── market_data.py        # Market data with validation
│   │   ├── trading_signal.py     # Trading signals and actions
│   │   └── portfolio.py          # Portfolio and position models
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   └── data_aggregator.py    # Multi-exchange data aggregation system
│   ├── repositories/             # Data access layer
│   │   └── __init__.py
│   ├── api/                      # API endpoints
│   │   └── __init__.py
│   ├── exchanges/                # Exchange integrations
│   │   ├── __init__.py
│   │   └── base.py               # Abstract exchange connector
│   ├── ml/                       # Machine learning components
│   │   ├── __init__.py
│   │   ├── base_models.py        # Abstract ML model classes
│   │   └── feature_engineering.py # Feature engineering pipeline
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logging.py            # Logging infrastructure
│       └── monitoring.py         # Monitoring and metrics
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_models.py            # Data model validation tests
│   ├── test_robinhood_connector.py # Robinhood integration tests
│   ├── test_oanda_connector.py   # OANDA forex trading tests
│   ├── test_coinbase_connector.py # Coinbase crypto trading tests
│   └── test_setup.py             # Test configuration
├── config/                       # Configuration files
│   ├── settings.yaml             # Development configuration
│   └── production.yaml           # Production configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Core Interfaces

### Exchange Connector Interface

The `ExchangeConnector` abstract base class defines the interface for all exchange integrations:

- **Robinhood**: Stocks, ETFs, indices, and options
- **OANDA**: Forex pairs and CFDs  
- **Coinbase**: Cryptocurrencies and perpetual futures

Key methods:

- `get_historical_data()`: Retrieve historical market data
- `get_real_time_data()`: Stream real-time market data
- `place_order()`: Execute trading orders
- `get_account_info()`: Account balances and information

### Feature Engineering Interface

The `FeatureTransformer` abstract base class enables pluggable feature engineering:

- **Technical Indicators**: Complete set including SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP
- **Advanced Features**: Wavelets, Fourier transforms, fractals, Hurst exponent
- **Cross-Asset Correlations**: Price-volume relationships and inter-asset dependencies
- **Market Microstructure Features**: Statistical patterns and anomaly detection

### ML Model Interface

Abstract base classes for machine learning models:

- `BaseMLModel`: General ML model interface
- `BasePyTorchModel`: PyTorch-specific implementation
- `BaseRLAgent`: Reinforcement learning agent interface

## Configuration Management

## Exchange Connectors

The platform now includes three fully implemented exchange connectors:

### Coinbase Connector ✅ **Complete**

The `CoinbaseConnector` provides comprehensive cryptocurrency trading capabilities:

**Key Features:**

- **24/7 Market Access**: Cryptocurrency markets never close
- **WebSocket Streaming**: Real-time price feeds via WebSocket connections
- **Comprehensive Order Types**: Market, limit, stop orders with crypto-specific parameters
- **Margin Trading**: Support for leveraged positions with configurable leverage ratios
- **Multi-Currency Support**: Major crypto pairs (BTC, ETH, LTC, BCH, ADA, DOT, LINK, etc.)
- **Sandbox Environment**: Full testing capabilities in Coinbase Pro sandbox

**Crypto-Specific Methods:**

- `get_current_prices()`: Real-time pricing for multiple crypto pairs
- `get_order_book()`: Market depth and liquidity information
- `get_trade_history()`: Recent trade execution data
- `place_margin_order()`: Leveraged trading with risk management
- `get_margin_profile()`: Account margin and leverage information

**Usage Example:**

```python
from src.exchanges.coinbase import CoinbaseConnector

# Initialize connector
connector = CoinbaseConnector(
    api_key="your_coinbase_key",
    api_secret="your_coinbase_secret", 
    passphrase="your_passphrase",
    sandbox=True  # Use sandbox for testing
)

# Connect and get real-time data
await connector.connect()
async for market_data in connector.get_real_time_data(["BTCUSD", "ETHUSD"]):
    print(f"{market_data.symbol}: ${market_data.close}")
```

## Data Aggregation System ✅ **Complete**

The platform now includes a sophisticated data aggregation system that normalizes and validates data from multiple exchanges in real-time.

### Key Features

**Real-time Data Aggregation:**

- Unified data streams from Robinhood, OANDA, and Coinbase
- Timestamp synchronization across exchanges
- Intelligent data quality validation and anomaly detection
- Confidence scoring for aggregated data points

**Data Quality Management:**

- **Price Validation**: Ensures proper OHLC relationships and detects anomalies
- **Volume Analysis**: Identifies unusual volume spikes and patterns
- **Timestamp Validation**: Prevents stale or future-dated data
- **Statistical Anomaly Detection**: Z-score based outlier detection
- **Quality Reporting**: Comprehensive data quality metrics and alerts

**Aggregation Strategies:**

- **Price Aggregation**: Median-based price consolidation (robust to outliers)
- **Volume Aggregation**: Summation across exchanges
- **Confidence Scoring**: Multi-factor confidence assessment
- **Source Tracking**: Maintains exchange attribution for all data

### Usage Example

```python
from src.services.data_aggregator import DataAggregator
from src.exchanges import RobinhoodConnector, OANDAConnector, CoinbaseConnector

# Initialize exchanges
exchanges = [
    RobinhoodConnector(api_key="..."),
    OANDAConnector(api_key="..."),
    CoinbaseConnector(api_key="...")
]

# Create aggregator
aggregator = DataAggregator(exchanges)

# Start real-time aggregation
symbols = ["AAPL", "EURUSD", "BTCUSD"]
async for aggregated_data in aggregator.start_aggregation(symbols):
    print(f"{aggregated_data.symbol}: ${aggregated_data.close}")
    print(f"Confidence: {aggregated_data.confidence_score:.2f}")
    print(f"Sources: {aggregated_data.source_count}")
```

### Data Quality Features

**Validation Types:**

- **Basic Constraints**: Price relationships, negative values, data completeness
- **Statistical Anomalies**: Z-score analysis for price and volume outliers
- **Temporal Validation**: Timestamp consistency and staleness detection
- **Cross-Exchange Consistency**: Price deviation analysis across sources

**Quality Reporting:**

```python
# Get data quality summary
quality_report = aggregator.get_data_quality_summary(hours=24)
print(f"Data points processed: {quality_report['total_data_points']}")
print(f"Quality issues detected: {quality_report['total_quality_issues']}")
print(f"Issue rate: {quality_report['issue_rate']:.2%}")
```

### Historical Data Aggregation

The system also provides historical data aggregation across exchanges:

```python
# Get aggregated historical data
df = await aggregator.get_historical_aggregated_data(
    symbol="AAPL",
    timeframe="1h", 
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31)
)
```

## Supported Trading Features

With all three exchange connectors and data aggregation now implemented, the platform supports:

### Asset Classes

- **Stocks & ETFs**: US equities via Robinhood
- **Options**: Equity options via Robinhood  
- **Forex**: 50+ currency pairs via OANDA
- **Cryptocurrencies**: 25+ crypto pairs via Coinbase
- **Futures**: Perpetual crypto futures via Coinbase

### Order Types

- **Market Orders**: Immediate execution at current market price
- **Limit Orders**: Execute at specified price or better
- **Stop Orders**: Stop-loss and stop-limit orders
- **Margin Orders**: Leveraged positions (Forex & Crypto)

### Market Data

- **Real-time Streaming**: WebSocket feeds from all exchanges
- **Historical Data**: OHLCV data with multiple timeframes
- **Market Depth**: Order book data (Coinbase & OANDA)
- **Trade History**: Recent execution data
- **Unified Aggregation**: Cross-exchange data normalization and validation

### Account Management

- **Multi-Exchange**: Unified interface across all platforms
- **Position Tracking**: Real-time P&L and exposure monitoring
- **Risk Management**: Automated position sizing and stop-losses
- **Paper Trading**: Sandbox environments for all exchanges

### Other Exchange Connectors

- **Robinhood Connector**: Stocks, ETFs, indices, and options ✅ **Complete**
- **OANDA Connector**: Forex pairs and CFDs ✅ **Complete**

## Configuration Management

The platform uses a hierarchical configuration system:

1. **Environment Variables**: Override any setting
2. **Configuration Files**: YAML/JSON format
3. **Default Values**: Sensible defaults for development

### Environment-Specific Settings

- **Development**: Local databases, sandbox APIs, debug logging
- **Production**: Cloud databases, live APIs, optimized settings
- **Testing**: In-memory databases, mocked services

## Logging and Monitoring

### Logging Features

- Colored console output for development
- Rotating file logs for production
- Structured logging with context
- Module-specific loggers

### Monitoring Features

- System resource monitoring (CPU, memory, disk)
- Application metrics collection
- Health checks for services
- Alert notifications via webhooks

## Getting Started

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:

   ```bash
   cp config/settings.yaml config/local.yaml
   # Edit config/local.yaml with your settings
   export CONFIG_FILE=config/local.yaml
   ```

3. **Set API Keys** (optional for development):

   ```bash
   export ROBINHOOD_API_KEY=your_key
   export OANDA_API_KEY=your_key  
   export COINBASE_API_KEY=your_key
   ```

4. **Run Application**:

   ```bash
   python src/main.py
   ```

## Architecture Principles

- **Modular Design**: Clear separation of concerns
- **Abstract Interfaces**: Pluggable components
- **Configuration-Driven**: Environment-specific settings
- **Observable**: Comprehensive logging and monitoring
- **Scalable**: Distributed computing with Ray
- **Testable**: Dependency injection and mocking support

## Data Models

The platform includes comprehensive data models with Pydantic validation:

### MarketData Model

The `MarketData` model represents OHLCV (Open, High, Low, Close, Volume) data with robust validation:

```python
from src.models import MarketData, ExchangeType
from datetime import datetime

# Create market data with validation
market_data = MarketData(
    symbol="AAPL",
    timestamp=datetime.now(),
    open=150.25,
    high=152.10,
    low=149.80,
    close=151.75,
    volume=1000000.0,
    exchange=ExchangeType.ROBINHOOD
)
```

**Key Features:**

- **Symbol Validation**: Automatically converts to uppercase and validates format
- **Price Relationship Validation**: Ensures high ≥ low, high ≥ open/close, low ≤ open/close
- **Timestamp Validation**: Prevents future timestamps
- **Exchange Support**: Robinhood, OANDA, and Coinbase exchanges
- **Type Safety**: Full Pydantic validation with proper error messages

### Other Models

- **TradingSignal**: Buy/sell/hold recommendations with confidence scores
- **Portfolio**: Complete portfolio management with positions and cash balance
- **Position**: Individual position tracking with P&L calculations

### Data Validation Features

The models include comprehensive validation to ensure data quality:

**MarketData Validation:**

- Price relationships (high ≥ low, etc.)
- Symbol format and normalization
- Timestamp validation (no future dates)
- Positive price and non-negative volume constraints

**Error Handling:**

```python
# This will raise a validation error
try:
    invalid_data = MarketData(
        symbol="AAPL",
        high=148.0,  # Error: high < low
        low=149.0,
        # ... other fields
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Usage Examples

See the examples directory for comprehensive usage demonstrations:

```bash
# Data models validation and usage
python examples/data_models_demo.py

# Data aggregation system demonstration
python examples/data_aggregation_demo.py

# Feature engineering pipeline demonstration
python examples/feature_engineering_demo.py
```

## Testing

The platform includes comprehensive test coverage for all major components:

### Exchange Connector Tests

- **`tests/test_coinbase_connector.py`**: Comprehensive cryptocurrency trading functionality ✅
  - Connection and authentication testing
  - WebSocket real-time data streaming
  - Market, limit, and margin order placement
  - Symbol formatting and validation
  - 24/7 crypto market hours testing
- **`tests/test_oanda_connector.py`**: Forex trading and market hours ✅  
- **`tests/test_robinhood_connector.py`**: Stock and options trading ✅

### Data Model Tests

- **`tests/test_models.py`**: Pydantic model validation and serialization ✅

### Data Aggregation Tests

- **`tests/test_data_aggregator.py`**: Data aggregation system functionality ✅
  - Multi-exchange data normalization
  - Timestamp synchronization and alignment
  - Data quality validation and anomaly detection
  - Confidence scoring and aggregation strategies
  - Historical data aggregation across exchanges

### Feature Engineering Tests

- **`tests/test_technical_indicators.py`**: Technical indicator calculations and validation ✅
  - Complete indicator set testing (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP)
  - Edge case handling (insufficient data, NaN values, zero volumes)
  - Mathematical accuracy validation against known formulas
- **`tests/test_advanced_features.py`**: Advanced feature transformers with edge cases ✅
  - Wavelet decomposition with multiple wavelets and levels
  - Fourier transform frequency analysis
  - Fractal dimension and Hurst exponent calculations
  - Cross-asset correlation and momentum features

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific exchange tests
pytest tests/test_coinbase_connector.py
pytest tests/test_oanda_connector.py
pytest tests/test_robinhood_connector.py

# Run data aggregation tests
pytest tests/test_data_aggregator.py

# Run feature engineering tests
pytest tests/test_technical_indicators.py
pytest tests/test_advanced_features.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Feature Engineering Pipeline ✅ **Complete**

The platform now includes a comprehensive feature engineering system with multiple transformer classes:

### Technical Indicators ✅ **Complete**

The `TechnicalIndicators` transformer provides a complete set of market indicators:

**Price-Based Indicators:**

- **SMA (Simple Moving Average)**: 20-period moving average for trend identification
- **EMA (Exponential Moving Average)**: 12-period exponential average for responsive trend tracking
- **RSI (Relative Strength Index)**: 14-period momentum oscillator (0-100 scale)
- **MACD (Moving Average Convergence Divergence)**:
  - MACD Line: 12-EMA minus 26-EMA
  - Signal Line: 9-period EMA of MACD line
  - Histogram: MACD line minus signal line
- **Bollinger Bands**: 20-period statistical bands with:
  - Upper Band: SMA + (2 × Standard Deviation)
  - Lower Band: SMA - (2 × Standard Deviation)
  - Band Width: Normalized width indicator
  - Band Position: Price position within bands (0-1 scale)

**Volume-Based Indicators:**

- **OBV (On-Balance Volume)**: Cumulative volume flow based on price direction
- **VWAP (Volume Weighted Average Price)**: Intraday price benchmark weighted by volume

**Volatility Indicators:**

- **ATR (Average True Range)**: 14-period true range average for volatility measurement

**Generated Feature Names:**
The technical indicators generate the following feature columns:

- `sma_20`, `ema_12`, `rsi_14`, `atr_14`, `obv`, `vwap`
- `macd_line`, `macd_signal`, `macd_histogram` (3 features from MACD)
- `bb_upper`, `bb_lower`, `bb_width`, `bb_position` (4 features from Bollinger Bands)

Total: **13 technical indicator features** per data point

### Advanced Feature Transformers ✅ **Complete**

**WaveletTransform:**

- Multi-resolution analysis using Daubechies wavelets
- Decomposes price series into different frequency components
- Configurable decomposition levels and wavelet types

**FourierFeatures:**

- Frequency domain analysis of price movements
- Extracts magnitude and phase components
- Rolling window FFT for time-varying spectral analysis

**FractalFeatures:**

- Hurst exponent calculation for trend persistence measurement
- Fractal dimension estimation using box-counting method
- Market regime identification through fractal analysis

**CrossAssetFeatures:**

- Price-volume correlation analysis
- Returns autocorrelation for momentum detection
- Volume momentum and flow indicators
- Cross-exchange price consistency metrics

### Usage Example

```python
from src.ml.feature_engineering import (
    FeatureEngineer, TechnicalIndicators, WaveletTransform, 
    FourierFeatures, FractalFeatures
)

# Create feature engineering pipeline
engineer = FeatureEngineer()

# Add complete technical indicators set
tech_indicators = TechnicalIndicators(config={
    "indicators": [
        "sma_20", "ema_12", "rsi_14", "macd", "bollinger_bands", 
        "atr_14", "obv", "vwap"
    ]
})
engineer.add_transformer(tech_indicators)

# Add advanced transformers
engineer.add_transformer(WaveletTransform(config={"levels": 3}))
engineer.add_transformer(FourierFeatures(config={"n_components": 10}))
engineer.add_transformer(FractalFeatures(config={"window_size": 50}))

# Process market data
features = engineer.fit_transform(market_data_df)
feature_names = engineer.get_feature_names()

print(f"Generated {features.shape[1]} features from {len(feature_names)} transformers")
print("Technical indicators:", [name for name in feature_names if any(
    indicator in name for indicator in ["sma", "ema", "rsi", "macd", "bb", "atr", "obv", "vwap"]
)])
```

### Feature Engineering Tests ✅ **Complete**

- **`tests/test_technical_indicators.py`**: Comprehensive technical indicator validation
- **`tests/test_advanced_features.py`**: Advanced transformer testing with edge cases
- **`examples/feature_engineering_demo.py`**: Complete usage demonstration

## Next Steps

This foundation provides the core structure and interfaces. The next tasks will implement:

1. ~~Data models and validation~~ ✅ **Complete**
2. ~~Exchange connector implementations~~ ✅ **Complete**
3. ~~Unified data aggregation system~~ ✅ **Complete**
4. ~~Feature engineering pipeline~~ ✅ **Complete**
5. CNN+LSTM model architecture
6. Reinforcement learning agents
7. Trading decision engine
8. API endpoints and user interface

## Requirements Addressed

This implementation addresses the following requirements:

### Data Models & Validation ✅ **Complete**

- **Requirement 4.4**: Data quality validation and anomaly detection ✅ **Complete**
- **Requirement 4.7**: Data integrity validation before model training ✅ **Complete**
- **Requirement 10.7**: Consistent timestamps and data formats ✅ **Complete**

### Exchange Integration ✅ **Complete**

- **Requirement 10.1**: Robinhood integration for stocks, ETFs, indices, and options ✅ **Complete**
- **Requirement 10.2**: OANDA integration for forex trading with real-time data ✅ **Complete**
- **Requirement 10.3**: Coinbase integration for cryptocurrency and perpetual futures ✅ **Complete**
- **Requirement 10.4**: Unified data format across all exchanges ✅ **Complete**
- **Requirement 10.5**: Secure API credential management ✅ **Complete**
- **Requirement 10.6**: Exchange-specific order routing and execution ✅ **Complete**
- **Requirement 10.8**: Rate limiting and connection management ✅ **Complete**
- **Requirement 10.9**: Failover mechanisms and retry logic ✅ **Complete**

### Data Aggregation System ✅ **Complete**

- **Requirement 4.1**: Multi-source data ingestion and normalization ✅ **Complete**
- **Requirement 4.4**: Data quality validation and anomaly detection ✅ **Complete**
- **Requirement 10.4**: Unified data aggregation from all exchanges ✅ **Complete**
- **Requirement 10.7**: Data synchronization and consistent timestamps ✅ **Complete**

### Feature Engineering Pipeline ✅ **Complete**

- **Requirement 4.2**: Technical indicators and statistical features ✅ **Complete**
- **Requirement 4.3**: Rolling windows and lag features for time series ✅ **Complete**
- **Requirement 4.5**: Custom feature engineering pipelines ✅ **Complete**

### ML/AI Infrastructure (In Progress)

- **Requirement 1.1**: CNN+LSTM models for feature extraction using PyTorch ⏳ **In Progress**

The platform now provides complete multi-asset trading capabilities across stocks, forex, and cryptocurrency markets with robust data validation, unified interfaces, and sophisticated data aggregation with quality assurance.
