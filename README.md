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
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   └── cnn_model.py          # CNN feature extraction model
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

# CNN feature extraction model demonstration
python examples/cnn_feature_extraction_demo.py

# LSTM temporal processing model demonstration
python examples/lstm_temporal_processing_demo.py
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

### ML Model Tests

- **`tests/test_cnn_model.py`**: CNN feature extraction model validation ✅
  - Model architecture and forward pass testing
  - Training pipeline with synthetic data
  - Model save/load functionality
  - Configuration validation and edge cases
  - Multi-head attention mechanism testing

- **`tests/test_lstm_model.py`**: LSTM temporal processing model validation ✅
  - Bidirectional LSTM architecture testing
  - Sequence-to-sequence prediction validation
  - Attention mechanism and skip connections testing
  - Gradient flow verification
  - Model persistence and configuration testing
  - End-to-end training pipeline validation

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

# Run ML model tests
pytest tests/test_cnn_model.py
pytest tests/test_lstm_model.py

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

## LSTM Temporal Processing Model ✅ **Complete**

The platform now includes a sophisticated LSTM temporal processing model that implements bidirectional sequence processing with attention mechanisms and skip connections for time series prediction.

### Key Features

**Bidirectional LSTM Architecture:**

- **Multi-Layer Processing**: 3-layer bidirectional LSTM with configurable hidden dimensions
- **Skip Connections**: Residual connections between LSTM layers for improved gradient flow
- **Layer Normalization**: Applied after each LSTM layer for stable training
- **Sequence-to-Sequence**: Full encoder-decoder architecture for temporal prediction

**Advanced Attention Mechanism:**

- **LSTM Attention**: Custom attention mechanism specifically designed for LSTM hidden states
- **Context Vector Generation**: Weighted combination of all timesteps for comprehensive context
- **Masking Support**: Handles variable-length sequences with padding masks
- **Interpretability**: Attention weights provide insights into temporal importance

**Training Features:**

- **Gradient Clipping**: Prevents exploding gradients during LSTM training
- **Learning Rate Scheduling**: Adaptive learning rate with plateau detection
- **Early Stopping**: Prevents overfitting with validation monitoring (25 epochs patience)
- **Dropout Regularization**: Applied at input, LSTM, and output layers

### LSTM Model Capabilities

**Sequence Processing:**

- **Variable Length Input**: Handles sequences of different lengths
- **Variable Length Output**: Generates sequences of specified target length
- **Encoder-Only Mode**: Extract temporal features without sequence generation
- **Bidirectional Processing**: Captures both forward and backward temporal dependencies

**Feature Extraction:**

```python
# Extract temporal features
encoded_sequences, context_vectors = lstm_model.extract_features(sequence_data)

# encoded_sequences: (batch_size, seq_len, hidden_dim * 2)  # Bidirectional
# context_vectors: (batch_size, hidden_dim * 2)  # Attended context
```

**Sequence Prediction:**

```python
# Predict future sequences
predictions = lstm_model.predict_sequence(input_sequences, target_length=20)

# predictions: (batch_size, target_length, output_dim)
```

### Model Architecture Options

The LSTM model supports extensive configuration:

- **Bidirectional vs Unidirectional**: Choose processing direction
- **Attention On/Off**: Enable or disable attention mechanism
- **Skip Connections**: Toggle residual connections between layers
- **Layer Count**: Configure number of LSTM layers (1-5 recommended)
- **Hidden Dimensions**: Adjust model capacity

### Integration with CNN

The LSTM model is designed to work seamlessly with the CNN feature extractor:

```python
# CNN extracts spatial features
cnn_features = cnn_model.extract_features(market_data)  # (batch, seq, cnn_dim)

# LSTM processes temporal dependencies
lstm_predictions = lstm_model.predict_sequence(cnn_features, target_length=10)

# Combined CNN+LSTM pipeline for comprehensive analysis
```

### Performance Characteristics

**Training Performance:**

- **Memory Efficient**: Optimized LSTM implementation with gradient checkpointing
- **Stable Training**: Layer normalization and gradient clipping prevent training issues
- **Fast Convergence**: Skip connections and attention improve learning speed

**Inference Performance:**

- **Batch Processing**: Efficient batch inference for multiple sequences
- **Real-time Capable**: Optimized for low-latency sequence prediction
- **Scalable**: Supports distributed inference with Ray

## Next Steps

This foundation provides the core structure and interfaces. The next tasks will implement:

1. ~~Data models and validation~~ ✅ **Complete**
2. ~~Exchange connector implementations~~ ✅ **Complete**
3. ~~Unified data aggregation system~~ ✅ **Complete**
4. ~~Feature engineering pipeline~~ ✅ **Complete**
5. ~~CNN+LSTM model architecture~~ ✅ **Complete**
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

### ML/AI Infrastructure ✅ **CNN+LSTM Hybrid Complete**

- **Requirement 1.1**: CNN+LSTM models for feature extraction using PyTorch ✅ **Complete**
- **Requirement 1.2**: CNN spatial pattern extraction from multi-dimensional market data ✅ **Complete**
- **Requirement 1.3**: LSTM temporal dependency capture in price movements ✅ **Complete**
- **Requirement 5.2**: CNN architecture with multiple filter sizes and attention ✅ **Complete**
- **Requirement 5.3**: LSTM sequence-to-sequence architecture for time series prediction ✅ **Complete**

## CNN+LSTM Hybrid Architecture ✅ **Complete**

The platform now includes both CNN feature extraction and LSTM temporal processing models that work together to form the complete CNN+LSTM hybrid architecture as specified in the design requirements.

### CNN Feature Extraction Model ✅ **Complete**

**Multi-Scale Convolutional Architecture:**

- **Multiple Filter Sizes**: Simultaneous convolutions with filters of sizes [3, 5, 7, 11] to capture patterns at different time scales
- **Batch Normalization**: Stable training with normalized activations
- **Residual Connections**: Skip connections to prevent vanishing gradients
- **Xavier Initialization**: Proper weight initialization for stable training

**Attention Mechanism:**

- **Multi-Head Attention**: 8-head attention mechanism for important feature selection
- **Self-Attention**: Learns relationships between different time steps
- **Attention Weights**: Provides interpretability for model decisions

### LSTM Temporal Processing Model ✅ **Complete**

**Bidirectional LSTM Architecture:**

- **Multi-Layer LSTM**: 3-layer bidirectional LSTM with configurable hidden dimensions
- **Skip Connections**: Residual connections between LSTM layers for improved gradient flow
- **Layer Normalization**: Stable training with normalized layer outputs
- **Sequence-to-Sequence**: Full encoder-decoder architecture for temporal prediction

**Advanced Attention Mechanism:**

- **LSTM Attention**: Custom attention mechanism for LSTM hidden states
- **Context Vector Generation**: Weighted combination of all timesteps
- **Masking Support**: Handles variable-length sequences with padding masks

**Training Features:**

- **Gradient Clipping**: Prevents exploding gradients during LSTM training
- **Learning Rate Scheduling**: Adaptive learning rate with plateau detection
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Model Checkpointing**: Automatic saving of best models during training

### CNN Architecture Details

The CNN feature extractor processes input data of shape `(batch_size, input_channels, sequence_length)` and outputs features of shape `(batch_size, sequence_length, output_dim)`.

**Network Components:**

1. **Parallel Convolutions**: Multiple 1D convolutions with different kernel sizes
2. **Feature Concatenation**: Combines outputs from all filter sizes
3. **Multi-Head Attention**: Applies attention mechanism to combined features
4. **Residual Connection**: Adds projected input to attended features
5. **Feature Projection**: Final linear layer to desired output dimensions

### LSTM Architecture Details

The LSTM temporal processor handles input sequences of shape `(batch_size, seq_len, input_dim)` and generates output sequences of shape `(batch_size, target_length, output_dim)`.

**Network Components:**

1. **Input Projection**: Linear transformation to match LSTM hidden dimensions
2. **Multi-Layer Bidirectional LSTM**: 3 layers with skip connections between layers
3. **Layer Normalization**: Applied after each LSTM layer for stable training
4. **Attention Mechanism**: Custom LSTM attention for context vector generation
5. **Sequence-to-Sequence Decoder**: Unidirectional LSTM decoder for output generation
6. **Output Projection**: Final linear layers for sequence prediction

### CNN Usage Example

```python
from src.ml.cnn_model import CNNFeatureExtractor, create_cnn_config, create_cnn_data_loader
import numpy as np

# Create CNN configuration
config = create_cnn_config(
    input_dim=20,           # Number of input features (e.g., OHLCV + indicators)
    output_dim=64,          # Number of output features
    filter_sizes=[3, 5, 7, 11],  # Multiple filter sizes
    num_filters=64,         # Filters per size
    use_attention=True,     # Enable attention mechanism
    num_attention_heads=8,  # Number of attention heads
    dropout_rate=0.3,       # Dropout for regularization
    learning_rate=0.001,    # Adam optimizer learning rate
    batch_size=32,          # Training batch size
    epochs=100,             # Training epochs
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize model
cnn_model = CNNFeatureExtractor(config)

# Prepare training data
# X_train shape: (samples, input_channels, sequence_length)
# y_train shape: (samples, sequence_length, target_dim)
train_loader = create_cnn_data_loader(X_train, y_train, batch_size=32)
val_loader = create_cnn_data_loader(X_val, y_val, batch_size=32)

# Train the model
training_result = cnn_model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)

# Extract features from new data
features = cnn_model.extract_features(X_new)
print(f"Extracted features shape: {features.shape}")

# Save trained model
cnn_model.save_model("checkpoints/cnn_feature_extractor.pth")
```

### LSTM Usage Example

```python
from src.ml.lstm_model import LSTMTemporalProcessor, create_lstm_config, create_lstm_data_loader
import numpy as np

# Create LSTM configuration
config = create_lstm_config(
    input_dim=64,           # Number of input features (from CNN or feature engineering)
    output_dim=32,          # Number of output features
    hidden_dim=128,         # LSTM hidden dimension
    num_layers=3,           # Number of LSTM layers
    sequence_length=60,     # Input sequence length
    bidirectional=True,     # Use bidirectional LSTM
    use_attention=True,     # Enable attention mechanism
    use_skip_connections=True,  # Enable skip connections
    dropout_rate=0.3,       # Dropout for regularization
    learning_rate=0.001,    # Adam optimizer learning rate
    batch_size=32,          # Training batch size
    epochs=100,             # Training epochs
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize model
lstm_model = LSTMTemporalProcessor(config)

# Prepare training data
# X_train shape: (samples, seq_len, input_dim)
# y_train shape: (samples, target_len, output_dim)
train_loader = create_lstm_data_loader(X_train, y_train, batch_size=32)
val_loader = create_lstm_data_loader(X_val, y_val, batch_size=32)

# Train the model
training_result = lstm_model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)

# Make sequence predictions
predictions = lstm_model.predict_sequence(X_new, target_length=20)
print(f"Predictions shape: {predictions.shape}")

# Extract temporal features
encoded_sequences, context_vectors = lstm_model.extract_features(X_new)
print(f"Encoded sequences shape: {encoded_sequences.shape}")
print(f"Context vectors shape: {context_vectors.shape}")

# Save trained model
lstm_model.save_model("checkpoints/lstm_temporal_processor.pth")
```

### Model Configuration Options

**CNN Configuration (`create_cnn_config()`):**

- `input_dim`: Number of input channels (market features)
- `output_dim`: Number of output features to extract
- `filter_sizes`: List of convolutional filter sizes (default: [3, 5, 7, 11])
- `num_filters`: Number of filters per size (default: 64)
- `use_attention`: Enable/disable attention mechanism (default: True)
- `num_attention_heads`: Number of attention heads (default: 8)
- `dropout_rate`: Dropout rate for regularization (default: 0.3)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 100)

**LSTM Configuration (`create_lstm_config()`):**

- `input_dim`: Number of input features
- `output_dim`: Number of output features
- `hidden_dim`: LSTM hidden dimension (default: 128)
- `num_layers`: Number of LSTM layers (default: 3)
- `sequence_length`: Length of input sequences (default: 60)
- `bidirectional`: Use bidirectional LSTM (default: True)
- `use_attention`: Enable attention mechanism (default: True)
- `use_skip_connections`: Enable skip connections (default: True)
- `dropout_rate`: Dropout rate for regularization (default: 0.3)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 100)

### Model Persistence

Both CNN and LSTM models include comprehensive save/load functionality:

**Saving Models:**

```python
# Save CNN model with metadata
cnn_model.save_model("models/cnn_extractor.pth")

# Save LSTM model with metadata
lstm_model.save_model("models/lstm_processor.pth")

# Each creates two files:
# - model.pth: PyTorch model checkpoint with full state
# - model_config.json: Human-readable configuration
```

**Loading Models:**

```python
# Load pre-trained CNN model
cnn_model = CNNFeatureExtractor(config)
cnn_model.load_model("models/cnn_extractor.pth")

# Load pre-trained LSTM model
lstm_model = LSTMTemporalProcessor(config)
lstm_model.load_model("models/lstm_processor.pth")

# Models are ready for inference
cnn_features = cnn_model.extract_features(new_data)
lstm_predictions = lstm_model.predict_sequence(sequence_data)
```

### Integration with Feature Engineering

The CNN+LSTM models integrate seamlessly with the feature engineering pipeline:

```python
from src.ml.feature_engineering import FeatureEngineer, TechnicalIndicators

# Create feature engineering pipeline
engineer = FeatureEngineer()
engineer.add_transformer(TechnicalIndicators())

# Process market data to create model input
market_features = engineer.fit_transform(market_data_df)

# For CNN: Reshape to (samples, features, sequence_length)
cnn_input = market_features.values.reshape(
    (num_samples, num_features, sequence_length)
)
spatial_features = cnn_model.extract_features(cnn_input)

# For LSTM: Use features directly (samples, seq_len, features)
lstm_input = market_features.values.reshape(
    (num_samples, sequence_length, num_features)
)
encoded_sequences, context_vectors = lstm_model.extract_features(lstm_input)

# Combine CNN and LSTM features for hybrid approach
hybrid_features = np.concatenate([
    spatial_features.reshape(num_samples, -1),
    context_vectors
], axis=1)
```

### Testing and Validation

Both CNN and LSTM models include comprehensive testing:

- **`tests/test_cnn_model.py`**: CNN architecture and training validation ✅
- **`tests/test_lstm_model.py`**: LSTM architecture and training validation ✅
- **`examples/cnn_feature_extraction_demo.py`**: CNN usage demonstration ✅
- **`examples/lstm_temporal_processing_demo.py`**: LSTM usage demonstration ✅

### Performance Characteristics

**Training Performance:**

- **GPU Acceleration**: Automatic CUDA detection and usage for both CNN and LSTM
- **Memory Efficient**: Gradient checkpointing and optimized memory usage
- **Stable Training**: Gradient clipping, learning rate scheduling, and early stopping
- **Regularization**: Dropout, layer normalization, and skip connections

**Inference Performance:**

- **Batch Processing**: Efficient batch inference for multiple samples
- **Low Latency**: Optimized forward pass for real-time applications
- **Sequence Processing**: Efficient sequence-to-sequence prediction
- **Feature Extraction**: Fast temporal and spatial feature extraction

**Model Capabilities:**

- **CNN Features**: Spatial pattern recognition across multiple time scales
- **LSTM Features**: Long-term temporal dependency modeling
- **Attention Mechanisms**: Interpretable attention weights for both models
- **Hybrid Architecture**: Combined CNN+LSTM for comprehensive market analysis

The platform now provides complete multi-asset trading capabilities across stocks, forex, and cryptocurrency markets with robust data validation, unified interfaces, sophisticated data aggregation with quality assurance, and advanced CNN+LSTM hybrid architecture for comprehensive spatial and temporal pattern recognition in market data.
