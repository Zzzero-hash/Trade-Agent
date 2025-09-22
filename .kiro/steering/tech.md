# Technology Stack

## Core Framework
- **Python 3.9+**: Primary development language
- **PyTorch 2.1+**: Deep learning framework for CNN+LSTM and RL models
- **PyTorch Lightning 2.1+**: Training orchestration and experiment management

## Machine Learning Libraries
- **Stable-Baselines3**: Reinforcement learning algorithms
- **Ray[rllib]**: Distributed RL training and hyperparameter optimization
- **Transformers**: Attention mechanisms and pre-trained models
- **Optuna**: Hyperparameter optimization
- **Scikit-learn**: Classical ML algorithms and evaluation metrics

## Experiment Tracking
- **MLflow**: Model registry, experiment tracking, and versioning
- **Weights & Biases (wandb)**: Advanced experiment tracking and visualization
- **TensorBoard**: Training metrics visualization

## Data Processing
- **yfinance**: Primary data source for real market data (OHLCV, metadata)
- **Pandas**: Data manipulation, time series processing, and analysis
- **NumPy**: Numerical computations and array operations
- **TA-Lib**: 200+ technical analysis indicators (RSI, MACD, Bollinger Bands)
- **Parquet/HDF5**: Efficient data storage and retrieval

## Development Tools
- **Black**: Code formatting (line length: 88)
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

## Infrastructure
- **CUDA 11.8+**: GPU acceleration
- **Mixed Precision Training**: Automatic mixed precision with torch.cuda.amp
- **DVC**: Data version control
- **Hydra**: Configuration management

## Build System
- **setuptools**: Package building
- **pip**: Dependency management
- **Virtual environments**: Isolation (.venv)

## Common Commands

### Environment Setup
```bash
# Automated setup (recommended)
python setup_environment.py

# Manual setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .

# Verify yfinance data access
python -c "import yfinance as yf; print(yf.download('AAPL', period='5d').head())"
```

### Development Workflow
```bash
# Code formatting and linting
make format
make lint

# Testing
make test
pytest tests/ -v

# GPU verification
make gpu-test
python -c "from utils.gpu_utils import get_device_info; print(get_device_info())"
```

### Experiment Management
```bash
# Start MLflow UI
make mlflow-ui
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Clean experiments
make clean-experiments
```

### Data and Model Training
```bash
# Download yfinance data
python data/ingestion/download_yfinance_data.py --symbols AAPL,GOOGL,MSFT --timeframes 1d,1h

# Train CNN+LSTM models on yfinance data
python experiments/runners/train_cnn_lstm.py --config configs/models/cnn_lstm_config.yaml

# Train RL agents with yfinance environment
python experiments/runners/train_rl_agents.py --config configs/training/rl_training.yaml

# Run comprehensive backtesting
python experiments/backtesting/run_backtest.py --models all --start-date 2022-01-01

# Hyperparameter optimization
python experiments/runners/optimize_hyperparams.py --data-source yfinance
```

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow Google-style docstrings with yfinance data examples
- Maximum line length: 88 characters (Black default)
- Use dataclasses for configuration objects with yfinance parameters
- Implement comprehensive logging with the logging module
- Use pathlib for file operations and data storage paths
- Follow PyTorch conventions for model implementations
- Include proper error handling for yfinance API calls and data validation
- Use proper temporal ordering for financial time series data (no look-ahead bias)
- Implement realistic transaction costs and slippage in backtesting