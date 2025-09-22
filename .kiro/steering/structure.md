# Project Structure

## Directory Organization

```
ml-trading-ensemble/
├── models/                 # ML model implementations
│   ├── cnn_lstm/          # CNN+LSTM feature extractors
│   ├── rl_agents/         # Reinforcement learning agents
│   ├── ensemble/          # Ensemble learning components
│   ├── checkpoints/       # Model checkpoints during training
│   └── saved/             # Final trained models
├── experiments/           # Training and evaluation orchestration
│   ├── runners/           # Training scripts for CNN+LSTM and RL agents
│   ├── backtesting/       # Realistic backtesting with transaction costs
│   ├── analysis/          # Performance analysis and visualization
│   ├── logs/              # Training and experiment logs
│   ├── results/           # Backtesting results and performance metrics
│   └── tracking.py        # MLflow experiment tracking utilities
├── data/                  # yfinance data pipeline and management
│   ├── ingestion/         # yfinance data download and collection
│   ├── preprocessing/     # Data cleaning, validation, and temporal splits
│   ├── features/          # Technical indicators and feature engineering
│   ├── processed/         # Processed datasets with train/val/test splits
│   └── raw/               # Raw OHLCV data from yfinance
├── configs/               # Configuration files
│   ├── models/            # Model architecture configurations
│   ├── experiments/       # Experiment setup configurations
│   ├── training/          # Training hyperparameters
│   ├── mlflow_config.yaml # MLflow tracking configuration
│   └── wandb_config.yaml  # Weights & Biases configuration
├── utils/                 # Utility functions and helpers
│   ├── gpu_utils.py       # GPU management and mixed precision
│   └── __init__.py
├── inference/             # Real-time inference and prediction system
│   ├── pipeline.py        # End-to-end inference pipeline
│   ├── api.py             # REST API for trading signals
│   └── monitoring.py      # Performance monitoring and alerts
├── outputs/               # Generated outputs and artifacts
└── .kiro/                 # Kiro IDE configuration and specs
```

## Architecture Patterns

### Model Organization
- **models/**: Each model type has its own subdirectory with clear separation
- **Inheritance**: Use base classes for common model functionality
- **Configuration**: Models accept configuration objects for hyperparameters
- **Checkpointing**: Automatic model checkpointing during training

### Training and Evaluation Management
- **experiments/runners/**: Training scripts for CNN+LSTM and RL agents on yfinance data
- **experiments/backtesting/**: Realistic backtesting with transaction costs and slippage
- **experiments/tracking.py**: MLflow tracking for experiments and model versioning
- **Reproducibility**: Deterministic training with fixed seeds and data splits
- **Performance Validation**: Statistical significance testing and confidence intervals

### yfinance Data Pipeline
- **data/raw/**: Raw OHLCV data from yfinance, never modified after download
- **data/processed/**: Clean datasets with proper train/validation/test temporal splits
- **data/features/**: Technical indicators and engineered features with metadata
- **Reproducible Processing**: All data transformations are deterministic and versioned
- **Real-time Updates**: Support for incremental data updates and live inference

### Configuration Management
- **YAML-based**: All configurations use YAML format for readability
- **Hierarchical**: Configs can inherit from base configurations
- **Environment-specific**: Separate configs for development, testing, production
- **Validation**: Use Pydantic for configuration validation

## Naming Conventions

### Files and Directories
- **snake_case**: All Python files and directories use snake_case
- **Descriptive names**: File names clearly indicate their purpose
- **Module structure**: Each directory has `__init__.py` for proper imports

### Python Code
- **Classes**: PascalCase (e.g., `CNNLSTMFeatureExtractor`)
- **Functions/Variables**: snake_case (e.g., `train_model`, `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_BATCH_SIZE`)
- **Private methods**: Leading underscore (e.g., `_validate_config`)

### Model and Experiment Names
- **Models**: Descriptive with version (e.g., `cnn_lstm_v2`, `ensemble_rl_v1`)
- **Experiments**: Include date and key parameters (e.g., `cnn_lstm_20240315_lr001`)
- **Checkpoints**: Include epoch and metric (e.g., `model_epoch_100_acc_0.85.pt`)

## Import Patterns

### Absolute Imports
```python
from models.cnn_lstm.feature_extractor import CNNLSTMFeatureExtractor
from experiments.tracking import ExperimentTracker
from utils.gpu_utils import GPUManager
```

### Configuration Imports
```python
from configs.models.cnn_lstm_config import CNNLSTMConfig
from configs.training.training_config import TrainingConfig
```

### Third-party Imports
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
```

## Code Organization Principles

### Separation of Concerns
- **Models**: Pure model implementations without training logic
- **Trainers**: Training orchestration and optimization
- **Data**: Data processing separate from model code
- **Utils**: Reusable utilities across the project

### Dependency Management
- **Minimal coupling**: Components should be loosely coupled
- **Interface-based**: Use abstract base classes for common interfaces
- **Dependency injection**: Pass dependencies through constructors

### Testing Structure
- **tests/**: Mirror the main project structure
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

## File Templates

### Model Implementation
```python
"""
Module docstring describing the model's purpose and architecture.
"""

import torch
import torch.nn as nn
import yfinance as yf
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for the model."""
    sequence_length: int = 60
    feature_dim: int = 200
    symbols: List[str] = None
    timeframes: List[str] = None

class YFinanceModelBase(nn.Module):
    """Base model for yfinance data processing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Implementation
    
    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        """Forward pass with yfinance market data."""
        pass
    
    def preprocess_yfinance_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """Convert yfinance data to model input format."""
        pass
```

### Training Runner
```python
"""
Training runner for yfinance-based models with MLflow tracking.
"""

import logging
import yfinance as yf
from pathlib import Path
from experiments.tracking import ExperimentTracker
from data.ingestion.yfinance_manager import YFinanceDataManager

logger = logging.getLogger(__name__)

def main():
    """Main training execution with yfinance data."""
    # Initialize data manager
    data_manager = YFinanceDataManager(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        timeframes=['1d', '1h'],
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Download and prepare data
    dataset = data_manager.prepare_training_data()
    
    # Initialize experiment tracking
    tracker = ExperimentTracker()
    
    # Train model with real market data
    # Implementation

if __name__ == "__main__":
    main()
```

## Best Practices

### Code Quality
- Use type hints for all public functions
- Write comprehensive docstrings
- Implement proper error handling and logging
- Follow the established code style (Black, Flake8, MyPy)

### Experiment Reproducibility
- Always set random seeds for reproducibility
- Version control all configurations and code
- Log all hyperparameters and metrics
- Save model checkpoints with metadata

### Performance Optimization
- Use GPU utilities for device management
- Implement mixed precision training where applicable
- Profile code for bottlenecks
- Use efficient data loading and preprocessing