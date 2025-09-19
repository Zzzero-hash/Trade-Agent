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
├── experiments/           # Experiment management and orchestration
│   ├── runners/           # Training and evaluation scripts
│   ├── analysis/          # Results analysis and visualization
│   ├── logs/              # Training and experiment logs
│   ├── results/           # Experiment outputs and metrics
│   └── tracking.py        # Experiment tracking utilities
├── data/                  # Data pipeline and management
│   ├── ingestion/         # Data collection and acquisition
│   ├── preprocessing/     # Data cleaning and normalization
│   ├── features/          # Feature engineering and extraction
│   ├── processed/         # Processed datasets ready for training
│   └── raw/               # Raw market data
├── configs/               # Configuration files
│   ├── models/            # Model architecture configurations
│   ├── experiments/       # Experiment setup configurations
│   ├── training/          # Training hyperparameters
│   ├── mlflow_config.yaml # MLflow tracking configuration
│   └── wandb_config.yaml  # Weights & Biases configuration
├── utils/                 # Utility functions and helpers
│   ├── gpu_utils.py       # GPU management and mixed precision
│   └── __init__.py
├── outputs/               # Generated outputs and artifacts
└── .kiro/                 # Kiro IDE configuration and specs
```

## Architecture Patterns

### Model Organization
- **models/**: Each model type has its own subdirectory with clear separation
- **Inheritance**: Use base classes for common model functionality
- **Configuration**: Models accept configuration objects for hyperparameters
- **Checkpointing**: Automatic model checkpointing during training

### Experiment Management
- **experiments/runners/**: Executable scripts for training and evaluation
- **experiments/tracking.py**: Centralized experiment tracking utilities
- **Reproducibility**: All experiments use deterministic seeds and version control

### Data Pipeline
- **data/raw/**: Never modify raw data, keep original sources intact
- **data/processed/**: Versioned processed datasets with DVC tracking
- **data/features/**: Feature engineering outputs with metadata
- **Immutable Processing**: Data transformations are reproducible and versioned

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
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the model."""
    pass

class ModelName(nn.Module):
    """Model implementation with clear documentation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with type hints."""
        pass
```

### Experiment Runner
```python
"""
Experiment runner with comprehensive logging and tracking.
"""

import logging
from pathlib import Path
from experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

def main():
    """Main experiment execution."""
    tracker = ExperimentTracker()
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