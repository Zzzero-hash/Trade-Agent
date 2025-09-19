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
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **TA-Lib**: Technical analysis indicators
- **yfinance**: Financial data acquisition

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

### Model Training
```bash
# Train models with configuration
python experiments/runners/train_models.py --config configs/experiments/cnn_lstm_experiment.yaml

# Hyperparameter optimization
python experiments/runners/optimize_hyperparams.py
```

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow Google-style docstrings
- Maximum line length: 88 characters (Black default)
- Use dataclasses for configuration objects
- Implement comprehensive logging with the logging module
- Use pathlib for file operations
- Follow PyTorch conventions for model implementations