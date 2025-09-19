# ML Trading Ensemble

Advanced Machine Learning Trading System with CNN+LSTM Feature Extractors and Ensemble RL Agents

## Overview

This project implements state-of-the-art machine learning models for financial market prediction and trading, featuring:

- **Revolutionary CNN+LSTM Architectures**: Multi-scale feature extraction with attention mechanisms
- **Advanced RL Agent Ensemble**: Sophisticated reinforcement learning agents with meta-learning capabilities  
- **Scientific Evaluation Framework**: Rigorous statistical validation and benchmarking
- **Production-Ready Infrastructure**: GPU acceleration, experiment tracking, and model versioning

## Project Structure

```
ml-trading-ensemble/
├── models/                 # ML model implementations
│   ├── cnn_lstm/          # CNN+LSTM feature extractors
│   ├── rl_agents/         # Reinforcement learning agents
│   └── ensemble/          # Ensemble learning components
├── experiments/           # Experiment management
│   ├── runners/           # Experiment orchestration
│   ├── analysis/          # Results analysis
│   └── tracking.py        # Experiment tracking utilities
├── data/                  # Data management
│   ├── ingestion/         # Data collection and ingestion
│   ├── preprocessing/     # Data cleaning and preprocessing
│   └── features/          # Feature engineering
├── configs/               # Configuration files
│   ├── models/            # Model configurations
│   ├── experiments/       # Experiment configurations
│   └── training/          # Training configurations
├── utils/                 # Utility functions
└── outputs/               # Generated outputs and results
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ml-trading-ensemble

# Run the automated setup script
python setup_environment.py
```

The setup script will:
- Validate Python version (3.9+ required)
- Install all dependencies from requirements.txt
- Configure GPU acceleration and mixed precision training
- Setup experiment tracking (MLflow, Weights & Biases)
- Create necessary directories
- Validate the installation

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Configuration

#### GPU Configuration
Edit `configs/training/gpu_config.yaml` to configure:
- Device selection (auto, cuda, cpu, mps)
- Mixed precision training settings
- Memory management
- Distributed training (if applicable)

#### Experiment Tracking
- **MLflow**: Configure in `configs/mlflow_config.yaml`
- **Weights & Biases**: Configure in `configs/wandb_config.yaml`

### 4. Verify Installation

```python
from utils.gpu_utils import get_device_info
from experiments.tracking import ExperimentTracker

# Check GPU setup
print(get_device_info())

# Test experiment tracking
tracker = ExperimentTracker()
run_id = tracker.start_run("test_run")
tracker.log_params({"test": 1.0})
tracker.end_run()
```

## Requirements

### System Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- 100GB+ storage for data and models

### Key Dependencies
- **PyTorch 2.1+**: Deep learning framework
- **MLflow**: Experiment tracking and model registry
- **Weights & Biases**: Advanced experiment tracking
- **Optuna**: Hyperparameter optimization
- **Ray**: Distributed computing and RL
- **Stable-Baselines3**: RL algorithms

## Development Workflow

### 1. Model Development
```bash
# Develop CNN+LSTM models
cd models/cnn_lstm/

# Develop RL agents  
cd models/rl_agents/

# Create ensemble methods
cd models/ensemble/
```

### 2. Experiment Management
```bash
# Configure experiments
cd experiments/

# Run training experiments
python runners/train_models.py --config configs/experiments/cnn_lstm_experiment.yaml

# Analyze results
python analysis/evaluate_models.py
```

### 3. Data Pipeline
```bash
# Setup data ingestion
cd data/ingestion/

# Run preprocessing
cd data/preprocessing/

# Generate features
cd data/features/
```

## Advanced Features

### Mixed Precision Training
Automatically enabled for compatible GPUs:
```python
from utils.gpu_utils import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(enabled=True, dtype="float16")
with trainer.autocast_context():
    outputs = model(inputs)
```

### Experiment Tracking
```python
from experiments.tracking import ExperimentTracker

tracker = ExperimentTracker()
run_id = tracker.start_run("my_experiment")
tracker.log_params({"lr": 0.001, "batch_size": 32})
tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
tracker.log_model(model, "cnn_lstm_v1")
tracker.end_run()
```

### Model Versioning
```python
from experiments.tracking import ModelVersionManager

manager = ModelVersionManager()
version = manager.register_model("runs:/run_id/model", "trading_model")
manager.promote_model("trading_model", version, "Production")
```

## Performance Optimization

### GPU Memory Management
```python
from utils.gpu_utils import GPUManager

gpu_manager = GPUManager()
gpu_manager.set_memory_fraction(0.9)  # Use 90% of GPU memory
gpu_manager.clear_cache()  # Clear memory cache
```

### Model Compilation (PyTorch 2.0+)
```python
from utils.gpu_utils import ModelOptimizer

# Compile model for optimization
model = ModelOptimizer.compile_model(model, mode="default")

# Use channels_last memory format
model = ModelOptimizer.to_channels_last(model)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configs
   - Enable gradient accumulation
   - Use mixed precision training

2. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Experiment Tracking Issues**
   - Verify MLflow database permissions
   - Check W&B authentication
   - Ensure network connectivity

### Getting Help

1. Check the logs in `experiments/logs/`
2. Run the validation script: `python setup_environment.py`
3. Review configuration files in `configs/`

## Contributing

1. Follow the established project structure
2. Add comprehensive tests for new features
3. Update documentation and configuration files
4. Use the experiment tracking system for all model development

## License

MIT License - see LICENSE file for details.

## Research and Citations

This project implements cutting-edge techniques from financial ML research. If you use this code in academic work, please cite relevant papers and this repository.

---

**Note**: This is a research-grade implementation focused on achieving state-of-the-art performance in financial ML. Ensure proper risk management and compliance when using in production trading systems.