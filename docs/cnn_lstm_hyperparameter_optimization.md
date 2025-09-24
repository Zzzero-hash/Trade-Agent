# CNN+LSTM Hyperparameter Optimization - Task 5.5

This document describes the comprehensive hyperparameter optimization implementation for CNN+LSTM models using Optuna.

## Overview

The hyperparameter optimization system implements the requirements for Task 5.5:

- **Optuna-based optimization**: Advanced hyperparameter search using Optuna framework
- **1000+ trials with early pruning**: Efficient optimization with automatic trial pruning
- **Multi-objective optimization**: Balances accuracy, training time, and model size
- **Best model retraining**: Automatically retrains the best configurations with full training

## Architecture

### Core Components

1. **CNNLSTMHyperparameterOptimizer**: Main optimization class
2. **Multi-objective optimization**: Pareto frontier optimization
3. **Early pruning**: Median and successive halving pruners
4. **Advanced sampling**: TPE, CMA-ES, and random samplers
5. **MLflow integration**: Experiment tracking and logging

### Search Space

The optimization covers all major model components:

#### CNN Architecture
- Number of filters: [32, 64, 128, 256]
- Filter sizes: Various combinations like [3,5,7], [3,5,7,11]
- Dropout rates: [0.1, 0.5]
- Attention mechanisms: Enable/disable with different head counts
- Residual connections: Enable/disable
- Activation functions: ReLU, GELU, Swish

#### LSTM Architecture
- Hidden dimensions: [64, 128, 256, 512, 1024]
- Number of layers: [1, 2, 3, 4]
- Dropout rates: [0.1, 0.5]
- Bidirectional: Enable/disable
- Attention mechanisms: Enable/disable with different head counts
- Skip connections: Enable/disable

#### Feature Fusion
- Fusion methods: Concatenation, attention, gated, bilinear
- Fusion dimensions: [128, 256, 512, 1024]
- Dropout rates: [0.1, 0.4]
- Number of attention heads: [4, 8, 16]

#### Training Parameters
- Learning rates: [1e-5, 1e-2] (log scale)
- Batch sizes: [16, 32, 64, 128]
- Weight decay: [1e-6, 1e-3] (log scale)
- Optimizers: Adam, AdamW, SGD, RMSprop
- Schedulers: Cosine, step, exponential, plateau
- Gradient clipping: [0.5, 5.0]

#### Regularization
- L1 regularization: [0.0, 1e-3] (log scale)
- L2 regularization: [1e-6, 1e-3] (log scale)
- Dropout scheduling: Constant, linear, cosine
- Label smoothing: [0.0, 0.2]

#### Multi-task Learning
- Classification weight: [0.3, 0.8]
- Regression weight: [0.2, 0.7]
- Task balancing: Fixed, dynamic, uncertainty-based

## Usage

### Basic Usage

```python
from src.ml.cnn_lstm_hyperopt import run_cnn_lstm_hyperparameter_optimization

# Run optimization
results = run_cnn_lstm_hyperparameter_optimization(
    data_loaders=(train_loader, val_loader, test_loader),
    input_dim=11,
    n_trials=1000,
    save_dir="hyperopt_results",
    retrain_best=True,
    top_k=5
)
```

### Advanced Usage

```python
from src.ml.cnn_lstm_hyperopt import CNNLSTMHyperparameterOptimizer

# Create optimizer
optimizer = CNNLSTMHyperparameterOptimizer(
    data_loaders=data_loaders,
    input_dim=11,
    save_dir="hyperopt_results",
    study_name="cnn_lstm_optimization"
)

# Run optimization
optimization_results = optimizer.optimize(
    n_trials=1000,
    pruner_type="median",
    sampler_type="tpe"
)

# Retrain best models
retrained_results = optimizer.retrain_best_models(
    top_k=5,
    full_epochs=200
)
```

### Command Line Usage

```bash
# Run with default settings
python scripts/run_cnn_lstm_hyperopt.py --n-trials 1000 --retrain-best

# Run with real data
python scripts/run_cnn_lstm_hyperopt.py \
    --use-real-data \
    --symbols AAPL GOOGL MSFT TSLA NVDA \
    --n-trials 1000 \
    --retrain-best \
    --top-k 5

# Run with custom configuration
python scripts/run_cnn_lstm_hyperopt.py \
    --n-trials 500 \
    --batch-size 64 \
    --pruner successive_halving \
    --sampler cmaes \
    --timeout 3600
```

## Multi-Objective Optimization

The system optimizes three objectives simultaneously:

1. **Accuracy** (maximize): Validation classification accuracy
2. **Training Time** (minimize): Time to train the model
3. **Model Size** (minimize): Number of parameters in MB

### Pareto Frontier

The optimization finds Pareto optimal solutions where no single objective can be improved without degrading others. This provides a set of trade-off solutions:

- High accuracy, longer training time, larger model
- Moderate accuracy, fast training, compact model
- Balanced performance across all objectives

## Pruning Strategies

### Median Pruner
- Prunes trials with performance below the median
- Configurable startup trials and warmup steps
- Suitable for most optimization scenarios

### Successive Halving Pruner
- Aggressively prunes poor-performing trials
- Allocates more resources to promising trials
- Better for large-scale optimizations

## Sampling Strategies

### TPE (Tree-structured Parzen Estimator)
- Default sampler with good general performance
- Builds probabilistic models of good/bad configurations
- Supports multivariate and grouped parameters

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Evolutionary algorithm for continuous optimization
- Good for high-dimensional search spaces
- Includes restart strategies

### Random Sampling
- Baseline sampling strategy
- Useful for comparison and small search spaces

## Results and Analysis

### Optimization Results

The system provides comprehensive results including:

```python
{
    'study_name': 'cnn_lstm_optimization',
    'n_trials_completed': 1000,
    'best_trials': [
        {
            'trial_number': 42,
            'values': [0.8543, 45.2, 12.3],  # [accuracy, time, size]
            'params': {...},
            'rank': 1
        },
        ...
    ],
    'optimization_history': [...],
    'timestamp': '2024-01-15T10:30:00'
}
```

### Retrained Model Results

```python
{
    'model_rank': 1,
    'config_params': {...},
    'training_results': {...},
    'test_metrics': {
        'test_class_acc': 0.8621,
        'test_reg_mse': 0.0234
    },
    'training_time': 1847.3,
    'model_path': 'hyperopt_results/retrained_model_1'
}
```

### Optimization Report

The system generates detailed reports including:

- Performance statistics across all trials
- Best configuration parameters
- Training efficiency metrics
- Model size analysis
- Convergence analysis

## Configuration

### YAML Configuration

```yaml
# configs/hyperopt/cnn_lstm_hyperopt_config.yaml
study:
  name: "cnn_lstm_optimization_task_5_5"
  n_trials: 1000
  directions: ["maximize", "minimize", "minimize"]

pruner:
  type: "median"
  n_startup_trials: 20
  n_warmup_steps: 10

sampler:
  type: "tpe"
  n_startup_trials: 50
  multivariate: true

search_space:
  cnn:
    num_filters: [32, 64, 128, 256]
    filter_sizes: [[3,5,7], [3,5,7,11]]
  lstm:
    hidden_dim: [64, 128, 256, 512, 1024]
    num_layers: [1, 4]
  # ... more parameters
```

## Integration with MLflow

The system integrates with MLflow for experiment tracking:

- Automatic experiment creation
- Parameter and metric logging
- Model artifact storage
- Visualization and comparison

```python
# MLflow tracking
mlflow.set_experiment("CNN_LSTM_Hyperopt_Task_5_5")
with mlflow.start_run():
    mlflow.log_params(hyperparams)
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, "model")
```

## Performance Considerations

### GPU Memory Management
- Automatic batch size adjustment
- Model cleanup between trials
- Memory monitoring and optimization

### Parallel Execution
- Single GPU execution to avoid conflicts
- CPU parallelization for data loading
- Distributed optimization support (future)

### Early Stopping
- Trial-level early stopping for efficiency
- Epoch-level early stopping within trials
- Adaptive patience based on performance

## Best Practices

### Data Preparation
- Use consistent data splits across trials
- Implement proper data augmentation
- Ensure reproducible data loading

### Search Space Design
- Start with broad ranges, then narrow down
- Use log scales for learning rates and regularization
- Include both architectural and training parameters

### Resource Management
- Monitor GPU memory usage
- Set appropriate timeouts
- Use checkpointing for long optimizations

### Result Analysis
- Analyze Pareto frontier trade-offs
- Validate results on independent test sets
- Consider ensemble methods with top models

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size range
   - Limit model size parameters
   - Use gradient checkpointing

2. **Slow Convergence**
   - Increase startup trials for samplers
   - Adjust pruning parameters
   - Use more aggressive pruning

3. **Poor Performance**
   - Check data quality and preprocessing
   - Verify search space ranges
   - Increase trial budget

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor trial progress
study.trials_dataframe()

# Analyze failed trials
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
```

## Future Enhancements

### Planned Features
- Distributed optimization across multiple GPUs
- Neural architecture search integration
- Automated feature selection
- Online learning and adaptation
- Advanced ensemble methods

### Research Directions
- Meta-learning for hyperparameter initialization
- Multi-fidelity optimization
- Bayesian optimization with constraints
- Automated machine learning (AutoML) integration

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hyperparameter Optimization Best Practices](https://arxiv.org/abs/1803.09820)

## Examples

See the following files for complete examples:

- `examples/cnn_lstm_hyperopt_demo.py`: Basic demonstration
- `scripts/run_cnn_lstm_hyperopt.py`: Full command-line interface
- `tests/test_cnn_lstm_hyperopt.py`: Comprehensive test suite
- `configs/hyperopt/cnn_lstm_hyperopt_config.yaml`: Configuration template