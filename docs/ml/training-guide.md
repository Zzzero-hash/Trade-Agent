# Model Training Guide

This comprehensive guide covers training CNN+LSTM hybrid models and reinforcement learning agents for the AI Trading Platform.

## Overview

The platform uses a multi-model approach combining:

- **CNN Models**: Spatial pattern recognition in market data
- **LSTM Models**: Temporal sequence processing
- **Hybrid CNN+LSTM**: Combined spatial-temporal feature extraction
- **RL Agents**: Decision-making and portfolio optimization
- **Ensemble Methods**: Combining multiple models for robust predictions

## Quick Start

### Prerequisites

```bash
# Ensure you have the required dependencies
pip install torch torchvision stable-baselines3 ray[tune]

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Basic Training Pipeline

```python
from src.ml.training_pipeline import TrainingPipeline
from src.ml.hybrid_model import HybridCNNLSTM
from src.services.data_aggregator import DataAggregator

# Initialize training pipeline
pipeline = TrainingPipeline(
    model_type="hybrid_cnn_lstm",
    config_path="config/training.yaml"
)

# Start training
results = pipeline.train(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

print(f"Training completed. Best validation loss: {results['best_val_loss']}")
```

## Model Architecture Deep Dive

### CNN Component

The CNN component extracts spatial patterns from market data using multiple filter sizes:

```python
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=256):
        super().__init__()
        
        # Multi-scale convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]  # Different time scales
        ])
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # (batch_size, features, sequence_length)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)
        
        # Concatenate and apply attention
        combined = torch.cat(conv_outputs, dim=1)
        attended, _ = self.attention(combined, combined, combined)
        
        return attended
```

### LSTM Component

The LSTM processes temporal sequences with bidirectional processing:

```python
class LSTMTemporalProcessor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Skip connections
        self.skip_connection = nn.Linear(input_dim, hidden_dim * 2)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply skip connection
        skip = self.skip_connection(x)
        output = lstm_out + skip
        
        return output, (h_n, c_n)
```

### Hybrid Fusion

The fusion module combines CNN and LSTM outputs:

```python
class FeatureFusion(nn.Module):
    def __init__(self, cnn_dim=256, lstm_dim=256, output_dim=256):
        super().__init__()
        
        # Cross-attention between CNN and LSTM features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Feature projection layers
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.lstm_proj = nn.Linear(lstm_dim, output_dim)
        
        # Output layers
        self.classifier = nn.Linear(output_dim, 3)  # Buy/Hold/Sell
        self.regressor = nn.Linear(output_dim, 1)   # Price prediction
        
    def forward(self, cnn_features, lstm_features):
        # Project features to common dimension
        cnn_proj = self.cnn_proj(cnn_features)
        lstm_proj = self.lstm_proj(lstm_features)
        
        # Apply cross-attention
        fused_features, attention_weights = self.cross_attention(
            cnn_proj, lstm_proj, lstm_proj
        )
        
        # Generate outputs
        classification = self.classifier(fused_features)
        regression = self.regressor(fused_features)
        
        return {
            'classification': classification,
            'regression': regression,
            'attention_weights': attention_weights,
            'features': fused_features
        }
```

## Training Configuration

### Configuration File Structure

Create a training configuration file (`config/training.yaml`):

```yaml
# Model Configuration
model:
  type: "hybrid_cnn_lstm"
  cnn:
    input_channels: 5
    hidden_dim: 256
    filter_sizes: [3, 5, 7, 11]
    num_heads: 8
    dropout: 0.1
  
  lstm:
    hidden_dim: 128
    num_layers: 2
    bidirectional: true
    dropout: 0.2
  
  fusion:
    output_dim: 256
    num_heads: 8

# Training Configuration
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  
  # Loss weights for multi-task learning
  loss_weights:
    classification: 0.6
    regression: 0.4
  
  # Optimizer settings
  optimizer:
    type: "adamw"
    weight_decay: 0.01
    betas: [0.9, 0.999]
  
  # Learning rate scheduler
  scheduler:
    type: "cosine_annealing"
    T_max: 100
    eta_min: 0.00001

# Data Configuration
data:
  sequence_length: 60  # 60 time steps
  prediction_horizon: 1  # Predict 1 step ahead
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  
  # Feature engineering
  features:
    - "open"
    - "high" 
    - "low"
    - "close"
    - "volume"
    - "rsi"
    - "macd"
    - "bollinger_bands"
  
  # Data augmentation
  augmentation:
    noise_std: 0.01
    time_shift_max: 5
    magnitude_warp: true

# Validation Configuration
validation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "mse"
    - "mae"
    - "sharpe_ratio"
  
  # Cross-validation
  cv_folds: 5
  time_series_split: true

# Hardware Configuration
hardware:
  device: "auto"  # auto, cpu, cuda
  mixed_precision: true
  num_workers: 4
  pin_memory: true
```

### Advanced Training Options

```python
# Custom training with advanced options
from src.ml.training_pipeline import AdvancedTrainingPipeline

pipeline = AdvancedTrainingPipeline(
    model_config="config/training.yaml",
    experiment_name="hybrid_model_v2",
    
    # Distributed training
    use_distributed=True,
    num_gpus=4,
    
    # Hyperparameter optimization
    use_ray_tune=True,
    num_trials=50,
    
    # Model checkpointing
    checkpoint_every=10,
    save_best_only=True,
    
    # Monitoring
    use_wandb=True,
    log_gradients=True,
    log_model_graph=True
)

# Define hyperparameter search space
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "hidden_dim": tune.choice([128, 256, 512]),
    "dropout": tune.uniform(0.1, 0.5)
}

# Start hyperparameter optimization
best_config = pipeline.optimize_hyperparameters(
    search_space=search_space,
    metric="val_sharpe_ratio",
    mode="max",
    max_concurrent_trials=4
)
```

## Reinforcement Learning Training

### Environment Setup

```python
from src.ml.trading_environment import TradingEnvironment
from src.ml.rl_agents import PPOAgent, SACAgent

# Create trading environment
env = TradingEnvironment(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    initial_balance=100000,
    transaction_cost=0.001,
    
    # Reward function configuration
    reward_config={
        "sharpe_weight": 0.4,
        "return_weight": 0.3,
        "drawdown_penalty": 0.3,
        "transaction_penalty": 0.1
    }
)

# Initialize RL agent
agent = PPOAgent(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    
    # Custom network architecture
    policy_kwargs={
        "net_arch": [256, 256, 128],
        "activation_fn": torch.nn.ReLU,
        "features_extractor_class": CNNLSTMFeatureExtractor
    }
)
```

### RL Training Pipeline

```python
from src.ml.rl_training_pipeline import RLTrainingPipeline

# Create RL training pipeline
rl_pipeline = RLTrainingPipeline(
    agent_type="ppo",
    environment_config="config/trading_env.yaml",
    training_config="config/rl_training.yaml"
)

# Train multiple agents for ensemble
agents = rl_pipeline.train_ensemble(
    agent_types=["ppo", "sac", "td3", "rainbow_dqn"],
    total_timesteps=1000000,
    eval_freq=10000,
    n_eval_episodes=10,
    
    # Ensemble configuration
    ensemble_method="weighted_voting",
    meta_learning=True,
    thompson_sampling=True
)

# Evaluate ensemble performance
results = rl_pipeline.evaluate_ensemble(
    agents=agents,
    test_episodes=100,
    render=False
)

print(f"Ensemble Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Best Practices

### Data Preparation

1. **Feature Engineering**:

   ```python
   from src.ml.feature_engineering import FeatureEngineer
   
   # Create comprehensive feature set
   feature_engineer = FeatureEngineer()
   
   # Technical indicators
   features = feature_engineer.add_technical_indicators(
       data, indicators=["rsi", "macd", "bollinger_bands", "atr"]
   )
   
   # Advanced features
   features = feature_engineer.add_wavelet_features(features)
   features = feature_engineer.add_fourier_features(features)
   features = feature_engineer.add_fractal_features(features)
   
   # Cross-asset features
   features = feature_engineer.add_cross_asset_features(
       features, correlation_window=30
   )
   ```

2. **Data Quality Validation**:

   ```python
   from src.services.data_quality_validator import DataQualityValidator
   
   validator = DataQualityValidator()
   
   # Validate data quality
   quality_report = validator.validate_batch(features)
   
   if quality_report.overall_score < 0.8:
       print("Warning: Data quality issues detected")
       print(quality_report.issues)
   ```

3. **Train/Validation/Test Splits**:

   ```python
   # Time-aware splitting for financial data
   from src.ml.data_utils import TimeSeriesSplitter
   
   splitter = TimeSeriesSplitter(
       train_ratio=0.7,
       val_ratio=0.15,
       test_ratio=0.15,
       gap_days=5  # Gap between train/val to prevent lookahead
   )
   
   train_data, val_data, test_data = splitter.split(features)
   ```

### Model Training Best Practices

1. **Regularization**:

   ```python
   # Use multiple regularization techniques
   model = HybridCNNLSTM(
       dropout=0.2,           # Dropout regularization
       weight_decay=0.01,     # L2 regularization
       batch_norm=True,       # Batch normalization
       layer_norm=True,       # Layer normalization
       gradient_clip=1.0      # Gradient clipping
   )
   ```

2. **Learning Rate Scheduling**:

   ```python
   # Cosine annealing with warm restarts
   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, T_0=10, T_mult=2, eta_min=1e-6
   )
   ```

3. **Early Stopping**:

   ```python
   from src.ml.callbacks import EarlyStopping
   
   early_stopping = EarlyStopping(
       monitor='val_sharpe_ratio',
       patience=15,
       mode='max',
       restore_best_weights=True
   )
   ```

4. **Model Checkpointing**:

   ```python
   from src.ml.callbacks import ModelCheckpoint
   
   checkpoint = ModelCheckpoint(
       filepath='checkpoints/best_model_{epoch:02d}_{val_loss:.4f}.pth',
       monitor='val_loss',
       save_best_only=True,
       save_weights_only=False
   )
   ```

### Hyperparameter Optimization

1. **Ray Tune Integration**:

   ```python
   import ray
   from ray import tune
   from ray.tune.schedulers import ASHAScheduler
   
   # Define search space
   config = {
       "lr": tune.loguniform(1e-5, 1e-1),
       "batch_size": tune.choice([16, 32, 64, 128]),
       "hidden_dim": tune.choice([64, 128, 256, 512]),
       "num_layers": tune.choice([1, 2, 3, 4]),
       "dropout": tune.uniform(0.1, 0.5)
   }
   
   # ASHA scheduler for early stopping
   scheduler = ASHAScheduler(
       metric="val_sharpe_ratio",
       mode="max",
       max_t=100,
       grace_period=10,
       reduction_factor=2
   )
   
   # Run hyperparameter optimization
   result = tune.run(
       train_model,
       config=config,
       scheduler=scheduler,
       num_samples=50,
       resources_per_trial={"cpu": 2, "gpu": 0.5}
   )
   ```

2. **Optuna Integration**:

   ```python
   import optuna
   
   def objective(trial):
       # Suggest hyperparameters
       lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
       batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
       hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
       
       # Train model with suggested parameters
       model = create_model(lr=lr, batch_size=batch_size, hidden_dim=hidden_dim)
       val_score = train_and_evaluate(model)
       
       return val_score
   
   # Create study and optimize
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   
   print(f"Best parameters: {study.best_params}")
   print(f"Best value: {study.best_value}")
   ```

### Model Evaluation

1. **Comprehensive Metrics**:

   ```python
   from src.ml.evaluation import ModelEvaluator
   
   evaluator = ModelEvaluator()
   
   # Evaluate on test set
   results = evaluator.evaluate(
       model=trained_model,
       test_data=test_loader,
       metrics=[
           'accuracy', 'precision', 'recall', 'f1_score',
           'mse', 'mae', 'sharpe_ratio', 'sortino_ratio',
           'max_drawdown', 'calmar_ratio'
       ]
   )
   
   # Generate evaluation report
   evaluator.generate_report(results, save_path='evaluation_report.html')
   ```

2. **Backtesting**:

   ```python
   from src.services.backtesting_engine import BacktestingEngine
   
   backtester = BacktestingEngine()
   
   # Run comprehensive backtest
   backtest_results = backtester.run_backtest(
       model=trained_model,
       start_date="2023-01-01",
       end_date="2023-12-31",
       initial_capital=100000,
       transaction_costs=0.001,
       
       # Risk management
       max_position_size=0.2,
       stop_loss_pct=0.05,
       take_profit_pct=0.15
   )
   
   # Analyze results
   print(f"Total Return: {backtest_results['total_return']:.2%}")
   print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
   print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
   ```

## Troubleshooting

### Common Training Issues

1. **Gradient Explosion/Vanishing**:

   ```python
   # Monitor gradients
   def monitor_gradients(model):
       total_norm = 0
       for p in model.parameters():
           if p.grad is not None:
               param_norm = p.grad.data.norm(2)
               total_norm += param_norm.item() ** 2
       total_norm = total_norm ** (1. / 2)
       return total_norm
   
   # Apply gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Overfitting**:

   ```python
   # Increase regularization
   model = HybridCNNLSTM(
       dropout=0.3,           # Increase dropout
       weight_decay=0.02,     # Increase L2 regularization
       batch_norm=True,       # Add batch normalization
   )
   
   # Use data augmentation
   augmenter = DataAugmenter(
       noise_std=0.02,
       time_shift_max=10,
       magnitude_warp=True
   )
   ```

3. **Poor Convergence**:

   ```python
   # Adjust learning rate
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=1e-4,              # Lower learning rate
       weight_decay=0.01,
       betas=(0.9, 0.999)
   )
   
   # Use learning rate scheduling
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=5
   )
   ```

### Performance Optimization

1. **Memory Optimization**:

   ```python
   # Use gradient checkpointing
   model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)
   
   # Mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   
   with torch.cuda.amp.autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Speed Optimization**:

   ```python
   # Use DataLoader optimizations
   train_loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4,        # Parallel data loading
       pin_memory=True,      # Faster GPU transfer
       persistent_workers=True  # Keep workers alive
   )
   
   # Compile model (PyTorch 2.0+)
   model = torch.compile(model)
   ```

## Advanced Topics

### Multi-GPU Training

```python
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Initialize distributed training
def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# Wrap model for distributed training
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
```

### Model Interpretability

```python
from src.ml.interpretability import SHAPExplainer, AttentionVisualizer

# SHAP explanations
explainer = SHAPExplainer(model)
shap_values = explainer.explain_prediction(sample_input)

# Attention visualization
visualizer = AttentionVisualizer(model)
attention_maps = visualizer.visualize_attention(sample_input)

# Feature importance
importance_scores = explainer.feature_importance(test_data)
```

### Model Deployment

```python
# Convert to TorchScript for production
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# ONNX export for cross-platform deployment
torch.onnx.export(
    model, sample_input, 'model.onnx',
    export_params=True, opset_version=11,
    input_names=['input'], output_names=['output']
)
```

## Next Steps

1. **Experiment Tracking**: Set up MLflow or Weights & Biases for experiment management
2. **Model Registry**: Implement model versioning and deployment pipeline
3. **A/B Testing**: Set up A/B testing framework for model comparison
4. **Monitoring**: Implement model drift detection and performance monitoring
5. **Automation**: Create automated retraining pipelines

For more advanced topics, see:

- [Model Serving Guide](./model-serving.md)
- [A/B Testing Framework](./ab-testing.md)
- [Interpretability Guide](../user-guide/interpretability.md)
