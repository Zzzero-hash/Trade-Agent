# Model Training Tutorials and Best Practices

This comprehensive guide covers training CNN+LSTM hybrid models and reinforcement learning agents for the AI Trading Platform.

## Overview

The platform uses a sophisticated ML pipeline combining:
- **CNN models** for spatial pattern recognition in market data
- **LSTM models** for temporal sequence processing
- **Hybrid CNN+LSTM** architecture with attention mechanisms
- **Reinforcement Learning** agents for decision making
- **Ensemble methods** for robust predictions

## Quick Start Training

### 1. Basic Model Training

```python
from src.ml.training_pipeline import TrainingPipeline
from src.ml.hybrid_model import HybridCNNLSTM
from src.config.settings import get_settings

# Initialize training pipeline
pipeline = TrainingPipeline()

# Load and prepare data
data = pipeline.load_training_data(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Train hybrid model
model = HybridCNNLSTM(
    cnn_channels=[32, 64, 128],
    lstm_hidden_size=256,
    num_classes=3,  # Buy, Hold, Sell
    dropout_rate=0.2
)

# Start training
results = pipeline.train_model(
    model=model,
    data=data,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

print(f"Training completed. Best validation accuracy: {results['best_val_acc']:.4f}")
```

### 2. Distributed Training

```python
from src.ml.distributed_training import DistributedTrainer
import ray

# Initialize Ray for distributed training
ray.init(address='ray://head-node:10001')

# Create distributed trainer
trainer = DistributedTrainer(
    num_workers=4,
    use_gpu=True,
    resources_per_worker={'CPU': 2, 'GPU': 0.5}
)

# Train with distributed setup
results = trainer.train_distributed(
    model_config={
        'model_type': 'hybrid_cnn_lstm',
        'cnn_channels': [32, 64, 128, 256],
        'lstm_hidden_size': 512,
        'attention_heads': 8
    },
    data_config={
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'lookback_window': 60,
        'prediction_horizon': 5
    },
    training_config={
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
)
```

## CNN Model Training

### Architecture Configuration

```python
from src.ml.cnn_model import CNNModel

# Configure CNN architecture
cnn_config = {
    'input_channels': 5,  # OHLCV data
    'conv_layers': [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1},
        {'out_channels': 64, 'kernel_size': 5, 'stride': 1},
        {'out_channels': 128, 'kernel_size': 7, 'stride': 1},
        {'out_channels': 256, 'kernel_size': 11, 'stride': 1}
    ],
    'attention_heads': 8,
    'dropout_rate': 0.3,
    'batch_norm': True
}

# Create and train CNN model
cnn_model = CNNModel(**cnn_config)

# Training loop with best practices
optimizer = torch.optim.AdamW(
    cnn_model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# Training with gradient clipping
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        outputs = cnn_model(batch['features'])
        loss = criterion(outputs, batch['targets'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
```

### CNN Training Best Practices

#### 1. Data Preprocessing
```python
from src.ml.feature_engineering import FeatureEngineer

# Comprehensive feature engineering
feature_engineer = FeatureEngineer()

def preprocess_cnn_data(raw_data):
    # Technical indicators
    features = feature_engineer.add_technical_indicators(raw_data)
    
    # Normalize features
    features = feature_engineer.normalize_features(features, method='robust')
    
    # Create sliding windows
    windowed_data = feature_engineer.create_sliding_windows(
        features, window_size=60, step_size=1
    )
    
    # Add channel dimension for CNN
    cnn_input = windowed_data.unsqueeze(1)  # [batch, 1, sequence, features]
    
    return cnn_input
```

#### 2. Regularization Techniques
```python
# Dropout scheduling
class DropoutScheduler:
    def __init__(self, initial_rate=0.5, final_rate=0.1, total_epochs=100):
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_epochs = total_epochs
    
    def get_dropout_rate(self, epoch):
        progress = epoch / self.total_epochs
        return self.initial_rate * (1 - progress) + self.final_rate * progress

# Data augmentation for financial data
class FinancialDataAugmentation:
    def __init__(self, noise_std=0.01, time_shift_max=5):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
    
    def augment(self, data):
        # Add Gaussian noise
        noise = torch.randn_like(data) * self.noise_std
        augmented = data + noise
        
        # Time shifting
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        if shift != 0:
            augmented = torch.roll(augmented, shift, dims=1)
        
        return augmented
```

## LSTM Model Training

### Architecture Configuration

```python
from src.ml.lstm_model import LSTMModel

# Configure bidirectional LSTM with attention
lstm_config = {
    'input_size': 128,  # Feature dimension
    'hidden_size': 256,
    'num_layers': 3,
    'bidirectional': True,
    'dropout': 0.2,
    'attention_heads': 8,
    'use_skip_connections': True
}

lstm_model = LSTMModel(**lstm_config)
```

### LSTM Training Strategies

#### 1. Sequence-to-Sequence Training
```python
def train_lstm_seq2seq(model, data_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in data_loader:
            sequences = batch['sequences']  # [batch, seq_len, features]
            targets = batch['targets']      # [batch, pred_len, features]
            
            # Teacher forcing during training
            outputs = model(sequences, targets[:, :-1])
            loss = criterion(outputs, targets[:, 1:])
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.6f}")
```

#### 2. Attention Mechanism Training
```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads, batch_first=True)
        self.output_layer = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Self-attention over LSTM outputs
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use attention weights for interpretability
        self.last_attention_weights = attn_weights
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        output = self.output_layer(pooled)
        
        return output
```

## Hybrid CNN+LSTM Training

### Architecture Integration

```python
from src.ml.hybrid_model import HybridCNNLSTM

# Configure hybrid architecture
hybrid_config = {
    'cnn_config': {
        'input_channels': 5,
        'conv_layers': [32, 64, 128, 256],
        'kernel_sizes': [3, 5, 7, 11]
    },
    'lstm_config': {
        'hidden_size': 256,
        'num_layers': 2,
        'bidirectional': True
    },
    'fusion_config': {
        'fusion_method': 'cross_attention',
        'attention_heads': 8,
        'fusion_dim': 512
    },
    'output_config': {
        'num_classes': 3,  # Classification
        'regression_dim': 1,  # Price prediction
        'uncertainty_estimation': True
    }
}

hybrid_model = HybridCNNLSTM(**hybrid_config)
```

### Multi-Task Learning

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {'classification': 1.0, 'regression': 1.0}
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        class_pred, reg_pred = predictions
        class_target, reg_target = targets
        
        class_loss = self.classification_loss(class_pred, class_target)
        reg_loss = self.regression_loss(reg_pred, reg_target)
        
        total_loss = (
            self.task_weights['classification'] * class_loss +
            self.task_weights['regression'] * reg_loss
        )
        
        return total_loss, {'classification': class_loss, 'regression': reg_loss}

# Training with multi-task loss
def train_hybrid_model(model, data_loader, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = MultiTaskLoss(task_weights={'classification': 0.7, 'regression': 0.3})
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            features = batch['features']
            class_targets = batch['class_targets']
            reg_targets = batch['reg_targets']
            
            # Forward pass
            class_pred, reg_pred, uncertainty = model(features)
            
            # Calculate loss
            total_loss, task_losses = criterion(
                (class_pred, reg_pred),
                (class_targets, reg_targets)
            )
            
            # Add uncertainty regularization
            uncertainty_loss = torch.mean(uncertainty)
            total_loss += 0.1 * uncertainty_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

## Reinforcement Learning Training

### Environment Setup

```python
from src.ml.trading_environment import TradingEnvironment
from src.ml.rl_agents import PPOAgent, SACAgent

# Create trading environment
env_config = {
    'symbols': ['AAPL', 'GOOGL', 'MSFT'],
    'initial_balance': 100000,
    'transaction_cost': 0.001,
    'max_position_size': 0.1,
    'lookback_window': 60,
    'reward_function': 'sharpe_ratio'
}

env = TradingEnvironment(**env_config)
```

### Agent Training

#### 1. PPO Agent Training
```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Configure PPO agent
ppo_config = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5
}

# Create and train PPO agent
ppo_agent = PPO(env=env, **ppo_config)

# Set up callbacks
eval_callback = EvalCallback(
    eval_env=env,
    best_model_save_path='./models/ppo_best/',
    log_path='./logs/ppo_eval/',
    eval_freq=10000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path='./models/ppo_checkpoints/',
    name_prefix='ppo_model'
)

# Train the agent
ppo_agent.learn(
    total_timesteps=1000000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)
```

#### 2. SAC Agent Training
```python
from stable_baselines3 import SAC

# Configure SAC for continuous action spaces
sac_config = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 'auto',
    'target_update_interval': 1
}

sac_agent = SAC(env=env, **sac_config)

# Train SAC agent
sac_agent.learn(
    total_timesteps=500000,
    log_interval=4,
    progress_bar=True
)
```

### Ensemble Training

```python
from src.ml.rl_ensemble import RLEnsemble

# Create ensemble of RL agents
ensemble_config = {
    'agents': [
        {'type': 'PPO', 'weight': 0.4},
        {'type': 'SAC', 'weight': 0.3},
        {'type': 'TD3', 'weight': 0.2},
        {'type': 'DQN', 'weight': 0.1}
    ],
    'combination_method': 'weighted_average',
    'uncertainty_estimation': True,
    'thompson_sampling': True
}

ensemble = RLEnsemble(**ensemble_config)

# Train ensemble with meta-learning
ensemble.train_ensemble(
    env=env,
    total_timesteps=2000000,
    meta_learning_freq=50000,
    adaptation_rate=0.01
)
```

## Hyperparameter Optimization

### Ray Tune Integration

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

def train_with_hyperopt():
    # Define search space
    search_space = {
        'learning_rate': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'hidden_size': tune.choice([128, 256, 512, 1024]),
        'dropout_rate': tune.uniform(0.1, 0.5),
        'num_layers': tune.choice([2, 3, 4, 5]),
        'attention_heads': tune.choice([4, 8, 16]),
        'weight_decay': tune.loguniform(1e-6, 1e-3)
    }
    
    # Configure scheduler
    scheduler = ASHAScheduler(
        metric='val_accuracy',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    
    # Configure search algorithm
    search_alg = OptunaSearch(
        metric='val_accuracy',
        mode='max'
    )
    
    # Run hyperparameter optimization
    analysis = tune.run(
        trainable=train_model_with_config,
        config=search_space,
        num_samples=50,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={'cpu': 4, 'gpu': 1},
        local_dir='./ray_results'
    )
    
    # Get best configuration
    best_config = analysis.best_config
    print(f"Best configuration: {best_config}")
    
    return best_config
```

### Grid Search for RL Hyperparameters

```python
from itertools import product

def rl_hyperparameter_search():
    # Define parameter grids
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [64, 128, 256],
        'gamma': [0.95, 0.99, 0.995],
        'tau': [0.001, 0.005, 0.01],
        'ent_coef': [0.01, 0.1, 'auto']
    }
    
    best_score = -float('inf')
    best_params = None
    
    # Grid search
    for params in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), params))
        
        # Train agent with current configuration
        agent = SAC(env=env, **config)
        agent.learn(total_timesteps=100000)
        
        # Evaluate performance
        mean_reward = evaluate_agent(agent, env, n_episodes=10)
        
        if mean_reward > best_score:
            best_score = mean_reward
            best_params = config
    
    return best_params, best_score
```

## Model Evaluation and Validation

### Cross-Validation for Time Series

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(model, data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(data):
        # Split data
        train_data = data[train_idx]
        val_data = data[val_idx]
        
        # Train model
        model.fit(train_data)
        
        # Evaluate
        predictions = model.predict(val_data)
        score = calculate_metrics(predictions, val_data.targets)
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }
```

### Walk-Forward Analysis

```python
def walk_forward_analysis(model, data, window_size=252, step_size=21):
    """
    Perform walk-forward analysis for trading models
    
    Args:
        model: Trading model to evaluate
        data: Historical market data
        window_size: Training window size (e.g., 252 trading days = 1 year)
        step_size: Step size for moving window (e.g., 21 days = 1 month)
    """
    results = []
    
    for start_idx in range(0, len(data) - window_size - step_size, step_size):
        # Define training and testing periods
        train_end = start_idx + window_size
        test_start = train_end
        test_end = test_start + step_size
        
        # Split data
        train_data = data[start_idx:train_end]
        test_data = data[test_start:test_end]
        
        # Train model on training period
        model.fit(train_data)
        
        # Test on out-of-sample period
        predictions = model.predict(test_data)
        
        # Calculate performance metrics
        metrics = {
            'period_start': test_data.index[0],
            'period_end': test_data.index[-1],
            'returns': calculate_returns(predictions, test_data),
            'sharpe_ratio': calculate_sharpe_ratio(predictions, test_data),
            'max_drawdown': calculate_max_drawdown(predictions, test_data),
            'win_rate': calculate_win_rate(predictions, test_data)
        }
        
        results.append(metrics)
    
    return pd.DataFrame(results)
```

## Training Best Practices

### 1. Data Management

```python
class TrainingDataManager:
    def __init__(self, data_path, cache_size=1000):
        self.data_path = data_path
        self.cache = {}
        self.cache_size = cache_size
    
    def load_data_batch(self, symbols, start_date, end_date):
        """Load and cache training data efficiently"""
        cache_key = f"{'-'.join(symbols)}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load data
        data = self._load_raw_data(symbols, start_date, end_date)
        
        # Preprocess
        data = self._preprocess_data(data)
        
        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = data
        
        return data
    
    def create_data_loader(self, data, batch_size=32, shuffle=True):
        """Create PyTorch DataLoader with proper collation"""
        dataset = TradingDataset(data)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
```

### 2. Training Monitoring

```python
import wandb
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    def __init__(self, project_name, experiment_name):
        # Initialize Weights & Biases
        wandb.init(project=project_name, name=experiment_name)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(f'runs/{experiment_name}')
        
        self.step = 0
    
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = self.step
            self.step += 1
        
        # Log to W&B
        wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_model_graph(self, model, input_sample):
        self.writer.add_graph(model, input_sample)
    
    def log_attention_weights(self, attention_weights, step):
        # Visualize attention patterns
        fig = plot_attention_heatmap(attention_weights)
        wandb.log({"attention_heatmap": wandb.Image(fig)}, step=step)
        self.writer.add_figure('attention_weights', fig, step)
```

### 3. Model Checkpointing

```python
class ModelCheckpoint:
    def __init__(self, checkpoint_dir, save_best_only=True, monitor='val_loss', mode='min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
    
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': model.config if hasattr(model, 'config') else None
        }
        
        # Always save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if improved
        current_score = metrics[self.monitor]
        is_best = (
            (self.mode == 'min' and current_score < self.best_score) or
            (self.mode == 'max' and current_score > self.best_score)
        )
        
        if is_best:
            self.best_score = current_score
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            
            # Save model for inference
            model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), model_path)
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_type='best'):
        checkpoint_file = f'{checkpoint_type}_checkpoint.pth'
        checkpoint_path = self.checkpoint_dir / checkpoint_file
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
```

### 4. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metrics):
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:
            return current_score > self.best_score + self.min_delta
```

## Advanced Training Techniques

### 1. Curriculum Learning

```python
class CurriculumLearning:
    def __init__(self, difficulty_levels, progression_criteria):
        self.difficulty_levels = difficulty_levels
        self.progression_criteria = progression_criteria
        self.current_level = 0
    
    def get_current_data(self, full_dataset):
        """Get data for current difficulty level"""
        level_config = self.difficulty_levels[self.current_level]
        
        # Filter data based on difficulty
        if level_config['type'] == 'volatility':
            # Start with low volatility periods
            filtered_data = full_dataset[
                full_dataset['volatility'] <= level_config['max_volatility']
            ]
        elif level_config['type'] == 'market_regime':
            # Start with trending markets
            filtered_data = full_dataset[
                full_dataset['market_regime'] == level_config['regime']
            ]
        
        return filtered_data
    
    def should_progress(self, current_performance):
        """Check if should move to next difficulty level"""
        criteria = self.progression_criteria[self.current_level]
        
        if current_performance['accuracy'] >= criteria['min_accuracy']:
            if self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                return True
        
        return False
```

### 2. Meta-Learning for Few-Shot Adaptation

```python
class MAMLTrainer:
    """Model-Agnostic Meta-Learning for quick adaptation to new markets"""
    
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    def meta_train_step(self, support_tasks, query_tasks):
        """Perform one meta-training step"""
        meta_loss = 0
        
        for support_data, query_data in zip(support_tasks, query_tasks):
            # Clone model for inner loop
            fast_model = copy.deepcopy(self.model)
            
            # Inner loop: adapt to support set
            for _ in range(self.num_inner_steps):
                support_loss = self._compute_loss(fast_model, support_data)
                
                # Compute gradients
                grads = torch.autograd.grad(
                    support_loss, fast_model.parameters(), create_graph=True
                )
                
                # Update fast model parameters
                for param, grad in zip(fast_model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
            
            # Outer loop: evaluate on query set
            query_loss = self._compute_loss(fast_model, query_data)
            meta_loss += query_loss
        
        # Update meta-model
        meta_loss /= len(support_tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
```

### 3. Adversarial Training

```python
class AdversarialTrainer:
    """Adversarial training for robust financial models"""
    
    def __init__(self, model, epsilon=0.01, alpha=0.001, num_steps=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate_adversarial_examples(self, x, y):
        """Generate adversarial examples using PGD"""
        x_adv = x.clone().detach()
        x_adv.requires_grad_(True)
        
        for _ in range(self.num_steps):
            # Forward pass
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, y)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            
            # Update adversarial example
            x_adv = x_adv + self.alpha * grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid range
        
        return x_adv.detach()
    
    def adversarial_training_step(self, x, y, optimizer):
        """Perform adversarial training step"""
        # Generate adversarial examples
        x_adv = self.generate_adversarial_examples(x, y)
        
        # Train on both clean and adversarial examples
        optimizer.zero_grad()
        
        # Clean loss
        clean_outputs = self.model(x)
        clean_loss = F.cross_entropy(clean_outputs, y)
        
        # Adversarial loss
        adv_outputs = self.model(x_adv)
        adv_loss = F.cross_entropy(adv_outputs, y)
        
        # Combined loss
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
```

This comprehensive training guide provides the foundation for building robust, high-performance trading models using the AI Trading Platform's ML infrastructure.