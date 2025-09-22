"""
CNN+LSTM Training Pipeline

This module implements a comprehensive training pipeline for CNN+LSTM feature extractors
with mixed precision training, advanced optimization, comprehensive metrics tracking,
and early stopping with model checkpointing.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import wandb

from .hybrid_cnn_lstm_transformer import (
    HybridCNNLSTMTransformer,
    HybridArchitectureConfig
)
from .multi_scale_price_cnn import MultiScalePriceCNN, MultiScalePriceCNNConfig
from .bidirectional_lstm_attention import (
    BidirectionalLSTMWithAttention,
    BidirectionalLSTMConfig
)
from experiments.tracking import ExperimentTracker


@dataclass
class TrainingConfig:
    """Configuration for CNN+LSTM training pipeline."""
    
    # Model configuration
    model_type: str = "hybrid"  # "hybrid", "cnn", "lstm"
    model_config: Optional[Dict[str, Any]] = None
    
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    loss_scale: str = "dynamic"  # "dynamic" or float value
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 10
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "val_loss"  # "val_loss", "val_accuracy"
    
    # Checkpointing
    save_every_n_epochs: int = 10
    save_best_only: bool = True
    checkpoint_dir: str = "models/checkpoints"
    
    # Validation
    validation_split: float = 0.2
    validation_frequency: int = 1  # Validate every N epochs
    
    # Logging and tracking
    log_every_n_steps: int = 100
    track_feature_quality: bool = True
    use_wandb: bool = False
    experiment_name: str = "cnn_lstm_training"
    
    # Multi-task learning
    tasks: List[str] = field(default_factory=lambda: ["price_prediction"])
    task_weights: Dict[str, float] = field(default_factory=lambda: {"price_prediction": 1.0})
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_std: float = 0.01
    temporal_jitter_prob: float = 0.1
    price_scaling_range: Tuple[float, float] = (0.95, 1.05)
    
    # Device and performance
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureQualityMetrics:
    """Metrics for evaluating feature quality."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.feature_correlations = []
        self.feature_variances = []
        self.information_coefficients = []
        self.feature_stability_scores = []
    
    def update(self, features: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch."""
        with torch.no_grad():
            # Convert to numpy for sklearn metrics
            features_np = features.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Feature variance (higher is generally better)
            feature_var = np.var(features_np, axis=0).mean()
            self.feature_variances.append(feature_var)
            
            # Information coefficient (correlation with targets)
            if len(targets_np.shape) == 1 or targets_np.shape[1] == 1:
                targets_flat = targets_np.flatten()
                correlations = []
                
                # Only calculate IC if we have sufficient variance in targets
                if np.std(targets_flat) > 1e-8:
                    for i in range(features_np.shape[1]):
                        feature_col = features_np[:, i]
                        # Only calculate correlation if feature has variance
                        if np.std(feature_col) > 1e-8:
                            try:
                                corr = np.corrcoef(feature_col, targets_flat)[0, 1]
                                if not np.isnan(corr) and not np.isinf(corr):
                                    correlations.append(abs(corr))
                            except:
                                continue
                
                if correlations:
                    mean_corr = np.mean(correlations)
                    self.information_coefficients.append(mean_corr)
                else:
                    self.information_coefficients.append(0.0)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}
        
        if self.feature_variances:
            metrics["feature_variance"] = np.mean(self.feature_variances)
        
        if self.information_coefficients:
            metrics["information_coefficient"] = np.mean(self.information_coefficients)
        
        return metrics


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        metric: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        """Get comparison function based on mode."""
        if self.mode == "min":
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if self.metric not in metrics:
            return False
        
        current_score = metrics[self.metric]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CNNLSTMTrainer:
    """
    Comprehensive CNN+LSTM training pipeline with advanced optimization.
    
    Features:
    - Mixed precision training with automatic loss scaling
    - Comprehensive metrics tracking (loss, accuracy, feature quality)
    - Early stopping and model checkpointing
    - Multi-task learning support
    - Data augmentation
    - Advanced optimization strategies
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics and tracking
        self.feature_quality_metrics = FeatureQualityMetrics()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            metric=config.early_stopping_metric
        )
        
        # Experiment tracking
        self.experiment_tracker = ExperimentTracker()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "feature_quality": []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"CNNLSTMTrainer_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        if self.config.model_type == "hybrid":
            model_config = HybridArchitectureConfig(**self.config.model_config)
            return HybridCNNLSTMTransformer(model_config)
        elif self.config.model_type == "cnn":
            model_config = MultiScalePriceCNNConfig(**self.config.model_config)
            return MultiScalePriceCNN(model_config)
        elif self.config.model_type == "lstm":
            model_config = BidirectionalLSTMConfig(**self.config.model_config)
            return BidirectionalLSTMWithAttention(model_config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            T_0 = max(1, self.config.num_epochs // 4)  # Ensure T_0 is at least 1
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=2
            )
        elif self.config.scheduler == "step":
            step_size = max(1, self.config.num_epochs // 3)  # Ensure step_size is at least 1
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=10,
                factor=0.5
            )
        else:
            return None    

    def _apply_data_augmentation(
        self,
        data: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply data augmentation techniques."""
        if not self.config.use_data_augmentation:
            return data, targets
        
        augmented_data = {}
        
        for key, tensor in data.items():
            # Add noise injection
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(tensor) * self.config.noise_std
                tensor = tensor + noise
            
            # Price scaling (for price-related features)
            if "price" in key.lower() or key in ["1min", "5min", "15min"]:
                if torch.rand(1).item() < 0.3:
                    scale_factor = (torch.rand(1).item() * 
                                  (self.config.price_scaling_range[1] - self.config.price_scaling_range[0]) + 
                                  self.config.price_scaling_range[0])
                    tensor = tensor * scale_factor
            
            # Temporal jittering
            if tensor.dim() >= 3 and torch.rand(1).item() < self.config.temporal_jitter_prob:
                # Small random shifts in time dimension
                shift = torch.randint(-2, 3, (1,)).item()
                if shift != 0:
                    tensor = torch.roll(tensor, shift, dims=-1)
            
            augmented_data[key] = tensor
        
        return augmented_data, targets
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss."""
        total_loss = 0.0
        task_losses = {}
        
        for task in self.config.tasks:
            if task in outputs and task in targets:
                if task == "price_prediction":
                    # MSE loss for price prediction
                    loss = nn.MSELoss()(outputs[task], targets[task])
                elif task == "volatility_estimation":
                    # MSE loss for volatility
                    loss = nn.MSELoss()(outputs[task], targets[task])
                elif task == "regime_detection":
                    # Cross-entropy for regime classification
                    loss = nn.CrossEntropyLoss()(outputs[task], targets[task])
                else:
                    # Default to MSE
                    loss = nn.MSELoss()(outputs[task], targets[task])
                
                task_losses[f"{task}_loss"] = loss
                total_loss += self.config.task_weights.get(task, 1.0) * loss
        
        return total_loss, task_losses
    
    def _compute_single_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss for single target case."""
        if isinstance(outputs, dict):
            if "price_prediction" in outputs:
                # Use task-specific output
                loss = nn.MSELoss()(outputs["price_prediction"], targets)
            elif "output" in outputs:
                # Fallback to generic output
                loss = nn.MSELoss()(outputs["output"], targets)
            else:
                # Use first available output
                first_key = next(iter(outputs.keys()))
                loss = nn.MSELoss()(outputs[first_key], targets)
            task_losses = {"main_loss": loss}
        else:
            loss = nn.MSELoss()(outputs, targets)
            task_losses = {"main_loss": loss}
        
        return loss, task_losses
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        with torch.no_grad():
            for task in self.config.tasks:
                if task in outputs and task in targets:
                    pred = outputs[task].detach().cpu().numpy()
                    true = targets[task].detach().cpu().numpy()
                    
                    if task == "price_prediction" or task == "volatility_estimation":
                        # Regression metrics
                        mse = np.mean((pred - true) ** 2)
                        mae = np.mean(np.abs(pred - true))
                        
                        # Direction accuracy for price prediction
                        if task == "price_prediction":
                            pred_flat = pred[:, 0] if len(pred.shape) == 2 and pred.shape[1] > 0 else pred.flatten()
                            true_flat = true[:, 0] if len(true.shape) == 2 and true.shape[1] > 0 else true.flatten()
                            
                            # Handle zero values properly for direction accuracy
                            pred_direction = np.where(pred_flat > 0, 1, -1)
                            true_direction = np.where(true_flat > 0, 1, -1)
                            
                            # Only calculate accuracy for non-zero true values
                            non_zero_mask = np.abs(true_flat) > 1e-8
                            if np.sum(non_zero_mask) > 0:
                                direction_acc = np.mean(
                                    pred_direction[non_zero_mask] == true_direction[non_zero_mask]
                                )
                                metrics[f"{task}_direction_accuracy"] = direction_acc
                            else:
                                metrics[f"{task}_direction_accuracy"] = 0.0
                        
                        metrics[f"{task}_mse"] = mse
                        metrics[f"{task}_mae"] = mae
                        
                    elif task == "regime_detection":
                        # Classification metrics
                        pred_classes = np.argmax(pred, axis=1)
                        true_classes = true.flatten() if len(true.shape) > 1 else true
                        
                        accuracy = accuracy_score(true_classes, pred_classes)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            true_classes, pred_classes, average='weighted', zero_division=0
                        )
                        
                        metrics[f"{task}_accuracy"] = accuracy
                        metrics[f"{task}_precision"] = precision
                        metrics[f"{task}_recall"] = recall
                        metrics[f"{task}_f1"] = f1
        
        return metrics
    
    def _forward_model(self, data):
        """Forward pass handling different model types."""
        if self.config.model_type == "hybrid":
            # Hybrid model expects multi-timeframe data and sequence data
            if isinstance(data, dict):
                if "sequence_data" in data:
                    sequence_data = data.pop("sequence_data")
                    lengths = data.pop("lengths", None)
                    return self.model(data, sequence_data, lengths)
                else:
                    # Multi-timeframe data only
                    return self.model(data)
            else:
                return self.model(data)
        elif self.config.model_type == "cnn":
            # CNN model expects multi-timeframe data
            if isinstance(data, dict):
                # Remove sequence_data if present (not needed for CNN)
                cnn_data = {k: v for k, v in data.items() if k != "sequence_data" and k != "lengths"}
                return self.model(cnn_data)
            else:
                return self.model(data)
        elif self.config.model_type == "lstm":
            # LSTM model expects sequence data
            if isinstance(data, dict) and "sequence_data" in data:
                sequence_data = data["sequence_data"]
                lengths = data.get("lengths", None)
                return self.model(sequence_data, lengths)
            else:
                return self.model(data)
        else:
            # Default behavior
            return self.model(data)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        # Reset feature quality metrics
        if self.config.track_feature_quality:
            self.feature_quality_metrics.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
            else:
                raise ValueError("Expected batch to be (data, targets) tuple")
            
            # Handle different data formats
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
            
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            # Apply data augmentation
            if isinstance(data, dict):
                data, targets = self._apply_data_augmentation(data, targets)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self._forward_model(data)
                    
                    # Compute loss
                    if isinstance(targets, dict):
                        loss, task_losses = self._compute_loss(outputs, targets)
                    else:
                        loss, task_losses = self._compute_single_loss(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self._forward_model(data)
                
                # Compute loss
                if isinstance(targets, dict):
                    loss, task_losses = self._compute_loss(outputs, targets)
                else:
                    loss, task_losses = self._compute_single_loss(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Compute batch metrics
            if isinstance(targets, dict):
                batch_metrics = self._compute_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
            
            # Track feature quality
            if self.config.track_feature_quality and isinstance(outputs, dict):
                if "output" in outputs:
                    features = outputs["output"]
                    if isinstance(targets, dict) and "price_prediction" in targets:
                        target_values = targets["price_prediction"]
                    elif not isinstance(targets, dict):
                        target_values = targets
                    else:
                        target_values = None
                    
                    if target_values is not None:
                        self.feature_quality_metrics.update(features, target_values)
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_results = {
            "train_loss": np.mean(epoch_losses)
        }
        
        # Average batch metrics
        for key, values in epoch_metrics.items():
            epoch_results[f"train_{key}"] = np.mean(values)
        
        # Feature quality metrics
        if self.config.track_feature_quality:
            feature_quality = self.feature_quality_metrics.compute()
            epoch_results.update({f"train_{k}": v for k, v in feature_quality.items()})
        
        return epoch_results
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {}
        
        # Reset feature quality metrics
        if self.config.track_feature_quality:
            val_feature_quality = FeatureQualityMetrics()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                else:
                    raise ValueError("Expected batch to be (data, targets) tuple")
                
                # Handle different data formats
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(data, dict):
                    if "sequence_data" in data:
                        sequence_data = data["sequence_data"]
                        lengths = data.get("lengths", None)
                        # For LSTM model, we need to pass the sequence_data directly
                        if self.config.model_type == "lstm":
                            outputs = self.model(sequence_data, lengths)
                        else:
                            outputs = self.model(data, sequence_data, lengths)
                    else:
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                
                # Compute loss
                if isinstance(targets, dict):
                    loss, task_losses = self._compute_loss(outputs, targets)
                else:
                    loss, task_losses = self._compute_single_loss(outputs, targets)
                
                # Track metrics
                epoch_losses.append(loss.item())
                
                # Compute batch metrics
                if isinstance(targets, dict):
                    batch_metrics = self._compute_metrics(outputs, targets)
                    for key, value in batch_metrics.items():
                        if key not in epoch_metrics:
                            epoch_metrics[key] = []
                        epoch_metrics[key].append(value)
                
                # Track feature quality
                if self.config.track_feature_quality and isinstance(outputs, dict):
                    if "output" in outputs:
                        features = outputs["output"]
                        if isinstance(targets, dict) and "price_prediction" in targets:
                            target_values = targets["price_prediction"]
                        elif not isinstance(targets, dict):
                            target_values = targets
                        else:
                            target_values = None
                        
                        if target_values is not None:
                            val_feature_quality.update(features, target_values)
        
        # Compute epoch metrics
        epoch_results = {
            "val_loss": np.mean(epoch_losses)
        }
        
        # Average batch metrics
        for key, values in epoch_metrics.items():
            epoch_results[f"val_{key}"] = np.mean(values)
        
        # Feature quality metrics
        if self.config.track_feature_quality:
            feature_quality = val_feature_quality.compute()
            epoch_results.update({f"val_{k}": v for k, v in feature_quality.items()})
        
        return epoch_results
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config,
            "metrics": metrics,
            "training_history": self.training_history,
            "global_step": self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model checkpoint at epoch {epoch}")
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.training_history = checkpoint["training_history"]
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training results and final metrics
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        # Start experiment tracking
        run_id = self.experiment_tracker.start_run(
            run_name=f"{self.config.experiment_name}_{int(time.time())}",
            tags={
                "model_type": self.config.model_type,
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
                "mixed_precision": str(self.config.use_mixed_precision)
            }
        )
        
        # Log configuration
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not callable(v) and not k.startswith('_')
        }
        self.experiment_tracker.log_params(config_dict)
        
        # Initialize wandb if configured
        if self.config.use_wandb:
            wandb.init(
                project=self.config.experiment_name,
                config=config_dict,
                name=f"run_{int(time.time())}"
            )
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Model type: {self.config.model_type}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.use_mixed_precision}")
        
        best_metric = float('inf') if self.config.early_stopping_metric.endswith('loss') else 0.0
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation phase
                val_metrics = {}
                if val_loader and epoch % self.config.validation_frequency == 0:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if "val_loss" in epoch_metrics:
                            self.scheduler.step(epoch_metrics["val_loss"])
                    else:
                        self.scheduler.step()
                
                # Update training history
                for key, value in epoch_metrics.items():
                    if key not in self.training_history:
                        self.training_history[key] = []
                    self.training_history[key].append(value)
                
                # Log metrics
                self.experiment_tracker.log_metrics(epoch_metrics, step=epoch)
                
                if self.config.use_wandb:
                    wandb.log(epoch_metrics, step=epoch)
                
                # Check if this is the best model
                current_metric = epoch_metrics.get(self.config.early_stopping_metric, float('inf'))
                is_best = False
                
                if self.config.early_stopping_metric.endswith('loss'):
                    if current_metric < best_metric:
                        best_metric = current_metric
                        is_best = True
                        self.best_metrics = epoch_metrics.copy()
                else:
                    if current_metric > best_metric:
                        best_metric = current_metric
                        is_best = True
                        self.best_metrics = epoch_metrics.copy()
                
                # Save checkpoint
                if (epoch % self.config.save_every_n_epochs == 0 or 
                    is_best or 
                    epoch == self.config.num_epochs - 1):
                    
                    if self.config.save_best_only and not is_best and epoch != self.config.num_epochs - 1:
                        pass  # Skip saving non-best checkpoints
                    else:
                        self.save_checkpoint(epoch, epoch_metrics, is_best)
                
                # Early stopping check
                if val_loader and self.early_stopping(epoch_metrics):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Log epoch summary
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs - 1} completed in {epoch_time:.2f}s"
                )
                self.logger.info(f"Train Loss: {train_metrics.get('train_loss', 0):.6f}")
                if val_metrics:
                    self.logger.info(f"Val Loss: {val_metrics.get('val_loss', 0):.6f}")
                
                # Log feature quality if available
                if self.config.track_feature_quality:
                    if "train_information_coefficient" in train_metrics:
                        self.logger.info(
                            f"Train IC: {train_metrics['train_information_coefficient']:.4f}"
                        )
                    if "val_information_coefficient" in val_metrics:
                        self.logger.info(
                            f"Val IC: {val_metrics['val_information_coefficient']:.4f}"
                        )
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # End experiment tracking
            self.experiment_tracker.end_run()
            
            if self.config.use_wandb:
                wandb.finish()
        
        # Final model save
        final_checkpoint_path = Path(self.config.checkpoint_dir) / "final_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
            "best_metrics": self.best_metrics
        }, final_checkpoint_path)
        
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best metrics: {self.best_metrics}")
        
        return {
            "training_history": self.training_history,
            "best_metrics": self.best_metrics,
            "final_epoch": self.current_epoch,
            "model_path": str(final_checkpoint_path)
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            checkpoint_path: Path to model checkpoint (optional)
        
        Returns:
            Evaluation metrics
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        test_losses = []
        test_metrics = {}
        
        # Reset feature quality metrics
        if self.config.track_feature_quality:
            test_feature_quality = FeatureQualityMetrics()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move data to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                else:
                    raise ValueError("Expected batch to be (data, targets) tuple")
                
                # Handle different data formats
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(data, dict):
                    if "sequence_data" in data:
                        sequence_data = data.pop("sequence_data")
                        lengths = data.pop("lengths", None)
                        outputs = self.model(data, sequence_data, lengths)
                    else:
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                
                # Compute loss
                if isinstance(targets, dict):
                    loss, task_losses = self._compute_loss(outputs, targets)
                else:
                    loss, task_losses = self._compute_single_loss(outputs, targets)
                
                # Track metrics
                test_losses.append(loss.item())
                
                # Compute batch metrics
                if isinstance(targets, dict):
                    batch_metrics = self._compute_metrics(outputs, targets)
                    for key, value in batch_metrics.items():
                        if key not in test_metrics:
                            test_metrics[key] = []
                        test_metrics[key].append(value)
                
                # Track feature quality
                if self.config.track_feature_quality and isinstance(outputs, dict):
                    if "output" in outputs:
                        features = outputs["output"]
                        if isinstance(targets, dict) and "price_prediction" in targets:
                            target_values = targets["price_prediction"]
                        elif not isinstance(targets, dict):
                            target_values = targets
                        else:
                            target_values = None
                        
                        if target_values is not None:
                            test_feature_quality.update(features, target_values)
        
        # Compute final metrics
        results = {
            "test_loss": np.mean(test_losses)
        }
        
        # Average batch metrics
        for key, values in test_metrics.items():
            results[f"test_{key}"] = np.mean(values)
        
        # Feature quality metrics
        if self.config.track_feature_quality:
            feature_quality = test_feature_quality.compute()
            results.update({f"test_{k}": v for k, v in feature_quality.items()})
        
        self.logger.info("Evaluation completed")
        self.logger.info(f"Test metrics: {results}")
        
        return results


def create_trainer(config: TrainingConfig) -> CNNLSTMTrainer:
    """
    Factory function to create a CNN+LSTM trainer.
    
    Args:
        config: Training configuration
    
    Returns:
        Initialized CNNLSTMTrainer
    """
    return CNNLSTMTrainer(config)


# Example usage and configuration templates
if __name__ == "__main__":
    # Example configuration for hybrid CNN+LSTM+Transformer training
    hybrid_config = TrainingConfig(
        model_type="hybrid",
        model_config={
            "input_dim": 5,
            "sequence_length": 100,
            "fusion_dim": 1024,
            "use_adaptive_selection": True,
            "output_dim": 512
        },
        num_epochs=200,
        batch_size=32,
        learning_rate=1e-4,
        use_mixed_precision=True,
        early_stopping_patience=20,
        track_feature_quality=True,
        experiment_name="hybrid_cnn_lstm_training"
    )
    
    # Create trainer
    trainer = create_trainer(hybrid_config)
    
    print(f"Created trainer for {hybrid_config.model_type} model")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
