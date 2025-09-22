#!/usr/bin/env python3
"""
CNN Multi-timeframe Training Runner with Real Data Support

This script extends the original training runner to support both synthetic
and real market data from Yahoo Finance.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.cnn_lstm.multi_scale_price_cnn import MultiScalePriceCNN, MultiScalePriceCNNConfig
from data.pipeline import create_data_loaders, create_augmentation_transform
from data.datasets.real_market_dataset import create_real_data_loaders
from utils.gpu_utils import get_device_info, GPUManager
from experiments.tracking import ExperimentTracker


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    name: str
    epochs: int
    complexity: float
    patterns: List[str]
    data_fraction: float = 1.0
    augmentation_strength: float = 1.0


@dataclass
class CNNTrainingConfig:
    """Configuration for CNN multi-timeframe training."""
    
    # Model configuration
    model_config: MultiScalePriceCNNConfig = field(default_factory=lambda: MultiScalePriceCNNConfig(
        sequence_length=100,
        num_features=5,
        timeframes=['1min', '5min', '15min'],
        cnn_filters=[64, 128, 256, 512],
        kernel_sizes=[3, 5, 7, 9],
        dilation_rates=[1, 2, 4, 8],
        attention_dim=512,
        num_attention_heads=8,
        output_dim=512,
        dropout_rate=0.1
    ))
    
    # Training parameters
    total_epochs: int = 50  # Minimum 50 epochs as per requirements
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Curriculum learning
    use_curriculum_learning: bool = True
    curriculum_stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            name="simple_patterns",
            epochs=15,
            complexity=0.3,
            patterns=["trend", "support_resistance"],
            data_fraction=0.5,
            augmentation_strength=0.5
        ),
        CurriculumStage(
            name="intermediate_patterns", 
            epochs=20,
            complexity=0.6,
            patterns=["head_shoulders", "triangles", "flags"],
            data_fraction=0.75,
            augmentation_strength=0.75
        ),
        CurriculumStage(
            name="complex_patterns",
            epochs=15,
            complexity=1.0,
            patterns=["elliott_waves", "harmonic_patterns", "multi_timeframe"],
            data_fraction=1.0,
            augmentation_strength=1.0
        )
    ])
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_std: float = 0.01
    temporal_jitter_prob: float = 0.1
    price_scaling_range: Tuple[float, float] = (0.95, 1.05)
    pattern_masking_prob: float = 0.1
    
    # Validation and metrics
    validation_split: float = 0.2
    track_feature_quality: bool = True
    correlation_analysis: bool = True
    downstream_task_validation: bool = True
    
    # Device and performance
    device: str = "auto"
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints/cnn_multi_timeframe"
    save_every_n_epochs: int = 10
    
    # Logging
    log_every_n_steps: int = 50
    experiment_name: str = "cnn_multi_timeframe_training"
    
    # Data source configuration
    use_real_data: bool = False
    real_data_symbol: str = "SPY"
    real_data_dir: str = "data/processed"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureQualityAnalyzer:
    """Analyzer for CNN feature quality validation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.feature_correlations = []
        self.feature_variances = []
        self.information_coefficients = []
        self.pattern_recognition_scores = []
        self.timeframe_consistency_scores = []
    
    def analyze_features(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        timeframe_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive feature quality analysis.
        
        Args:
            features: Combined CNN features (batch_size, feature_dim)
            targets: Target values (batch_size, target_dim)
            timeframe_features: Individual timeframe features
            
        Returns:
            Dictionary of quality metrics
        """
        with torch.no_grad():
            features_np = features.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            metrics = {}
            
            # Feature variance (higher is generally better for diversity)
            feature_var = np.var(features_np, axis=0).mean()
            metrics["feature_variance"] = feature_var
            
            # Information coefficient (correlation with targets)
            if len(targets_np.shape) == 1 or targets_np.shape[1] == 1:
                targets_flat = targets_np.flatten()
                correlations = []
                for i in range(features_np.shape[1]):
                    corr = np.corrcoef(features_np[:, i], targets_flat)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                if correlations:
                    mean_ic = np.mean(correlations)
                    max_ic = np.max(correlations)
                    metrics["information_coefficient"] = mean_ic
                    metrics["max_information_coefficient"] = max_ic
            
            # Feature stability (consistency across batches)
            feature_stability = 1.0 - np.std(features_np, axis=0).mean() / (np.mean(features_np, axis=0).std() + 1e-8)
            metrics["feature_stability"] = feature_stability
            
            # Timeframe consistency analysis
            if timeframe_features is not None:
                consistency_scores = []
                timeframes = list(timeframe_features.keys())
                
                for i in range(len(timeframes)):
                    for j in range(i + 1, len(timeframes)):
                        tf1_features = timeframe_features[timeframes[i]].detach().cpu().numpy()
                        tf2_features = timeframe_features[timeframes[j]].detach().cpu().numpy()
                        
                        # Compute correlation between timeframe features
                        if tf1_features.shape[1] == tf2_features.shape[1]:
                            corr_matrix = np.corrcoef(tf1_features.T, tf2_features.T)
                            # Extract cross-correlation block
                            n_features = tf1_features.shape[1]
                            cross_corr = corr_matrix[:n_features, n_features:]
                            consistency_score = np.mean(np.abs(np.diag(cross_corr)))
                            consistency_scores.append(consistency_score)
                
                if consistency_scores:
                    metrics["timeframe_consistency"] = np.mean(consistency_scores)
            
            return metrics
    
    def update(self, metrics: Dict[str, float]):
        """Update running metrics."""
        if "feature_variance" in metrics:
            self.feature_variances.append(metrics["feature_variance"])
        if "information_coefficient" in metrics:
            self.information_coefficients.append(metrics["information_coefficient"])
        if "timeframe_consistency" in metrics:
            self.timeframe_consistency_scores.append(metrics["timeframe_consistency"])
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        summary = {}
        
        if self.feature_variances:
            summary["avg_feature_variance"] = np.mean(self.feature_variances)
            summary["std_feature_variance"] = np.std(self.feature_variances)
        
        if self.information_coefficients:
            summary["avg_information_coefficient"] = np.mean(self.information_coefficients)
            summary["std_information_coefficient"] = np.std(self.information_coefficients)
        
        if self.timeframe_consistency_scores:
            summary["avg_timeframe_consistency"] = np.mean(self.timeframe_consistency_scores)
            summary["std_timeframe_consistency"] = np.std(self.timeframe_consistency_scores)
        
        return summary


class PatternComplexityDataset:
    """Dataset wrapper that filters data based on pattern complexity."""
    
    def __init__(self, base_dataset, complexity: float, data_fraction: float = 1.0):
        self.base_dataset = base_dataset
        self.complexity = complexity
        self.data_fraction = data_fraction
        
        # Create indices based on complexity and data fraction
        total_samples = len(base_dataset)
        num_samples = int(total_samples * data_fraction)
        
        # For curriculum learning, we can simulate complexity by:
        # - Using a subset of data (simpler patterns first)
        # - Potentially filtering based on volatility or other complexity measures
        
        if complexity < 0.5:
            # Simple patterns: use first portion of data (typically more stable)
            self.indices = list(range(0, num_samples))
        elif complexity < 0.8:
            # Intermediate patterns: use middle portion
            start_idx = total_samples // 4
            end_idx = start_idx + num_samples
            self.indices = list(range(start_idx, min(end_idx, total_samples)))
        else:
            # Complex patterns: use all available data
            self.indices = list(range(0, num_samples))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


class AugmentedDataset:
    """Dataset wrapper that applies data augmentation transforms."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, targets = self.dataset[idx]
        if self.transform:
            data, targets = self.transform(data, targets)
        return data, targets


class CNNMultiTimeframeTrainer:
    """Trainer for CNN multi-timeframe price pattern recognition."""
    
    def __init__(self, config: CNNTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = MultiScalePriceCNN(config.model_config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Initialize metrics
        self.feature_analyzer = FeatureQualityAnalyzer()
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
            "feature_quality": [],
            "information_coefficient": [],
            "timeframe_consistency": []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"CNNMultiTimeframeTrainer_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_data_loaders(self):
        """Create data loaders based on configuration."""
        if self.config.use_real_data:
            self.logger.info(f"Using real market data for symbol: {self.config.real_data_symbol}")
            return create_real_data_loaders(
                symbol=self.config.real_data_symbol,
                data_dir=self.config.real_data_dir,
                timeframes=self.config.model_config.timeframes,
                sequence_length=self.config.model_config.sequence_length,
                target_columns=["price_prediction"],
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        else:
            self.logger.info("Using synthetic market data")
            return create_data_loaders(
                timeframes=self.config.model_config.timeframes,
                sequence_length=self.config.model_config.sequence_length,
                target_columns=["price_prediction"],
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
    
    def _create_augmented_data_loader(
        self, 
        base_dataset, 
        stage: CurriculumStage,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader with curriculum learning and augmentation."""
        
        # Apply curriculum learning filtering
        curriculum_dataset = PatternComplexityDataset(
            base_dataset, 
            stage.complexity, 
            stage.data_fraction
        )
        
        # Create augmentation transform
        if self.config.use_data_augmentation:
            augmentation_strength = stage.augmentation_strength
            transform = create_augmentation_transform(
                noise_std=self.config.noise_std * augmentation_strength,
                jitter_prob=self.config.temporal_jitter_prob * augmentation_strength,
                scale_range=(
                    1.0 - (1.0 - self.config.price_scaling_range[0]) * augmentation_strength,
                    1.0 + (self.config.price_scaling_range[1] - 1.0) * augmentation_strength
                )
            )
        else:
            transform = None
        
        # Apply transform to dataset if needed
        if transform:
            curriculum_dataset = AugmentedDataset(curriculum_dataset, transform)
        
        return DataLoader(
            curriculum_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def _compute_loss_and_metrics(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and evaluation metrics."""
        
        # Handle case where outputs is a dictionary
        if isinstance(outputs, dict):
            # Use the price_prediction output if available, otherwise use the main output
            if 'price_prediction' in outputs:
                outputs_tensor = outputs['price_prediction']
            elif 'output' in outputs:
                outputs_tensor = outputs['output']
            else:
                # Use the first available tensor output
                outputs_tensor = next(iter(outputs.values()))
        else:
            outputs_tensor = outputs
        
        # Primary loss (MSE for price prediction)
        loss = nn.MSELoss()(outputs_tensor, targets)
        
        metrics = {}
        
        with torch.no_grad():
            # Convert to numpy for metrics computation
            outputs_np = outputs_tensor.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # MSE and MAE
            mse = np.mean((outputs_np - targets_np) ** 2)
            mae = np.mean(np.abs(outputs_np - targets_np))
            
            metrics["mse"] = mse
            metrics["mae"] = mae
            
            # Direction accuracy (for price prediction)
            if outputs_np.shape[1] == 1:  # Single output
                pred_flat = outputs_np.flatten()
                true_flat = targets_np.flatten()
                
                # Handle zero values properly for direction accuracy
                pred_direction = np.where(pred_flat > 0, 1, -1)
                true_direction = np.where(true_flat > 0, 1, -1)
                
                # Only calculate accuracy for non-zero true values
                non_zero_mask = np.abs(true_flat) > 1e-8
                if np.sum(non_zero_mask) > 0:
                    direction_acc = np.mean(
                        pred_direction[non_zero_mask] == true_direction[non_zero_mask]
                    )
                    metrics["direction_accuracy"] = direction_acc
                else:
                    metrics["direction_accuracy"] = 0.0
        
        return loss, metrics
    
    def train_stage(
        self,
        stage: CurriculumStage,
        train_dataset,
        val_dataset
    ) -> Dict[str, float]:
        """Train for a specific curriculum stage."""
        
        self.logger.info(f"Starting curriculum stage: {stage.name}")
        self.logger.info(f"Stage parameters: epochs={stage.epochs}, complexity={stage.complexity}")
        
        # Create stage-specific data loaders
        train_loader = self._create_augmented_data_loader(
            train_dataset, stage, self.config.batch_size, shuffle=True
        )
        val_loader = self._create_augmented_data_loader(
            val_dataset, stage, self.config.batch_size, shuffle=False
        )
        
        stage_metrics = {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "feature_qualities": []
        }
        
        for epoch in range(stage.epochs):
            # Training phase
            train_results = self._train_epoch(train_loader, epoch)
            stage_metrics["train_losses"].append(train_results["loss"])
            stage_metrics["train_accuracies"].append(train_results.get("direction_accuracy", 0.0))
            
            # Validation phase
            val_results = self._validate_epoch(val_loader, epoch)
            stage_metrics["val_losses"].append(val_results["loss"])
            stage_metrics["val_accuracies"].append(val_results.get("direction_accuracy", 0.0))
            
            # Feature quality analysis
            if self.config.track_feature_quality:
                feature_quality = self._analyze_feature_quality(val_loader)
                stage_metrics["feature_qualities"].append(feature_quality)
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Stage {stage.name}, Epoch {epoch+1}/{stage.epochs}: "
                f"Train Loss: {train_results['loss']:.6f}, "
                f"Val Loss: {val_results['loss']:.6f}, "
                f"Val Acc: {val_results.get('direction_accuracy', 0.0):.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_results, stage.name)
            
            self.current_epoch += 1
        
        # Compute stage summary
        stage_summary = {
            "avg_train_loss": np.mean(stage_metrics["train_losses"]),
            "avg_val_loss": np.mean(stage_metrics["val_losses"]),
            "avg_train_accuracy": np.mean(stage_metrics["train_accuracies"]),
            "avg_val_accuracy": np.mean(stage_metrics["val_accuracies"]),
            "final_train_loss": stage_metrics["train_losses"][-1],
            "final_val_loss": stage_metrics["val_losses"][-1],
            "final_train_accuracy": stage_metrics["train_accuracies"][-1],
            "final_val_accuracy": stage_metrics["val_accuracies"][-1]
        }
        
        if stage_metrics["feature_qualities"]:
            avg_feature_quality = {}
            for key in stage_metrics["feature_qualities"][0].keys():
                values = [fq[key] for fq in stage_metrics["feature_qualities"] if key in fq]
                if values:
                    avg_feature_quality[f"avg_{key}"] = np.mean(values)
            stage_summary.update(avg_feature_quality)
        
        self.logger.info(f"Completed stage {stage.name}: {stage_summary}")
        return stage_summary
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
            
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss and metrics
            loss, batch_metrics = self._compute_loss_and_metrics(outputs, targets)
            
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
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                self.logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )
            
            self.global_step += 1
        
        # Compute epoch averages
        results = {"loss": np.mean(epoch_losses)}
        for key, values in epoch_metrics.items():
            results[key] = np.mean(values)
        
        return results
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {}
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss and metrics
                loss, batch_metrics = self._compute_loss_and_metrics(outputs, targets)
                
                # Track metrics
                epoch_losses.append(loss.item())
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
        
        # Compute epoch averages
        results = {"loss": np.mean(epoch_losses)}
        for key, values in epoch_metrics.items():
            results[key] = np.mean(values)
        
        return results
    
    def _analyze_feature_quality(self, data_loader: DataLoader) -> Dict[str, float]:
        """Analyze feature quality on validation data."""
        self.model.eval()
        
        all_features = []
        all_targets = []
        timeframe_features = {tf: [] for tf in self.config.model_config.timeframes}
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit analysis to first 10 batches for efficiency
                    break
                
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                
                targets = targets.to(self.device)
                
                # Get features
                features = self.model(data)
                
                # Handle case where features is a dictionary
                if isinstance(features, dict):
                    # Use the main output if available, otherwise use the first available tensor
                    if 'output' in features:
                        features_tensor = features['output']
                    else:
                        features_tensor = next(iter(features.values()))
                else:
                    features_tensor = features
                
                # Get individual timeframe features for consistency analysis
                batch_timeframe_features = {}
                for timeframe in self.config.model_config.timeframes:
                    if timeframe in data:
                        tf_features = self.model.timeframe_branches[timeframe](data[timeframe])
                        batch_timeframe_features[timeframe] = tf_features
                        timeframe_features[timeframe].append(tf_features)
                
                all_features.append(features_tensor)
                all_targets.append(targets)
        
        if not all_features:
            return {}
        
        # Concatenate all features and targets
        combined_features = torch.cat(all_features, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Concatenate timeframe features
        combined_timeframe_features = {}
        for timeframe, feature_list in timeframe_features.items():
            if feature_list:
                combined_timeframe_features[timeframe] = torch.cat(feature_list, dim=0)
        
        # Analyze feature quality
        quality_metrics = self.feature_analyzer.analyze_features(
            combined_features,
            combined_targets,
            combined_timeframe_features if combined_timeframe_features else None
        )
        
        return quality_metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], stage_name: str = ""):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "training_history": self.training_history,
            "stage_name": stage_name
        }
        
        # Save checkpoint
        checkpoint_name = f"cnn_checkpoint_epoch_{epoch}"
        if stage_name:
            checkpoint_name += f"_{stage_name}"
        checkpoint_name += ".pt"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with curriculum learning."""
        
        self.logger.info("Starting CNN multi-timeframe training")
        self.logger.info(f"Total epochs: {self.config.total_epochs}")
        self.logger.info(f"Curriculum learning: {self.config.use_curriculum_learning}")
        self.logger.info(f"Using real data: {self.config.use_real_data}")
        
        # Create data loaders
        try:
            train_loader, val_loader, test_loader = self._create_data_loaders()
        except Exception as e:
            self.logger.error(f"Error creating data loaders: {e}")
            raise
        
        # Get base datasets from loaders
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        results = {
            "stage_results": {},
            "overall_metrics": {},
            "training_history": self.training_history
        }
        
        if self.config.use_curriculum_learning:
            # Train through curriculum stages
            for stage in self.config.curriculum_stages:
                stage_results = self.train_stage(stage, train_dataset, val_dataset)
                results["stage_results"][stage.name] = stage_results
                
                # Update overall training history
                self.training_history["train_loss"].extend([stage_results["final_train_loss"]])
                self.training_history["val_loss"].extend([stage_results["final_val_loss"]])
                self.training_history["train_accuracy"].extend([stage_results["final_train_accuracy"]])
                self.training_history["val_accuracy"].extend([stage_results["final_val_accuracy"]])
        else:
            # Standard training without curriculum
            single_stage = CurriculumStage(
                name="standard_training",
                epochs=self.config.total_epochs,
                complexity=1.0,
                patterns=["all"],
                data_fraction=1.0,
                augmentation_strength=1.0
            )
            stage_results = self.train_stage(single_stage, train_dataset, val_dataset)
            results["stage_results"]["standard_training"] = stage_results
        
        # Final validation and feature quality analysis
        final_val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        final_metrics = self._validate_epoch(final_val_loader, self.current_epoch)
        final_feature_quality = self._analyze_feature_quality(final_val_loader)
        
        results["overall_metrics"] = {
            **final_metrics,
            **final_feature_quality
        }
        
        # Compute feature quality summary
        feature_summary = self.feature_analyzer.compute_summary()
        results["feature_quality_summary"] = feature_summary
        
        # Save final model
        final_checkpoint_path = Path(self.config.checkpoint_dir) / "final_cnn_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "results": results
        }, final_checkpoint_path)
        
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Final metrics: {results['overall_metrics']}")
        
        return results


def main():
    """Main function to run CNN multi-timeframe training."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiments/logs/cnn_multi_timeframe_real_training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup GPU
    gpu_manager = GPUManager()
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Create configuration for real data training
    config = CNNTrainingConfig(
        total_epochs=50,  # Minimum 50 epochs as per requirements
        batch_size=64,
        learning_rate=2e-4,
        use_curriculum_learning=True,
        use_data_augmentation=True,
        track_feature_quality=True,
        correlation_analysis=True,
        downstream_task_validation=True,
        use_real_data=True,
        real_data_symbol="SPY",
        real_data_dir="data/processed"
    )
    
    # Create trainer
    trainer = CNNMultiTimeframeTrainer(config)
    
    # Create logs directory
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    
    try:
        # Run training
        results = trainer.train()
        
        # Save results
        results_path = Path("experiments/results") / "cnn_multi_timeframe_real_results.json"
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("CNN MULTI-TIMEFRAME TRAINING WITH REAL DATA COMPLETED")
        print("="*80)
        
        print(f"\nFinal Metrics:")
        for key, value in results["overall_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        if "feature_quality_summary" in results:
            print(f"\nFeature Quality Summary:")
            for key, value in results["feature_quality_summary"].items():
                print(f"  {key}: {value:.6f}")
        
        print(f"\nStage Results:")
        for stage_name, stage_results in results["stage_results"].items():
            print(f"  {stage_name}:")
            print(f"    Final Val Loss: {stage_results['final_val_loss']:.6f}")
            print(f"    Final Val Accuracy: {stage_results['final_val_accuracy']:.6f}")
        
        # Check if requirements are met
        print(f"\nRequirement Validation:")
        
        # Check if >70% accuracy achieved (Requirement 1.1)
        final_accuracy = results["overall_metrics"].get("direction_accuracy", 0.0)
        print(f"  Price Direction Accuracy: {final_accuracy:.4f} (Target: >0.70)")
        
        # Check information coefficient (Requirement 1.1)
        ic = results["overall_metrics"].get("information_coefficient", 0.0)
        print(f"  Information Coefficient: {ic:.4f} (Target: >0.15)")
        
        # Check feature quality
        if "feature_quality_summary" in results:
            avg_ic = results["feature_quality_summary"].get("avg_information_coefficient", 0.0)
            print(f"  Average Information Coefficient: {avg_ic:.4f}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
