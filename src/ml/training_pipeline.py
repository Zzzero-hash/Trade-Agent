"""Training Pipeline for CNN Feature Extraction Model

This module provides utilities for training CNN models with proper data loading,
model checkpointing, versioning, and monitoring capabilities.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import logging

from .cnn_model import CNNFeatureExtractor, create_cnn_config, create_cnn_data_loader
from .base_models import ModelConfig, TrainingResult
from .decision_auditor import DecisionAuditor, create_decision_auditor
from .uncertainty_calibrator import UncertaintyCalibrator, create_uncertainty_calibrator


class MarketDataset(Dataset):
    """Custom Dataset for market data with sliding windows"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 50,
        prediction_horizon: int = 1
    ):
        """Initialize market dataset
        
        Args:
            features: Feature array of shape (samples, features)
            targets: Target array of shape (samples, targets)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Calculate valid sample indices
        self.valid_indices = self._calculate_valid_indices()
        
    def _calculate_valid_indices(self) -> List[int]:
        """Calculate valid starting indices for sequences"""
        max_start_idx = len(self.features) - self.sequence_length - self.prediction_horizon + 1
        return list(range(max_start_idx))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample"""
        start_idx = self.valid_indices[idx]
        
        # Extract sequence
        feature_seq = self.features[start_idx:start_idx + self.sequence_length]
        target_seq = self.targets[start_idx + self.prediction_horizon:
                                 start_idx + self.sequence_length + self.prediction_horizon]
        
        # Transpose features for CNN: (features, sequence_length)
        feature_seq = feature_seq.T
        
        return torch.FloatTensor(feature_seq), torch.FloatTensor(target_seq)


class ModelCheckpoint:
    """Model checkpointing utility with versioning"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = "cnn_feature_extractor",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        """Initialize model checkpoint
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Base name for model files
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitored metric
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        model: CNNFeatureExtractor,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> bool:
        """Save model checkpoint
        
        Args:
            model: CNN model to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        current_metric = metrics.get(self.monitor, None)
        if current_metric is None:
            self.logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return False
        
        # Check if this is the best model
        is_best = False
        if self.mode == 'min':
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric
        
        # Save checkpoint if not save_best_only or if this is the best
        if not self.save_best_only or is_best:
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
            
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
                'timestamp': datetime.now().isoformat(),
                'model_config': model.config
            }
            
            # Add optimizer and scheduler states if provided
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Generate filename
            if is_best:
                filename = f"{self.model_name}_best.pth"
            else:
                filename = f"{self.model_name}_epoch_{epoch:04d}.pth"
            
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Save checkpoint
            torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
            
            # Save metadata
            metadata_path = filepath.replace('.pth', '_metadata.json')
            metadata = {
                'epoch': epoch,
                'metrics': metrics,
                'is_best': is_best,
                'timestamp': checkpoint['timestamp'],
                'model_type': 'CNNFeatureExtractor',
                'filepath': filepath
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved checkpoint: {filename}")
            if is_best:
                self.logger.info(f"New best {self.monitor}: {current_metric:.6f}")
            
            return True
        
        return False
    
    def load_best_checkpoint(self, model: CNNFeatureExtractor) -> Dict[str, Any]:
        """Load the best checkpoint
        
        Args:
            model: CNN model to load weights into
            
        Returns:
            Checkpoint data dictionary
        """
        best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        checkpoint = torch.load(best_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class CNNTrainingPipeline:
    """Complete training pipeline for CNN feature extractor"""
    
    def __init__(
        self,
        config: ModelConfig,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        decision_auditor: Optional[DecisionAuditor] = None,
        uncertainty_calibrator: Optional[UncertaintyCalibrator] = None
    ):
        """Initialize training pipeline
        
        Args:
            config: Model configuration
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = CNNFeatureExtractor(config)
        
        # Setup checkpointing
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            model_name="cnn_feature_extractor"
        )
        
        # Setup TensorBoard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Initialize decision auditor
        self.decision_auditor = decision_auditor or create_decision_auditor()

        # Initialize uncertainty calibrator
        self.uncertainty_calibrator = uncertainty_calibrator or create_uncertainty_calibrator(self.model)
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = os.path.join(self.log_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 50,
        prediction_horizon: int = 1,
        train_split: float = 0.8,
        val_split: float = 0.1,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training
        
        Args:
            features: Feature array of shape (samples, features)
            targets: Target array of shape (samples, targets)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            batch_size: Batch size (uses config if None)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Create dataset
        dataset = MarketDataset(
            features=features,
            targets=targets,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Data prepared: Train={len(train_dataset)}, "
                        f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 20,
        gradient_clip_norm: float = 1.0
    ) -> TrainingResult:
        """Train the CNN model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            early_stopping_patience: Patience for early stopping
            gradient_clip_norm: Maximum gradient norm for clipping
            
        Returns:
            Training results
        """
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, gradient_clip_norm
            )
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Prepare metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            # Log metrics
            self._log_metrics(metrics, epoch)
            
            # Save checkpoint
            self.checkpoint.save_checkpoint(
                self.model, epoch, metrics, optimizer, scheduler
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Store history
            self.training_history.append(metrics)
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Load best model
        try:
            self.checkpoint.load_best_checkpoint(self.model)
            self.model.is_trained = True
        except FileNotFoundError:
            self.logger.warning("Best checkpoint not found, using current model")

        # Calibrate uncertainty
        try:
            self.logger.info("Calibrating model uncertainty...")
            # Extract validation data for calibration
            X_val = np.concatenate([batch[0].numpy() for batch in val_loader], axis=0)
            y_class_val = np.concatenate([batch[1].numpy() for batch in val_loader], axis=0)
            y_reg_val = np.concatenate([batch[1].numpy() for batch in val_loader], axis=0)

            self.uncertainty_calibrator.calibrate_uncertainty_isotonic(
                X_val=X_val,
                y_class_val=y_class_val,
                y_reg_val=y_reg_val
            )
            self.logger.info("Uncertainty calibration completed.")
        except Exception as e:
            self.logger.error(f"Failed to calibrate uncertainty: {e}")

        # Register model version with the auditor
        self.decision_auditor.register_model_version(
            model=self.model,
            training_data_hash="<placeholder_training_data_hash>",  # TODO: Implement data hashing
            hyperparameters=self.config.__dict__,
            performance_metrics={'val_loss': best_val_loss}
        )
        
        return TrainingResult(
            train_loss=train_loss,
            val_loss=val_loss,
            epochs_trained=epoch + 1,
            best_epoch=self.checkpoint.best_epoch
        )
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        gradient_clip_norm: float
    ) -> float:
        """Train for one epoch"""
        self.model.train(True)  # Use PyTorch's train method
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.model.device)
            batch_y = batch_y.to(self.model.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model.forward(batch_x)
            
            # Compute loss
            loss = criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=gradient_clip_norm
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.model.device)
                batch_y = batch_y.to(self.model.device)
                
                output = self.model.forward(batch_x)
                loss = criterion(output, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log training metrics"""
        # Console logging
        self.logger.info(
            f"Epoch {epoch+1:4d}: "
            f"Train Loss: {metrics['train_loss']:.6f}, "
            f"Val Loss: {metrics['val_loss']:.6f}, "
            f"LR: {metrics['learning_rate']:.2e}"
        )
        
        # TensorBoard logging
        if self.writer is not None:
            for key, value in metrics.items():
                if key != 'epoch':
                    self.writer.add_scalar(key, value, epoch)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.model.device)
                batch_y = batch_y.to(self.model.device)
                
                output = self.model.forward(batch_x)
                
                loss = criterion(output, batch_y)
                mae = mae_criterion(output, batch_y)
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        metrics = {
            'test_loss': total_loss / num_batches,
            'test_mae': total_mae / num_batches,
            'test_rmse': np.sqrt(total_loss / num_batches)
        }
        
        self.logger.info("Test Results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        return metrics
    
    def close(self) -> None:
        """Close training pipeline and cleanup resources"""
        if self.writer is not None:
            self.writer.close()
        
        self.logger.info("Training pipeline closed")


def create_training_pipeline(
    input_dim: int,
    output_dim: int,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    **config_kwargs
) -> CNNTrainingPipeline:
    """Create a complete CNN training pipeline
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured training pipeline
    """
    config = create_cnn_config(
        input_dim=input_dim,
        output_dim=output_dim,
        **config_kwargs
    )
    
    pipeline = CNNTrainingPipeline(
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    return pipeline