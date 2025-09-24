"""CNN+LSTM Hybrid Model for Trading Platform

This module implements the CNN+LSTM hybrid architecture that combines
CNN feature extraction with LSTM temporal processing for multi-task
learning (classification and regression) with ensemble capabilities
and uncertainty quantification.

Requirements: 1.1, 1.4, 5.6
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime
from dataclasses import dataclass

from .base_models import BasePyTorchModel, ModelConfig, TrainingResult
from .cnn_model import CNNFeatureExtractor, MultiHeadAttention
from .lstm_model import LSTMTemporalProcessor, LSTMAttention


@dataclass
class HybridModelConfig(ModelConfig):
    """Configuration for CNN+LSTM hybrid model"""
    # CNN configuration
    cnn_filter_sizes: List[int] = None
    cnn_num_filters: int = 64
    cnn_use_attention: bool = True
    cnn_attention_heads: int = 8
    
    # LSTM configuration
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 3
    lstm_bidirectional: bool = True
    lstm_use_attention: bool = True
    lstm_use_skip_connections: bool = True
    
    # Hybrid configuration
    feature_fusion_dim: int = 256
    sequence_length: int = 60
    prediction_horizon: int = 10
    
    # Multi-task configuration
    num_classes: int = 3  # Buy, Hold, Sell
    regression_targets: int = 2  # Price prediction + volatility estimation
    
    # Ensemble configuration
    num_ensemble_models: int = 5
    ensemble_dropout_rate: float = 0.1
    
    # Uncertainty quantification
    use_monte_carlo_dropout: bool = True
    mc_dropout_samples: int = 100
    
    # Training configuration
    classification_weight: float = 0.4
    regression_weight: float = 0.6
    dropout_rate: float = 0.3
    
    # Optimizer configuration
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential", "none"
    step_size: int = 20
    gamma: float = 0.5
    
    # Feature fusion configuration
    fusion_dim: int = 512
    fusion_heads: int = 8
    
    def __post_init__(self):
        if self.cnn_filter_sizes is None:
            self.cnn_filter_sizes = [3, 5, 7, 11]


class FeatureFusion(nn.Module):
    """Feature fusion module for combining CNN and LSTM features"""
    
    def __init__(
        self,
        cnn_feature_dim: int,
        lstm_feature_dim: int,
        fusion_dim: int,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.cnn_projection = nn.Linear(cnn_feature_dim, fusion_dim)
        self.lstm_projection = nn.Linear(lstm_feature_dim, fusion_dim)
        
        # Cross-attention between CNN and LSTM features
        self.cross_attention = MultiHeadAttention(fusion_dim, num_heads=8)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(
        self,
        cnn_features: torch.Tensor,
        lstm_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: (batch_size, seq_len, cnn_feature_dim)
            lstm_features: (batch_size, seq_len, lstm_feature_dim)
            
        Returns:
            fused_features: (batch_size, seq_len, fusion_dim)
        """
        # Project features to common dimension
        cnn_proj = self.cnn_projection(cnn_features)
        lstm_proj = self.lstm_projection(lstm_features)
        
        # Cross-attention: CNN attends to LSTM
        cnn_attended, _ = self.cross_attention(cnn_proj, lstm_proj, lstm_proj)
        
        # Cross-attention: LSTM attends to CNN
        lstm_attended, _ = self.cross_attention(lstm_proj, cnn_proj, cnn_proj)
        
        # Concatenate attended features
        fused = torch.cat([cnn_attended, lstm_attended], dim=-1)
        
        # Apply fusion layers
        fused_features = self.fusion_layers(fused)
        
        return fused_features


class UncertaintyQuantification(nn.Module):
    """Uncertainty quantification module using Monte Carlo Dropout"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 4, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using Monte Carlo Dropout
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            mean_prediction: Mean prediction
            uncertainty: Prediction uncertainty (std)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (num_samples, batch_size, output_dim)
        
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_prediction, uncertainty


class CNNLSTMHybridModel(BasePyTorchModel):
    """CNN+LSTM Hybrid Model with Multi-task Learning and Ensemble Capabilities
    
    This model integrates CNN feature extraction with LSTM temporal processing
    for multi-task learning (classification and regression) with uncertainty
    quantification and ensemble capabilities.
    """
    
    def __init__(self, config: HybridModelConfig):
        super().__init__(config)  # Base class handles device optimization
        
        self.config = config
        self.sequence_length = config.sequence_length
        self.prediction_horizon = config.prediction_horizon
        
        # Build model architecture
        self.build_model()
        
    def build_model(self) -> None:
        """Build CNN+LSTM hybrid architecture"""
        
        # CNN Feature Extractor
        # After transposing data: (batch, sequence_length, features)
        # CNN expects (batch, channels, sequence) so channels = sequence_length
        cnn_config = ModelConfig(
            model_type="CNNFeatureExtractor",
            input_dim=self.config.sequence_length,  # Use sequence_length as input channels
            output_dim=self.config.feature_fusion_dim // 2,
            hidden_dims=[self.config.cnn_num_filters * len(self.config.cnn_filter_sizes)],
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            device=self.config.device
        )
        
        # Add CNN-specific attributes
        cnn_config.filter_sizes = self.config.cnn_filter_sizes
        cnn_config.num_filters = self.config.cnn_num_filters
        cnn_config.use_attention = self.config.cnn_use_attention
        cnn_config.num_attention_heads = self.config.cnn_attention_heads
        cnn_config.dropout_rate = self.config.dropout_rate
        
        self.cnn_extractor = CNNFeatureExtractor(cnn_config)
        
        # LSTM Temporal Processor
        lstm_config = ModelConfig(
            model_type="LSTMTemporalProcessor",
            input_dim=self.config.feature_fusion_dim // 2,  # CNN output
            output_dim=self.config.feature_fusion_dim // 2,
            hidden_dims=[self.config.lstm_hidden_dim],
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            device=self.config.device
        )
        
        # Add LSTM-specific attributes
        lstm_config.hidden_dim = self.config.lstm_hidden_dim
        lstm_config.num_layers = self.config.lstm_num_layers
        lstm_config.sequence_length = self.config.sequence_length
        lstm_config.bidirectional = self.config.lstm_bidirectional
        lstm_config.use_attention = self.config.lstm_use_attention
        lstm_config.use_skip_connections = self.config.lstm_use_skip_connections
        lstm_config.dropout_rate = self.config.dropout_rate
        
        self.lstm_processor = LSTMTemporalProcessor(lstm_config)
        
        # Feature Fusion Module
        cnn_output_dim = self.config.feature_fusion_dim // 2
        lstm_output_dim = (self.config.lstm_hidden_dim * 2 
                          if self.config.lstm_bidirectional 
                          else self.config.lstm_hidden_dim)
        
        self.feature_fusion = FeatureFusion(
            cnn_feature_dim=cnn_output_dim,
            lstm_feature_dim=lstm_output_dim,
            fusion_dim=self.config.feature_fusion_dim,
            dropout_rate=self.config.dropout_rate
        )
        
        # Multi-task Output Heads
        
        # Classification head (Buy/Hold/Sell)
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.feature_fusion_dim, self.config.feature_fusion_dim // 2),
            nn.LayerNorm(self.config.feature_fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.feature_fusion_dim // 2, self.config.num_classes)
        )
        
        # Regression head (Price prediction)
        self.regression_head = UncertaintyQuantification(
            input_dim=self.config.feature_fusion_dim,
            output_dim=self.config.regression_targets,
            dropout_rate=self.config.dropout_rate
        )
        
        # Ensemble components
        self.ensemble_models = nn.ModuleList()
        for i in range(self.config.num_ensemble_models):
            ensemble_model = nn.Sequential(
                nn.Linear(self.config.feature_fusion_dim, self.config.feature_fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.config.ensemble_dropout_rate),
                nn.Linear(self.config.feature_fusion_dim // 2, self.config.num_classes + self.config.regression_targets)
            )
            self.ensemble_models.append(ensemble_model)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(self.config.num_ensemble_models))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        use_ensemble: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model
        
        Args:
            x: Input tensor (batch_size, input_channels, sequence_length)
            return_features: Whether to return intermediate features
            use_ensemble: Whether to use ensemble predictions
            
        Returns:
            Dictionary containing predictions and optionally features
        """
        batch_size, input_channels, seq_len = x.shape
        
        # CNN Feature Extraction
        # Reshape for CNN: (batch_size, input_channels, sequence_length)
        cnn_features = self.cnn_extractor.forward(x)  # (batch_size, seq_len, cnn_output_dim)
        
        # LSTM Temporal Processing
        # Use CNN features as input to LSTM
        lstm_features, lstm_context = self.lstm_processor.forward_encoder_only(cnn_features)
        
        # Feature Fusion
        fused_features = self.feature_fusion(cnn_features, lstm_features)
        
        # Multi-task Predictions
        
        # Classification (use last timestep)
        classification_logits = self.classification_head(fused_features[:, -1, :])
        classification_probs = F.softmax(classification_logits, dim=-1)
        
        # Regression with uncertainty
        if self.config.use_monte_carlo_dropout:
            regression_mean, regression_uncertainty = self.regression_head.predict_with_uncertainty(
                fused_features[:, -1, :],
                num_samples=self.config.mc_dropout_samples
            )
        else:
            regression_mean = self.regression_head(fused_features[:, -1, :])
            regression_uncertainty = torch.zeros_like(regression_mean)
        
        # Ensemble Predictions
        ensemble_outputs = []
        if use_ensemble:
            for ensemble_model in self.ensemble_models:
                ensemble_out = ensemble_model(fused_features[:, -1, :])
                ensemble_outputs.append(ensemble_out)
            
            # Weighted ensemble
            ensemble_stack = torch.stack(ensemble_outputs)  # (num_models, batch_size, output_dim)
            ensemble_weights_norm = F.softmax(self.ensemble_weights, dim=0)
            
            # Separate classification and regression parts
            total_output_dim = self.config.num_classes + self.config.regression_targets
            ensemble_classification = ensemble_stack[:, :, :self.config.num_classes]
            ensemble_regression = ensemble_stack[:, :, self.config.num_classes:]
            
            # Weighted average
            ensemble_class_logits = torch.sum(
                ensemble_classification * ensemble_weights_norm.view(-1, 1, 1), dim=0
            )
            ensemble_class_probs = F.softmax(ensemble_class_logits, dim=-1)
            
            ensemble_regression_pred = torch.sum(
                ensemble_regression * ensemble_weights_norm.view(-1, 1, 1), dim=0
            )
        else:
            ensemble_class_probs = classification_probs
            ensemble_regression_pred = regression_mean
        
        # Prepare output dictionary
        output = {
            'classification_logits': classification_logits,
            'classification_probs': classification_probs,
            'regression_mean': regression_mean,
            'regression_uncertainty': regression_uncertainty,
            'ensemble_classification': ensemble_class_probs if use_ensemble else classification_probs,
            'ensemble_regression': ensemble_regression_pred if use_ensemble else regression_mean,
            'ensemble_weights': F.softmax(self.ensemble_weights, dim=0) if use_ensemble else None
        }
        
        if return_features:
            output.update({
                'cnn_features': cnn_features,
                'lstm_features': lstm_features,
                'lstm_context': lstm_context,
                'fused_features': fused_features
            })
        
        return output
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        classification_targets: torch.Tensor,
        regression_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions dictionary
            classification_targets: Classification targets (batch_size,)
            regression_targets: Regression targets (batch_size, regression_targets)
            
        Returns:
            Dictionary of losses
        """
        # Classification loss
        classification_loss = F.cross_entropy(
            predictions['classification_logits'],
            classification_targets.long()
        )
        
        # Regression loss (MSE)
        regression_loss = F.mse_loss(
            predictions['regression_mean'],
            regression_targets
        )
        
        # Uncertainty loss (encourage calibrated uncertainty)
        uncertainty_loss = torch.mean(predictions['regression_uncertainty'])
        
        # Ensemble losses
        ensemble_class_loss = F.cross_entropy(
            predictions['ensemble_classification'],
            classification_targets.long()
        )
        
        ensemble_regression_loss = F.mse_loss(
            predictions['ensemble_regression'],
            regression_targets
        )
        
        # Total loss
        total_loss = (
            self.config.classification_weight * classification_loss +
            self.config.regression_weight * regression_loss +
            0.1 * uncertainty_loss +
            0.2 * ensemble_class_loss +
            0.2 * ensemble_regression_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'uncertainty_loss': uncertainty_loss,
            'ensemble_class_loss': ensemble_class_loss,
            'ensemble_regression_loss': ensemble_regression_loss
        }
    
    def fit(
        self,
        X_train: np.ndarray,
        y_class_train: np.ndarray,
        y_reg_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_class_val: Optional[np.ndarray] = None,
        y_reg_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train the hybrid model"""
        return self.train_model(
            X_train, y_class_train, y_reg_train,
            X_val, y_class_val, y_reg_val
        )
    
    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = True,
        use_ensemble: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with the hybrid model
        
        Args:
            X: Input data (batch_size, input_channels, sequence_length)
            return_uncertainty: Whether to return uncertainty estimates
            use_ensemble: Whether to use ensemble predictions
            
        Returns:
            Dictionary of predictions
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.forward(x_tensor, use_ensemble=use_ensemble)
            
            result = {
                'classification_probs': predictions['classification_probs'].cpu().numpy(),
                'classification_pred': torch.argmax(predictions['classification_probs'], dim=-1).cpu().numpy(),
                'regression_pred': predictions['regression_mean'].cpu().numpy(),
                'ensemble_classification': predictions['ensemble_classification'].cpu().numpy(),
                'ensemble_regression': predictions['ensemble_regression'].cpu().numpy()
            }
            
            if return_uncertainty:
                result['regression_uncertainty'] = predictions['regression_uncertainty'].cpu().numpy()
                result['ensemble_weights'] = predictions['ensemble_weights'].cpu().numpy() if predictions['ensemble_weights'] is not None else None
            
            return result
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_class_train: np.ndarray,
        y_reg_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_class_val: Optional[np.ndarray] = None,
        y_reg_val: Optional[np.ndarray] = None,
        num_epochs: Optional[int] = None
    ) -> TrainingResult:
        """Train the hybrid model with multi-task learning"""
        
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        # Create data loaders
        train_loader = self._create_data_loader(
            X_train, y_class_train, y_reg_train, shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_class_val is not None and y_reg_val is not None:
            val_loader = self._create_data_loader(
                X_val, y_class_val, y_reg_val, shuffle=False
            )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stopping_patience = 25
        
        for epoch in range(num_epochs):
            # Training phase
            self.train(True)
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y_class, batch_y_reg in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.forward(batch_x, use_ensemble=True)
                
                # Compute losses
                losses = self.compute_loss(predictions, batch_y_class, batch_y_reg)
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += losses['total_loss'].item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            avg_val_loss = 0.0
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y_class, batch_y_reg in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y_class = batch_y_class.to(self.device)
                        batch_y_reg = batch_y_reg.to(self.device)
                        
                        predictions = self.forward(batch_x, use_ensemble=True)
                        losses = self.compute_loss(predictions, batch_y_class, batch_y_reg)
                        
                        val_loss += losses['total_loss'].item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Early stopping and best model tracking
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        self.is_trained = True
        
        return TrainingResult(
            train_loss=train_losses[-1],
            val_loss=val_losses[-1] if val_losses else 0.0,
            epochs_trained=len(train_losses),
            best_epoch=best_epoch
        )
    
    def _create_data_loader(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader for multi-task learning"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_class_tensor = torch.LongTensor(y_class)
        y_reg_tensor = torch.FloatTensor(y_reg)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_class_tensor, y_reg_tensor)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader
    
    def save_model(self, filepath: str) -> None:
        """Save hybrid model with metadata"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and metadata
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'model_type': 'CNNLSTMHybridModel',
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
        
        # Also save config as JSON for easy inspection
        config_path = filepath.replace('.pth', '_config.json')
        config_dict = {
            'model_type': checkpoint['model_type'],
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'sequence_length': self.config.sequence_length,
            'prediction_horizon': self.config.prediction_horizon,
            'num_classes': self.config.num_classes,
            'regression_targets': self.config.regression_targets,
            'num_ensemble_models': self.config.num_ensemble_models,
            'use_monte_carlo_dropout': self.config.use_monte_carlo_dropout,
            'timestamp': checkpoint['timestamp']
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load hybrid model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore configuration
        loaded_config = checkpoint['config']
        
        # Update current config with loaded config attributes
        for attr_name in dir(loaded_config):
            if not attr_name.startswith('_'):
                setattr(self.config, attr_name, getattr(loaded_config, attr_name))
        
        # Rebuild model with loaded config
        self.build_model()
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        
        print(f"Loaded hybrid model from {filepath}")
        print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
        print(f"Timestamp: {checkpoint.get('timestamp', 'Unknown')}")


def create_hybrid_config(
    input_dim: int,
    sequence_length: int = 60,
    prediction_horizon: int = 10,
    num_classes: int = 3,
    regression_targets: int = 2,  # Price prediction + volatility estimation
    **kwargs
) -> HybridModelConfig:
    """Create configuration for CNN+LSTM hybrid model"""
    
    config = HybridModelConfig(
        model_type="CNNLSTMHybridModel",
        input_dim=input_dim,
        output_dim=num_classes + regression_targets,
        hidden_dims=[256],  # Feature fusion dimension
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        num_classes=num_classes,
        regression_targets=regression_targets,
        **kwargs
    )
    
    return config


def create_hybrid_data_loader(
    features: np.ndarray,
    class_targets: np.ndarray,
    reg_targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader for hybrid model training"""
    
    # Convert to tensors
    feature_tensor = torch.FloatTensor(features)
    class_tensor = torch.LongTensor(class_targets)
    reg_tensor = torch.FloatTensor(reg_targets)
    
    # Create dataset and dataloader
    dataset = TensorDataset(feature_tensor, class_tensor, reg_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader