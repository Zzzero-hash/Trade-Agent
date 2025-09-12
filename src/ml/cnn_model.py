"""CNN Feature Extraction Model for Trading Platform

This module implements a Convolutional Neural Network for extracting spatial
patterns from multi-dimensional market data. The CNN uses multiple filter sizes
and attention mechanisms as specified in the design document.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime

from .base_models import BasePyTorchModel, ModelConfig, TrainingResult


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for CNN features"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        
        return output, attention_weights


class CNNFeatureExtractor(BasePyTorchModel):
    """CNN Feature Extraction Model with Multiple Filter Sizes and Attention
    
    This model implements the CNN component of the CNN+LSTM hybrid architecture
    as specified in requirements 1.1, 1.2, and 5.2.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # CNN configuration
        self.input_channels = config.input_dim
        self.filter_sizes = getattr(config, 'filter_sizes', [3, 5, 7, 11])
        self.num_filters = getattr(config, 'num_filters', 64)
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.use_attention = getattr(config, 'use_attention', True)
        self.num_attention_heads = getattr(config, 'num_attention_heads', 8)
        
        # Build model architecture
        self.build_model()
        
    def build_model(self) -> None:
        """Build CNN architecture with multiple filter sizes and attention"""
        
        # Multiple 1D Convolutional layers with different filter sizes
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for filter_size in self.filter_sizes:
            conv = nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=filter_size,
                padding=filter_size // 2,  # Same padding
                bias=False
            )
            bn = nn.BatchNorm1d(self.num_filters)
            
            self.conv_layers.append(conv)
            self.batch_norms.append(bn)
        
        # Total output channels from all conv layers
        total_filters = self.num_filters * len(self.filter_sizes)
        
        # Attention mechanism
        if self.use_attention:
            self.attention = MultiHeadAttention(
                d_model=total_filters,
                num_heads=self.num_attention_heads,
                dropout=self.dropout_rate
            )
        
        # Residual connections
        self.residual_projection = nn.Linear(self.input_channels, total_filters)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Output projection layers
        self.feature_projection = nn.Linear(total_filters, self.config.output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN feature extractor
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Extracted features of shape (batch_size, sequence_length, output_dim)
        """
        batch_size, channels, seq_len = x.shape
        
        # Apply multiple convolutional layers with different filter sizes
        conv_outputs = []
        
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            # Convolution -> Batch Norm -> ReLU
            conv_out = F.relu(bn(conv(x)))
            conv_outputs.append(conv_out)
        
        # Concatenate outputs from all filter sizes
        # Shape: (batch_size, total_filters, sequence_length)
        combined_features = torch.cat(conv_outputs, dim=1)
        
        # Transpose for attention: (batch_size, sequence_length, total_filters)
        combined_features = combined_features.transpose(1, 2)
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            attended_features, attention_weights = self.attention(
                combined_features, combined_features, combined_features
            )
            
            # Residual connection
            # Project input to match feature dimensions
            residual = self.residual_projection(x.transpose(1, 2))
            attended_features = attended_features + residual
            
            features = attended_features
        else:
            features = combined_features
        
        # Apply dropout
        features = self.dropout(features)
        
        # Final feature projection
        output_features = self.feature_projection(features)
        
        return output_features
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train the CNN model using the base class method"""
        return self.train_model(X_train, y_train, X_val, y_val)
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract features from input data
        
        Args:
            x: Input data of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Extracted features as numpy array
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            features = self.forward(x_tensor)
            return features.cpu().numpy()
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> TrainingResult:
        """Train the CNN feature extractor
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs (uses config if None)
            
        Returns:
            Training results with loss and metrics
        """
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Use MSE loss for feature extraction (reconstruction-like objective)
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Training phase
            super().train(True)  # Use PyTorch's train method
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.forward(batch_x)
                
                # For unsupervised feature learning, we can use reconstruction loss
                # or in supervised case, predict the target
                if features.shape[-1] == batch_y.shape[-1]:
                    loss = criterion(features, batch_y)
                else:
                    # Reconstruction loss: try to reconstruct input from features
                    reconstructed = torch.mean(features, dim=-1, keepdim=True)
                    target = torch.mean(batch_y, dim=-1, keepdim=True)
                    loss = criterion(reconstructed, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
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
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        features = self.forward(batch_x)
                        
                        if features.shape[-1] == batch_y.shape[-1]:
                            loss = criterion(features, batch_y)
                        else:
                            reconstructed = torch.mean(features, dim=-1, keepdim=True)
                            target = torch.mean(batch_y, dim=-1, keepdim=True)
                            loss = criterion(reconstructed, target)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Track best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
            
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
            epochs_trained=num_epochs,
            best_epoch=best_epoch
        )
    
    def save_model(self, filepath: str) -> None:
        """Save CNN model with metadata"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and metadata
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'model_type': 'CNNFeatureExtractor',
            'filter_sizes': self.filter_sizes,
            'num_filters': self.num_filters,
            'use_attention': self.use_attention,
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
            'filter_sizes': self.filter_sizes,
            'num_filters': self.num_filters,
            'use_attention': self.use_attention,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.config.learning_rate,
            'timestamp': checkpoint['timestamp']
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load CNN model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.config = checkpoint['config']
        self.filter_sizes = checkpoint.get('filter_sizes', [3, 5, 7, 11])
        self.num_filters = checkpoint.get('num_filters', 64)
        self.use_attention = checkpoint.get('use_attention', True)
        
        # Rebuild model with loaded config
        self.build_model()
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        
        print(f"Loaded CNN model from {filepath}")
        print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
        print(f"Timestamp: {checkpoint.get('timestamp', 'Unknown')}")


def create_cnn_data_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader for CNN training
    
    Args:
        features: Input features of shape (samples, channels, sequence_length)
        targets: Target values of shape (samples, sequence_length, target_dim)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch DataLoader
    """
    # Convert to tensors
    feature_tensor = torch.FloatTensor(features)
    target_tensor = torch.FloatTensor(targets)
    
    # Create dataset and dataloader
    dataset = TensorDataset(feature_tensor, target_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def create_cnn_config(
    input_dim: int,
    output_dim: int,
    filter_sizes: Optional[List[int]] = None,
    num_filters: int = 64,
    use_attention: bool = True,
    num_attention_heads: int = 8,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cpu"
) -> ModelConfig:
    """Create configuration for CNN feature extractor
    
    Args:
        input_dim: Number of input channels (features)
        output_dim: Number of output features
        filter_sizes: List of convolutional filter sizes
        num_filters: Number of filters per size
        use_attention: Whether to use attention mechanism
        num_attention_heads: Number of attention heads
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        ModelConfig object with CNN-specific parameters
    """
    if filter_sizes is None:
        filter_sizes = [3, 5, 7, 11]
    
    config = ModelConfig(
        model_type="CNNFeatureExtractor",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[num_filters * len(filter_sizes)],
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    
    # Add CNN-specific attributes
    config.filter_sizes = filter_sizes
    config.num_filters = num_filters
    config.use_attention = use_attention
    config.num_attention_heads = num_attention_heads
    config.dropout_rate = dropout_rate
    
    return config