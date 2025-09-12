"""LSTM Temporal Processing Model for Trading Platform

This module implements a bidirectional LSTM with attention mechanisms and skip
connections for temporal sequence processing in time series prediction.
The model is designed for sequence-to-sequence architecture as specified
in requirements 1.1, 1.3, and 5.3.
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


class LSTMAttention(nn.Module):
    """Attention mechanism for LSTM hidden states"""
    
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, lstm_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - optional padding mask
            
        Returns:
            context: (batch_size, hidden_dim) - attended context vector
            attention_weights: (batch_size, seq_len) - attention weights
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.shape
        
        # Apply linear transformation
        # (batch_size, seq_len, attention_dim)
        attention_hidden = torch.tanh(self.attention_linear(lstm_outputs))
        attention_hidden = self.dropout(attention_hidden)
        
        # Calculate attention scores
        # (batch_size, seq_len, 1) -> (batch_size, seq_len)
        attention_scores = self.context_vector(attention_hidden).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to LSTM outputs
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim)
        # -> (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return context, attention_weights


class LSTMTemporalProcessor(BasePyTorchModel):
    """Bidirectional LSTM with Attention and Skip Connections
    
    This model implements the LSTM component for temporal sequence processing
    with sequence-to-sequence architecture for time series prediction.
    Supports bidirectional processing, attention mechanisms, and skip connections
    for improved gradient flow and temporal modeling.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # LSTM configuration
        self.input_dim = config.input_dim
        self.hidden_dim = getattr(config, 'hidden_dim', 128)
        self.num_layers = getattr(config, 'num_layers', 3)
        self.output_dim = config.output_dim
        self.dropout_rate = getattr(config, 'dropout_rate', 0.3)
        self.bidirectional = getattr(config, 'bidirectional', True)
        self.use_attention = getattr(config, 'use_attention', True)
        self.use_skip_connections = getattr(config, 'use_skip_connections', True)
        self.sequence_length = getattr(config, 'sequence_length', 60)
        
        # Build model architecture
        self.build_model()
        
    def build_model(self) -> None:
        """Build LSTM architecture with attention and skip connections"""
        
        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Multi-layer bidirectional LSTM
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        for i in range(self.num_layers):
            # LSTM layer
            lstm_input_dim = self.hidden_dim if i == 0 else (
                self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
            )
            
            lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,  # We'll apply dropout manually
                bidirectional=self.bidirectional
            )
            self.lstm_layers.append(lstm)
            
            # Layer normalization
            lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
            self.layer_norms.append(nn.LayerNorm(lstm_output_dim))
            
            # Skip connection projection (if dimensions don't match)
            if self.use_skip_connections and i > 0:
                prev_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
                if prev_dim != lstm_output_dim:
                    self.skip_projections.append(nn.Linear(prev_dim, lstm_output_dim))
                else:
                    self.skip_projections.append(nn.Identity())
            else:
                self.skip_projections.append(nn.Identity())
        
        # Attention mechanism
        final_hidden_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        if self.use_attention:
            self.attention = LSTMAttention(final_hidden_dim)
        
        # Dropout layers
        self.input_dropout = nn.Dropout(self.dropout_rate * 0.5)  # Lower dropout for input
        self.lstm_dropout = nn.Dropout(self.dropout_rate)
        self.output_dropout = nn.Dropout(self.dropout_rate)
        
        # Output layers for sequence-to-sequence prediction
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)  # Decoder LSTM output dim
        self.sequence_decoder = nn.Linear(self.hidden_dim, self.output_dim)
        
        # For sequence-to-sequence, we need to generate outputs for each timestep
        self.decoder_lstm = nn.LSTM(
            input_size=self.output_dim + final_hidden_dim,  # Previous output + context
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # Decoder is unidirectional
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif param.dim() > 1:  # Other weight matrices
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor, 
                target_length: Optional[int] = None) -> torch.Tensor:
        """Forward pass through LSTM temporal processor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target_length: Length of output sequence (defaults to input length)
            
        Returns:
            Output sequences of shape (batch_size, target_length, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        if target_length is None:
            target_length = seq_len
        
        # Input projection and dropout
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # Multi-layer LSTM with skip connections
        lstm_outputs = []
        current_input = x
        
        for i, (lstm, layer_norm, skip_proj) in enumerate(
            zip(self.lstm_layers, self.layer_norms, self.skip_projections)
        ):
            # LSTM forward pass
            lstm_out, _ = lstm(current_input)
            
            # Layer normalization
            lstm_out = layer_norm(lstm_out)
            
            # Skip connection (residual connection)
            if self.use_skip_connections and i > 0:
                # Project previous layer output to match current dimensions
                skip_input = skip_proj(lstm_outputs[-1])
                lstm_out = lstm_out + skip_input
            
            # Apply dropout
            lstm_out = self.lstm_dropout(lstm_out)
            
            lstm_outputs.append(lstm_out)
            current_input = lstm_out
        
        # Final LSTM output
        final_lstm_output = lstm_outputs[-1]  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention mechanism
        if self.use_attention:
            # Get attended context vector
            context, attention_weights = self.attention(final_lstm_output)
            # Expand context to match sequence length for decoder
            context_expanded = context.unsqueeze(1).expand(-1, target_length, -1)
        else:
            # Use mean pooling as fallback
            context = torch.mean(final_lstm_output, dim=1)
            context_expanded = context.unsqueeze(1).expand(-1, target_length, -1)
        
        # Sequence-to-sequence decoder
        decoder_outputs = []
        decoder_hidden = None
        
        # Initialize first decoder input (can be learned parameter or zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=x.device)
        
        for t in range(target_length):
            # Concatenate previous output with context
            decoder_lstm_input = torch.cat([decoder_input, context_expanded[:, t:t+1, :]], dim=-1)
            
            # LSTM decoder step
            decoder_out, decoder_hidden = self.decoder_lstm(decoder_lstm_input, decoder_hidden)
            
            # Project to output dimension
            output_proj = self.output_projection(decoder_out)
            output_proj = self.output_dropout(output_proj)
            
            # Generate output for this timestep
            timestep_output = self.sequence_decoder(output_proj)
            decoder_outputs.append(timestep_output)
            
            # Use current output as next input (teacher forcing during training)
            decoder_input = timestep_output
        
        # Concatenate all decoder outputs
        output_sequence = torch.cat(decoder_outputs, dim=1)  # (batch_size, target_length, output_dim)
        
        return output_sequence
    
    def forward_encoder_only(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder only (for feature extraction)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            encoded_sequence: (batch_size, seq_len, hidden_dim)
            context_vector: (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and dropout
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # Multi-layer LSTM with skip connections
        lstm_outputs = []
        current_input = x
        
        for i, (lstm, layer_norm, skip_proj) in enumerate(
            zip(self.lstm_layers, self.layer_norms, self.skip_projections)
        ):
            # LSTM forward pass
            lstm_out, _ = lstm(current_input)
            
            # Layer normalization
            lstm_out = layer_norm(lstm_out)
            
            # Skip connection
            if self.use_skip_connections and i > 0:
                skip_input = skip_proj(lstm_outputs[-1])
                lstm_out = lstm_out + skip_input
            
            # Apply dropout
            lstm_out = self.lstm_dropout(lstm_out)
            
            lstm_outputs.append(lstm_out)
            current_input = lstm_out
        
        # Final LSTM output
        encoded_sequence = lstm_outputs[-1]
        
        # Apply attention to get context vector
        if self.use_attention:
            context_vector, _ = self.attention(encoded_sequence)
        else:
            context_vector = torch.mean(encoded_sequence, dim=1)
        
        return encoded_sequence, context_vector
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """Train the LSTM model using the base class method"""
        return self.train_model(X_train, y_train, X_val, y_val)
    
    def predict_sequence(self, x: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """Predict sequences from input data
        
        Args:
            x: Input data of shape (batch_size, seq_len, input_dim)
            target_length: Length of output sequence
            
        Returns:
            Predicted sequences as numpy array
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            predictions = self.forward(x_tensor, target_length)
            return predictions.cpu().numpy()
    
    def extract_features(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract temporal features from input data
        
        Args:
            x: Input data of shape (batch_size, seq_len, input_dim)
            
        Returns:
            encoded_sequences: Sequence features (batch_size, seq_len, hidden_dim)
            context_vectors: Context vectors (batch_size, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            encoded_seq, context_vec = self.forward_encoder_only(x_tensor)
            return encoded_seq.cpu().numpy(), context_vec.cpu().numpy()
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> TrainingResult:
        """Train the LSTM temporal processor
        
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
        
        # Use MSE loss for sequence prediction
        criterion = nn.MSELoss()
        
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
            super().train(True)  # Use PyTorch's train method
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.forward(batch_x, target_length=batch_y.shape[1])
                
                # Calculate loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for LSTM stability
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
                        
                        predictions = self.forward(batch_x, target_length=batch_y.shape[1])
                        loss = criterion(predictions, batch_y)
                        
                        val_loss += loss.item()
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
    
    def save_model(self, filepath: str) -> None:
        """Save LSTM model with metadata"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and metadata
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'model_type': 'LSTMTemporalProcessor',
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'use_skip_connections': self.use_skip_connections,
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
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'use_skip_connections': self.use_skip_connections,
            'dropout_rate': self.dropout_rate,
            'sequence_length': self.sequence_length,
            'learning_rate': self.config.learning_rate,
            'timestamp': checkpoint['timestamp']
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load LSTM model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.config = checkpoint['config']
        self.hidden_dim = checkpoint.get('hidden_dim', 128)
        self.num_layers = checkpoint.get('num_layers', 3)
        self.bidirectional = checkpoint.get('bidirectional', True)
        self.use_attention = checkpoint.get('use_attention', True)
        self.use_skip_connections = checkpoint.get('use_skip_connections', True)
        
        # Rebuild model with loaded config
        self.build_model()
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)
        
        print(f"Loaded LSTM model from {filepath}")
        print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
        print(f"Timestamp: {checkpoint.get('timestamp', 'Unknown')}")


def create_lstm_data_loader(
    sequences: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader for LSTM training
    
    Args:
        sequences: Input sequences of shape (samples, seq_len, input_dim)
        targets: Target sequences of shape (samples, target_len, output_dim)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch DataLoader
    """
    # Convert to tensors
    sequence_tensor = torch.FloatTensor(sequences)
    target_tensor = torch.FloatTensor(targets)
    
    # Create dataset and dataloader
    dataset = TensorDataset(sequence_tensor, target_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def create_lstm_config(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    sequence_length: int = 60,
    bidirectional: bool = True,
    use_attention: bool = True,
    use_skip_connections: bool = True,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cpu"
) -> ModelConfig:
    """Create configuration for LSTM temporal processor
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        sequence_length: Length of input sequences
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use attention mechanism
        use_skip_connections: Whether to use skip connections
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        ModelConfig object with LSTM-specific parameters
    """
    config = ModelConfig(
        model_type="LSTMTemporalProcessor",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[hidden_dim],
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    
    # Add LSTM-specific attributes
    config.hidden_dim = hidden_dim
    config.num_layers = num_layers
    config.sequence_length = sequence_length
    config.bidirectional = bidirectional
    config.use_attention = use_attention
    config.use_skip_connections = use_skip_connections
    config.dropout_rate = dropout_rate
    
    return config


def create_sequence_data(
    data: np.ndarray,
    sequence_length: int = 60,
    prediction_length: int = 10,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence data for LSTM training
    
    Args:
        data: Time series data of shape (timesteps, features)
        sequence_length: Length of input sequences
        prediction_length: Length of prediction sequences
        stride: Stride between sequences
        
    Returns:
        sequences: Input sequences (num_sequences, sequence_length, features)
        targets: Target sequences (num_sequences, prediction_length, features)
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    timesteps, features = data.shape
    
    # Calculate number of sequences
    num_sequences = (timesteps - sequence_length - prediction_length + 1) // stride
    
    sequences = []
    targets = []
    
    for i in range(0, num_sequences * stride, stride):
        # Input sequence
        seq = data[i:i + sequence_length]
        sequences.append(seq)
        
        # Target sequence (next prediction_length timesteps)
        target = data[i + sequence_length:i + sequence_length + prediction_length]
        targets.append(target)
    
    return np.array(sequences), np.array(targets)