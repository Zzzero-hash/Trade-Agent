"""
Multi-scale Price CNN with Attention Mechanisms

This module implements a sophisticated CNN architecture for multi-timeframe
price pattern recognition with self-attention mechanisms and residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class MultiScalePriceCNNConfig:
    """Configuration for Multi-scale Price CNN."""
    
    # Input dimensions
    sequence_length: int = 100
    num_features: int = 5  # OHLCV
    
    # Multi-scale timeframes
    timeframes: List[str] = None  # ['1min', '5min', '15min']
    
    # CNN architecture
    cnn_filters: List[int] = None  # [64, 128, 256]
    kernel_sizes: List[int] = None  # [3, 5, 7]
    dilation_rates: List[int] = None  # [1, 2, 4]
    
    # Attention mechanism
    attention_dim: int = 256
    num_attention_heads: int = 8
    
    # Residual connections
    use_residual: bool = True
    
    # Regularization
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # Output
    output_dim: int = 512
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1min', '5min', '15min']
        if self.cnn_filters is None:
            self.cnn_filters = [64, 128, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4]


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with residual connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Dilated convolution
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual connection projection if needed
        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        # Main path
        out = self.conv(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for CNN features."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.query(x)  # (batch_size, seq_len, embed_dim)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output


class TimeframeCNNBranch(nn.Module):
    """CNN branch for a specific timeframe with dilated convolutions."""
    
    def __init__(
        self,
        config: MultiScalePriceCNNConfig,
        timeframe: str
    ):
        super().__init__()
        
        self.timeframe = timeframe
        self.config = config
        
        # Build dilated convolution layers
        layers = []
        in_channels = config.num_features
        
        for i, (filters, kernel_size, dilation) in enumerate(
            zip(config.cnn_filters, config.kernel_sizes, config.dilation_rates)
        ):
            layers.append(
                DilatedConvBlock(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    use_residual=config.use_residual,
                    dropout_rate=config.dropout_rate
                )
            )
            in_channels = filters
        
        self.conv_layers = nn.ModuleList(layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection
        feature_dim = config.output_dim // len(config.timeframes)
        self.feature_proj = nn.Linear(config.cnn_filters[-1], feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for timeframe-specific CNN branch.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, seq_len)
            
        Returns:
            Feature tensor of shape (batch_size, output_dim // num_timeframes)
        """
        # Apply dilated convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, filters, 1)
        x = x.squeeze(-1)  # (batch_size, filters)
        
        # Feature projection
        x = self.feature_proj(x)
        
        return x


class MultiScalePriceCNN(nn.Module):
    """
    Multi-scale Price CNN with Attention Mechanisms.
    
    This model processes price data at multiple timeframes using parallel CNN branches
    with dilated convolutions, then fuses the features using self-attention mechanisms.
    """
    
    def __init__(self, config: MultiScalePriceCNNConfig):
        super().__init__()
        
        self.config = config
        
        # Create CNN branches for each timeframe
        self.timeframe_branches = nn.ModuleDict({
            timeframe: TimeframeCNNBranch(config, timeframe)
            for timeframe in config.timeframes
        })
        
        # Calculate actual feature dimension after concatenation
        feature_per_timeframe = config.output_dim // len(config.timeframes)
        actual_feature_dim = feature_per_timeframe * len(config.timeframes)
        
        # Ensure attention dimension is divisible by number of heads
        attention_dim = (actual_feature_dim // config.num_attention_heads) * config.num_attention_heads
        if attention_dim != actual_feature_dim:
            # Add a projection layer to make dimensions compatible
            self.feature_adapter = nn.Linear(actual_feature_dim, attention_dim)
        else:
            self.feature_adapter = nn.Identity()
        
        # Self-attention for feature fusion
        self.attention = MultiHeadSelfAttention(
            embed_dim=attention_dim,
            num_heads=config.num_attention_heads,
            dropout_rate=config.dropout_rate
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(attention_dim)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(attention_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.output_dim, config.output_dim)
        )
        
        # Task-specific prediction heads
        self.prediction_heads = nn.ModuleDict({
            'price_prediction': nn.Linear(config.output_dim, 1),
            'volatility_estimation': nn.Linear(config.output_dim, 1),
            'regime_detection': nn.Linear(config.output_dim, 4)  # 4 market regimes
        })
        
    def forward(self, multi_timeframe_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of multi-scale price CNN.
        
        Args:
            multi_timeframe_data: Dictionary mapping timeframe names to tensors
                                 of shape (batch_size, num_features, seq_len)
                                 
        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        batch_size = next(iter(multi_timeframe_data.values())).shape[0]
        
        # Process each timeframe through its CNN branch
        timeframe_features = []
        for timeframe in self.config.timeframes:
            if timeframe in multi_timeframe_data:
                features = self.timeframe_branches[timeframe](multi_timeframe_data[timeframe])
                timeframe_features.append(features)
            else:
                # Handle missing timeframe data with zeros
                device = next(iter(multi_timeframe_data.values())).device
                zero_features = torch.zeros(
                    batch_size, 
                    self.config.output_dim // len(self.config.timeframes),
                    device=device
                )
                timeframe_features.append(zero_features)
        
        # Concatenate timeframe features
        combined_features = torch.cat(timeframe_features, dim=1)  # (batch_size, actual_feature_dim)
        
        # Adapt features for attention if needed
        adapted_features = self.feature_adapter(combined_features)  # (batch_size, attention_dim)
        
        # Reshape for attention (add sequence dimension)
        adapted_features = adapted_features.unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Apply self-attention
        attended_features = self.attention(adapted_features)  # (batch_size, 1, attention_dim)
        attended_features = attended_features.squeeze(1)  # (batch_size, attention_dim)
        
        # Layer normalization and residual connection
        attended_features = self.layer_norm(attended_features + adapted_features.squeeze(1))
        
        # Final projection
        output = self.final_proj(attended_features)
        
        # Generate task-specific predictions
        predictions = {}
        for task_name, head in self.prediction_heads.items():
            predictions[task_name] = head(output)
        
        return {
            'output': output,
            'price_prediction': predictions['price_prediction'],
            'volatility_estimation': predictions['volatility_estimation'],
            'regime_detection': predictions['regime_detection']
        }
    
    def get_attention_weights(self, multi_timeframe_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            multi_timeframe_data: Dictionary mapping timeframe names to tensors
                                 
        Returns:
            Attention weights tensor
        """
        with torch.no_grad():
            # Process timeframe features
            timeframe_features = []
            for timeframe in self.config.timeframes:
                if timeframe in multi_timeframe_data:
                    features = self.timeframe_branches[timeframe](multi_timeframe_data[timeframe])
                    timeframe_features.append(features)
                else:
                    device = next(iter(multi_timeframe_data.values())).device
                    zero_features = torch.zeros(
                        multi_timeframe_data[next(iter(multi_timeframe_data.keys()))].shape[0],
                        self.config.output_dim // len(self.config.timeframes),
                        device=device
                    )
                    timeframe_features.append(zero_features)
            
            combined_features = torch.cat(timeframe_features, dim=1).unsqueeze(1)
            
            # Get attention weights from the attention module
            q = self.attention.query(combined_features)
            k = self.attention.key(combined_features)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.attention.scale
            attention_weights = F.softmax(scores, dim=-1)
            
            return attention_weights


def create_multi_scale_price_cnn(config: Optional[MultiScalePriceCNNConfig] = None) -> MultiScalePriceCNN:
    """
    Factory function to create a Multi-scale Price CNN model.
    
    Args:
        config: Optional configuration. If None, uses default configuration.
        
    Returns:
        Initialized MultiScalePriceCNN model
    """
    if config is None:
        config = MultiScalePriceCNNConfig()
    
    return MultiScalePriceCNN(config)


if __name__ == "__main__":
    # Example usage and testing
    config = MultiScalePriceCNNConfig(
        sequence_length=100,
        num_features=5,
        timeframes=['1min', '5min', '15min'],
        cnn_filters=[64, 128, 256],
        kernel_sizes=[3, 5, 7],
        dilation_rates=[1, 2, 4],
        attention_dim=256,
        num_attention_heads=8,
        output_dim=512
    )
    
    model = create_multi_scale_price_cnn(config)
    
    # Create sample data
    batch_size = 32
    sample_data = {
        '1min': torch.randn(batch_size, 5, 100),
        '5min': torch.randn(batch_size, 5, 100),
        '15min': torch.randn(batch_size, 5, 100)
    }
    
    # Forward pass
    output = model(sample_data)
    print(f"Output shape: {output.shape}")
    
    # Get attention weights
    attention_weights = model.get_attention_weights(sample_data)
    print(f"Attention weights shape: {attention_weights.shape}")