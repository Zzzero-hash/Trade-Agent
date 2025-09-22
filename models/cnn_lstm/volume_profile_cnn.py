"""
Advanced Volume Profile CNN

This module implements a sophisticated 2D CNN architecture for volume-at-price
distribution analysis, order book imbalance detection, and market microstructure
feature extraction with attention pooling mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class VolumeProfileCNNConfig:
    """Configuration for Volume Profile CNN."""

    # Input dimensions
    price_levels: int = 100  # Number of price levels in volume profile
    time_steps: int = 50  # Number of time steps
    volume_features: int = 4  # [volume, buy_volume, sell_volume, imbalance]

    # Order book dimensions
    order_book_depth: int = 20  # Number of bid/ask levels
    order_book_features: int = (
        6  # [bid_price, bid_size, ask_price, ask_size, spread, imbalance]
    )

    # 2D CNN architecture
    conv2d_filters: List[int] = None  # [32, 64, 128, 256]
    conv2d_kernels: List[Tuple[int, int]] = None  # [(3,3), (3,3), (5,5), (5,5)]
    conv2d_strides: List[Tuple[int, int]] = None  # [(1,1), (2,2), (1,1), (2,2)]

    # 1D CNN for order book
    conv1d_filters: List[int] = None  # [64, 128, 256]
    conv1d_kernels: List[int] = None  # [3, 5, 7]

    # Attention pooling
    attention_dim: int = 256
    num_attention_heads: int = 8

    # Microstructure features
    microstructure_dim: int = 128

    # Regularization
    dropout_rate: float = 0.1
    batch_norm: bool = True

    # Output
    output_dim: int = 512

    def __post_init__(self):
        if self.conv2d_filters is None:
            self.conv2d_filters = [32, 64, 128, 256]
        if self.conv2d_kernels is None:
            self.conv2d_kernels = [(3, 3), (3, 3), (5, 5), (5, 5)]
        if self.conv2d_strides is None:
            self.conv2d_strides = [(1, 1), (2, 2), (1, 1), (2, 2)]
        if self.conv1d_filters is None:
            self.conv1d_filters = [64, 128, 256]
        if self.conv1d_kernels is None:
            self.conv1d_kernels = [3, 5, 7]


class Conv2DBlock(nn.Module):
    """2D Convolution block with batch normalization and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = None,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 2D convolution block."""
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class Conv1DBlock(nn.Module):
    """1D Convolution block for order book processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 1D convolution block."""
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class AttentionPooling(nn.Module):
    """Attention pooling mechanism for important level identification."""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()

        self.attention_dim = attention_dim

        # Attention mechanism
        self.attention_linear = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))

        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple of (pooled_output, attention_weights)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute attention scores
        attention_scores = torch.tanh(
            self.attention_linear(x)
        )  # (batch_size, seq_len, attention_dim)
        attention_scores = torch.matmul(
            attention_scores, self.context_vector
        )  # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Apply attention weights
        attended_output = torch.sum(
            x * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, input_dim)

        # Output projection
        output = self.output_proj(attended_output)

        return output, attention_weights


class MarketMicrostructureExtractor(nn.Module):
    """Extract market microstructure features from order book and trade data."""

    def __init__(self, config: VolumeProfileCNNConfig):
        super().__init__()

        self.config = config

        # Order book imbalance calculator
        self.imbalance_proj = nn.Linear(
            config.order_book_features, config.microstructure_dim // 4
        )

        # Spread analyzer
        self.spread_proj = nn.Linear(
            config.order_book_features, config.microstructure_dim // 4
        )

        # Volume flow analyzer
        self.volume_flow_proj = nn.Linear(
            config.volume_features, config.microstructure_dim // 4
        )

        # Price impact estimator
        self.price_impact_proj = nn.Linear(
            config.volume_features, config.microstructure_dim // 4
        )

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.microstructure_dim, config.microstructure_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.microstructure_dim, config.microstructure_dim),
        )

    def forward(
        self, order_book_data: torch.Tensor, volume_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract microstructure features.

        Args:
            order_book_data: Order book tensor (batch_size, time_steps, order_book_features)
            volume_data: Volume data tensor (batch_size, time_steps, volume_features)

        Returns:
            Microstructure features tensor (batch_size, microstructure_dim)
        """
        batch_size = order_book_data.shape[0]

        # Calculate order book imbalance features
        imbalance_features = self.imbalance_proj(order_book_data.mean(dim=1))

        # Calculate spread features
        spread_features = self.spread_proj(order_book_data.mean(dim=1))

        # Calculate volume flow features
        volume_flow_features = self.volume_flow_proj(volume_data.mean(dim=1))

        # Calculate price impact features
        price_impact_features = self.price_impact_proj(volume_data.mean(dim=1))

        # Concatenate all microstructure features
        microstructure_features = torch.cat(
            [
                imbalance_features,
                spread_features,
                volume_flow_features,
                price_impact_features,
            ],
            dim=1,
        )

        # Apply fusion layer
        output = self.fusion_layer(microstructure_features)

        return output


class VolumeProfile2DCNN(nn.Module):
    """2D CNN for volume-at-price distribution analysis."""

    def __init__(self, config: VolumeProfileCNNConfig):
        super().__init__()

        self.config = config

        # 2D CNN layers for volume profile
        conv_layers = []
        in_channels = config.volume_features

        for i, (filters, kernel, stride) in enumerate(
            zip(config.conv2d_filters, config.conv2d_kernels, config.conv2d_strides)
        ):
            conv_layers.append(
                Conv2DBlock(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=stride,
                    dropout_rate=config.dropout_rate,
                    batch_norm=config.batch_norm,
                )
            )
            in_channels = filters

        self.conv_layers = nn.ModuleList(conv_layers)

        # Calculate output dimensions after convolutions
        self._calculate_conv_output_size()

        # Attention pooling for important level identification
        # The input_dim will be the number of channels from the last conv layer
        self.attention_pooling = AttentionPooling(
            input_dim=config.conv2d_filters[-1], attention_dim=config.attention_dim
        )

    def _calculate_conv_output_size(self):
        """Calculate the output size after all convolution layers."""
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(
            1,
            self.config.volume_features,
            self.config.time_steps,
            self.config.price_levels,
        )

        with torch.no_grad():
            x = dummy_input
            for conv_layer in self.conv_layers:
                x = conv_layer(x)

            # Flatten spatial dimensions
            self.conv_output_size = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(
        self, volume_profile: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of volume profile 2D CNN.

        Args:
            volume_profile: Volume profile tensor
                           (batch_size, volume_features, time_steps, price_levels)

        Returns:
            Tuple of (features, attention_weights)
        """
        batch_size = volume_profile.shape[0]

        # Apply 2D convolution layers
        x = volume_profile
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten spatial dimensions and reshape for attention
        x = x.view(
            batch_size, x.shape[1], -1
        )  # (batch_size, channels, spatial_features)
        x = x.transpose(1, 2)  # (batch_size, spatial_features, channels)

        # Apply attention pooling
        features, attention_weights = self.attention_pooling(x)

        return features, attention_weights


class OrderBookCNN(nn.Module):
    """1D CNN for order book depth pattern recognition."""

    def __init__(self, config: VolumeProfileCNNConfig):
        super().__init__()

        self.config = config

        # 1D CNN layers for order book
        conv_layers = []
        in_channels = config.order_book_features

        for filters, kernel in zip(config.conv1d_filters, config.conv1d_kernels):
            conv_layers.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel,
                    dropout_rate=config.dropout_rate,
                    batch_norm=config.batch_norm,
                )
            )
            in_channels = filters

        self.conv_layers = nn.ModuleList(conv_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Feature projection
        self.feature_proj = nn.Linear(config.conv1d_filters[-1], config.output_dim // 4)

    def forward(self, order_book_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of order book CNN.

        Args:
            order_book_data: Order book tensor
                           (batch_size, order_book_features, order_book_depth)

        Returns:
            Order book features tensor (batch_size, output_dim // 4)
        """
        # Apply 1D convolution layers
        x = order_book_data
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch_size, filters, 1)
        x = x.squeeze(-1)  # (batch_size, filters)

        # Feature projection
        features = self.feature_proj(x)

        return features


class VolumeProfileCNN(nn.Module):
    """
    Advanced Volume Profile CNN for market microstructure analysis.

    This model combines 2D CNN for volume-at-price analysis, 1D CNN for order book
    pattern recognition, and microstructure feature extraction with attention pooling.
    """

    def __init__(self, config: VolumeProfileCNNConfig):
        super().__init__()

        self.config = config

        # Volume profile 2D CNN
        self.volume_profile_cnn = VolumeProfile2DCNN(config)

        # Order book 1D CNN
        self.order_book_cnn = OrderBookCNN(config)

        # Market microstructure extractor
        self.microstructure_extractor = MarketMicrostructureExtractor(config)

        # Feature fusion
        total_features = (
            config.conv2d_filters[-1]  # volume profile features
            + config.output_dim // 4  # order book features
            + config.microstructure_dim  # microstructure features
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.output_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(self, market_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of volume profile CNN.

        Args:
            market_data: Dictionary containing:
                - 'volume_profile': (batch_size, volume_features, time_steps, price_levels)
                - 'order_book': (batch_size, order_book_features, order_book_depth)
                - 'order_book_time_series': (batch_size, time_steps, order_book_features)
                - 'volume_time_series': (batch_size, time_steps, volume_features)

        Returns:
            Dictionary containing:
                - 'features': Final feature tensor (batch_size, output_dim)
                - 'volume_attention': Attention weights from volume profile
                - 'microstructure_features': Raw microstructure features
        """
        # Extract volume profile features
        volume_features, volume_attention = self.volume_profile_cnn(
            market_data["volume_profile"]
        )

        # Extract order book features
        order_book_features = self.order_book_cnn(market_data["order_book"])

        # Extract microstructure features
        microstructure_features = self.microstructure_extractor(
            market_data["order_book_time_series"], market_data["volume_time_series"]
        )

        # Concatenate all features
        combined_features = torch.cat(
            [volume_features, order_book_features, microstructure_features], dim=1
        )

        # Apply feature fusion
        fused_features = self.feature_fusion(combined_features)

        # Layer normalization
        output_features = self.layer_norm(fused_features)

        return {
            "features": output_features,
            "volume_attention": volume_attention,
            "microstructure_features": microstructure_features,
        }

    def get_feature_importance(
        self, market_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores for interpretability.

        Args:
            market_data: Market data dictionary

        Returns:
            Dictionary of feature importance scores
        """
        with torch.no_grad():
            # Get attention weights
            _, volume_attention = self.volume_profile_cnn(market_data["volume_profile"])

            # Calculate gradient-based importance (simplified)
            output = self.forward(market_data)

            return {
                "volume_attention": volume_attention,
                "feature_magnitude": torch.abs(output["features"]).mean(dim=0),
            }


def create_volume_profile_cnn(
    config: Optional[VolumeProfileCNNConfig] = None,
) -> VolumeProfileCNN:
    """
    Factory function to create a Volume Profile CNN model.

    Args:
        config: Optional configuration. If None, uses default configuration.

    Returns:
        Initialized VolumeProfileCNN model
    """
    if config is None:
        config = VolumeProfileCNNConfig()

    return VolumeProfileCNN(config)


if __name__ == "__main__":
    # Example usage and testing
    config = VolumeProfileCNNConfig(
        price_levels=100,
        time_steps=50,
        volume_features=4,
        order_book_depth=20,
        order_book_features=6,
        conv2d_filters=[32, 64, 128, 256],
        conv1d_filters=[64, 128, 256],
        attention_dim=256,
        output_dim=512,
    )

    model = create_volume_profile_cnn(config)

    # Create sample data
    batch_size = 16
    sample_data = {
        "volume_profile": torch.randn(batch_size, 4, 50, 100),
        "order_book": torch.randn(batch_size, 6, 20),
        "order_book_time_series": torch.randn(batch_size, 50, 6),
        "volume_time_series": torch.randn(batch_size, 50, 4),
    }

    # Forward pass
    output = model(sample_data)
    print(f"Output features shape: {output['features'].shape}")
    print(f"Volume attention shape: {output['volume_attention'].shape}")
    print(f"Microstructure features shape: {output['microstructure_features'].shape}")

    # Get feature importance
    importance = model.get_feature_importance(sample_data)
    print(f"Feature importance keys: {importance.keys()}")
