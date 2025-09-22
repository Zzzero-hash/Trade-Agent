"""
Hybrid CNN-LSTM-Transformer Architecture

This module implements a revolutionary hybrid architecture that combines CNN spatial
features with LSTM/Transformer temporal modeling, featuring learned feature fusion
with attention weights, cross-attention between spatial and temporal representations,
and adaptive architecture selection based on market conditions.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_scale_price_cnn import MultiScalePriceCNN, MultiScalePriceCNNConfig
from .bidirectional_lstm_attention import (
    BidirectionalLSTMWithAttention,
    BidirectionalLSTMConfig
)
from .transformer_temporal_encoder import (
    TransformerTemporalEncoder,
    TransformerTemporalConfig
)


@dataclass
class HybridArchitectureConfig:
    """Configuration for Hybrid CNN-LSTM-Transformer Architecture."""

    # Input dimensions
    input_dim: int = 5  # OHLCV
    sequence_length: int = 100

    # CNN configuration
    cnn_config: Optional[MultiScalePriceCNNConfig] = None

    # LSTM configuration
    lstm_config: Optional[BidirectionalLSTMConfig] = None

    # Transformer configuration
    transformer_config: Optional[TransformerTemporalConfig] = None

    # Feature fusion
    fusion_dim: int = 1024
    fusion_heads: int = 8
    fusion_dropout: float = 0.1

    # Cross-attention
    cross_attention_dim: int = 512
    cross_attention_heads: int = 8

    # Adaptive architecture selection
    use_adaptive_selection: bool = True
    market_condition_dim: int = 64
    num_market_conditions: int = 4  # bull, bear, sideways, volatile

    # Output
    output_dim: int = 512

    def __post_init__(self):
        if self.cnn_config is None:
            self.cnn_config = MultiScalePriceCNNConfig(
                sequence_length=self.sequence_length,
                num_features=self.input_dim,
                output_dim=self.fusion_dim // 3
            )

        if self.lstm_config is None:
            self.lstm_config = BidirectionalLSTMConfig(
                input_dim=self.input_dim,
                sequence_length=self.sequence_length,
                output_dim=self.fusion_dim // 3
            )

        if self.transformer_config is None:
            self.transformer_config = TransformerTemporalConfig(
                input_dim=self.input_dim,
                sequence_length=self.sequence_length,
                output_dim=self.fusion_dim // 3
            )


class CrossModalAttention(nn.Module):
    """Cross-attention mechanism between spatial and temporal representations."""

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Projections for spatial features (queries)
        self.spatial_q_proj = nn.Linear(spatial_dim, attention_dim)

        # Projections for temporal features (keys and values)
        self.temporal_k_proj = nn.Linear(temporal_dim, attention_dim)
        self.temporal_v_proj = nn.Linear(temporal_dim, attention_dim)

        # Output projection
        self.out_proj = nn.Linear(attention_dim, spatial_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention between spatial and temporal features.

        Args:
            spatial_features: Spatial features (batch_size, spatial_dim)
            temporal_features: Temporal features (batch_size, seq_len, temporal_dim)
            temporal_mask: Mask for temporal features (batch_size, seq_len)

        Returns:
            Enhanced spatial features (batch_size, spatial_dim)
        """
        batch_size = spatial_features.shape[0]
        seq_len = temporal_features.shape[1]

        # Project features
        q = self.spatial_q_proj(spatial_features)  # (batch_size, attention_dim)
        k = self.temporal_k_proj(temporal_features)  # (batch_size, seq_len, attention_dim)
        v = self.temporal_v_proj(temporal_features)  # (batch_size, seq_len, attention_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply temporal mask if provided
        if temporal_mask is not None:
            scores = scores.masked_fill(
                temporal_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.attention_dim
        ).squeeze(1)

        output = self.out_proj(attn_output)

        return output


class MarketConditionClassifier(nn.Module):
    """Classifier for market condition detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conditions: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_conditions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify market conditions."""
        return self.classifier(x)


class AdaptiveArchitectureSelector(nn.Module):
    """Adaptive architecture selection based on market conditions."""

    def __init__(self, config: HybridArchitectureConfig):
        super().__init__()

        self.config = config

        # Market condition classifier
        self.market_classifier = MarketConditionClassifier(
            input_dim=config.fusion_dim,
            hidden_dim=config.market_condition_dim,
            num_conditions=config.num_market_conditions
        )

        # Architecture weight generators for each market condition
        self.architecture_weights = nn.ModuleDict()
        for i in range(config.num_market_conditions):
            self.architecture_weights[str(i)] = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(config.fusion_dim // 2, 3),  # CNN, LSTM, Transformer weights
                nn.Softmax(dim=-1)
            )

    def forward(
        self,
        fused_features: torch.Tensor,
        cnn_features: torch.Tensor,
        lstm_features: torch.Tensor,
        transformer_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive architecture selection and feature weighting.

        Args:
            fused_features: Fused features for market condition detection
            cnn_features: CNN features
            lstm_features: LSTM features
            transformer_features: Transformer features

        Returns:
            Tuple of (weighted_features, market_condition_probs)
        """
        # Classify market conditions
        market_condition_logits = self.market_classifier(fused_features)
        market_condition_probs = F.softmax(market_condition_logits, dim=-1)

        # Generate architecture weights for each condition
        batch_size = fused_features.shape[0]
        weighted_features = torch.zeros_like(cnn_features)

        for i in range(self.config.num_market_conditions):
            # Get weights for this market condition
            condition_weights = self.architecture_weights[str(i)](fused_features)

            # Weight the features
            condition_features = (
                condition_weights[:, 0:1] * cnn_features +
                condition_weights[:, 1:2] * lstm_features +
                condition_weights[:, 2:3] * transformer_features
            )

            # Weight by market condition probability
            condition_prob = market_condition_probs[:, i:i+1]
            weighted_features += condition_prob * condition_features

        return weighted_features, market_condition_probs


class LearnedFeatureFusion(nn.Module):
    """Learned feature fusion with attention weights."""

    def __init__(self, config: HybridArchitectureConfig):
        super().__init__()

        self.config = config
        feature_dim = config.fusion_dim // 3  # Each component contributes equally

        # Feature projection layers
        self.cnn_proj = nn.Linear(feature_dim, config.fusion_dim)
        self.lstm_proj = nn.Linear(feature_dim, config.fusion_dim)
        self.transformer_proj = nn.Linear(feature_dim, config.fusion_dim)

        # Multi-head attention for feature fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            dropout=config.fusion_dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.fusion_dim)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_dim * 3, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim)
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        lstm_features: torch.Tensor,
        transformer_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Learned feature fusion with attention.

        Args:
            cnn_features: CNN features (batch_size, feature_dim)
            lstm_features: LSTM features (batch_size, feature_dim)
            transformer_features: Transformer features (batch_size, feature_dim)

        Returns:
            Fused features (batch_size, fusion_dim)
        """
        # Project features to fusion dimension
        cnn_proj = self.cnn_proj(cnn_features)
        lstm_proj = self.lstm_proj(lstm_features)
        transformer_proj = self.transformer_proj(transformer_features)

        # Stack features for attention
        stacked_features = torch.stack([cnn_proj, lstm_proj, transformer_proj], dim=1)

        # Apply self-attention for feature fusion
        attended_features, attention_weights = self.fusion_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Layer normalization with residual connection
        attended_features = self.layer_norm(attended_features + stacked_features)

        # Flatten and fuse
        flattened_features = attended_features.view(attended_features.shape[0], -1)
        fused_features = self.fusion_layer(flattened_features)

        return fused_features


class HybridCNNLSTMTransformer(nn.Module):
    """
    Hybrid CNN-LSTM-Transformer Architecture.

    This revolutionary architecture combines CNN spatial features with LSTM/Transformer
    temporal modeling, featuring learned feature fusion, cross-attention between
    spatial and temporal representations, and adaptive architecture selection.
    """

    def __init__(self, config: HybridArchitectureConfig):
        super().__init__()

        self.config = config

        # Component architectures
        self.cnn = MultiScalePriceCNN(config.cnn_config)
        self.lstm = BidirectionalLSTMWithAttention(config.lstm_config)
        self.transformer = TransformerTemporalEncoder(config.transformer_config)

        # Cross-modal attention modules
        # Ensure attention_dim is compatible with spatial_dim
        cross_attention_dim = min(config.cross_attention_dim, config.cnn_config.output_dim)
        # Make sure cross_attention_dim is divisible by num_heads
        cross_attention_dim = (cross_attention_dim // config.cross_attention_heads) * config.cross_attention_heads
        
        self.cnn_temporal_attention = CrossModalAttention(
            spatial_dim=config.cnn_config.output_dim,
            temporal_dim=config.lstm_config.lstm_hidden_dim * 2,
            attention_dim=cross_attention_dim,
            num_heads=config.cross_attention_heads
        )

        self.cnn_transformer_attention = CrossModalAttention(
            spatial_dim=config.cnn_config.output_dim,
            temporal_dim=config.transformer_config.d_model,
            attention_dim=cross_attention_dim,
            num_heads=config.cross_attention_heads
        )

        # Learned feature fusion
        self.feature_fusion = LearnedFeatureFusion(config)

        # Adaptive architecture selection
        if config.use_adaptive_selection:
            self.adaptive_selector = AdaptiveArchitectureSelector(config)

        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.fusion_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.output_dim, config.output_dim)
        )
        
        # Task-specific prediction heads
        self.prediction_heads = nn.ModuleDict({
            'price_prediction': nn.Linear(config.output_dim, 1),
            'volatility_estimation': nn.Linear(config.output_dim, 1),
            'regime_detection': nn.Linear(config.output_dim, 4)  # 4 market regimes
        })

    def forward(
        self,
        multi_timeframe_data: Dict[str, torch.Tensor],
        sequence_data: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of hybrid architecture.

        Args:
            multi_timeframe_data: Multi-timeframe data for CNN
            sequence_data: Sequential data for LSTM/Transformer (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for padding (batch_size,)

        Returns:
            Dictionary containing model outputs and intermediate features
        """
        batch_size = sequence_data.shape[0]

        # CNN spatial feature extraction
        cnn_results = self.cnn(multi_timeframe_data)
        if isinstance(cnn_results, dict):
            cnn_features = cnn_results['output']
        else:
            cnn_features = cnn_results

        # LSTM temporal modeling
        lstm_results = self.lstm(sequence_data, lengths)
        if isinstance(lstm_results, dict):
            lstm_features = lstm_results['output']
            lstm_temporal_features = lstm_results.get('lstm_output')
        else:
            lstm_features = lstm_results
            lstm_temporal_features = None

        # Transformer temporal modeling
        transformer_results = self.transformer(sequence_data, lengths)
        if isinstance(transformer_results, dict):
            transformer_features = transformer_results.get('pooled_output', transformer_results.get('output'))
            transformer_temporal_features = transformer_results.get('output')
        else:
            transformer_features = transformer_results
            transformer_temporal_features = None

        # Cross-modal attention enhancement
        enhanced_cnn_features = cnn_features.clone() if not isinstance(cnn_features, dict) else cnn_features['output'].clone()

        # CNN-LSTM cross-attention
        if lstm_temporal_features is not None:
            cnn_lstm_enhanced = self.cnn_temporal_attention(
                cnn_features, lstm_temporal_features
            )
            enhanced_cnn_features = enhanced_cnn_features + cnn_lstm_enhanced

        # CNN-Transformer cross-attention
        if transformer_temporal_features is not None:
            cnn_transformer_enhanced = self.cnn_transformer_attention(
                cnn_features, transformer_temporal_features
            )
            enhanced_cnn_features = enhanced_cnn_features + cnn_transformer_enhanced

        # Learned feature fusion
        fused_features = self.feature_fusion(
            enhanced_cnn_features, lstm_features, transformer_features
        )

        # Adaptive architecture selection
        if self.config.use_adaptive_selection and hasattr(self, 'adaptive_selector'):
            final_features, market_condition_probs = self.adaptive_selector(
                fused_features, enhanced_cnn_features, lstm_features, transformer_features
            )
        else:
            final_features = fused_features
            market_condition_probs = None

        # Final output projection
        output = self.output_projection(final_features)
        
        # Generate task-specific predictions
        predictions = {}
        for task_name, head in self.prediction_heads.items():
            predictions[task_name] = head(output)

        return {
            'output': output,
            'price_prediction': predictions['price_prediction'],
            'volatility_estimation': predictions['volatility_estimation'],
            'regime_detection': predictions['regime_detection'],
            'cnn_features': cnn_features,
            'enhanced_cnn_features': enhanced_cnn_features,
            'lstm_features': lstm_features,
            'transformer_features': transformer_features,
            'fused_features': fused_features,
            'final_features': final_features,
            'market_condition_probs': market_condition_probs,
            'lstm_attention_weights': lstm_results.get('attention_weights'),
            'transformer_layer_outputs': transformer_results.get('layer_outputs')
        }

    def get_feature_importance(
        self,
        multi_timeframe_data: Dict[str, torch.Tensor],
        sequence_data: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze feature importance across different components.

        Args:
            multi_timeframe_data: Multi-timeframe data for CNN
            sequence_data: Sequential data for LSTM/Transformer
            lengths: Sequence lengths for padding

        Returns:
            Dictionary containing feature importance analysis
        """
        with torch.no_grad():
            results = self.forward(multi_timeframe_data, sequence_data, lengths)

            # Compute feature norms as importance measures
            cnn_importance = torch.norm(results['cnn_features'], dim=-1)
            lstm_importance = torch.norm(results['lstm_features'], dim=-1)
            transformer_importance = torch.norm(results['transformer_features'], dim=-1)

            # Normalize to get relative importance
            total_importance = cnn_importance + lstm_importance + transformer_importance
            cnn_relative = cnn_importance / (total_importance + 1e-8)
            lstm_relative = lstm_importance / (total_importance + 1e-8)
            transformer_relative = transformer_importance / (total_importance + 1e-8)

            return {
                'cnn_importance': cnn_importance,
                'lstm_importance': lstm_importance,
                'transformer_importance': transformer_importance,
                'cnn_relative_importance': cnn_relative,
                'lstm_relative_importance': lstm_relative,
                'transformer_relative_importance': transformer_relative,
                'market_condition_probs': results['market_condition_probs']
            }

    def freeze_component(self, component: str):
        """Freeze parameters of a specific component."""
        if component == 'cnn':
            for param in self.cnn.parameters():
                param.requires_grad = False
        elif component == 'lstm':
            for param in self.lstm.parameters():
                param.requires_grad = False
        elif component == 'transformer':
            for param in self.transformer.parameters():
                param.requires_grad = False

    def unfreeze_component(self, component: str):
        """Unfreeze parameters of a specific component."""
        if component == 'cnn':
            for param in self.cnn.parameters():
                param.requires_grad = True
        elif component == 'lstm':
            for param in self.lstm.parameters():
                param.requires_grad = True
        elif component == 'transformer':
            for param in self.transformer.parameters():
                param.requires_grad = True


def create_hybrid_cnn_lstm_transformer(
    config: Optional[HybridArchitectureConfig] = None
) -> HybridCNNLSTMTransformer:
    """
    Factory function to create a Hybrid CNN-LSTM-Transformer model.

    Args:
        config: Optional configuration. If None, uses default configuration.

    Returns:
        Initialized HybridCNNLSTMTransformer model
    """
    if config is None:
        config = HybridArchitectureConfig()

    return HybridCNNLSTMTransformer(config)


if __name__ == "__main__":
    # Example usage and testing
    config = HybridArchitectureConfig(
        input_dim=5,
        sequence_length=100,
        fusion_dim=1024,
        use_adaptive_selection=True,
        output_dim=512
    )

    model = create_hybrid_cnn_lstm_transformer(config)

    # Create sample data
    batch_size = 32
    seq_len = 100
    input_dim = 5

    # Multi-timeframe data for CNN
    multi_timeframe_data = {
        '1min': torch.randn(batch_size, input_dim, seq_len),
        '5min': torch.randn(batch_size, input_dim, seq_len),
        '15min': torch.randn(batch_size, input_dim, seq_len)
    }

    # Sequential data for LSTM/Transformer
    sequence_data = torch.randn(batch_size, seq_len, input_dim)
    sample_lengths = torch.randint(50, seq_len + 1, (batch_size,))

    # Forward pass
    results = model(multi_timeframe_data, sequence_data, sample_lengths)
    print(f"Output shape: {results['output'].shape}")
    print(f"CNN features shape: {results['cnn_features'].shape}")
    print(f"LSTM features shape: {results['lstm_features'].shape}")
    print(f"Transformer features shape: {results['transformer_features'].shape}")
    print(f"Fused features shape: {results['fused_features'].shape}")

    if results['market_condition_probs'] is not None:
        print(f"Market condition probs shape: {results['market_condition_probs'].shape}")

    # Feature importance analysis
    importance = model.get_feature_importance(
        multi_timeframe_data, sequence_data, sample_lengths
    )
    print(f"CNN relative importance: {importance['cnn_relative_importance'].mean():.4f}")
    print(f"LSTM relative importance: {importance['lstm_relative_importance'].mean():.4f}")
    print(f"Transformer relative importance: {importance['transformer_relative_importance'].mean():.4f}")
