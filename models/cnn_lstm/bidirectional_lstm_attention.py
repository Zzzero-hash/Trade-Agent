"""
Bidirectional LSTM with Multi-Head Attention

This module implements a sophisticated bidirectional LSTM architecture with
multi-head attention mechanisms for temporal dependency capture and hierarchical
feature extraction across multiple time horizons.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BidirectionalLSTMConfig:
    """Configuration for Bidirectional LSTM with Multi-Head Attention."""

    # Input dimensions
    input_dim: int = 512
    sequence_length: int = 100

    # LSTM architecture
    lstm_hidden_dim: int = 256
    num_lstm_layers: int = 2
    lstm_dropout: float = 0.2

    # Multi-head attention
    attention_dim: int = 512
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Hierarchical time horizons
    time_horizons: List[int] = None  # [10, 30, 60] steps

    # Layer normalization and regularization
    use_layer_norm: bool = True
    gradient_clip_val: float = 1.0
    dropout_rate: float = 0.1

    # Output
    output_dim: int = 512

    def __post_init__(self):
        if self.time_horizons is None:
            self.time_horizons = [10, 30, 60]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism optimized for temporal sequences."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            attn_mask: Attention mask (seq_len, seq_len)
            key_padding_mask: Key padding mask (batch_size, seq_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.shape

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)

        return output, attn_weights.mean(dim=1)  # Average over heads


class HierarchicalTemporalEncoder(nn.Module):
    """Hierarchical temporal encoder for multi-horizon feature extraction."""

    def __init__(self, config: BidirectionalLSTMConfig):
        super().__init__()

        self.config = config
        self.time_horizons = config.time_horizons

        # Create separate encoders for each time horizon
        self.horizon_encoders = nn.ModuleDict()
        for horizon in self.time_horizons:
            self.horizon_encoders[str(horizon)] = nn.Sequential(
                nn.Linear(config.lstm_hidden_dim * 2, config.attention_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.attention_dim, config.attention_dim // len(self.time_horizons))
            )

        # Calculate actual concatenated dimension
        horizon_dim = config.attention_dim // len(self.time_horizons)
        total_concat_dim = horizon_dim * len(self.time_horizons)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_concat_dim, config.attention_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Extract hierarchical features across multiple time horizons.

        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_dim * 2)

        Returns:
            Hierarchical features (batch_size, attention_dim)
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape
        horizon_features = []

        for horizon in self.time_horizons:
            if horizon <= seq_len:
                # Extract features for this time horizon
                horizon_data = lstm_output[:, -horizon:, :]  # Last 'horizon' steps
                # Global average pooling over time dimension
                pooled_features = torch.mean(horizon_data, dim=1)
                # Process through horizon-specific encoder
                encoded_features = self.horizon_encoders[str(horizon)](pooled_features)
                horizon_features.append(encoded_features)
            else:
                # Handle case where horizon is larger than sequence length
                device = lstm_output.device
                zero_features = torch.zeros(
                    batch_size,
                    self.config.attention_dim // len(self.time_horizons),
                    device=device
                )
                horizon_features.append(zero_features)

        # Concatenate and fuse horizon features
        combined_features = torch.cat(horizon_features, dim=1)
        fused_features = self.fusion_layer(combined_features)

        return fused_features


class BidirectionalLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Attention.

    This model implements bidirectional LSTM for forward/backward temporal modeling
    with multi-head attention mechanisms and hierarchical feature extraction.
    """

    def __init__(self, config: BidirectionalLSTMConfig):
        super().__init__()

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.lstm_hidden_dim)

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention
        lstm_output_dim = config.lstm_hidden_dim * 2  # Bidirectional
        self.attention = MultiHeadAttention(
            embed_dim=lstm_output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        )

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_dim)
            self.layer_norm2 = nn.LayerNorm(lstm_output_dim)

        # Hierarchical temporal encoder
        self.hierarchical_encoder = HierarchicalTemporalEncoder(config)

        # Final projection layers
        self.final_projection = nn.Sequential(
            nn.Linear(config.attention_dim, config.output_dim),
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

        # Gradient clipping value
        self.gradient_clip_val = config.gradient_clip_val

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of bidirectional LSTM with attention.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for padding (batch_size,)

        Returns:
            Dictionary containing:
                - 'output': Final output features (batch_size, output_dim)
                - 'attention_weights': Attention weights (batch_size, seq_len, seq_len)
                - 'lstm_output': Raw LSTM output (batch_size, seq_len, hidden_dim*2)
                - 'hierarchical_features': Hierarchical features (batch_size, attention_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Input projection
        x_proj = self.input_proj(x)

        # Pack sequences if lengths are provided
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_output_packed, (hidden, cell) = self.lstm(x_packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output_packed, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(x_proj)

        # Apply layer normalization
        if self.config.use_layer_norm:
            lstm_output = self.layer_norm1(lstm_output)

        # Multi-head attention (self-attention)
        attended_output, attention_weights = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=self._create_padding_mask(lengths, seq_len) if lengths is not None else None
        )

        # Residual connection and layer normalization
        lstm_output = lstm_output + attended_output
        if self.config.use_layer_norm:
            lstm_output = self.layer_norm2(lstm_output)

        # Hierarchical temporal encoding
        hierarchical_features = self.hierarchical_encoder(lstm_output)

        # Final projection
        output = self.final_projection(hierarchical_features)
        
        # Generate task-specific predictions
        predictions = {}
        for task_name, head in self.prediction_heads.items():
            predictions[task_name] = head(output)

        return {
            'output': output,
            'price_prediction': predictions['price_prediction'],
            'volatility_estimation': predictions['volatility_estimation'],
            'regime_detection': predictions['regime_detection'],
            'attention_weights': attention_weights,
            'lstm_output': lstm_output,
            'hierarchical_features': hierarchical_features
        }

    def _create_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Create padding mask for variable length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def clip_gradients(self):
        """Clip gradients for training stability."""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.gradient_clip_val
            )

    def get_attention_patterns(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention patterns for interpretability analysis.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for padding (batch_size,)

        Returns:
            Dictionary containing attention analysis results
        """
        with torch.no_grad():
            results = self.forward(x, lengths)

            # Compute attention entropy for each position
            attention_weights = results['attention_weights']
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8),
                dim=-1
            )

            # Find most attended positions
            max_attention_positions = torch.argmax(
                torch.sum(attention_weights, dim=1), dim=-1
            )

            return {
                'attention_weights': attention_weights,
                'attention_entropy': attention_entropy,
                'max_attention_positions': max_attention_positions,
                'mean_attention_entropy': torch.mean(attention_entropy)
            }


def create_bidirectional_lstm_attention(
    config: Optional[BidirectionalLSTMConfig] = None
) -> BidirectionalLSTMWithAttention:
    """
    Factory function to create a Bidirectional LSTM with Attention model.

    Args:
        config: Optional configuration. If None, uses default configuration.

    Returns:
        Initialized BidirectionalLSTMWithAttention model
    """
    if config is None:
        config = BidirectionalLSTMConfig()

    return BidirectionalLSTMWithAttention(config)


if __name__ == "__main__":
    # Example usage and testing
    config = BidirectionalLSTMConfig(
        input_dim=512,
        sequence_length=100,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        attention_dim=512,
        num_attention_heads=8,
        time_horizons=[10, 30, 60],
        output_dim=512
    )

    model = create_bidirectional_lstm_attention(config)

    # Create sample data
    batch_size = 32
    seq_len = 100
    input_dim = 512

    sample_data = torch.randn(batch_size, seq_len, input_dim)
    sample_lengths = torch.randint(50, seq_len + 1, (batch_size,))

    # Forward pass
    results = model(sample_data, sample_lengths)
    print(f"Output shape: {results['output'].shape}")
    print(f"Attention weights shape: {results['attention_weights'].shape}")
    print(f"Hierarchical features shape: {results['hierarchical_features'].shape}")

    # Get attention patterns
    attention_analysis = model.get_attention_patterns(sample_data, sample_lengths)
    print(f"Mean attention entropy: {attention_analysis['mean_attention_entropy']:.4f}")