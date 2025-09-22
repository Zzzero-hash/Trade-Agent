"""
Transformer-based Temporal Encoder

This module implements a sophisticated Transformer encoder architecture optimized
for time series data with positional encoding, causal masking, multi-scale temporal
attention, and efficient attention mechanisms.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerTemporalConfig:
    """Configuration for Transformer-based Temporal Encoder."""

    # Input dimensions
    input_dim: int = 512
    sequence_length: int = 100

    # Transformer architecture
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048

    # Multi-scale temporal attention
    temporal_scales: List[int] = None  # [1, 4, 16] for different attention windows
    use_multi_scale: bool = True

    # Positional encoding
    max_position_embeddings: int = 1000
    use_learned_positional: bool = False

    # Causal masking
    use_causal_mask: bool = True

    # Efficient attention mechanisms
    attention_type: str = "standard"  # "standard", "linear", "performer"
    linear_attention_features: int = 256

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1

    # Layer normalization
    layer_norm_eps: float = 1e-5
    pre_norm: bool = True  # Pre-norm vs post-norm

    # Output
    output_dim: int = 512

    def __post_init__(self):
        if self.temporal_scales is None:
            self.temporal_scales = [1, 4, 16]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series data."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for time series data."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input embeddings."""
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings


class LinearAttention(nn.Module):
    """Linear attention mechanism for efficient computation."""

    def __init__(self, d_model: int, num_heads: int, feature_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feature mapping for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, feature_dim),
            nn.ReLU()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Linear attention forward pass."""
        batch_size, seq_len, d_model = query.shape

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply feature mapping
        q_features = self.feature_map(q)  # (batch, seq_len, heads, feature_dim)
        k_features = self.feature_map(k)

        # Linear attention computation
        # Compute K^T V first (more efficient)
        kv = torch.einsum('bshf,bshd->bhfd', k_features, v)
        # Then compute Q(K^T V)
        output = torch.einsum('bshf,bhfd->bshd', q_features, kv)

        # Normalize by sum of keys
        k_sum = torch.sum(k_features, dim=1, keepdim=True)  # (batch, 1, heads, feature_dim)
        q_k_sum = torch.einsum('bshf,bthf->bsh', q_features, k_sum)
        output = output / (q_k_sum.unsqueeze(-1) + 1e-8)

        # Reshape and project
        output = output.contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(output)


class MultiScaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention across different time horizons."""

    def __init__(self, config: TransformerTemporalConfig):
        super().__init__()
        self.config = config
        self.temporal_scales = config.temporal_scales

        # Create attention modules for each scale
        self.scale_attentions = nn.ModuleDict()
        for scale in self.temporal_scales:
            if config.attention_type == "linear":
                attention = LinearAttention(
                    config.d_model,
                    config.num_heads,
                    config.linear_attention_features
                )
            else:
                attention = nn.MultiheadAttention(
                    config.d_model,
                    config.num_heads,
                    dropout=config.attention_dropout,
                    batch_first=True
                )
            self.scale_attentions[str(scale)] = attention

        # Scale fusion layer
        self.scale_fusion = nn.Linear(
            config.d_model * len(self.temporal_scales),
            config.d_model
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Multi-scale temporal attention forward pass."""
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []

        for scale in self.temporal_scales:
            # Create scale-specific input by downsampling or windowing
            if scale == 1:
                scale_input = x
                scale_mask = attn_mask
            else:
                # Downsample by taking every 'scale' timesteps
                scale_input = x[:, ::scale, :]
                if attn_mask is not None:
                    scale_mask = attn_mask[::scale, ::scale]
                else:
                    scale_mask = None

            # Apply attention at this scale
            attention = self.scale_attentions[str(scale)]
            if isinstance(attention, LinearAttention):
                scale_output = attention(scale_input, scale_input, scale_input, scale_mask)
            else:
                scale_output, _ = attention(
                    scale_input, scale_input, scale_input,
                    attn_mask=scale_mask
                )

            # Upsample back to original sequence length if needed
            if scale > 1:
                scale_output = F.interpolate(
                    scale_output.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_outputs.append(scale_output)

        # Fuse multi-scale outputs
        fused_output = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused_output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Enhanced Transformer encoder layer with multi-scale attention."""

    def __init__(self, config: TransformerTemporalConfig):
        super().__init__()
        self.config = config

        # Multi-scale attention or standard attention
        if config.use_multi_scale:
            self.self_attn = MultiScaleTemporalAttention(config)
        else:
            if config.attention_type == "linear":
                self.self_attn = LinearAttention(
                    config.d_model,
                    config.num_heads,
                    config.linear_attention_features
                )
            else:
                self.self_attn = nn.MultiheadAttention(
                    config.d_model,
                    config.num_heads,
                    dropout=config.attention_dropout,
                    batch_first=True
                )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.activation_dropout),
            nn.Linear(config.d_ff, config.d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Dropout
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.pre_norm = config.pre_norm

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transformer encoder layer forward pass."""
        if self.pre_norm:
            # Pre-norm architecture
            # Self-attention
            x_norm = self.norm1(x)
            if isinstance(self.self_attn, (MultiScaleTemporalAttention, LinearAttention)):
                attn_output = self.self_attn(x_norm, attn_mask)
            else:
                attn_output, _ = self.self_attn(
                    x_norm, x_norm, x_norm,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask
                )
            x = x + self.dropout1(attn_output)

            # Feed-forward
            x_norm = self.norm2(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.dropout2(ff_output)
        else:
            # Post-norm architecture
            # Self-attention
            if isinstance(self.self_attn, (MultiScaleTemporalAttention, LinearAttention)):
                attn_output = self.self_attn(x, attn_mask)
            else:
                attn_output, _ = self.self_attn(
                    x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask
                )
            x = self.norm1(x + self.dropout1(attn_output))

            # Feed-forward
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based Temporal Encoder for Time Series.

    This model implements a Transformer encoder optimized for time series data
    with positional encoding, causal masking, multi-scale temporal attention,
    and efficient attention mechanisms.
    """

    def __init__(self, config: TransformerTemporalConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding
        if config.use_learned_positional:
            self.pos_encoding = LearnedPositionalEncoding(
                config.d_model, config.max_position_embeddings
            )
        else:
            self.pos_encoding = PositionalEncoding(
                config.d_model, config.max_position_embeddings
            )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive modeling."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for allowed positions, False for masked

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

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Transformer temporal encoder.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for padding (batch_size,)
            return_all_layers: Whether to return outputs from all layers

        Returns:
            Dictionary containing:
                - 'output': Final output features (batch_size, seq_len, output_dim)
                - 'pooled_output': Global pooled features (batch_size, output_dim)
                - 'layer_outputs': Outputs from all layers (if requested)
        """
        batch_size, seq_len, input_dim = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        if isinstance(self.pos_encoding, PositionalEncoding):
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        else:
            x = self.pos_encoding(x)

        # Create masks
        attn_mask = None
        key_padding_mask = None

        if self.config.use_causal_mask:
            attn_mask = self._create_causal_mask(seq_len, x.device)

        if lengths is not None:
            key_padding_mask = self._create_padding_mask(lengths, seq_len)

        # Apply Transformer layers
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
            if return_all_layers:
                layer_outputs.append(x.clone())

        # Final layer normalization
        x = self.final_norm(x)

        # Output projection
        output = self.output_projection(x)

        # Global pooling for sequence-level representation
        if lengths is not None:
            # Mask out padding tokens for pooling
            mask = ~key_padding_mask.unsqueeze(-1)
            masked_output = output * mask
            pooled_output = torch.sum(masked_output, dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled_output = torch.mean(output, dim=1)

        result = {
            'output': output,
            'pooled_output': pooled_output
        }

        if return_all_layers:
            result['layer_outputs'] = layer_outputs

        return result

    def get_attention_maps(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for interpretability.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for padding (batch_size,)

        Returns:
            Dictionary containing attention analysis results
        """
        # This is a simplified version - in practice, you'd need to modify
        # the attention modules to return attention weights
        with torch.no_grad():
            results = self.forward(x, lengths)
            # Placeholder for attention analysis
            return {
                'attention_entropy': torch.zeros(x.shape[0], x.shape[1]),
                'attention_patterns': torch.zeros(x.shape[0], x.shape[1], x.shape[1])
            }


def create_transformer_temporal_encoder(
    config: Optional[TransformerTemporalConfig] = None
) -> TransformerTemporalEncoder:
    """
    Factory function to create a Transformer Temporal Encoder.

    Args:
        config: Optional configuration. If None, uses default configuration.

    Returns:
        Initialized TransformerTemporalEncoder model
    """
    if config is None:
        config = TransformerTemporalConfig()

    return TransformerTemporalEncoder(config)


if __name__ == "__main__":
    # Example usage and testing
    config = TransformerTemporalConfig(
        input_dim=512,
        sequence_length=100,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        temporal_scales=[1, 4, 16],
        use_multi_scale=True,
        attention_type="standard",
        output_dim=512
    )

    model = create_transformer_temporal_encoder(config)

    # Create sample data
    batch_size = 32
    seq_len = 100
    input_dim = 512

    sample_data = torch.randn(batch_size, seq_len, input_dim)
    sample_lengths = torch.randint(50, seq_len + 1, (batch_size,))

    # Forward pass
    results = model(sample_data, sample_lengths, return_all_layers=True)
    print(f"Output shape: {results['output'].shape}")
    print(f"Pooled output shape: {results['pooled_output'].shape}")
    print(f"Number of layer outputs: {len(results['layer_outputs'])}")

    # Test with different attention types
    for attention_type in ["standard", "linear"]:
        config.attention_type = attention_type
        model = create_transformer_temporal_encoder(config)
        results = model(sample_data, sample_lengths)
        print(f"{attention_type} attention - Output shape: {results['output'].shape}")