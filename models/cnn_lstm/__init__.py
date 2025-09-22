"""
CNN+LSTM Models Module

This module contains state-of-the-art architectures for financial market
feature extraction and temporal modeling, including:
- Multi-scale price CNNs with attention mechanisms
- Volume profile CNNs for market microstructure analysis
- CNN ensemble architectures with neural architecture search
- Bidirectional LSTM with multi-head attention for temporal dependencies
- Transformer-based temporal encoders with causal masking and multi-scale attention
- Hybrid CNN-LSTM-Transformer architectures with cross-modal attention and adaptive selection
"""

from .multi_scale_price_cnn import (
    MultiScalePriceCNN,
    MultiScalePriceCNNConfig,
    create_multi_scale_price_cnn
)

from .volume_profile_cnn import (
    VolumeProfileCNN,
    VolumeProfileCNNConfig,
    create_volume_profile_cnn
)

from .cnn_ensemble_nas import (
    CNNEnsembleNAS,
    CNNEnsembleConfig,
    NASConfig,
    NASTrainer,
    create_cnn_ensemble_nas
)

from .bidirectional_lstm_attention import (
    BidirectionalLSTMWithAttention,
    BidirectionalLSTMConfig,
    create_bidirectional_lstm_attention
)

from .transformer_temporal_encoder import (
    TransformerTemporalEncoder,
    TransformerTemporalConfig,
    create_transformer_temporal_encoder
)

from .hybrid_cnn_lstm_transformer import (
    HybridCNNLSTMTransformer,
    HybridArchitectureConfig,
    create_hybrid_cnn_lstm_transformer
)

__all__ = [
    # Multi-scale Price CNN
    'MultiScalePriceCNN',
    'MultiScalePriceCNNConfig',
    'create_multi_scale_price_cnn',
    
    # Volume Profile CNN
    'VolumeProfileCNN',
    'VolumeProfileCNNConfig',
    'create_volume_profile_cnn',
    
    # CNN Ensemble with NAS
    'CNNEnsembleNAS',
    'CNNEnsembleConfig',
    'NASConfig',
    'NASTrainer',
    'create_cnn_ensemble_nas',
    
    # Bidirectional LSTM with Attention
    'BidirectionalLSTMWithAttention',
    'BidirectionalLSTMConfig',
    'create_bidirectional_lstm_attention',
    
    # Transformer Temporal Encoder
    'TransformerTemporalEncoder',
    'TransformerTemporalConfig',
    'create_transformer_temporal_encoder',
    
    # Hybrid CNN-LSTM-Transformer
    'HybridCNNLSTMTransformer',
    'HybridArchitectureConfig',
    'create_hybrid_cnn_lstm_transformer'
]