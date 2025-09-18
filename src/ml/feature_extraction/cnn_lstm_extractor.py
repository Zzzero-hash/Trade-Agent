"""CNN+LSTM feature extractor implementation

This module provides the core CNN+LSTM feature extraction functionality
with proper error handling, input validation, and resource management.
"""

import logging
from typing import Dict, Optional, Any
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn

from .base import (
    FeatureExtractor, 
    FeatureExtractionError, 
    DataValidationError,
    FeatureComputationError,
    ModelLoadError
)


class CNNLSTMExtractor(FeatureExtractor):
    """CNN+LSTM feature extractor for trading environments
    
    This extractor uses a pre-trained CNN+LSTM hybrid model to generate
    rich feature representations from market data for RL environments.
    """
    
    def __init__(self, hybrid_model, device: Optional[str] = None):
        """Initialize CNN+LSTM feature extractor
        
        Args:
            hybrid_model: Pre-trained CNN+LSTM hybrid model
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model = hybrid_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Move model to device and set to eval mode
        try:
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize model: {e}") from e
    
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract CNN+LSTM features from market data
        
        Args:
            data: Market data window as numpy array
            
        Returns:
            Dictionary containing extracted features:
            - 'fused_features': Combined CNN+LSTM features
            - 'cnn_features': CNN-only features  
            - 'lstm_features': LSTM-only features
            - 'classification_confidence': Model confidence scores
            - 'regression_uncertainty': Uncertainty estimates
            - 'ensemble_weights': Ensemble weights (if available)
            
        Raises:
            DataValidationError: If input data is invalid
            FeatureComputationError: If feature extraction fails
        """
        # Validate input data
        self.validate_input_data(data)
        
        try:
            with self._inference_context():
                # Convert to tensor and add batch dimension if needed
                if data.ndim == 2:
                    data = data[np.newaxis, ...]  # Add batch dimension
                
                input_tensor = torch.FloatTensor(data).to(self.device)
                
                # Extract features from model
                with torch.no_grad():
                    outputs = self.model.forward(
                        input_tensor,
                        return_features=True,
                        use_ensemble=True
                    )
                
                # Convert outputs to numpy and extract features
                features = self._process_model_outputs(outputs)
                
                return features
                
        except torch.cuda.OutOfMemoryError as e:
            raise FeatureComputationError(
                f"GPU out of memory during feature extraction: {e}"
            ) from e
        except RuntimeError as e:
            raise FeatureComputationError(
                f"Model inference failed: {e}"
            ) from e
        except Exception as e:
            raise FeatureComputationError(
                f"Unexpected error during feature extraction: {e}"
            ) from e
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of extracted features
        
        Returns:
            Dictionary mapping feature names to their dimensions
        """
        return {
            'fused_features': 256,  # Standard fused feature dimension
            'cnn_features': 128,    # CNN feature dimension
            'lstm_features': 128,   # LSTM feature dimension
            'classification_confidence': 1,  # Single confidence score
            'regression_uncertainty': 1,     # Single uncertainty value
            'ensemble_weights': 4    # Number of ensemble models
        }
    
    def _process_model_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Process model outputs into feature dictionary
        
        Args:
            outputs: Raw model outputs as tensors
            
        Returns:
            Processed features as numpy arrays
        """
        features = {}
        
        # Extract fused features (last timestep)
        if 'fused_features' in outputs:
            features['fused_features'] = outputs['fused_features'][:, -1, :].cpu().numpy()
        
        # Extract CNN features
        if 'cnn_features' in outputs:
            features['cnn_features'] = outputs['cnn_features'][:, -1, :].cpu().numpy()
        
        # Extract LSTM features  
        if 'lstm_features' in outputs:
            features['lstm_features'] = outputs['lstm_features'][:, -1, :].cpu().numpy()
        
        # Extract classification confidence
        if 'classification_probs' in outputs:
            features['classification_confidence'] = torch.max(
                outputs['classification_probs'], dim=1
            )[0].cpu().numpy()
        
        # Extract regression uncertainty
        if 'regression_uncertainty' in outputs:
            features['regression_uncertainty'] = outputs['regression_uncertainty'].cpu().numpy()
        
        # Extract ensemble weights
        if 'ensemble_weights' in outputs and outputs['ensemble_weights'] is not None:
            features['ensemble_weights'] = outputs['ensemble_weights'].cpu().numpy()
        
        return features
    
    @contextmanager
    def _inference_context(self):
        """Context manager for model inference with proper resource cleanup"""
        try:
            yield
        finally:
            # Clear GPU cache if using CUDA
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def validate_input_data(self, data: np.ndarray) -> None:
        """Enhanced input validation for CNN+LSTM model
        
        Args:
            data: Input data to validate
            
        Raises:
            DataValidationError: If data is invalid
        """
        # Call parent validation first
        super().validate_input_data(data)
        
        # Additional CNN+LSTM specific validation
        if data.ndim == 2 and data.shape[0] < 10:
            raise DataValidationError(
                f"Insufficient time steps: got {data.shape[0]}, need at least 10"
            )
        
        if data.ndim == 3 and data.shape[1] < 10:
            raise DataValidationError(
                f"Insufficient time steps: got {data.shape[1]}, need at least 10"
            )