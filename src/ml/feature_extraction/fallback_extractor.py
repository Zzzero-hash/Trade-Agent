"""Fallback feature extractor implementation

This module provides a fallback feature extractor that uses basic technical
indicators when the primary CNN+LSTM extractor fails.
"""

import logging
from typing import Dict
import numpy as np

from .base import FeatureExtractor, FeatureExtractionError


class FallbackFeatureExtractor(FeatureExtractor):
    """Fallback feature extractor using basic technical indicators
    
    This extractor provides a reliable fallback when the primary CNN+LSTM
    extractor fails, using simple technical indicators that are always
    computable from price/volume data.
    """
    
    def __init__(self, primary_extractor: FeatureExtractor):
        """Initialize fallback feature extractor
        
        Args:
            primary_extractor: Primary extractor to try first
        """
        self.primary_extractor = primary_extractor
        self.logger = logging.getLogger(__name__)
        self._fallback_active = False
    
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features with fallback capability
        
        Args:
            data: Input market data as numpy array
            
        Returns:
            Dictionary containing extracted features
            
        Raises:
            FeatureExtractionError: If both primary and fallback extraction fail
        """
        try:
            # Try primary extractor first
            result = self.primary_extractor.extract_features(data)
            
            # Reset fallback flag if primary succeeds
            if self._fallback_active:
                self.logger.info("Primary feature extractor recovered")
                self._fallback_active = False
            
            return result
            
        except Exception as e:
            # Log fallback activation
            if not self._fallback_active:
                self.logger.warning(
                    f"Primary feature extractor failed, using fallback: {e}"
                )
                self._fallback_active = True
            
            # Use fallback extraction
            return self._extract_fallback_features(data)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions
        
        Returns fallback dimensions when in fallback mode, otherwise
        returns primary extractor dimensions.
        """
        if self._fallback_active:
            return self._get_fallback_dimensions()
        else:
            try:
                return self.primary_extractor.get_feature_dimensions()
            except Exception:
                return self._get_fallback_dimensions()
    
    def _extract_fallback_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract basic technical indicator features
        
        Args:
            data: Market data array (assumes OHLCV format)
            
        Returns:
            Dictionary with basic technical indicators
        """
        try:
            # Validate input for fallback
            self.validate_input_data(data)
            
            # Assume data is in OHLCV format
            if data.ndim == 3:
                # Batch dimension present, take first sample
                prices = data[0, :, :]
            else:
                prices = data
            
            # Extract OHLCV (assuming standard format)
            if prices.shape[1] >= 5:
                open_prices = prices[:, 0]
                high_prices = prices[:, 1]
                low_prices = prices[:, 2]
                close_prices = prices[:, 3]
                volumes = prices[:, 4]
            else:
                # Fallback to close prices only
                close_prices = prices[:, -1]
                open_prices = high_prices = low_prices = close_prices
                volumes = np.ones_like(close_prices)
            
            # Calculate basic technical indicators
            features = self._calculate_technical_indicators(
                open_prices, high_prices, low_prices, close_prices, volumes
            )
            
            return features
            
        except Exception as e:
            raise FeatureExtractionError(
                f"Fallback feature extraction failed: {e}"
            ) from e
    
    def _calculate_technical_indicators(self, open_p: np.ndarray, high_p: np.ndarray,
                                      low_p: np.ndarray, close_p: np.ndarray,
                                      volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate basic technical indicators
        
        Args:
            open_p: Open prices
            high_p: High prices  
            low_p: Low prices
            close_p: Close prices
            volume: Volume data
            
        Returns:
            Dictionary with technical indicators
        """
        # Simple moving averages
        sma_5 = self._sma(close_p, 5)
        sma_20 = self._sma(close_p, 20)
        
        # Price ratios
        price_ratio = close_p[-1] / close_p[0] if len(close_p) > 0 else 1.0
        
        # Volatility (simple standard deviation)
        returns = np.diff(close_p) / close_p[:-1] if len(close_p) > 1 else np.array([0.0])
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        
        # Volume indicators
        avg_volume = np.mean(volume) if len(volume) > 0 else 0.0
        volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # RSI approximation
        rsi = self._simple_rsi(close_p, 14)
        
        # Combine into feature vector
        basic_features = np.array([
            sma_5, sma_20, price_ratio, volatility, avg_volume,
            volume_ratio, rsi, close_p[-1], high_p[-1], low_p[-1],
            np.mean(close_p), np.max(close_p), np.min(close_p),
            np.mean(volume), np.std(close_p)
        ], dtype=np.float32)
        
        return {
            'fused_features': basic_features.reshape(1, -1),  # Match expected shape
            'classification_confidence': np.array([0.5]),     # Neutral confidence
            'regression_uncertainty': np.array([1.0])         # High uncertainty
        }
    
    def _sma(self, prices: np.ndarray, window: int) -> float:
        """Calculate simple moving average"""
        if len(prices) >= window:
            return np.mean(prices[-window:])
        else:
            return np.mean(prices) if len(prices) > 0 else 0.0
    
    def _simple_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate simplified RSI"""
        if len(prices) < 2:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) >= window:
            avg_gain = np.mean(gains[-window:])
            avg_loss = np.mean(losses[-window:])
        else:
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_fallback_dimensions(self) -> Dict[str, int]:
        """Get fallback feature dimensions"""
        return {
            'fused_features': 15,  # Basic technical indicators
            'classification_confidence': 1,
            'regression_uncertainty': 1
        }
    
    @property
    def is_fallback_active(self) -> bool:
        """Check if fallback mode is currently active"""
        return self._fallback_active