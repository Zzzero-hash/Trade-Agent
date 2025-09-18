"""Base classes and exceptions for feature extraction

This module provides the abstract base class and custom exceptions
for the feature extraction system, enabling clean separation of concerns
and proper error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class FeatureExtractionError(Exception):
    """Base exception for feature extraction errors"""
    pass


class DataValidationError(FeatureExtractionError):
    """Raised when input data validation fails"""
    pass


class FeatureComputationError(FeatureExtractionError):
    """Raised when feature computation fails"""
    pass


class ModelLoadError(FeatureExtractionError):
    """Raised when model loading fails"""
    pass


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors
    
    This interface defines the contract for all feature extraction implementations,
    enabling clean separation of concerns, easy testing and mocking.
    """
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from input data
        
        Args:
            data: Input market data as numpy array
            
        Returns:
            Dictionary containing extracted features with descriptive names
            
        Raises:
            DataValidationError: If input data is invalid
            FeatureComputationError: If feature extraction fails
        """
        pass
    
    @abstractmethod
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of extracted features
        
        Returns:
            Dictionary mapping feature names to their dimensions
        """
        pass
    
    def validate_input_data(self, data: np.ndarray) -> None:
        """Validate input data
        
        Args:
            data: Input data to validate
            
        Raises:
            DataValidationError: If data is invalid
        """
        if data is None:
            raise DataValidationError("Input data cannot be None")
        
        if not isinstance(data, np.ndarray):
            raise DataValidationError(f"Expected np.ndarray, got {type(data)}")
        
        if data.ndim not in [2, 3]:
            raise DataValidationError(f"Expected 2D or 3D array, got {data.ndim}D array")
        
        if data.size == 0:
            raise DataValidationError("Input data cannot be empty")
        
        if np.isnan(data).any() or np.isinf(data).any():
            raise DataValidationError("Input data contains NaN or infinite values")