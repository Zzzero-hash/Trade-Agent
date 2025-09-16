"""Attention Visualization for CNN+LSTM Hybrid Model

This module provides visualization capabilities for attention weights in the 
CNN+LSTM hybrid model, including CNN attention, LSTM attention, and cross-attention
between CNN and LSTM features.

Requirements: 12.2 - WHEN model explanations are requested THEN the system SHALL
use attention mechanisms to highlight influential data points.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, List

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with 'pip install matplotlib' for visualization.")


class AttentionVisualizer:
    """Visualization tools for attention weights in CNN+LSTM hybrid model."""
    
    def __init__(self):
        """Initialize attention visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available. Visualization functions will raise RuntimeError.")
    
    def visualize_cnn_attention(
        self,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sequence_labels: Optional[List[str]] = None,
        title: str = "CNN Attention Weights",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create heatmap visualization of CNN attention weights.
        
        Args:
            attention_weights: Attention weights of shape (batch_size, sequence_length, features)
                              or (sequence_length, features) for single sample
            feature_names: Optional list of feature names for y-axis labels
            sequence_labels: Optional list of sequence labels for x-axis labels
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        # Handle batch dimension
        if attention_weights.ndim == 3:
            # Use first sample in batch
            attention_weights = attention_weights[0]
        
        # Transpose for proper orientation (features as rows, sequence as columns)
        attention_weights = attention_weights.T
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=sequence_labels or True,
            yticklabels=feature_names or True,
            cmap="viridis",
            cbar=True,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(title)
        plt.xlabel("Sequence Steps")
        plt.ylabel("Features")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_lstm_attention(
        self,
        attention_weights: np.ndarray,
        sequence_labels: Optional[List[str]] = None,
        title: str = "LSTM Attention Weights",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create heatmap visualization of LSTM attention weights.
        
        Args:
            attention_weights: Attention weights of shape (batch_size, sequence_length)
                              or (sequence_length,) for single sample
            sequence_labels: Optional list of sequence labels for x-axis labels
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        # Handle batch dimension
        if attention_weights.ndim == 2:
            # Use first sample in batch
            attention_weights = attention_weights[0]
        
        # Reshape for heatmap (1 row, sequence_length columns)
        attention_weights = attention_weights.reshape(1, -1)
        
        # Create heatmap
        plt.figure(figsize=(12, 3))
        sns.heatmap(
            attention_weights,
            xticklabels=sequence_labels or True,
            yticklabels=["Attention"],
            cmap="viridis",
            cbar=True,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(title)
        plt.xlabel("Sequence Steps")
        plt.ylabel("")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_cross_attention(
        self,
        cross_attention_weights: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = "Cross-Attention Weights",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create heatmap visualization of cross-attention weights between CNN and LSTM.
        
        Args:
            cross_attention_weights: Cross-attention weights of shape 
                                   (batch_size, cnn_features, lstm_features)
                                   or (cnn_features, lstm_features) for single sample
            row_labels: Optional list of row labels (CNN features)
            col_labels: Optional list of column labels (LSTM features)
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        # Handle batch dimension
        if cross_attention_weights.ndim == 3:
            # Use first sample in batch
            cross_attention_weights = cross_attention_weights[0]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cross_attention_weights,
            xticklabels=col_labels or True,
            yticklabels=row_labels or True,
            cmap="viridis",
            cbar=True,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(title)
        plt.xlabel("LSTM Features")
        plt.ylabel("CNN Features")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def create_attention_visualizer() -> AttentionVisualizer:
    """
    Factory function to create attention visualizer.
    
    Returns:
        Configured AttentionVisualizer instance
    """
    return AttentionVisualizer()