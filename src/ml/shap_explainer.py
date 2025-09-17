"""SHAP Explainer for CNN+LSTM Hybrid Model

This module implements SHAP (SHapley Additive exPlanations) integration for
local model interpretability. It provides feature importance scores for
individual predictions and integrates with the existing CNN+LSTM hybrid model.

Requirements: 12.4 - WHEN model explanations are requested THEN the system SHALL
use attention mechanisms to highlight influential data points.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import warnings

# SHAP library imports
try:
    import shap
    from shap import KernelExplainer, DeepExplainer, Explanation
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Install with 'pip install shap' for full functionality.")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with 'pip install matplotlib' for visualization.")

# Captum for integrated gradients
try:
    from captum.attr import IntegratedGradients, NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. Install with 'pip install captum' for integrated gradients.")

from .hybrid_model import CNNLSTMHybridModel


class SHAPCache:
    """Simple cache for SHAP computations to improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}  # type: Dict[str, Dict[str, Any]]
        self.access_times = {}  # type: Dict[str, datetime]
    
    def _hash_input(self, data: np.ndarray) -> str:
        """Create a hash of the input data for caching."""
        # Convert to bytes and hash
        data_bytes = data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Retrieve cached SHAP values for input data."""
        key = self._hash_input(data)
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def put(self, data: np.ndarray, shap_values: Dict[str, Any]) -> None:
        """Store SHAP values in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            # Find the key with the minimum access time
            oldest_key = None
            oldest_time = None
            for key, access_time in self.access_times.items():
                if oldest_time is None or access_time < oldest_time:
                    oldest_time = access_time
                    oldest_key = key
            
            if oldest_key is not None:
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
        
        key = self._hash_input(data)
        self.cache[key] = shap_values
        self.access_times[key] = datetime.now()


class SHAPExplainer:
    """SHAP explainer for CNN+LSTM hybrid model."""
    
    def __init__(self, model: CNNLSTMHybridModel, cache_size: int = 100):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained CNN+LSTM hybrid model
            cache_size: Maximum number of SHAP computations to cache
        """
        self.model = model
        self.device = model.device
        self.cache = SHAPCache(max_size=cache_size)
        self.is_trained = model.is_trained
        
        # Validate that model is trained
        if not self.is_trained:
            raise ValueError("Model must be trained before creating SHAP explainer")
    
    def _model_predict_fn(self, data: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP explainer.
        
        Args:
            data: Input data of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Model predictions of shape (batch_size, num_outputs)
        """
        # Convert to tensor and move to device
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.forward(data_tensor, use_ensemble=True)
            
            # Combine classification and regression outputs
            # For classification, use probabilities
            cls_probs = predictions['classification_probs']  # (batch_size, num_classes)
            # For regression, use mean predictions
            reg_preds = predictions['regression_mean']  # (batch_size, targets)
            
            # Concatenate outputs
            combined_output = torch.cat([cls_probs, reg_preds], dim=1)
            return combined_output.cpu().numpy()
    
    def _model_predict_classification_fn(self, data: np.ndarray) -> np.ndarray:
        """
        Classification-only prediction function for SHAP explainer.
        
        Args:
            data: Input data of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        # Convert to tensor and move to device
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.forward(data_tensor, use_ensemble=True)
            
            # Return only classification probabilities
            cls_probs = predictions['classification_probs']  # (batch_size, num_classes)
            return cls_probs.cpu().numpy()
    
    def _model_predict_regression_fn(self, data: np.ndarray) -> np.ndarray:
        """
        Regression-only prediction function for SHAP explainer.
        
        Args:
            data: Input data of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Regression predictions of shape (batch_size, targets)
        """
        # Convert to tensor and move to device
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.forward(data_tensor, use_ensemble=True)
            
            # Return only regression predictions
            reg_preds = predictions['regression_mean']  # (batch_size, targets)
            return reg_preds.cpu().numpy()
    
    def compute_shap_values(
        self, 
        background_data: np.ndarray, 
        test_data: np.ndarray,
        explainer_type: str = "kernel",
        output_type: str = "combined",  # "combined", "classification", "regression"
        cache_result: bool = True,
        max_evals: int = 1000
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for test data using background data.
        
        Args:
            background_data: Background data for SHAP explainer of shape
                           (background_samples, input_channels, sequence_length)
            test_data: Test data to explain of shape
                      (test_samples, input_channels, sequence_length)
            explainer_type: Type of SHAP explainer to use ("kernel", "deep")
            output_type: Type of output to explain ("combined", "classification", "regression")
            cache_result: Whether to cache the computed SHAP values
            max_evals: Maximum number of evaluations for KernelExplainer
            
        Returns:
            Dictionary containing SHAP values and metadata
        """
        # Check cache first
        if cache_result:
            cached_result = self.cache.get(test_data)
            if cached_result is not None:
                return cached_result
        
        # Check if SHAP is available
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not available. Please install with 'pip install shap'.")
        
        # Select prediction function based on output type
        if output_type == "classification":
            predict_fn = self._model_predict_classification_fn
        elif output_type == "regression":
            predict_fn = self._model_predict_regression_fn
        else:  # combined
            predict_fn = self._model_predict_fn
        
        # Compute SHAP values based on explainer type
        if explainer_type == "kernel":
            # Use KernelExplainer for model-agnostic explanations
            explainer = KernelExplainer(predict_fn, background_data)
            shap_values = explainer.shap_values(test_data, nsamples=max_evals)
        elif explainer_type == "deep":
            # Use DeepExplainer for deep learning models (if model supports it)
            try:
                # For DeepExplainer, we need to convert data to tensors
                background_tensor = torch.FloatTensor(background_data).to(self.device)
                test_tensor = torch.FloatTensor(test_data).to(self.device)
                
                # Create a wrapper model for DeepExplainer
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model, output_type):
                        super().__init__()
                        self.model = model
                        self.output_type = output_type
                    
                    def forward(self, x):
                        predictions = self.model.forward(x, use_ensemble=True)
                        if self.output_type == "classification":
                            return predictions['classification_probs']
                        elif self.output_type == "regression":
                            return predictions['regression_mean']
                        else:  # combined
                            return torch.cat([predictions['classification_probs'], 
                                            predictions['regression_mean']], dim=1)
                
                wrapper = ModelWrapper(self.model, output_type)
                explainer = DeepExplainer(wrapper, background_tensor)
                shap_values = explainer.shap_values(test_tensor)
            except Exception as e:
                # Fallback to KernelExplainer if DeepExplainer fails
                warnings.warn(f"DeepExplainer failed: {e}. Falling back to KernelExplainer.")
                explainer = KernelExplainer(predict_fn, background_data)
                shap_values = explainer.shap_values(test_data, nsamples=max_evals)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
        
        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        elif isinstance(shap_values, list):
            shap_values = [sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv for sv in shap_values]
        
        # Get expected values for the explainer
        expected_value = None
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value
            if isinstance(expected_value, torch.Tensor):
                expected_value = expected_value.cpu().numpy()
            elif isinstance(expected_value, list):
                expected_value = [ev.cpu().numpy() if isinstance(ev, torch.Tensor) else ev for ev in expected_value]
        
        # Prepare result dictionary
        result = {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'background_data_shape': background_data.shape,
            'test_data_shape': test_data.shape,
            'explainer_type': explainer_type,
            'output_type': output_type,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'CNNLSTMHybridModel'
        }
        
        # Cache result if requested
        if cache_result:
            self.cache.put(test_data, result)
        
        return result
    
    def get_feature_importance(
        self, 
        shap_result: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract feature importance from SHAP values.
        
        Args:
            shap_result: Result from compute_shap_values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with feature importance scores
        """
        shap_values = shap_result['shap_values']
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multiple outputs (classification + regression)
            # Combine all SHAP values
            combined_shap = np.concatenate(shap_values, axis=-1) if len(shap_values) > 1 else shap_values[0]
        else:
            combined_shap = shap_values
        
        # Calculate mean absolute SHAP values across samples and timesteps
        # Shape: (input_channels,)
        feature_importance = np.mean(np.abs(combined_shap), axis=(0, 2))
        
        # If we have feature names, create a mapping
        if feature_names and len(feature_names) == len(feature_importance):
            importance_dict = dict(zip(feature_names, feature_importance))
        else:
            importance_dict = {f"feature_{i}": val for i, val in enumerate(feature_importance)}
        
        return {
            'feature_importance': feature_importance,
            'importance_by_feature': importance_dict,
            'total_importance': np.sum(feature_importance)
        }
    
    def visualize_shap_summary(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        feature_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
        plot_type: str = "dot",  # "dot", "bar", "violin"
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values from compute_shap_values
            feature_data: Original feature data for coloring (optional)
            feature_names: Optional list of feature names
            max_display: Maximum number of features to display
            plot_type: Type of summary plot ("dot", "bar", "violin")
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not available. Please install with 'pip install shap'.")
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For multi-output, visualize the first output (classification)
            shap_values_to_plot = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_values_to_plot = shap_values
        
        # Reshape SHAP values for visualization if needed
        # SHAP expects (n_samples, n_features) but we have (n_samples, n_features, n_timesteps)
        if shap_values_to_plot.ndim == 3:
            # Average over timesteps or take the last timestep
            shap_values_to_plot = np.mean(shap_values_to_plot, axis=2)
        
        # Prepare feature data for coloring
        feature_data_for_plot = None
        if feature_data is not None:
            if feature_data.ndim == 3:
                # Average over timesteps or take the last timestep
                feature_data_for_plot = np.mean(feature_data, axis=2)
            else:
                feature_data_for_plot = feature_data
        
        # Create SHAP summary plot using the SHAP library
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_to_plot, 
            features=feature_data_for_plot,
            feature_names=feature_names, 
            max_display=max_display,
            plot_type=plot_type,
            show=False  # We'll handle show/save ourselves
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_shap_force(
        self,
        shap_values: np.ndarray,
        expected_value: Union[float, np.ndarray],
        feature_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        matplotlib: bool = True,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP force plot.
        
        Args:
            shap_values: SHAP values for a single prediction
            expected_value: Expected value of the model output
            feature_values: Feature values for the prediction (optional)
            feature_names: Optional list of feature names
            matplotlib: Whether to use matplotlib backend
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not available. Please install with 'pip install shap'.")
        
        # Handle multi-dimensional SHAP values
        if shap_values.ndim > 1:
            # For time series data, average over timesteps or take last timestep
            if shap_values.ndim == 2:  # (features, timesteps)
                shap_values = np.mean(shap_values, axis=1)
            elif shap_values.ndim == 3:  # (samples, features, timesteps)
                shap_values = np.mean(shap_values[0], axis=1)  # Take first sample and average over timesteps
        
        # Handle feature values
        if feature_values is not None and feature_values.ndim > 1:
            if feature_values.ndim == 2:  # (features, timesteps)
                feature_values = np.mean(feature_values, axis=1)
            elif feature_values.ndim == 3:  # (samples, features, timesteps)
                feature_values = np.mean(feature_values[0], axis=1)  # Take first sample and average over timesteps
        
        # Handle expected value
        if isinstance(expected_value, np.ndarray):
            if expected_value.ndim > 0:
                expected_value = expected_value.item() if expected_value.size == 1 else expected_value[0]
        
        try:
            # Create SHAP force plot
            force_plot = shap.force_plot(
                expected_value,
                shap_values,
                features=feature_values,
                feature_names=feature_names,
                matplotlib=matplotlib,
                show=False  # We'll handle show/save ourselves
            )
            
            if matplotlib:
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
            else:
                # For HTML output, we can't easily save/show, so just return the plot
                if show:
                    return force_plot
                    
        except Exception as e:
            warnings.warn(f"Force plot failed: {e}. This may be due to incompatible data shapes.")
            # Fallback: create a simple bar plot
            if matplotlib:
                plt.figure(figsize=(12, 6))
                colors = ['red' if val < 0 else 'blue' for val in shap_values]
                plt.barh(range(len(shap_values)), shap_values, color=colors)
                plt.xlabel('SHAP Value')
                plt.ylabel('Features')
                if feature_names:
                    plt.yticks(range(len(feature_names)), feature_names)
                plt.title(f'SHAP Values (Expected: {expected_value:.3f})')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
    
    def visualize_shap_decision(
        self,
        shap_values: np.ndarray,
        expected_value: Union[float, np.ndarray],
        feature_values: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        highlight: Optional[int] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP decision plot.
        
        Args:
            shap_values: SHAP values for predictions
            expected_value: Expected value of the model output
            feature_values: Feature values for the predictions (optional)
            feature_names: Optional list of feature names
            highlight: Index of prediction to highlight (optional)
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available. Please install with 'pip install matplotlib'.")
        
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library not available. Please install with 'pip install shap'.")
        
        # Handle multi-dimensional SHAP values
        if shap_values.ndim == 3:
            # For time series data, average over timesteps
            shap_values = np.mean(shap_values, axis=2)
        
        # Handle feature values
        if feature_values is not None and feature_values.ndim == 3:
            feature_values = np.mean(feature_values, axis=2)
        
        try:
            # Create SHAP decision plot
            plt.figure(figsize=(10, 8))
            shap.decision_plot(
                expected_value,
                shap_values,
                features=feature_values,
                feature_names=feature_names,
                highlight=highlight,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            warnings.warn(f"Decision plot failed: {e}. This may be due to incompatible SHAP version or data shapes.")
            # Fallback: create a waterfall-style plot
            self._create_waterfall_plot(shap_values, expected_value, feature_names, show, save_path)
    
    def _create_waterfall_plot(
        self,
        shap_values: np.ndarray,
        expected_value: Union[float, np.ndarray],
        feature_names: Optional[List[str]] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Create a waterfall-style plot as fallback for decision plot."""
        if shap_values.ndim > 1:
            shap_values = shap_values[0]  # Take first sample
        
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value.item() if expected_value.size == 1 else expected_value[0]
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        sorted_shap = shap_values[sorted_indices]
        
        if feature_names:
            sorted_names = [feature_names[i] for i in sorted_indices]
        else:
            sorted_names = [f"Feature {i}" for i in sorted_indices]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative values
        cumulative = [expected_value]
        for val in sorted_shap:
            cumulative.append(cumulative[-1] + val)
        
        # Plot bars
        colors = ['red' if val < 0 else 'blue' for val in sorted_shap]
        x_pos = range(len(sorted_shap))
        
        for i, (pos, val, color) in enumerate(zip(x_pos, sorted_shap, colors)):
            plt.bar(pos, val, bottom=cumulative[i], color=color, alpha=0.7)
        
        # Add expected value line
        plt.axhline(y=expected_value, color='black', linestyle='--', alpha=0.5, label=f'Expected: {expected_value:.3f}')
        
        # Add final prediction line
        final_pred = cumulative[-1]
        plt.axhline(y=final_pred, color='green', linestyle='-', alpha=0.7, label=f'Prediction: {final_pred:.3f}')
        
        plt.xlabel('Features (sorted by |SHAP value|)')
        plt.ylabel('SHAP Value')
        plt.title('SHAP Decision Plot (Waterfall Style)')
        plt.xticks(x_pos, sorted_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_shap_values(
        self,
        shap_result: Dict[str, Any],
        test_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create visualization of SHAP values (deprecated, use specific visualization methods).
        
        Args:
            shap_result: Result from compute_shap_values
            test_data: Test data used for explanation
            feature_names: Optional list of feature names
            output_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        warnings.warn(
            "visualize_shap_values is deprecated. Use visualize_shap_summary or visualize_shap_force instead.",
            DeprecationWarning
        )
        
        # Use the new summary plot method instead
        shap_values = shap_result['shap_values']
        self.visualize_shap_summary(
            shap_values,
            feature_data=test_data,
            feature_names=feature_names,
            show=show_plot,
            save_path=output_path
        )
    
    def get_attention_weights(
        self,
        input_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract attention weights from the model to highlight influential data points.
        
        This implements requirement 12.4: WHEN model explanations are requested 
        THEN the system SHALL use attention mechanisms to highlight influential data points
        
        Args:
            input_data: Input data of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Dictionary containing attention weights from different model components
        """
        # Convert to tensor and move to device
        data_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Get model predictions with features
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.forward(data_tensor, return_features=True, use_ensemble=True)
        
        attention_info = {}
        
        # Extract CNN attention if available
        if 'cnn_attention_weights' in predictions:
            attention_info['cnn_attention'] = predictions['cnn_attention_weights'].cpu().numpy()
        
        # Extract LSTM attention if available
        if 'lstm_attention_weights' in predictions:
            attention_info['lstm_attention'] = predictions['lstm_attention_weights'].cpu().numpy()
        
        # Extract fusion attention if available
        if 'fusion_attention_weights' in predictions:
            attention_info['fusion_attention'] = predictions['fusion_attention_weights'].cpu().numpy()
        
        # Extract attention from CNN component if available
        if hasattr(self.model, 'cnn_extractor') and hasattr(self.model.cnn_extractor, 'attention'):
            try:
                # Get CNN features with attention
                cnn_features = self.model.cnn_extractor.forward(data_tensor)
                # If the CNN has attention mechanism, extract weights
                if hasattr(self.model.cnn_extractor.attention, 'attention_weights'):
                    attention_info['cnn_multihead_attention'] = self.model.cnn_extractor.attention.attention_weights.cpu().numpy()
            except Exception as e:
                warnings.warn(f"Could not extract CNN attention weights: {e}")
        
        # Extract attention from LSTM component if available
        if hasattr(self.model, 'lstm_processor') and hasattr(self.model.lstm_processor, 'attention'):
            try:
                # Get LSTM features with attention
                lstm_features, lstm_context = self.model.lstm_processor.forward_encoder_only(data_tensor)
                # If the LSTM has attention mechanism, extract weights
                if hasattr(self.model.lstm_processor.attention, 'attention_weights'):
                    attention_info['lstm_attention_weights'] = self.model.lstm_processor.attention.attention_weights.cpu().numpy()
            except Exception as e:
                warnings.warn(f"Could not extract LSTM attention weights: {e}")
        
        # If no specific attention weights, compute based on feature activations
        if not attention_info:
            # Use the fused features as a proxy for attention
            if 'fused_features' in predictions:
                fused_features = predictions['fused_features'].cpu().numpy()
                # Compute attention as normalized feature magnitudes
                attention_weights = np.abs(fused_features)
                # Normalize across feature dimension
                attention_weights = attention_weights / (np.sum(attention_weights, axis=-1, keepdims=True) + 1e-8)
                attention_info['computed_attention'] = attention_weights
            
            # Compute gradient-based attention as fallback
            try:
                attention_info.update(self._compute_gradient_attention(data_tensor))
            except Exception as e:
                warnings.warn(f"Could not compute gradient-based attention: {e}")
        
        return attention_info
    
    def _compute_gradient_attention(self, input_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based attention weights.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Dictionary with gradient-based attention weights
        """
        input_data.requires_grad_(True)
        
        # Forward pass
        predictions = self.model.forward(input_data, use_ensemble=True)
        
        # Get the maximum prediction for gradient computation
        if 'classification_probs' in predictions:
            max_class_score = torch.max(predictions['classification_probs'], dim=1)[0]
            target_score = torch.mean(max_class_score)
        elif 'regression_mean' in predictions:
            target_score = torch.mean(predictions['regression_mean'])
        else:
            return {}
        
        # Compute gradients
        target_score.backward()
        
        # Get gradients
        gradients = input_data.grad.cpu().numpy()
        
        # Compute attention as absolute gradients normalized
        gradient_attention = np.abs(gradients)
        gradient_attention = gradient_attention / (np.sum(gradient_attention, axis=(1, 2), keepdims=True) + 1e-8)
        
        return {'gradient_attention': gradient_attention}
    
    def explain_prediction(
        self,
        background_data: np.ndarray,
        test_sample: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Provide a complete explanation for a single prediction.
        
        Args:
            background_data: Background data for SHAP explainer
            test_sample: Single test sample to explain
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary containing comprehensive explanation
        """
        # Ensure test_sample has batch dimension
        if test_sample.ndim == 2:
            test_sample = np.expand_dims(test_sample, axis=0)
        
        # Compute SHAP values
        shap_result = self.compute_shap_values(background_data, test_sample)
        
        # Get feature importance
        importance = self.get_feature_importance(shap_result, feature_names)
        
        # Get attention weights
        attention = self.get_attention_weights(test_sample)
        
        # Combine all explanation components
        explanation = {
            'shap_values': shap_result,
            'feature_importance': importance,
            'attention_weights': attention,
            'explanation_timestamp': datetime.now().isoformat()
        }
        
        return explanation


def create_shap_explainer(
    model: CNNLSTMHybridModel,
    cache_size: int = 100
) -> SHAPExplainer:
    """
    Factory function to create SHAP explainer for CNN+LSTM hybrid model.
    
    Args:
        model: Trained CNN+LSTM hybrid model
        cache_size: Maximum number of SHAP computations to cache
        
    Returns:
        Configured SHAPExplainer instance
    """
    return SHAPExplainer(model, cache_size=cache_size)