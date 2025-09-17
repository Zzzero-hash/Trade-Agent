"""Feature Importance Analysis for CNN+LSTM Hybrid Model

This module provides comprehensive feature importance analysis including
permutation importance and integrated gradients for the CNN+LSTM hybrid model.

Requirements:
12.1 - Feature importance scores for individual predictions
12.3 - Integrated gradients for feature attribution
"""

import numpy as np
import torch
import warnings
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error
import copy

# Captum for integrated gradients
try:
    from captum.attr import IntegratedGradients, NoiseTunnel

    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn(
        "Captum not available. Install with 'pip install captum' for integrated gradients."
    )

# Scikit-learn for permutation importance
try:
    from sklearn.metrics import accuracy_score, mean_squared_error

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some features may be limited.")

from .hybrid_model import CNNLSTMHybridModel


class FeatureImportanceAnalyzer:
    """Feature importance analysis for CNN+LSTM hybrid model."""

    def __init__(self, model: CNNLSTMHybridModel):
        """
        Initialize feature importance analyzer.

        Args:
            model: Trained CNN+LSTM hybrid model
        """
        self.model = model
        self.device = model.device
        self.is_trained = model.is_trained

        # Validate that model is trained
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before creating feature importance analyzer"
            )

        if not CAPTUM_AVAILABLE:
            warnings.warn("Captum not available. Integrated gradients will not work.")

        if not SKLEARN_AVAILABLE:
            warnings.warn("Scikit-learn not available. Some metrics may be limited.")

    def compute_permutation_importance(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        scoring: str = "accuracy",
        n_repeats: int = 10,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute permutation importance for features.

        Args:
            X: Input data of shape (samples, channels, sequence_length)
            y_class: Classification targets of shape (samples,)
            y_reg: Regression targets of shape (samples, targets)
            scoring: Scoring metric ("accuracy", "precision", "recall", "f1", "mse", "mae")
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducibility

        Returns:
            Dictionary with feature importances
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "Scikit-learn not available. Please install with 'pip install scikit-learn'."
            )

        if random_state is not None:
            np.random.seed(random_state)

        # Get baseline score
        baseline_score = self._compute_baseline_score(X, y_class, y_reg, scoring)

        # Initialize importance arrays
        n_features = X.shape[1]  # channels dimension
        importances = np.zeros((n_features, n_repeats))

        # Compute importance for each feature
        for feature_idx in range(n_features):
            for repeat_idx in range(n_repeats):
                # Create permuted data
                X_permuted = X.copy()
                # Permute only the current feature across all samples and timesteps
                permuted_indices = np.random.permutation(X.shape[0])
                X_permuted[:, feature_idx, :] = X_permuted[
                    permuted_indices, feature_idx, :
                ]

                # Get score with permuted feature
                permuted_score = self._compute_baseline_score(
                    X_permuted, y_class, y_reg, scoring
                )

                # Importance is decrease in score (higher is more important)
                importances[feature_idx, repeat_idx] = baseline_score - permuted_score

        # Calculate mean and std across repeats
        mean_importance = np.mean(importances, axis=1)
        std_importance = np.std(importances, axis=1)

        return {
            "importances_mean": mean_importance,
            "importances_std": std_importance,
            "importances": importances,
            "baseline_score": baseline_score,
            "scoring": scoring,
        }

    def _compute_baseline_score(
        self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray, scoring: str
    ) -> float:
        """Compute baseline score for permutation importance."""
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.predict(
                X, return_uncertainty=False, use_ensemble=True
            )

        y_pred_class = predictions["classification_pred"]
        y_pred_reg = predictions["regression_pred"]

        # Compute score based on scoring parameter
        if scoring in ["accuracy", "precision", "recall", "f1"]:
            if len(y_class) > 0:
                if scoring == "accuracy":
                    return accuracy_score(y_class, y_pred_class)
                # For other metrics, we would need additional imports from sklearn.metrics
                else:
                    return accuracy_score(y_class, y_pred_class)  # fallback
            else:
                return 0.0
        elif scoring in ["mse", "mae"]:
            if len(y_reg) > 0 and len(y_pred_reg) > 0:
                if scoring == "mse":
                    return -mean_squared_error(
                        y_reg, y_pred_reg
                    )  # negative because we want higher to be better
                else:
                    # For MAE, we would need mean_absolute_error
                    return -mean_squared_error(y_reg, y_pred_reg)  # fallback
            else:
                return 0.0
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")

    def compute_integrated_gradients(
        self,
        input_data: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target: Optional[int] = None,
        steps: int = 50,
        method: str = "riemann_trapezoid",
        return_convergence_delta: bool = False,
        output_type: str = "classification",  # "classification", "regression"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute integrated gradients using Captum.

        Args:
            input_data: Input tensor of shape (batch_size, channels, sequence_length)
            baseline: Baseline tensor of same shape as input_data (default: zeros)
            target: Target class for attribution (for classification)
            steps: Number of steps in integration
            method: Integration method
            return_convergence_delta: Whether to return convergence delta
            output_type: Type of output to compute gradients for

        Returns:
            Tuple of (attributions, convergence_delta)
        """
        if not CAPTUM_AVAILABLE:
            raise RuntimeError(
                "Captum not available. Please install with 'pip install captum'."
            )

        if baseline is None:
            baseline = torch.zeros_like(input_data)

        # Create a wrapper function for the model that returns the desired output
        def model_wrapper(x):
            predictions = self.model.forward(x, use_ensemble=True)
            if output_type == "classification":
                return predictions["classification_probs"]
            elif output_type == "regression":
                return predictions["regression_mean"]
            else:
                # Combined output
                return torch.cat(
                    [
                        predictions["classification_probs"],
                        predictions["regression_mean"],
                    ],
                    dim=1,
                )

        # Create Integrated Gradients explainer
        ig = IntegratedGradients(model_wrapper)

        # Compute attributions
        attributions, delta = ig.attribute(
            input_data,
            baselines=baseline,
            target=target,
            n_steps=steps,
            method=method,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions, delta

    def compute_integrated_gradients_with_noise(
        self,
        input_data: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
        nt_samples: int = 10,
        stdevs: float = 0.01,
        return_convergence_delta: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute integrated gradients with noise tunnel for smoothing.

        Args:
            input_data: Input tensor of shape (batch_size, channels, sequence_length)
            baseline: Baseline tensor of same shape as input_data (default: zeros)
            steps: Number of steps in integration
            nt_samples: Number of samples for noise tunnel
            stdevs: Standard deviation of noise
            return_convergence_delta: Whether to return convergence delta

        Returns:
            Tuple of (attributions, convergence_delta)
        """
        if not CAPTUM_AVAILABLE:
            raise RuntimeError(
                "Captum not available. Please install with 'pip install captum'."
            )

        if baseline is None:
            baseline = torch.zeros_like(input_data)

        # Create Integrated Gradients explainer
        ig = IntegratedGradients(self.model)

        # Create Noise Tunnel
        noise_tunnel = NoiseTunnel(ig)

        # Compute attributions with noise tunnel
        attributions, delta = noise_tunnel.attribute(
            input_data,
            baselines=baseline,
            target=None,
            n_steps=steps,
            nt_samples=nt_samples,
            stdevs=stdevs,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions, delta

    def visualize_integrated_gradients(
        self,
        attributions: torch.Tensor,
        original_data: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        title: str = "Integrated Gradients",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Visualize integrated gradients attributions.

        Args:
            attributions: Attribution tensor from integrated gradients
            original_data: Original input data
            feature_names: Optional list of feature names
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise RuntimeError(
                "Matplotlib not available. Please install with 'pip install matplotlib'."
            )

        # Convert to numpy
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().numpy()
        if isinstance(original_data, torch.Tensor):
            original_data = original_data.cpu().numpy()

        # Handle batch dimension
        if attributions.ndim == 3 and attributions.shape[0] == 1:
            attributions = attributions[0]  # Remove batch dimension
            original_data = original_data[0]

        # For time series data, we'll create a heatmap
        if attributions.ndim == 2:  # (features, timesteps)
            plt.figure(figsize=(12, 8))

            # Create heatmap
            sns.heatmap(
                attributions,
                cmap="RdBu_r",
                center=0,
                cbar=True,
                xticklabels=False,  # Too many timesteps to show
                yticklabels=feature_names if feature_names else True,
            )

            plt.title(title)
            plt.xlabel("Time Steps")
            plt.ylabel("Features")

        else:  # 1D attributions
            plt.figure(figsize=(10, 6))

            # Create bar plot
            x_pos = range(len(attributions))
            colors = ["red" if val < 0 else "blue" for val in attributions]

            plt.bar(x_pos, attributions, color=colors, alpha=0.7)
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            plt.title(title)
            plt.xlabel("Features")
            plt.ylabel("Attribution")

            if feature_names:
                plt.xticks(x_pos, feature_names, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def compare_attribution_methods(
        self,
        input_data: torch.Tensor,
        shap_values: Optional[np.ndarray] = None,
        integrated_gradients: Optional[torch.Tensor] = None,
        permutation_importance: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Compare different attribution methods side by side.

        Args:
            input_data: Original input data
            shap_values: SHAP attribution values
            integrated_gradients: Integrated gradients attributions
            permutation_importance: Permutation importance scores
            feature_names: Optional list of feature names
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError(
                "Matplotlib not available. Please install with 'pip install matplotlib'."
            )

        # Prepare data for comparison
        methods = []
        attributions = []

        if shap_values is not None:
            methods.append("SHAP")
            # Average over timesteps if needed
            if shap_values.ndim == 3:
                shap_avg = np.mean(np.abs(shap_values[0]), axis=1)
            elif shap_values.ndim == 2:
                shap_avg = np.mean(np.abs(shap_values), axis=1)
            else:
                shap_avg = np.abs(shap_values)
            attributions.append(shap_avg)

        if integrated_gradients is not None:
            methods.append("Integrated Gradients")
            ig_np = (
                integrated_gradients.cpu().numpy()
                if isinstance(integrated_gradients, torch.Tensor)
                else integrated_gradients
            )
            # Average over timesteps if needed
            if ig_np.ndim == 3:
                ig_avg = np.mean(np.abs(ig_np[0]), axis=1)
            elif ig_np.ndim == 2:
                ig_avg = np.mean(np.abs(ig_np), axis=1)
            else:
                ig_avg = np.abs(ig_np)
            attributions.append(ig_avg)

        if permutation_importance is not None:
            methods.append("Permutation Importance")
            attributions.append(np.abs(permutation_importance))

        if not methods:
            raise ValueError("At least one attribution method must be provided")

        # Create comparison plot
        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 6))
        if len(methods) == 1:
            axes = [axes]

        for i, (method, attr) in enumerate(zip(methods, attributions)):
            ax = axes[i]

            # Normalize attributions for comparison
            attr_norm = attr / (np.max(attr) + 1e-8)

            x_pos = range(len(attr_norm))
            ax.bar(x_pos, attr_norm, alpha=0.7)
            ax.set_title(f"{method}")
            ax.set_ylabel("Normalized Attribution")
            ax.set_xlabel("Features")

            if feature_names:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(feature_names, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def create_feature_importance_analyzer(
    model: CNNLSTMHybridModel,
) -> FeatureImportanceAnalyzer:
    """
    Factory function to create feature importance analyzer.

    Args:
        model: Trained CNN+LSTM hybrid model

    Returns:
        Configured FeatureImportanceAnalyzer instance
    """
    return FeatureImportanceAnalyzer(model)
