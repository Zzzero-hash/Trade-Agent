"""Uncertainty Calibration for CNN+LSTM Hybrid Model

This module provides uncertainty calibration and confidence score validation
for the CNN+LSTM hybrid model to ensure reliable decision making.

Requirements: 
12.6 - Uncertainty calibration and confidence score validation
"""

import numpy as np
import torch
import warnings
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

from .hybrid_model import CNNLSTMHybridModel


class UncertaintyCalibrator:
    """Uncertainty calibration for CNN+LSTM hybrid model."""
    
    def __init__(self, model: CNNLSTMHybridModel):
        """
        Initialize uncertainty calibrator.
        
        Args:
            model: Trained CNN+LSTM hybrid model
        """
        self.model = model
        self.device = model.device
        self.is_trained = model.is_trained
        
        # Validate that model is trained
        if not self.is_trained:
            raise ValueError("Model must be trained before creating uncertainty calibrator")
        
        # Calibration models
        self.platt_scaler_cls = None
        self.platt_scaler_reg = None
        self.isotonic_regressor_cls = None
        self.isotonic_regressor_reg = None
        
        # Calibration status
        self.is_calibrated = False
        self.calibration_method = None
    
    def calibrate_uncertainty_platt(
        self,
        X_val: np.ndarray,
        y_class_val: np.ndarray,
        y_reg_val: np.ndarray
    ) -> Tuple[LogisticRegression, LogisticRegression]:
        """
        Calibrate uncertainty using Platt scaling.
        
        Args:
            X_val: Validation data of shape (samples, channels, sequence_length)
            y_class_val: Classification validation targets of shape (samples,)
            y_reg_val: Regression validation targets of shape (samples, targets)
            
        Returns:
            Tuple of (classification_scaler, regression_scaler)
        """
        # Get model predictions with uncertainty
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.predict(
                X_val, 
                return_uncertainty=True, 
                use_ensemble=True
            )
        
        # Extract classification predictions and uncertainties
        y_pred_class_proba = predictions['classification_probs']  # (samples, classes)
        y_pred_class = np.argmax(y_pred_class_proba, axis=1)  # (samples,)
        
        # For binary classification, use probability of positive class
        # For multiclass, we'll calibrate each class separately (simplified approach)
        if y_pred_class_proba.shape[1] == 2:
            # Binary classification
            y_pred_class_binary = y_pred_class_proba[:, 1]  # Probability of positive class
            
            # Fit Platt scaler for classification
            self.platt_scaler_cls = LogisticRegression()
            self.platt_scaler_cls.fit(y_pred_class_binary.reshape(-1, 1), y_class_val)
        else:
            # Multiclass - simplified approach: calibrate using predicted class probabilities
            y_pred_class_max_proba = np.max(y_pred_class_proba, axis=1)  # Max probability
            
            # Fit Platt scaler for classification
            self.platt_scaler_cls = LogisticRegression()
            self.platt_scaler_cls.fit(y_pred_class_max_proba.reshape(-1, 1), 
                                    (y_pred_class == y_class_val).astype(int))
        
        # Extract regression predictions and uncertainties
        y_pred_reg = predictions['regression_pred']  # (samples, targets)
        y_pred_reg_uncertainty = predictions['regression_uncertainty']  # (samples, targets)
        
        # For regression, we'll calibrate using the predicted values vs actual values
        # This is a simplified approach - in practice, you might want to calibrate uncertainty separately
        if y_pred_reg.ndim == 1:
            y_pred_reg = y_pred_reg.reshape(-1, 1)
            y_reg_val = y_reg_val.reshape(-1, 1)
            y_pred_reg_uncertainty = y_pred_reg_uncertainty.reshape(-1, 1)
        
        # Fit Platt scaler for regression (one for each target)
        self.platt_scaler_reg = []
        for target_idx in range(y_pred_reg.shape[1]):
            # Simple approach: calibrate based on absolute error
            abs_errors = np.abs(y_pred_reg[:, target_idx] - y_reg_val[:, target_idx])
            
            # Fit scaler to map predicted uncertainty to actual errors
            scaler = LogisticRegression()
            scaler.fit(y_pred_reg_uncertainty[:, target_idx].reshape(-1, 1), 
                      (abs_errors < y_pred_reg_uncertainty[:, target_idx]).astype(int))
            self.platt_scaler_reg.append(scaler)
        
        self.is_calibrated = True
        self.calibration_method = "platt"
        
        return self.platt_scaler_cls, self.platt_scaler_reg
    
    def calibrate_uncertainty_isotonic(
        self,
        X_val: np.ndarray,
        y_class_val: np.ndarray,
        y_reg_val: np.ndarray
    ) -> Tuple[IsotonicRegression, List[IsotonicRegression]]:
        """
        Calibrate uncertainty using isotonic regression.
        
        Args:
            X_val: Validation data of shape (samples, channels, sequence_length)
            y_class_val: Classification validation targets of shape (samples,)
            y_reg_val: Regression validation targets of shape (samples, targets)
            
        Returns:
            Tuple of (classification_regressor, regression_regressors)
        """
        # Get model predictions with uncertainty
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.predict(
                X_val, 
                return_uncertainty=True, 
                use_ensemble=True
            )
        
        # Extract classification predictions and uncertainties
        y_pred_class_proba = predictions['classification_probs']  # (samples, classes)
        y_pred_class = np.argmax(y_pred_class_proba, axis=1)  # (samples,)
        
        # For binary classification, use probability of positive class
        if y_pred_class_proba.shape[1] == 2:
            # Binary classification
            y_pred_class_binary = y_pred_class_proba[:, 1]  # Probability of positive class
            
            # Fit isotonic regressor for classification
            self.isotonic_regressor_cls = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor_cls.fit(y_pred_class_binary, y_class_val)
        else:
            # Multiclass - simplified approach: calibrate using predicted class probabilities
            y_pred_class_max_proba = np.max(y_pred_class_proba, axis=1)  # Max probability
            y_correct = (y_pred_class == y_class_val).astype(int)  # Correct predictions
            
            # Fit isotonic regressor for classification
            self.isotonic_regressor_cls = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor_cls.fit(y_pred_class_max_proba, y_correct)
        
        # Extract regression predictions and uncertainties
        y_pred_reg = predictions['regression_pred']  # (samples, targets)
        y_pred_reg_uncertainty = predictions['regression_uncertainty']  # (samples, targets)
        
        # For regression, calibrate using the predicted values vs actual values
        if y_pred_reg.ndim == 1:
            y_pred_reg = y_pred_reg.reshape(-1, 1)
            y_reg_val = y_reg_val.reshape(-1, 1)
            y_pred_reg_uncertainty = y_pred_reg_uncertainty.reshape(-1, 1)
        
        # Fit isotonic regressor for regression (one for each target)
        self.isotonic_regressor_reg = []
        for target_idx in range(y_pred_reg.shape[1]):
            # Simple approach: calibrate based on absolute error
            abs_errors = np.abs(y_pred_reg[:, target_idx] - y_reg_val[:, target_idx])
            y_within_uncertainty = (abs_errors < y_pred_reg_uncertainty[:, target_idx]).astype(int)
            
            # Fit regressor to map predicted uncertainty to actual coverage
            regressor = IsotonicRegression(out_of_bounds='clip')
            regressor.fit(y_pred_reg_uncertainty[:, target_idx], y_within_uncertainty)
            self.isotonic_regressor_reg.append(regressor)
        
        self.is_calibrated = True
        self.calibration_method = "isotonic"
        
        return self.isotonic_regressor_cls, self.isotonic_regressor_reg
    
    def apply_calibration(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply calibration to model predictions.
        
        Args:
            predictions: Model predictions dictionary
            
        Returns:
            Calibrated predictions dictionary
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before applying calibration")
        
        calibrated_predictions = predictions.copy()
        
        # Apply classification calibration
        if 'classification_probs' in predictions:
            y_pred_class_proba = predictions['classification_probs']
            
            if self.calibration_method == "platt" and self.platt_scaler_cls is not None:
                if y_pred_class_proba.shape[1] == 2:
                    # Binary classification
                    y_pred_calibrated = y_pred_class_proba[:, 1]  # Probability of positive class
                    y_pred_calibrated = self.platt_scaler_cls.predict_proba(
                        y_pred_calibrated.reshape(-1, 1)
                    )[:, 1]
                    calibrated_predictions['classification_probs'][:, 1] = y_pred_calibrated
                    calibrated_predictions['classification_probs'][:, 0] = 1 - y_pred_calibrated
                else:
                    # Multiclass - simplified approach
                    y_pred_max_proba = np.max(y_pred_class_proba, axis=1)
                    y_pred_calibrated = self.platt_scaler_cls.predict_proba(
                        y_pred_max_proba.reshape(-1, 1)
                    )[:, 1]
                    # Apply calibration factor to all classes proportionally
                    calibration_factor = y_pred_calibrated / (y_pred_max_proba + 1e-8)
                    calibrated_predictions['classification_probs'] = (
                        y_pred_class_proba * calibration_factor.reshape(-1, 1)
                    )
                    # Renormalize
                    calibrated_predictions['classification_probs'] = (
                        calibrated_predictions['classification_probs'] / 
                        np.sum(calibrated_predictions['classification_probs'], axis=1, keepdims=True)
                    )
            
            elif self.calibration_method == "isotonic" and self.isotonic_regressor_cls is not None:
                if y_pred_class_proba.shape[1] == 2:
                    # Binary classification
                    y_pred_calibrated = y_pred_class_proba[:, 1]  # Probability of positive class
                    y_pred_calibrated = self.isotonic_regressor_cls.predict(y_pred_calibrated)
                    calibrated_predictions['classification_probs'][:, 1] = y_pred_calibrated
                    calibrated_predictions['classification_probs'][:, 0] = 1 - y_pred_calibrated
                else:
                    # Multiclass - simplified approach
                    y_pred_max_proba = np.max(y_pred_class_proba, axis=1)
                    y_pred_calibrated = self.isotonic_regressor_cls.predict(y_pred_max_proba)
                    # Apply calibration factor to all classes proportionally
                    calibration_factor = y_pred_calibrated / (y_pred_max_proba + 1e-8)
                    calibrated_predictions['classification_probs'] = (
                        y_pred_class_proba * calibration_factor.reshape(-1, 1)
                    )
                    # Renormalize
                    calibrated_predictions['classification_probs'] = (
                        calibrated_predictions['classification_probs'] / 
                        np.sum(calibrated_predictions['classification_probs'], axis=1, keepdims=True)
                    )
        
        # Apply regression calibration
        if 'regression_uncertainty' in predictions and self.calibration_method == "platt":
            y_pred_reg_uncertainty = predictions['regression_uncertainty']
            
            if y_pred_reg_uncertainty.ndim == 1:
                y_pred_reg_uncertainty = y_pred_reg_uncertainty.reshape(-1, 1)
            
            calibrated_uncertainty = np.zeros_like(y_pred_reg_uncertainty)
            for target_idx, scaler in enumerate(self.platt_scaler_reg):
                if scaler is not None:
                    calibrated_uncertainty[:, target_idx] = scaler.predict(
                        y_pred_reg_uncertainty[:, target_idx].reshape(-1, 1)
                    ).ravel()
            
            calibrated_predictions['regression_uncertainty'] = calibrated_uncertainty
        
        elif 'regression_uncertainty' in predictions and self.calibration_method == "isotonic":
            y_pred_reg_uncertainty = predictions['regression_uncertainty']
            
            if y_pred_reg_uncertainty.ndim == 1:
                y_pred_reg_uncertainty = y_pred_reg_uncertainty.reshape(-1, 1)
            
            calibrated_uncertainty = np.zeros_like(y_pred_reg_uncertainty)
            for target_idx, regressor in enumerate(self.isotonic_regressor_reg):
                if regressor is not None:
                    calibrated_uncertainty[:, target_idx] = regressor.predict(
                        y_pred_reg_uncertainty[:, target_idx]
                    )
            
            calibrated_predictions['regression_uncertainty'] = calibrated_uncertainty
        
        return calibrated_predictions
    
    def validate_calibration(
        self,
        X_test: np.ndarray,
        y_class_test: np.ndarray,
        y_reg_test: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Validate calibration quality using various metrics.
        
        Args:
            X_test: Test data of shape (samples, channels, sequence_length)
            y_class_test: Classification test targets of shape (samples,)
            y_reg_test: Regression test targets of shape (samples, targets)
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration quality metrics
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before validating calibration")
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.predict(
                X_test, 
                return_uncertainty=True, 
                use_ensemble=True
            )
        
        # Apply calibration
        calibrated_predictions = self.apply_calibration(predictions)
        
        metrics = {}
        
        # Classification calibration metrics
        if 'classification_probs' in predictions:
            y_pred_class_proba = predictions['classification_probs']
            y_pred_class_calibrated = calibrated_predictions['classification_probs']
            
            # For binary classification
            if y_pred_class_proba.shape[1] == 2:
                y_pred_binary = y_pred_class_proba[:, 1]
                y_pred_calibrated_binary = y_pred_class_calibrated[:, 1]
                
                # Brier score (lower is better)
                metrics['brier_score_uncalibrated'] = brier_score_loss(y_class_test, y_pred_binary)
                metrics['brier_score_calibrated'] = brier_score_loss(y_class_test, y_pred_calibrated_binary)
                
                # Log loss (lower is better)
                metrics['log_loss_uncalibrated'] = log_loss(y_class_test, y_pred_binary)
                metrics['log_loss_calibrated'] = log_loss(y_class_test, y_pred_calibrated_binary)
                
                # Expected calibration error (lower is better)
                fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
                    y_class_test, y_pred_binary, n_bins=n_bins
                )
                ece_uncal = np.sum(
                    np.abs(fraction_of_positives_uncal - mean_predicted_value_uncal) * 
                    np.histogram(y_pred_binary, bins=n_bins)[0] / len(y_pred_binary)
                )
                metrics['ece_uncalibrated'] = ece_uncal
                
                fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                    y_class_test, y_pred_calibrated_binary, n_bins=n_bins
                )
                ece_cal = np.sum(
                    np.abs(fraction_of_positives_cal - mean_predicted_value_cal) * 
                    np.histogram(y_pred_calibrated_binary, bins=n_bins)[0] / len(y_pred_calibrated_binary)
                )
                metrics['ece_calibrated'] = ece_cal
                
                # Improvement
                metrics['brier_score_improvement'] = (
                    metrics['brier_score_uncalibrated'] - metrics['brier_score_calibrated']
                )
                metrics['ece_improvement'] = (
                    metrics['ece_uncalibrated'] - metrics['ece_calibrated']
                )
            else:
                # Multiclass - simplified metrics
                y_pred_class = np.argmax(y_pred_class_proba, axis=1)
                y_pred_calibrated_class = np.argmax(y_pred_class_calibrated, axis=1)
                
                # Accuracy
                metrics['accuracy_uncalibrated'] = np.mean(y_pred_class == y_class_test)
                metrics['accuracy_calibrated'] = np.mean(y_pred_calibrated_class == y_class_test)
                
                # Brier score (multiclass)
                metrics['brier_score_uncalibrated'] = np.mean(
                    np.sum((y_pred_class_proba - np.eye(y_pred_class_proba.shape[1])[y_class_test]) ** 2, axis=1)
                )
                metrics['brier_score_calibrated'] = np.mean(
                    np.sum((y_pred_class_calibrated - np.eye(y_pred_class_calibrated.shape[1])[y_class_test]) ** 2, axis=1)
                )
                
                metrics['brier_score_improvement'] = (
                    metrics['brier_score_uncalibrated'] - metrics['brier_score_calibrated']
                )
        
        # Regression calibration metrics
        if 'regression_pred' in predictions and 'regression_uncertainty' in predictions:
            y_pred_reg = predictions['regression_pred']
            y_pred_reg_calibrated = calibrated_predictions['regression_pred']
            y_pred_reg_uncertainty = predictions['regression_uncertainty']
            y_pred_reg_uncertainty_calibrated = calibrated_predictions['regression_uncertainty']
            
            # Prediction interval coverage (should be close to nominal coverage)
            if y_pred_reg.ndim == 1:
                y_pred_reg = y_pred_reg.reshape(-1, 1)
                y_reg_test = y_reg_test.reshape(-1, 1)
                y_pred_reg_uncertainty = y_pred_reg_uncertainty.reshape(-1, 1)
                y_pred_reg_uncertainty_calibrated = y_pred_reg_uncertainty_calibrated.reshape(-1, 1)
            
            # Coverage at 95% confidence (assuming 2*uncertainty covers 95% of cases)
            coverage_uncal = np.mean(
                np.abs(y_pred_reg - y_reg_test) <= 2 * y_pred_reg_uncertainty
            )
            coverage_cal = np.mean(
                np.abs(y_pred_reg_calibrated - y_reg_test) <= 2 * y_pred_reg_uncertainty_calibrated
            )
            
            metrics['coverage_95_uncalibrated'] = coverage_uncal
            metrics['coverage_95_calibrated'] = coverage_cal
            metrics['coverage_improvement'] = coverage_cal - coverage_uncal
            
            # Sharpness (mean prediction interval width, lower is better)
            sharpness_uncal = np.mean(4 * y_pred_reg_uncertainty)  # 4 * std for 95% interval
            sharpness_cal = np.mean(4 * y_pred_reg_uncertainty_calibrated)
            
            metrics['sharpness_uncalibrated'] = sharpness_uncal
            metrics['sharpness_calibrated'] = sharpness_cal
            metrics['sharpness_improvement'] = sharpness_cal - sharpness_uncal
        
        return metrics
    
    def visualize_reliability_diagram(
        self,
        X_test: np.ndarray,
        y_class_test: np.ndarray,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create reliability diagram visualization.
        
        Args:
            X_test: Test data of shape (samples, channels, sequence_length)
            y_class_test: Classification test targets of shape (samples,)
            n_bins: Number of bins for calibration curve
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before visualizing reliability")
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.predict(
                X_test, 
                return_uncertainty=True, 
                use_ensemble=True
            )
        
        # Apply calibration
        calibrated_predictions = self.apply_calibration(predictions)
        
        # Extract classification predictions
        if 'classification_probs' in predictions:
            y_pred_class_proba = predictions['classification_probs']
            y_pred_class_calibrated = calibrated_predictions['classification_probs']
            
            # For binary classification
            if y_pred_class_proba.shape[1] == 2:
                y_pred_binary = y_pred_class_proba[:, 1]
                y_pred_calibrated_binary = y_pred_class_calibrated[:, 1]
                
                # Compute calibration curves
                fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
                    y_class_test, y_pred_binary, n_bins=n_bins
                )
                fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                    y_class_test, y_pred_calibrated_binary, n_bins=n_bins
                )
                
                # Create reliability diagram
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                plt.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, "s-", 
                        label="Uncalibrated")
                plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", 
                        label="Calibrated")
                
                plt.ylabel("Fraction of positives")
                plt.xlabel("Mean predicted probability")
                plt.title(title)
                plt.legend()
                plt.grid()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
            else:
                warnings.warn("Reliability diagram only implemented for binary classification")


def create_uncertainty_calibrator(model: CNNLSTMHybridModel) -> UncertaintyCalibrator:
    """
    Factory function to create uncertainty calibrator.
    
    Args:
        model: Trained CNN+LSTM hybrid model
        
    Returns:
        Configured UncertaintyCalibrator instance
    """
    return UncertaintyCalibrator(model)