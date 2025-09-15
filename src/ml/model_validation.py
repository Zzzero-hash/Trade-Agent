"""
Automated Model Validation and Selection Pipeline

This module provides comprehensive model validation, comparison, and selection
capabilities with support for cross-validation, statistical testing, and
automated model deployment decisions.
"""

import os
import json
import logging
import pickle
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, classification_report, confusion_matrix
)
import torch
import torch.nn as nn
import gymnasium as gym

# Optional imports
try:
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .base_models import ModelConfig, TrainingResult


@dataclass
class ValidationConfig:
    """Configuration for model validation"""
    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, time_series, standard
    test_size: float = 0.2
    random_state: int = 42
    
    # Metrics to compute
    classification_metrics: List[str] = None
    regression_metrics: List[str] = None
    rl_metrics: List[str] = None
    
    # Statistical testing
    significance_level: float = 0.05
    statistical_tests: List[str] = None  # t_test, wilcoxon, friedman
    
    # Performance thresholds
    min_accuracy: float = 0.6
    min_f1_score: float = 0.5
    max_mse: float = 1.0
    min_reward: float = 0.0
    
    # Validation data requirements
    min_samples: int = 1000
    class_balance_threshold: float = 0.1  # Minimum class proportion
    
    # Model comparison
    comparison_metric: str = "f1_score"  # Primary metric for comparison
    improvement_threshold: float = 0.01  # Minimum improvement to consider significant
    
    # Output settings
    save_predictions: bool = True
    save_reports: bool = True
    generate_plots: bool = True

    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
            ]
        
        if self.regression_metrics is None:
            self.regression_metrics = [
                'mse', 'mae', 'rmse', 'r2_score', 'mape'
            ]
        
        if self.rl_metrics is None:
            self.rl_metrics = [
                'mean_reward', 'std_reward', 'success_rate', 'episode_length'
            ]
        
        if self.statistical_tests is None:
            self.statistical_tests = ['t_test', 'wilcoxon']


@dataclass
class ValidationResult:
    """Results from model validation"""
    model_id: str
    model_type: str
    validation_metrics: Dict[str, float]
    cross_validation_scores: Dict[str, List[float]]
    statistical_significance: Dict[str, Dict[str, float]]
    performance_summary: Dict[str, Any]
    validation_time: float
    timestamp: datetime
    passed_thresholds: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict


class ModelValidator:
    """Comprehensive model validation system"""
    
    def __init__(
        self,
        config: ValidationConfig,
        output_dir: str = "validation_results"
    ):
        """Initialize model validator
        
        Args:
            config: Validation configuration
            output_dir: Directory for validation outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
    
    def _setup_logging(self) -> None:
        """Setup logging for validation"""
        log_file = self.output_dir / "validation.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def validate_classification_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_id: str,
        model_type: str = "classification"
    ) -> ValidationResult:
        """Validate classification model
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            model_id: Unique model identifier
            model_type: Type of model
            
        Returns:
            Validation results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Validating classification model: {model_id}")
        
        # Data quality checks
        self._validate_data_quality(X, y, "classification")
        
        # Cross-validation
        cv_scores = self._perform_cross_validation(
            model, X, y, self.config.classification_metrics, "classification"
        )
        
        # Hold-out validation
        holdout_metrics = self._holdout_validation(
            model, X, y, self.config.classification_metrics, "classification"
        )
        
        # Statistical significance testing
        statistical_results = self._statistical_testing(cv_scores)
        
        # Performance summary
        performance_summary = self._create_performance_summary(
            holdout_metrics, cv_scores, "classification"
        )
        
        # Check thresholds
        passed_thresholds = self._check_classification_thresholds(holdout_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            holdout_metrics, cv_scores, "classification"
        )
        
        # Create validation result
        validation_time = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            model_id=model_id,
            model_type=model_type,
            validation_metrics=holdout_metrics,
            cross_validation_scores=cv_scores,
            statistical_significance=statistical_results,
            performance_summary=performance_summary,
            validation_time=validation_time,
            timestamp=start_time,
            passed_thresholds=passed_thresholds,
            recommendations=recommendations
        )
        
        # Save results
        self._save_validation_result(result)
        
        # Add to history
        self.validation_history.append(result)
        
        self.logger.info(f"Validation completed for {model_id}")
        return result
    
    def validate_regression_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_id: str,
        model_type: str = "regression"
    ) -> ValidationResult:
        """Validate regression model
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            model_id: Unique model identifier
            model_type: Type of model
            
        Returns:
            Validation results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Validating regression model: {model_id}")
        
        # Data quality checks
        self._validate_data_quality(X, y, "regression")
        
        # Cross-validation
        cv_scores = self._perform_cross_validation(
            model, X, y, self.config.regression_metrics, "regression"
        )
        
        # Hold-out validation
        holdout_metrics = self._holdout_validation(
            model, X, y, self.config.regression_metrics, "regression"
        )
        
        # Statistical significance testing
        statistical_results = self._statistical_testing(cv_scores)
        
        # Performance summary
        performance_summary = self._create_performance_summary(
            holdout_metrics, cv_scores, "regression"
        )
        
        # Check thresholds
        passed_thresholds = self._check_regression_thresholds(holdout_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            holdout_metrics, cv_scores, "regression"
        )
        
        # Create validation result
        validation_time = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            model_id=model_id,
            model_type=model_type,
            validation_metrics=holdout_metrics,
            cross_validation_scores=cv_scores,
            statistical_significance=statistical_results,
            performance_summary=performance_summary,
            validation_time=validation_time,
            timestamp=start_time,
            passed_thresholds=passed_thresholds,
            recommendations=recommendations
        )
        
        # Save results
        self._save_validation_result(result)
        
        # Add to history
        self.validation_history.append(result)
        
        self.logger.info(f"Validation completed for {model_id}")
        return result
    
    def validate_rl_model(
        self,
        model,
        env: gym.Env,
        model_id: str,
        n_eval_episodes: int = 100,
        model_type: str = "reinforcement_learning"
    ) -> ValidationResult:
        """Validate reinforcement learning model
        
        Args:
            model: Trained RL model
            env: Environment for evaluation
            model_id: Unique model identifier
            n_eval_episodes: Number of evaluation episodes
            model_type: Type of model
            
        Returns:
            Validation results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Validating RL model: {model_id}")
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines3 required for RL model validation")
        
        # Evaluate model
        episode_rewards, episode_lengths = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
        )
        
        # Compute metrics
        holdout_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'success_rate': np.mean(np.array(episode_rewards) > 0)  # Assuming positive reward = success
        }
        
        # Cross-validation for RL (multiple environment seeds)
        cv_scores = self._rl_cross_validation(model, env, n_eval_episodes // 5)
        
        # Statistical significance testing
        statistical_results = self._statistical_testing(cv_scores)
        
        # Performance summary
        performance_summary = self._create_performance_summary(
            holdout_metrics, cv_scores, "reinforcement_learning"
        )
        
        # Check thresholds
        passed_thresholds = self._check_rl_thresholds(holdout_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            holdout_metrics, cv_scores, "reinforcement_learning"
        )
        
        # Create validation result
        validation_time = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            model_id=model_id,
            model_type=model_type,
            validation_metrics=holdout_metrics,
            cross_validation_scores=cv_scores,
            statistical_significance=statistical_results,
            performance_summary=performance_summary,
            validation_time=validation_time,
            timestamp=start_time,
            passed_thresholds=passed_thresholds,
            recommendations=recommendations
        )
        
        # Save results
        self._save_validation_result(result)
        
        # Add to history
        self.validation_history.append(result)
        
        self.logger.info(f"Validation completed for {model_id}")
        return result
    
    def _validate_data_quality(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> None:
        """Validate data quality"""
        # Check minimum samples
        if len(X) < self.config.min_samples:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.config.min_samples}")
        
        # Check for missing values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains missing values")
        
        # Check class balance for classification
        if task_type == "classification":
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_proportions = class_counts / len(y)
            
            if np.min(class_proportions) < self.config.class_balance_threshold:
                self.logger.warning(f"Class imbalance detected: {dict(zip(unique_classes, class_proportions))}")
    
    def _perform_cross_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str],
        task_type: str
    ) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        # Choose CV strategy
        if self.config.cv_strategy == "stratified" and task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        elif self.config.cv_strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
        
        cv_scores = {}
        
        for metric in metrics:
            if task_type == "classification":
                scoring = self._get_sklearn_scoring(metric, "classification")
            else:
                scoring = self._get_sklearn_scoring(metric, "regression")
            
            if scoring:
                try:
                    scores = cross_val_score(
                        model, X, y, cv=cv, scoring=scoring, n_jobs=-1
                    )
                    cv_scores[metric] = scores.tolist()
                except Exception as e:
                    self.logger.warning(f"Failed to compute {metric} in CV: {e}")
                    cv_scores[metric] = []
        
        return cv_scores
    
    def _holdout_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """Perform hold-out validation"""
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        # Train model on training set
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        holdout_metrics = {}
        
        for metric in metrics:
            try:
                if task_type == "classification":
                    value = self._compute_classification_metric(metric, y_test, y_pred, model, X_test)
                else:
                    value = self._compute_regression_metric(metric, y_test, y_pred)
                
                holdout_metrics[metric] = value
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric}: {e}")
                holdout_metrics[metric] = np.nan
        
        return holdout_metrics
    
    def _rl_cross_validation(
        self,
        model,
        env: gym.Env,
        n_eval_episodes: int
    ) -> Dict[str, List[float]]:
        """Cross-validation for RL models using different environment seeds"""
        cv_scores = {metric: [] for metric in self.config.rl_metrics}
        
        for seed in range(self.config.cv_folds):
            # Set environment seed
            env.reset(seed=seed)
            
            # Evaluate model
            episode_rewards, episode_lengths = evaluate_policy(
                model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
            )
            
            # Compute metrics for this fold
            fold_metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'success_rate': np.mean(np.array(episode_rewards) > 0),
                'episode_length': np.mean(episode_lengths)
            }
            
            # Add to CV scores
            for metric in self.config.rl_metrics:
                if metric in fold_metrics:
                    cv_scores[metric].append(fold_metrics[metric])
        
        return cv_scores
    
    def _statistical_testing(
        self,
        cv_scores: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance testing"""
        statistical_results = {}
        
        for metric, scores in cv_scores.items():
            if len(scores) < 2:
                continue
            
            metric_results = {}
            
            # T-test against zero (or baseline)
            if 't_test' in self.config.statistical_tests:
                t_stat, p_value = stats.ttest_1samp(scores, 0)
                metric_results['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level
                }
            
            # Normality test
            if len(scores) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(scores)
                metric_results['normality_test'] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > self.config.significance_level
                }
            
            # Confidence interval
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1)
            n = len(scores)
            
            # 95% confidence interval
            ci_margin = stats.t.ppf(0.975, n-1) * (std_score / np.sqrt(n))
            metric_results['confidence_interval'] = {
                'mean': mean_score,
                'lower': mean_score - ci_margin,
                'upper': mean_score + ci_margin,
                'margin': ci_margin
            }
            
            statistical_results[metric] = metric_results
        
        return statistical_results
    
    def _create_performance_summary(
        self,
        holdout_metrics: Dict[str, float],
        cv_scores: Dict[str, List[float]],
        task_type: str
    ) -> Dict[str, Any]:
        """Create performance summary"""
        summary = {
            'task_type': task_type,
            'holdout_performance': holdout_metrics,
            'cross_validation_summary': {}
        }
        
        # CV summary statistics
        for metric, scores in cv_scores.items():
            if scores:
                summary['cross_validation_summary'][metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                }
        
        # Overall assessment
        if task_type == "classification":
            primary_metric = holdout_metrics.get('f1_score', 0)
        elif task_type == "regression":
            primary_metric = holdout_metrics.get('r2_score', 0)
        else:  # RL
            primary_metric = holdout_metrics.get('mean_reward', 0)
        
        summary['primary_metric_value'] = primary_metric
        summary['performance_tier'] = self._classify_performance(primary_metric, task_type)
        
        return summary
    
    def _classify_performance(self, metric_value: float, task_type: str) -> str:
        """Classify performance into tiers"""
        if task_type == "classification":
            if metric_value >= 0.9:
                return "excellent"
            elif metric_value >= 0.8:
                return "good"
            elif metric_value >= 0.7:
                return "fair"
            else:
                return "poor"
        elif task_type == "regression":
            if metric_value >= 0.9:
                return "excellent"
            elif metric_value >= 0.8:
                return "good"
            elif metric_value >= 0.6:
                return "fair"
            else:
                return "poor"
        else:  # RL
            if metric_value >= 100:
                return "excellent"
            elif metric_value >= 50:
                return "good"
            elif metric_value >= 0:
                return "fair"
            else:
                return "poor"
    
    def _check_classification_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if classification model meets thresholds"""
        checks = [
            metrics.get('accuracy', 0) >= self.config.min_accuracy,
            metrics.get('f1_score', 0) >= self.config.min_f1_score
        ]
        return all(checks)
    
    def _check_regression_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if regression model meets thresholds"""
        checks = [
            metrics.get('mse', float('inf')) <= self.config.max_mse
        ]
        return all(checks)
    
    def _check_rl_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if RL model meets thresholds"""
        checks = [
            metrics.get('mean_reward', -float('inf')) >= self.config.min_reward
        ]
        return all(checks)
    
    def _generate_recommendations(
        self,
        holdout_metrics: Dict[str, float],
        cv_scores: Dict[str, List[float]],
        task_type: str
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check for overfitting
        for metric in holdout_metrics:
            if metric in cv_scores and cv_scores[metric]:
                cv_mean = np.mean(cv_scores[metric])
                holdout_value = holdout_metrics[metric]
                
                if task_type in ["classification", "reinforcement_learning"]:
                    # Higher is better
                    if cv_mean - holdout_value > 0.1:
                        recommendations.append(f"Possible overfitting detected in {metric}")
                else:  # regression
                    # Lower is better for error metrics
                    if holdout_value - cv_mean > 0.1:
                        recommendations.append(f"Possible overfitting detected in {metric}")
        
        # Performance-based recommendations
        if task_type == "classification":
            if holdout_metrics.get('accuracy', 0) < 0.7:
                recommendations.append("Consider feature engineering or model complexity adjustment")
            if holdout_metrics.get('precision', 0) < holdout_metrics.get('recall', 0) - 0.1:
                recommendations.append("Model has low precision - consider adjusting decision threshold")
        
        elif task_type == "regression":
            if holdout_metrics.get('r2_score', 0) < 0.5:
                recommendations.append("Low R² score - consider feature selection or model complexity")
        
        elif task_type == "reinforcement_learning":
            if holdout_metrics.get('std_reward', 0) > holdout_metrics.get('mean_reward', 0):
                recommendations.append("High reward variance - consider longer training or hyperparameter tuning")
        
        # Variance recommendations
        for metric, scores in cv_scores.items():
            if scores and np.std(scores) > np.mean(scores) * 0.2:
                recommendations.append(f"High variance in {metric} across folds - consider more stable training")
        
        return recommendations
    
    def _get_sklearn_scoring(self, metric: str, task_type: str) -> Optional[str]:
        """Get sklearn scoring string for metric"""
        if task_type == "classification":
            mapping = {
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1_score': 'f1_macro',
                'roc_auc': 'roc_auc_ovr'
            }
        else:  # regression
            mapping = {
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2_score': 'r2'
            }
        
        return mapping.get(metric)
    
    def _compute_classification_metric(
        self,
        metric: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model,
        X_test: np.ndarray
    ) -> float:
        """Compute classification metric"""
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'f1_score':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'roc_auc':
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    return roc_auc_score(y_true, y_proba[:, 1])
                else:
                    return roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                return np.nan
        else:
            return np.nan
    
    def _compute_regression_metric(
        self,
        metric: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute regression metric"""
        if metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'r2_score':
            return r2_score(y_true, y_pred)
        elif metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        else:
            return np.nan
    
    def _save_validation_result(self, result: ValidationResult) -> None:
        """Save validation result to file"""
        # Create model-specific directory
        model_dir = self.output_dir / result.model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save main result
        result_file = model_dir / f"validation_result_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save detailed report if enabled
        if self.config.save_reports:
            self._generate_detailed_report(result, model_dir)
    
    def _generate_detailed_report(self, result: ValidationResult, output_dir: Path) -> None:
        """Generate detailed validation report"""
        report_file = output_dir / "validation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Model Validation Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model ID: {result.model_id}\n")
            f.write(f"Model Type: {result.model_type}\n")
            f.write(f"Validation Time: {result.validation_time:.2f} seconds\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Passed Thresholds: {result.passed_thresholds}\n\n")
            
            f.write(f"Validation Metrics:\n")
            f.write(f"-" * 20 + "\n")
            for metric, value in result.validation_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write(f"\nCross-Validation Scores:\n")
            f.write(f"-" * 25 + "\n")
            for metric, scores in result.cross_validation_scores.items():
                if scores:
                    f.write(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
            
            f.write(f"\nRecommendations:\n")
            f.write(f"-" * 15 + "\n")
            for rec in result.recommendations:
                f.write(f"• {rec}\n")
    
    def compare_models(
        self,
        model_ids: List[str],
        comparison_metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple validated models
        
        Args:
            model_ids: List of model IDs to compare
            comparison_metric: Metric to use for comparison
            
        Returns:
            Comparison results
        """
        if comparison_metric is None:
            comparison_metric = self.config.comparison_metric
        
        # Get validation results for specified models
        model_results = []
        for model_id in model_ids:
            for result in self.validation_history:
                if result.model_id == model_id:
                    model_results.append(result)
                    break
        
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Extract comparison metric values
        metric_values = []
        model_names = []
        
        for result in model_results:
            if comparison_metric in result.validation_metrics:
                metric_values.append(result.validation_metrics[comparison_metric])
                model_names.append(result.model_id)
        
        # Statistical comparison
        comparison_results = {
            'models': model_names,
            'metric': comparison_metric,
            'values': metric_values,
            'best_model': model_names[np.argmax(metric_values)],
            'best_value': np.max(metric_values),
            'ranking': sorted(zip(model_names, metric_values), key=lambda x: x[1], reverse=True)
        }
        
        # Pairwise statistical tests
        if len(metric_values) == 2:
            # T-test for two models
            cv_scores_1 = model_results[0].cross_validation_scores.get(comparison_metric, [])
            cv_scores_2 = model_results[1].cross_validation_scores.get(comparison_metric, [])
            
            if cv_scores_1 and cv_scores_2:
                t_stat, p_value = stats.ttest_ind(cv_scores_1, cv_scores_2)
                comparison_results['statistical_test'] = {
                    'test': 'independent_t_test',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level
                }
        
        return comparison_results
    
    def select_best_model(
        self,
        model_ids: List[str],
        selection_criteria: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, ValidationResult]:
        """Select the best model based on validation results
        
        Args:
            model_ids: List of model IDs to consider
            selection_criteria: Custom selection criteria
            
        Returns:
            Tuple of (best_model_id, validation_result)
        """
        if selection_criteria is None:
            selection_criteria = {
                'primary_metric': self.config.comparison_metric,
                'require_threshold_pass': True,
                'consider_stability': True
            }
        
        # Get validation results
        candidates = []
        for model_id in model_ids:
            for result in self.validation_history:
                if result.model_id == model_id:
                    candidates.append(result)
                    break
        
        if not candidates:
            raise ValueError("No validation results found for specified models")
        
        # Filter by threshold requirements
        if selection_criteria.get('require_threshold_pass', False):
            candidates = [c for c in candidates if c.passed_thresholds]
        
        if not candidates:
            raise ValueError("No models passed the required thresholds")
        
        # Score models
        primary_metric = selection_criteria['primary_metric']
        best_model = None
        best_score = float('-inf')
        
        for candidate in candidates:
            if primary_metric not in candidate.validation_metrics:
                continue
            
            score = candidate.validation_metrics[primary_metric]
            
            # Adjust score for stability if requested
            if selection_criteria.get('consider_stability', False):
                cv_scores = candidate.cross_validation_scores.get(primary_metric, [])
                if cv_scores:
                    stability_penalty = np.std(cv_scores) * 0.1  # 10% penalty for instability
                    score -= stability_penalty
            
            if score > best_score:
                best_score = score
                best_model = candidate
        
        if best_model is None:
            raise ValueError("No suitable model found")
        
        return best_model.model_id, best_model


def create_model_validator(
    config: Optional[ValidationConfig] = None,
    output_dir: str = "validation_results"
) -> ModelValidator:
    """Create model validator
    
    Args:
        config: Validation configuration (uses defaults if None)
        output_dir: Directory for validation outputs
        
    Returns:
        Configured model validator
    """
    if config is None:
        config = ValidationConfig()
    
    return ModelValidator(config=config, output_dir=output_dir)