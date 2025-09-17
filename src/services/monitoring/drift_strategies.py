"""
Enhanced drift detection strategies with statistical significance testing.

This module provides various drift detection strategies using statistical tests,
machine learning methods, and domain-specific approaches for trading models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import accuracy_score
import logging

from src.models.monitoring import DriftDetectionResult, DriftType, AlertSeverity
from src.services.monitoring.exceptions import DriftDetectionError, InsufficientDataError
from src.utils.logging import get_logger

logger = get_logger("drift_strategies")


class DriftDetectionMethod(Enum):
    """Available drift detection methods"""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    MANN_WHITNEY_U = "mann_whitney_u"
    ANDERSON_DARLING = "anderson_darling"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"
    PERFORMANCE_BASED = "performance_based"
    TRADING_SPECIFIC = "trading_specific"


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""
    method: DriftDetectionMethod
    significance_level: float = 0.05
    min_samples: int = 100
    window_size: int = 1000
    sensitivity: float = 1.0  # Multiplier for thresholds
    enable_early_detection: bool = True
    bootstrap_iterations: int = 1000


class DriftDetectionStrategy(ABC):
    """Abstract base class for drift detection strategies"""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect drift between reference and current data"""
        pass
    
    def _validate_data(self, reference_data: np.ndarray, current_data: np.ndarray) -> None:
        """Validate input data"""
        if len(reference_data) < self.config.min_samples:
            raise InsufficientDataError(
                "drift_detection", 
                self.config.min_samples, 
                len(reference_data)
            )
        
        if len(current_data) < self.config.min_samples:
            raise InsufficientDataError(
                "drift_detection", 
                self.config.min_samples, 
                len(current_data)
            )
    
    def _calculate_severity(self, p_value: float, effect_size: float = 0.0) -> AlertSeverity:
        """Calculate alert severity based on statistical significance and effect size"""
        
        # Adjust thresholds based on sensitivity
        alpha = self.config.significance_level / self.config.sensitivity
        
        if p_value < alpha / 100 or effect_size > 1.0:  # Very strong evidence
            return AlertSeverity.CRITICAL
        elif p_value < alpha / 10 or effect_size > 0.8:  # Strong evidence
            return AlertSeverity.HIGH
        elif p_value < alpha or effect_size > 0.5:  # Moderate evidence
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class KolmogorovSmirnovStrategy(DriftDetectionStrategy):
    """Kolmogorov-Smirnov test for distribution drift"""
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        self._validate_data(reference_data, current_data)
        
        try:
            # Flatten arrays for univariate test
            ref_flat = reference_data.flatten()
            cur_flat = current_data.flatten()
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(ref_flat, cur_flat)
            
            # Calculate effect size (difference in means normalized by pooled std)
            ref_mean, ref_std = np.mean(ref_flat), np.std(ref_flat)
            cur_mean, cur_std = np.mean(cur_flat), np.std(cur_flat)
            
            pooled_std = np.sqrt((ref_std**2 + cur_std**2) / 2)
            effect_size = abs(ref_mean - cur_mean) / pooled_std if pooled_std > 0 else 0
            
            # Determine drift detection
            drift_detected = p_value < self.config.significance_level
            severity = self._calculate_severity(p_value, effect_size)
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=ks_statistic,
                threshold=self.config.significance_level,
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'kolmogorov_smirnov',
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'reference_mean': ref_mean,
                    'current_mean': cur_mean,
                    'reference_std': ref_std,
                    'current_std': cur_std,
                    'reference_samples': len(ref_flat),
                    'current_samples': len(cur_flat)
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "data_drift", f"KS test failed: {str(e)}")


class MannWhitneyUStrategy(DriftDetectionStrategy):
    """Mann-Whitney U test for median shift detection"""
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        self._validate_data(reference_data, current_data)
        
        try:
            ref_flat = reference_data.flatten()
            cur_flat = current_data.flatten()
            
            # Perform Mann-Whitney U test
            mw_statistic, p_value = stats.mannwhitneyu(
                ref_flat, cur_flat, alternative='two-sided'
            )
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(ref_flat), len(cur_flat)
            effect_size = 1 - (2 * mw_statistic) / (n1 * n2)
            
            drift_detected = p_value < self.config.significance_level
            severity = self._calculate_severity(p_value, abs(effect_size))
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=abs(effect_size),
                threshold=self.config.significance_level,
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'mann_whitney_u',
                    'mw_statistic': float(mw_statistic),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'reference_median': float(np.median(ref_flat)),
                    'current_median': float(np.median(cur_flat)),
                    'reference_samples': len(ref_flat),
                    'current_samples': len(cur_flat)
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "data_drift", f"Mann-Whitney U test failed: {str(e)}")


class PopulationStabilityIndexStrategy(DriftDetectionStrategy):
    """Population Stability Index (PSI) for feature drift detection"""
    
    def __init__(self, config: DriftDetectionConfig, n_bins: int = 10):
        super().__init__(config)
        self.n_bins = n_bins
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        self._validate_data(reference_data, current_data)
        
        try:
            ref_flat = reference_data.flatten()
            cur_flat = current_data.flatten()
            
            # Calculate PSI
            psi_score = self._calculate_psi(ref_flat, cur_flat)
            
            # PSI thresholds (industry standard)
            if psi_score < 0.1:
                severity = AlertSeverity.LOW
                drift_detected = False
            elif psi_score < 0.2:
                severity = AlertSeverity.MEDIUM
                drift_detected = True
            else:
                severity = AlertSeverity.HIGH
                drift_detected = True
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=psi_score,
                threshold=0.1,  # Standard PSI threshold
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'population_stability_index',
                    'psi_score': psi_score,
                    'n_bins': self.n_bins,
                    'reference_samples': len(ref_flat),
                    'current_samples': len(cur_flat)
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "data_drift", f"PSI calculation failed: {str(e)}")
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins based on reference data quantiles
        bin_edges = np.quantile(reference, np.linspace(0, 1, self.n_bins + 1))
        bin_edges[0] = -np.inf  # Handle edge cases
        bin_edges[-1] = np.inf
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi


class JensenShannonDivergenceStrategy(DriftDetectionStrategy):
    """Jensen-Shannon divergence for distribution comparison"""
    
    def __init__(self, config: DriftDetectionConfig, n_bins: int = 50):
        super().__init__(config)
        self.n_bins = n_bins
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        self._validate_data(reference_data, current_data)
        
        try:
            ref_flat = reference_data.flatten()
            cur_flat = current_data.flatten()
            
            # Calculate JS divergence
            js_divergence = self._calculate_js_divergence(ref_flat, cur_flat)
            
            # JS divergence thresholds (0 = identical, 1 = completely different)
            threshold = 0.1 * self.config.sensitivity
            
            if js_divergence > threshold * 3:
                severity = AlertSeverity.CRITICAL
            elif js_divergence > threshold * 2:
                severity = AlertSeverity.HIGH
            elif js_divergence > threshold:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            drift_detected = js_divergence > threshold
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=js_divergence,
                threshold=threshold,
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'jensen_shannon_divergence',
                    'js_divergence': js_divergence,
                    'threshold': threshold,
                    'n_bins': self.n_bins,
                    'reference_samples': len(ref_flat),
                    'current_samples': len(cur_flat)
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "data_drift", f"JS divergence calculation failed: {str(e)}")
    
    def _calculate_js_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence"""
        
        # Create common bins
        combined = np.concatenate([reference, current])
        bin_edges = np.histogram_bin_edges(combined, bins=self.n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        ref_hist = ref_hist / np.sum(ref_hist)
        cur_hist = cur_hist / np.sum(cur_hist)
        
        # Avoid log(0)
        ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
        cur_hist = np.where(cur_hist == 0, 1e-10, cur_hist)
        
        # Calculate JS divergence
        m = 0.5 * (ref_hist + cur_hist)
        js_div = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(cur_hist * np.log(cur_hist / m))
        
        return js_div


class PerformanceBasedDriftStrategy(DriftDetectionStrategy):
    """Performance-based drift detection using model accuracy degradation"""
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        # This strategy requires performance metrics in metadata
        if not metadata or 'reference_performance' not in metadata or 'current_performance' not in metadata:
            raise DriftDetectionError("model", "performance_drift", "Performance metrics required in metadata")
        
        try:
            ref_performance = metadata['reference_performance']
            cur_performance = metadata['current_performance']
            
            # Calculate performance degradation
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            degradations = {}
            
            for metric in performance_metrics:
                if metric in ref_performance and metric in cur_performance:
                    ref_value = ref_performance[metric]
                    cur_value = cur_performance[metric]
                    
                    if ref_value > 0:
                        degradation = (ref_value - cur_value) / ref_value
                        degradations[metric] = {
                            'reference': ref_value,
                            'current': cur_value,
                            'degradation': degradation
                        }
            
            if not degradations:
                raise DriftDetectionError("model", "performance_drift", "No valid performance metrics found")
            
            # Calculate overall drift score (maximum degradation)
            max_degradation = max(d['degradation'] for d in degradations.values())
            
            # Determine severity based on degradation
            threshold = 0.05  # 5% degradation threshold
            
            if max_degradation > threshold * 4:  # 20% degradation
                severity = AlertSeverity.CRITICAL
            elif max_degradation > threshold * 2:  # 10% degradation
                severity = AlertSeverity.HIGH
            elif max_degradation > threshold:  # 5% degradation
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            drift_detected = max_degradation > threshold
            
            return DriftDetectionResult(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=severity,
                drift_score=max_degradation,
                threshold=threshold,
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'performance_based',
                    'degradations': degradations,
                    'max_degradation': max_degradation,
                    'threshold': threshold,
                    'reference_performance': ref_performance,
                    'current_performance': cur_performance
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "performance_drift", f"Performance drift detection failed: {str(e)}")


class TradingSpecificDriftStrategy(DriftDetectionStrategy):
    """Trading-specific drift detection for market regime changes"""
    
    async def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        
        self._validate_data(reference_data, current_data)
        
        try:
            # Assume data contains market features (returns, volatility, volume, etc.)
            ref_features = self._extract_trading_features(reference_data)
            cur_features = self._extract_trading_features(current_data)
            
            # Detect regime changes
            regime_changes = self._detect_regime_changes(ref_features, cur_features)
            
            # Calculate overall drift score
            drift_score = np.mean([change['score'] for change in regime_changes])
            
            # Determine severity
            threshold = 0.3  # Trading-specific threshold
            
            if drift_score > threshold * 2:
                severity = AlertSeverity.HIGH
            elif drift_score > threshold:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            drift_detected = drift_score > threshold
            
            return DriftDetectionResult(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                drift_score=drift_score,
                threshold=threshold,
                detected=drift_detected,
                timestamp=datetime.now(),
                details={
                    'method': 'trading_specific',
                    'regime_changes': regime_changes,
                    'reference_features': ref_features,
                    'current_features': cur_features,
                    'drift_score': drift_score,
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            raise DriftDetectionError("model", "concept_drift", f"Trading drift detection failed: {str(e)}")
    
    def _extract_trading_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract trading-specific features from market data"""
        
        # Assume data is time series of returns or prices
        if data.ndim > 1:
            data = data.flatten()
        
        # Calculate trading features
        returns = np.diff(data) / data[:-1] if len(data) > 1 else np.array([0])
        
        features = {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'skewness': float(stats.skew(returns)) if len(returns) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(data),
            'autocorrelation': self._calculate_autocorrelation(returns)
        }
        
        return features
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag"""
        if len(returns) <= lag:
            return 0.0
        
        return float(np.corrcoef(returns[:-lag], returns[lag:])[0, 1])
    
    def _detect_regime_changes(self, ref_features: Dict[str, float], cur_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect regime changes in trading features"""
        
        regime_changes = []
        
        # Define regime change thresholds for each feature
        thresholds = {
            'volatility': 0.5,  # 50% change in volatility
            'mean_return': 0.3,  # 30% change in mean return
            'sharpe_ratio': 0.4,  # 40% change in Sharpe ratio
            'max_drawdown': 0.3,  # 30% change in max drawdown
            'skewness': 1.0,     # Change in skewness
            'kurtosis': 2.0      # Change in kurtosis
        }
        
        for feature, threshold in thresholds.items():
            if feature in ref_features and feature in cur_features:
                ref_value = ref_features[feature]
                cur_value = cur_features[feature]
                
                # Calculate relative change
                if abs(ref_value) > 1e-6:  # Avoid division by very small numbers
                    relative_change = abs(cur_value - ref_value) / abs(ref_value)
                else:
                    relative_change = abs(cur_value - ref_value)
                
                if relative_change > threshold:
                    regime_changes.append({
                        'feature': feature,
                        'reference_value': ref_value,
                        'current_value': cur_value,
                        'relative_change': relative_change,
                        'threshold': threshold,
                        'score': min(1.0, relative_change / threshold)
                    })
        
        return regime_changes


class DriftDetectionContext:
    """Context class that manages different drift detection strategies"""
    
    def __init__(self):
        self.strategies: Dict[DriftDetectionMethod, DriftDetectionStrategy] = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Initialize all drift detection strategies"""
        
        base_config = DriftDetectionConfig(method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV)
        
        # Initialize strategies with appropriate configurations
        self.strategies[DriftDetectionMethod.KOLMOGOROV_SMIRNOV] = KolmogorovSmirnovStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV)
        )
        
        self.strategies[DriftDetectionMethod.MANN_WHITNEY_U] = MannWhitneyUStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.MANN_WHITNEY_U)
        )
        
        self.strategies[DriftDetectionMethod.POPULATION_STABILITY_INDEX] = PopulationStabilityIndexStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.POPULATION_STABILITY_INDEX)
        )
        
        self.strategies[DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE] = JensenShannonDivergenceStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE)
        )
        
        self.strategies[DriftDetectionMethod.PERFORMANCE_BASED] = PerformanceBasedDriftStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.PERFORMANCE_BASED)
        )
        
        self.strategies[DriftDetectionMethod.TRADING_SPECIFIC] = TradingSpecificDriftStrategy(
            DriftDetectionConfig(method=DriftDetectionMethod.TRADING_SPECIFIC)
        )
    
    async def detect_drift(
        self, 
        drift_type: DriftType, 
        model_name: str, 
        data: Dict[str, Any], 
        threshold: float = 0.05,
        method: Optional[DriftDetectionMethod] = None
    ) -> Optional[DriftDetectionResult]:
        """Detect drift using specified or appropriate strategy"""
        
        try:
            # Select appropriate method based on drift type and available data
            if method is None:
                method = self._select_method(drift_type, data)
            
            if method not in self.strategies:
                logger.warning(f"Drift detection method {method.value} not available")
                return None
            
            strategy = self.strategies[method]
            
            # Extract data for drift detection
            reference_data, current_data, metadata = self._prepare_data(data, drift_type)
            
            if reference_data is None or current_data is None:
                logger.warning(f"Insufficient data for drift detection: {drift_type.value}")
                return None
            
            # Perform drift detection
            # Pass threshold to strategy
            strategy.config.significance_level = threshold
            result = await strategy.detect_drift(reference_data, current_data, metadata)
            
            # Log result
            logger.info(f"Drift detection completed for {model_name}: "
                       f"method={method.value}, detected={result.detected}, "
                       f"score={result.drift_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Drift detection failed for {model_name}: {e}")
            return None
    
    def _select_method(self, drift_type: DriftType, data: Dict[str, Any]) -> DriftDetectionMethod:
        """Select appropriate drift detection method based on drift type and data"""
        
        if drift_type == DriftType.PERFORMANCE_DRIFT:
            if 'reference_performance' in data and 'current_performance' in data:
                return DriftDetectionMethod.PERFORMANCE_BASED
            else:
                return DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        
        elif drift_type == DriftType.CONCEPT_DRIFT:
            # Use trading-specific method for concept drift in trading models
            return DriftDetectionMethod.TRADING_SPECIFIC
        
        elif drift_type == DriftType.DATA_QUALITY_DRIFT:
            # Use PSI for data quality drift
            return DriftDetectionMethod.POPULATION_STABILITY_INDEX
        
        elif drift_type == DriftType.DATA_DRIFT:
            # Use KS test as default for data drift
            return DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        
        elif drift_type == DriftType.DATA_QUALITY_DRIFT:
            # Use PSI for data quality drift
            return DriftDetectionMethod.POPULATION_STABILITY_INDEX
        
        else:
            return DriftDetectionMethod.KOLMOGOROV_SMIRNOV
    
    def _prepare_data(self, data: Dict[str, Any], drift_type: DriftType) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Prepare data for drift detection based on drift type"""
        
        reference_data = None
        current_data = None
        metadata = {}
        
        if drift_type == DriftType.DATA_DRIFT:
            # Extract feature data
            if 'feature_history' in data:
                feature_history = data['feature_history']
                if len(feature_history) >= 100:  # Minimum samples
                    split_point = len(feature_history) // 2
                    reference_data = np.array(feature_history[:split_point])
                    current_data = np.array(feature_history[split_point:])
        
        elif drift_type == DriftType.CONCEPT_DRIFT:
            # Extract feature data for concept drift
            if 'feature_history' in data:
                feature_history = data['feature_history']
                if len(feature_history) >= 100:  # Minimum samples
                    split_point = len(feature_history) // 2
                    reference_data = np.array(feature_history[:split_point])
                    current_data = np.array(feature_history[split_point:])
        
        elif drift_type == DriftType.DATA_QUALITY_DRIFT:
            # Extract data quality metrics
            if 'feature_history' in data:
                feature_history = data['feature_history']
                if len(feature_history) >= 50:
                    # Use recent data for quality drift detection
                    reference_data = np.array(feature_history[-100:-50])
                    current_data = np.array(feature_history[-50:])
        
        elif drift_type == DriftType.PERFORMANCE_DRIFT:
            # Extract performance data
            if 'baseline_metrics' in data and 'current_metrics' in data:
                baseline = data['baseline_metrics']
                current = data['current_metrics']
                
                if baseline and current:
                    # Create dummy arrays for the strategy (actual comparison is in metadata)
                    reference_data = np.array([1.0])
                    current_data = np.array([1.0])
                    
                    metadata = {
                        'reference_performance': baseline.__dict__ if hasattr(baseline, '__dict__') else baseline,
                        'current_performance': current.__dict__ if hasattr(current, '__dict__') else current
                    }
        
        elif drift_type == DriftType.DATA_QUALITY_DRIFT:
            # Extract data quality metrics
            if 'feature_history' in data:
                feature_history = data['feature_history']
                if len(feature_history) >= 50:
                    # Use recent data for quality drift detection
                    reference_data = np.array(feature_history[-100:-50])
                    current_data = np.array(feature_history[-50:])
        
        return reference_data, current_data, metadata
    
    def get_available_methods(self) -> List[DriftDetectionMethod]:
        """Get list of available drift detection methods"""
        return list(self.strategies.keys())
    
    def configure_strategy(self, method: DriftDetectionMethod, config: DriftDetectionConfig) -> None:
        """Configure a specific drift detection strategy"""
        
        if method in self.strategies:
            self.strategies[method].config = config
            logger.info(f"Configuration updated for drift detection method: {method.value}")
        else:
            logger.warning(f"Drift detection method {method.value} not found")
