"""
Comprehensive Feature Engineering Framework

This module provides a unified interface for generating all types of features:
- 100+ technical indicators
- Market microstructure features
- Regime detection features
- Alternative data features

The framework is designed for scalability, modularity, and production use.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path
import joblib
from datetime import datetime

from .technical_indicators import TechnicalIndicatorEngine, IndicatorConfig
from .microstructure import MicrostructureEngine, MicrostructureConfig
from .regime_detection import RegimeDetectionEngine, RegimeConfig
from .alternative_data import AlternativeDataEngine, AlternativeDataConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineConfig:
    """Master configuration for the feature engineering framework."""
    
    # Component configurations
    technical_config: IndicatorConfig = field(default_factory=IndicatorConfig)
    microstructure_config: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    alternative_config: AlternativeDataConfig = field(default_factory=AlternativeDataConfig)
    
    # Feature selection and processing
    enable_technical: bool = True
    enable_microstructure: bool = True
    enable_regime: bool = True
    enable_alternative: bool = True
    
    # Data processing options
    fill_method: str = 'forward'  # 'forward', 'backward', 'interpolate', 'drop'
    normalize_features: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Performance options
    parallel_processing: bool = True
    n_jobs: int = -1
    chunk_size: int = 10000
    
    # Caching options
    enable_caching: bool = True
    cache_dir: str = "data/features/cache"
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = 'variance'  # 'variance', 'correlation', 'mutual_info'
    correlation_threshold: float = 0.95


class FeatureValidator:
    """Validates and cleans feature data."""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
    
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data format and completeness."""
        errors = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} is not numeric")
        
        # Check for sufficient data
        if len(data) < 100:
            errors.append("Insufficient data: need at least 100 rows")
        
        # Check for data consistency
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                errors.append(f"Found {invalid_hl} rows where high < low")
        
        return len(errors) == 0, errors
    
    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate generated features."""
        logger.info(f"Cleaning features: {features.shape}")
        
        # Handle missing values
        if self.config.fill_method == 'forward':
            features = features.fillna(method='ffill')
        elif self.config.fill_method == 'backward':
            features = features.fillna(method='bfill')
        elif self.config.fill_method == 'interpolate':
            features = features.interpolate()
        elif self.config.fill_method == 'drop':
            features = features.dropna()
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Remove outliers
        if self.config.remove_outliers:
            features = self._remove_outliers(features)
        
        # Normalize features
        if self.config.normalize_features:
            features = self._normalize_features(features)
        
        logger.info(f"Cleaned features: {features.shape}")
        return features
    
    def _remove_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if features[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
                features.loc[z_scores > self.config.outlier_threshold, col] = np.nan
        
        # Fill outliers with median
        features = features.fillna(features.median())
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_min = features[col].min()
            col_max = features[col].max()
            
            if col_max > col_min:  # Avoid division by zero
                features[col] = (features[col] - col_min) / (col_max - col_min)
        
        return features


class FeatureSelector:
    """Selects most relevant features based on various criteria."""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
    
    def select_features(self, features: pd.DataFrame, 
                       target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Select most relevant features."""
        if self.config.max_features is None or len(features.columns) <= self.config.max_features:
            return features, list(features.columns)
        
        logger.info(f"Selecting {self.config.max_features} features from {len(features.columns)}")
        
        if self.config.feature_selection_method == 'variance':
            selected_features = self._select_by_variance(features)
        elif self.config.feature_selection_method == 'correlation':
            selected_features = self._select_by_correlation(features)
        elif self.config.feature_selection_method == 'mutual_info' and target is not None:
            selected_features = self._select_by_mutual_info(features, target)
        else:
            # Fallback to variance-based selection
            selected_features = self._select_by_variance(features)
        
        return features[selected_features], selected_features
    
    def _select_by_variance(self, features: pd.DataFrame) -> List[str]:
        """Select features with highest variance."""
        variances = features.var().sort_values(ascending=False)
        return variances.head(self.config.max_features).index.tolist()
    
    def _select_by_correlation(self, features: pd.DataFrame) -> List[str]:
        """Select features with low inter-correlation."""
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.config.correlation_threshold)]
        
        # Keep features not in drop list
        selected = [col for col in features.columns if col not in to_drop]
        
        # If still too many, select by variance
        if len(selected) > self.config.max_features:
            variances = features[selected].var().sort_values(ascending=False)
            selected = variances.head(self.config.max_features).index.tolist()
        
        return selected
    
    def _select_by_mutual_info(self, features: pd.DataFrame, 
                              target: pd.Series) -> List[str]:
        """Select features with highest mutual information with target."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(features.fillna(0), target.fillna(0))
            
            # Create series and sort
            mi_series = pd.Series(mi_scores, index=features.columns)
            mi_series = mi_series.sort_values(ascending=False)
            
            return mi_series.head(self.config.max_features).index.tolist()
            
        except ImportError:
            logger.warning("sklearn not available, falling back to variance selection")
            return self._select_by_variance(features)


class FeatureCache:
    """Caches computed features for performance."""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, data: pd.DataFrame, feature_types: List[str]) -> str:
        """Generate cache key based on data and configuration."""
        data_hash = pd.util.hash_pandas_object(data).sum()
        config_str = f"{feature_types}_{self.config.technical_config}_{self.config.microstructure_config}"
        return f"{data_hash}_{hash(config_str)}"
    
    def load_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache."""
        if not self.config.enable_caching:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def save_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Save features to cache."""
        if not self.config.enable_caching:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            joblib.dump(features, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class ComprehensiveFeatureEngine:
    """Main feature engineering engine that orchestrates all components."""
    
    def __init__(self, config: FeatureEngineConfig = None):
        self.config = config or FeatureEngineConfig()
        
        # Initialize component engines
        self.technical_engine = TechnicalIndicatorEngine(self.config.technical_config)
        self.microstructure_engine = MicrostructureEngine(self.config.microstructure_config)
        self.regime_engine = RegimeDetectionEngine(self.config.regime_config)
        self.alternative_engine = AlternativeDataEngine(self.config.alternative_config)
        
        # Initialize utilities
        self.validator = FeatureValidator(self.config)
        self.selector = FeatureSelector(self.config)
        self.cache = FeatureCache(self.config)
        
        logger.info("Comprehensive Feature Engine initialized")
    
    def generate_all_features(self, data: pd.DataFrame, 
                            target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate all available features."""
        feature_types = []
        if self.config.enable_technical:
            feature_types.append('technical')
        if self.config.enable_microstructure:
            feature_types.append('microstructure')
        if self.config.enable_regime:
            feature_types.append('regime')
        if self.config.enable_alternative:
            feature_types.append('alternative')
        
        return self.generate_features(data, feature_types, target)
    
    def generate_features(self, data: pd.DataFrame, 
                         feature_types: List[str],
                         target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate specified types of features."""
        
        # Validate input data
        is_valid, errors = self.validator.validate_input_data(data)
        if not is_valid:
            raise ValueError(f"Invalid input data: {errors}")
        
        # Check cache
        cache_key = self.cache.get_cache_key(data, feature_types)
        cached_features = self.cache.load_features(cache_key)
        if cached_features is not None:
            logger.info("Loaded features from cache")
            return cached_features
        
        logger.info(f"Generating features: {feature_types}")
        all_features = pd.DataFrame(index=data.index)
        
        # Generate each type of features
        for feature_type in feature_types:
            try:
                if feature_type == 'technical' and self.config.enable_technical:
                    features = self.technical_engine.calculate_all_indicators(data)
                    all_features = pd.concat([all_features, features], axis=1)
                    logger.info(f"Generated {len(features.columns)} technical indicators")
                
                elif feature_type == 'microstructure' and self.config.enable_microstructure:
                    features = self.microstructure_engine.calculate_all_features(data)
                    all_features = pd.concat([all_features, features], axis=1)
                    logger.info(f"Generated {len(features.columns)} microstructure features")
                
                elif feature_type == 'regime' and self.config.enable_regime:
                    features = self.regime_engine.calculate_all_features(data)
                    all_features = pd.concat([all_features, features], axis=1)
                    logger.info(f"Generated {len(features.columns)} regime detection features")
                
                elif feature_type == 'alternative' and self.config.enable_alternative:
                    features = self.alternative_engine.calculate_all_features(data)
                    all_features = pd.concat([all_features, features], axis=1)
                    logger.info(f"Generated {len(features.columns)} alternative data features")
                
            except Exception as e:
                logger.error(f"Error generating {feature_type} features: {e}")
                continue
        
        # Clean and validate features
        all_features = self.validator.clean_features(all_features)
        
        # Feature selection
        if self.config.max_features is not None and len(all_features.columns) > self.config.max_features:
            all_features, selected_features = self.selector.select_features(all_features, target)
            logger.info(f"Selected {len(selected_features)} features")
        
        # Cache results
        self.cache.save_features(all_features, cache_key)
        
        logger.info(f"Feature generation complete: {all_features.shape}")
        return all_features
    
    def get_feature_importance(self, features: pd.DataFrame, 
                             target: pd.Series) -> pd.Series:
        """Calculate feature importance scores."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression
            
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features.fillna(0), target.fillna(0))
            rf_importance = pd.Series(rf.feature_importances_, index=features.columns)
            
            # Mutual information
            mi_scores = mutual_info_regression(features.fillna(0), target.fillna(0))
            mi_importance = pd.Series(mi_scores, index=features.columns)
            
            # Combine scores (weighted average)
            combined_importance = 0.7 * rf_importance + 0.3 * mi_importance
            return combined_importance.sort_values(ascending=False)
            
        except ImportError:
            logger.warning("sklearn not available, using correlation-based importance")
            return features.corrwith(target).abs().sort_values(ascending=False)
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive summary of generated features."""
        summary = {
            'total_features': len(features.columns),
            'feature_types': {},
            'data_quality': {
                'missing_values': features.isnull().sum().sum(),
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
                'zero_variance_features': (features.var() == 0).sum()
            },
            'statistics': {
                'mean_correlation': features.corr().abs().mean().mean(),
                'max_correlation': features.corr().abs().max().max(),
                'feature_ranges': {
                    'min': features.min().min(),
                    'max': features.max().max(),
                    'mean': features.mean().mean(),
                    'std': features.std().mean()
                }
            }
        }
        
        # Count features by type
        for col in features.columns:
            if col.startswith('momentum_'):
                summary['feature_types']['momentum'] = summary['feature_types'].get('momentum', 0) + 1
            elif col.startswith('volatility_'):
                summary['feature_types']['volatility'] = summary['feature_types'].get('volatility', 0) + 1
            elif col.startswith('volume_'):
                summary['feature_types']['volume'] = summary['feature_types'].get('volume', 0) + 1
            elif col.startswith('micro_'):
                summary['feature_types']['microstructure'] = summary['feature_types'].get('microstructure', 0) + 1
            elif col.startswith('regime_'):
                summary['feature_types']['regime'] = summary['feature_types'].get('regime', 0) + 1
            elif col.startswith('alt_'):
                summary['feature_types']['alternative'] = summary['feature_types'].get('alternative', 0) + 1
            else:
                summary['feature_types']['other'] = summary['feature_types'].get('other', 0) + 1
        
        return summary
    
    def save_features(self, features: pd.DataFrame, filepath: str) -> None:
        """Save features to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.csv':
            features.to_csv(filepath)
        elif filepath.suffix == '.parquet':
            features.to_parquet(filepath)
        elif filepath.suffix == '.pkl':
            features.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load features from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath, index_col=0)
        elif filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.pkl':
            return pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


# Convenience function for quick feature generation
def generate_comprehensive_features(data: pd.DataFrame, 
                                  config: Optional[FeatureEngineConfig] = None,
                                  target: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Convenience function to generate comprehensive features with default configuration.
    
    Args:
        data: Market data with OHLCV columns
        config: Optional configuration (uses defaults if None)
        target: Optional target variable for feature selection
    
    Returns:
        DataFrame with comprehensive features
    """
    engine = ComprehensiveFeatureEngine(config)
    return engine.generate_all_features(data, target)