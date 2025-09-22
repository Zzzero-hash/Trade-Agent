"""
Robust Data Preprocessing Pipeline for Financial Time Series

This module implements comprehensive data preprocessing for financial market data including:
- Data cleaning with forward-fill, interpolation, and outlier detection
- Proper train/validation/test splits with temporal ordering (no look-ahead bias)
- Data normalization and scaling appropriate for financial time series
- Data quality monitoring with automated validation checks
- Missing data handling and outlier treatment
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import scipy.stats as stats
from scipy import interpolate

logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """Scaling methods for financial data."""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"     # Min-max scaling
    ROBUST = "robust"     # Robust scaling (median and IQR)
    NONE = "none"         # No scaling


class ImputationMethod(Enum):
    """Methods for handling missing data."""
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    LINEAR_INTERPOLATION = "linear"
    SPLINE_INTERPOLATION = "spline"
    KNN_IMPUTATION = "knn"
    MEAN_IMPUTATION = "mean"
    MEDIAN_IMPUTATION = "median"
    DROP = "drop"


class OutlierMethod(Enum):
    """Methods for outlier detection and treatment."""
    Z_SCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation"
    MODIFIED_Z_SCORE = "modified_zscore"
    NONE = "none"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Data cleaning
    handle_missing_data: bool = True
    imputation_method: ImputationMethod = ImputationMethod.FORWARD_FILL
    max_consecutive_missing: int = 5  # Max consecutive missing values to impute
    
    # Outlier detection and treatment
    detect_outliers: bool = True
    outlier_method: OutlierMethod = OutlierMethod.Z_SCORE
    outlier_threshold: float = 3.0
    outlier_treatment: str = "clip"  # "clip", "remove", "impute"
    
    # Data splits
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    min_train_samples: int = 1000
    
    # Scaling and normalization
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    scale_features: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    
    # Data quality
    min_data_quality_score: float = 70.0
    require_complete_days: bool = True  # For intraday data
    
    # Validation
    validate_temporal_order: bool = True
    validate_ohlc_consistency: bool = True
    validate_volume_positive: bool = True


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_dates: Dict[str, datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingResults:
    """Results from data preprocessing."""
    processed_data: pd.DataFrame
    data_splits: DataSplit
    scalers: Dict[str, Any]
    preprocessing_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class FinancialDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for financial time series data.
    
    Features:
    - Temporal-aware data cleaning and imputation
    - Proper time series splits with no look-ahead bias
    - Financial-specific scaling and normalization
    - Comprehensive data quality monitoring
    - Outlier detection and treatment
    - Data validation and consistency checks
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor."""
        self.config = config or PreprocessingConfig()
        self.scalers = {}
        self.imputers = {}
        self.preprocessing_stats = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate preprocessing configuration."""
        if abs(self.config.train_ratio + self.config.validation_ratio + self.config.test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        if self.config.train_ratio <= 0 or self.config.validation_ratio <= 0 or self.config.test_ratio <= 0:
            raise ValueError("All data split ratios must be positive")
        
        if self.config.outlier_threshold <= 0:
            raise ValueError("Outlier threshold must be positive")
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data with appropriate methods.
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned data
        """
        logger.info(f"Starting data cleaning for {len(data)} records")
        cleaned_data = data.copy()
        
        # Ensure temporal ordering
        if 'timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('timestamp')
        
        # Basic data validation
        cleaned_data = self._validate_ohlcv_data(cleaned_data)
        
        # Handle missing data
        if self.config.handle_missing_data:
            cleaned_data = self._handle_missing_data(cleaned_data)
        
        # Detect and treat outliers
        if self.config.detect_outliers:
            cleaned_data = self._detect_and_treat_outliers(cleaned_data)
        
        # Final validation
        cleaned_data = self._final_validation(cleaned_data)
        
        logger.info(f"Data cleaning completed. Final records: {len(cleaned_data)}")
        return cleaned_data
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data consistency."""
        if not self.config.validate_ohlc_consistency:
            return data
        
        logger.debug("Validating OHLCV data consistency")
        initial_count = len(data)
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with invalid OHLC relationships
        valid_mask = (
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['open'] > 0) &
            (data['high'] > 0) &
            (data['low'] > 0) &
            (data['close'] > 0)
        )
        
        if 'volume' in data.columns and self.config.validate_volume_positive:
            valid_mask &= (data['volume'] >= 0)
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} rows with invalid OHLCV data")
            data = data[valid_mask].copy()
        
        logger.debug(f"OHLCV validation: {initial_count} -> {len(data)} records")
        return data
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using specified method."""
        logger.debug(f"Handling missing data using method: {self.config.imputation_method.value}")
        
        # Check for missing data
        missing_counts = data.isnull().sum()
        if missing_counts.sum() == 0:
            logger.debug("No missing data found")
            return data
        
        logger.info(f"Found missing data: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Handle different imputation methods
        if self.config.imputation_method == ImputationMethod.FORWARD_FILL:
            data = self._forward_fill_imputation(data)
        elif self.config.imputation_method == ImputationMethod.BACKWARD_FILL:
            data = data.bfill()
        elif self.config.imputation_method == ImputationMethod.LINEAR_INTERPOLATION:
            data = self._linear_interpolation(data)
        elif self.config.imputation_method == ImputationMethod.SPLINE_INTERPOLATION:
            data = self._spline_interpolation(data)
        elif self.config.imputation_method == ImputationMethod.KNN_IMPUTATION:
            data = self._knn_imputation(data)
        elif self.config.imputation_method == ImputationMethod.MEAN_IMPUTATION:
            data = self._statistical_imputation(data, strategy='mean')
        elif self.config.imputation_method == ImputationMethod.MEDIAN_IMPUTATION:
            data = self._statistical_imputation(data, strategy='median')
        elif self.config.imputation_method == ImputationMethod.DROP:
            data = data.dropna()
        
        # Check for remaining missing data
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Still have {remaining_missing} missing values after imputation")
        
        return data
    
    def _forward_fill_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Forward fill with limits for consecutive missing values."""
        # Forward fill with limit
        filled_data = data.ffill(limit=self.config.max_consecutive_missing)
        
        # For any remaining missing values at the beginning, use backward fill
        filled_data = filled_data.bfill()
        
        return filled_data
    
    def _linear_interpolation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Linear interpolation for missing values."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].interpolate(method='linear', limit=self.config.max_consecutive_missing)
        
        return data
    
    def _spline_interpolation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Spline interpolation for missing values."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().any() and len(data[col].dropna()) >= 4:  # Need at least 4 points for spline
                try:
                    data[col] = data[col].interpolate(method='spline', order=3, limit=self.config.max_consecutive_missing)
                except:
                    # Fallback to linear interpolation
                    data[col] = data[col].interpolate(method='linear', limit=self.config.max_consecutive_missing)
        
        return data
    
    def _knn_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """KNN imputation for missing values."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            self.imputers['knn'] = imputer
        
        return data
    
    def _statistical_imputation(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Statistical imputation (mean/median) for missing values."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            self.imputers[strategy] = imputer
        
        return data
    
    def _detect_and_treat_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and treat outliers in financial data."""
        logger.debug(f"Detecting outliers using method: {self.config.outlier_method.value}")
        
        if self.config.outlier_method == OutlierMethod.NONE:
            return data
        
        numeric_cols = [col for col in self.config.scale_features if col in data.columns]
        outlier_mask = pd.Series(False, index=data.index)
        
        for col in numeric_cols:
            if col in data.columns:
                col_outliers = self._detect_column_outliers(data[col])
                outlier_mask |= col_outliers
        
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outlier records ({outlier_count/len(data)*100:.1f}%)")
            
            # Treat outliers based on configuration
            if self.config.outlier_treatment == "remove":
                data = data[~outlier_mask].copy()
                logger.info(f"Removed {outlier_count} outlier records")
            elif self.config.outlier_treatment == "clip":
                data = self._clip_outliers(data, outlier_mask, numeric_cols)
            elif self.config.outlier_treatment == "impute":
                data = self._impute_outliers(data, outlier_mask, numeric_cols)
        
        return data
    
    def _detect_column_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers in a single column."""
        if self.config.outlier_method == OutlierMethod.Z_SCORE:
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = pd.Series(False, index=series.index)
            outliers.loc[series.dropna().index] = z_scores > self.config.outlier_threshold
            return outliers
        
        elif self.config.outlier_method == OutlierMethod.MODIFIED_Z_SCORE:
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > self.config.outlier_threshold
        
        elif self.config.outlier_method == OutlierMethod.IQR:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        return pd.Series(False, index=series.index)
    
    def _clip_outliers(self, data: pd.DataFrame, outlier_mask: pd.Series, columns: List[str]) -> pd.DataFrame:
        """Clip outliers to reasonable bounds."""
        for col in columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.01)
                Q99 = data[col].quantile(0.99)
                data.loc[outlier_mask, col] = data.loc[outlier_mask, col].clip(Q1, Q99)
        
        logger.info("Clipped outliers to 1st-99th percentile range")
        return data
    
    def _impute_outliers(self, data: pd.DataFrame, outlier_mask: pd.Series, columns: List[str]) -> pd.DataFrame:
        """Impute outliers with median values."""
        for col in columns:
            if col in data.columns:
                median_val = data[col].median()
                data.loc[outlier_mask, col] = median_val
        
        logger.info("Imputed outliers with median values")
        return data
    
    def _final_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final validation of cleaned data."""
        # Remove any remaining rows with NaN values
        initial_count = len(data)
        data = data.dropna()
        
        if len(data) < initial_count:
            logger.warning(f"Dropped {initial_count - len(data)} rows with remaining NaN values")
        
        # Ensure minimum data requirements
        if len(data) < self.config.min_train_samples:
            raise ValueError(f"Insufficient data after cleaning: {len(data)} < {self.config.min_train_samples}")
        
        return data
    
    def create_temporal_splits(self, data: pd.DataFrame) -> DataSplit:
        """
        Create train/validation/test splits with proper temporal ordering.
        
        Args:
            data: Cleaned data to split
            
        Returns:
            DataSplit object with train/validation/test data
        """
        logger.info("Creating temporal data splits")
        
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column for temporal splits")
        
        # Sort by timestamp to ensure proper temporal ordering
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(data)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        # Create indices
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        test_indices = np.arange(val_end, n_samples)
        
        # Create data splits
        train_data = data.iloc[train_indices].copy()
        val_data = data.iloc[val_indices].copy()
        test_data = data.iloc[test_indices].copy()
        
        # Get split dates
        split_dates = {
            'train_start': train_data['timestamp'].min(),
            'train_end': train_data['timestamp'].max(),
            'val_start': val_data['timestamp'].min(),
            'val_end': val_data['timestamp'].max(),
            'test_start': test_data['timestamp'].min(),
            'test_end': test_data['timestamp'].max()
        }
        
        # Validate splits
        self._validate_temporal_splits(split_dates)
        
        # Create metadata
        metadata = {
            'total_samples': n_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_ratio_actual': len(train_data) / n_samples,
            'val_ratio_actual': len(val_data) / n_samples,
            'test_ratio_actual': len(test_data) / n_samples
        }
        
        logger.info(f"Created temporal splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return DataSplit(
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data,
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates=split_dates,
            metadata=metadata
        )
    
    def _validate_temporal_splits(self, split_dates: Dict[str, datetime]):
        """Validate that temporal splits don't have look-ahead bias."""
        if not self.config.validate_temporal_order:
            return
        
        # Check temporal ordering
        if not (split_dates['train_end'] <= split_dates['val_start']):
            raise ValueError("Train data overlaps with validation data")
        
        if not (split_dates['val_end'] <= split_dates['test_start']):
            raise ValueError("Validation data overlaps with test data")
        
        logger.debug("Temporal split validation passed - no look-ahead bias detected")
    
    def scale_features(self, data_split: DataSplit) -> Tuple[DataSplit, Dict[str, Any]]:
        """
        Scale features using appropriate financial scaling methods.
        
        Args:
            data_split: Data splits to scale
            
        Returns:
            Tuple of (scaled_data_split, scalers)
        """
        if self.config.scaling_method == ScalingMethod.NONE:
            return data_split, {}
        
        logger.info(f"Scaling features using method: {self.config.scaling_method.value}")
        
        scalers = {}
        scaled_split = DataSplit(
            train_data=data_split.train_data.copy(),
            validation_data=data_split.validation_data.copy(),
            test_data=data_split.test_data.copy(),
            train_indices=data_split.train_indices,
            validation_indices=data_split.validation_indices,
            test_indices=data_split.test_indices,
            split_dates=data_split.split_dates,
            metadata=data_split.metadata.copy()
        )
        
        # Scale each feature column
        for feature in self.config.scale_features:
            if feature in data_split.train_data.columns:
                scaler = self._create_scaler()
                
                # Fit scaler on training data only
                train_values = data_split.train_data[feature].values.reshape(-1, 1)
                scaler.fit(train_values)
                
                # Transform all splits
                scaled_split.train_data[feature] = scaler.transform(
                    data_split.train_data[feature].values.reshape(-1, 1)
                ).flatten()
                
                scaled_split.validation_data[feature] = scaler.transform(
                    data_split.validation_data[feature].values.reshape(-1, 1)
                ).flatten()
                
                scaled_split.test_data[feature] = scaler.transform(
                    data_split.test_data[feature].values.reshape(-1, 1)
                ).flatten()
                
                scalers[feature] = scaler
                
                logger.debug(f"Scaled feature '{feature}' using {self.config.scaling_method.value}")
        
        # Store scaling statistics
        scaling_stats = {}
        for feature, scaler in scalers.items():
            if hasattr(scaler, 'mean_'):
                scaling_stats[feature] = {
                    'mean': scaler.mean_[0],
                    'scale': scaler.scale_[0]
                }
            elif hasattr(scaler, 'center_'):
                scaling_stats[feature] = {
                    'center': scaler.center_[0],
                    'scale': scaler.scale_[0]
                }
        
        scaled_split.metadata['scaling_stats'] = scaling_stats
        
        logger.info(f"Scaled {len(scalers)} features")
        return scaled_split, scalers
    
    def _create_scaler(self):
        """Create scaler based on configuration."""
        if self.config.scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler()
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")
    
    def calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive data quality metrics."""
        metrics = {}
        
        # Basic completeness metrics
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        metrics['completeness'] = (1 - missing_cells / total_cells) * 100
        
        # Temporal consistency
        if 'timestamp' in data.columns:
            time_diffs = data['timestamp'].diff().dropna()
            metrics['temporal_consistency'] = (time_diffs > pd.Timedelta(0)).mean() * 100
        
        # OHLCV consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            valid_ohlc = (
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            )
            metrics['ohlc_consistency'] = valid_ohlc.mean() * 100
        
        # Volume consistency
        if 'volume' in data.columns:
            metrics['volume_consistency'] = (data['volume'] >= 0).mean() * 100
        
        # Calculate overall quality score
        quality_components = [
            metrics.get('completeness', 0) * 0.3,
            metrics.get('temporal_consistency', 0) * 0.2,
            metrics.get('ohlc_consistency', 0) * 0.3,
            metrics.get('volume_consistency', 0) * 0.2
        ]
        metrics['overall_quality'] = sum(quality_components)
        
        return metrics
    
    def preprocess(self, data: pd.DataFrame) -> PreprocessingResults:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw financial data
            
        Returns:
            PreprocessingResults with all preprocessing outputs
        """
        logger.info(f"Starting complete preprocessing pipeline for {len(data)} records")
        
        warnings_list = []
        errors_list = []
        
        try:
            # Step 1: Clean data
            cleaned_data = self.clean_data(data)
            
            # Step 2: Create temporal splits
            data_splits = self.create_temporal_splits(cleaned_data)
            
            # Step 3: Scale features
            scaled_splits, scalers = self.scale_features(data_splits)
            
            # Step 4: Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(cleaned_data)
            
            # Check quality threshold
            if quality_metrics['overall_quality'] < self.config.min_data_quality_score:
                warnings_list.append(
                    f"Data quality score {quality_metrics['overall_quality']:.1f}% "
                    f"below threshold {self.config.min_data_quality_score}%"
                )
            
            # Collect preprocessing statistics
            preprocessing_stats = {
                'original_records': len(data),
                'cleaned_records': len(cleaned_data),
                'records_removed': len(data) - len(cleaned_data),
                'removal_percentage': (len(data) - len(cleaned_data)) / len(data) * 100,
                'split_info': scaled_splits.metadata,
                'scaling_method': self.config.scaling_method.value,
                'imputation_method': self.config.imputation_method.value,
                'outlier_method': self.config.outlier_method.value
            }
            
            logger.info("Preprocessing pipeline completed successfully")
            
            return PreprocessingResults(
                processed_data=cleaned_data,
                data_splits=scaled_splits,
                scalers=scalers,
                preprocessing_stats=preprocessing_stats,
                quality_metrics=quality_metrics,
                warnings=warnings_list,
                errors=errors_list
            )
        
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            errors_list.append(str(e))
            raise


def main():
    """Example usage of the data preprocessor."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 0,  # Will be calculated
        'low': 0,   # Will be calculated
        'close': 0, # Will be calculated
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Calculate OHLC relationships
    for i in range(len(sample_data)):
        open_price = sample_data.loc[i, 'open']
        daily_range = abs(np.random.randn() * 2)
        
        sample_data.loc[i, 'high'] = open_price + daily_range
        sample_data.loc[i, 'low'] = open_price - daily_range
        sample_data.loc[i, 'close'] = open_price + np.random.randn() * 1
    
    # Add some missing values and outliers for testing
    sample_data.loc[10:15, 'volume'] = np.nan
    sample_data.loc[50, 'close'] = sample_data.loc[50, 'close'] * 10  # Outlier
    
    print(f"Created sample data with {len(sample_data)} records")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    
    # Create preprocessor
    config = PreprocessingConfig(
        imputation_method=ImputationMethod.FORWARD_FILL,
        outlier_method=OutlierMethod.Z_SCORE,
        scaling_method=ScalingMethod.ROBUST,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15
    )
    
    preprocessor = FinancialDataPreprocessor(config)
    
    # Run preprocessing
    results = preprocessor.preprocess(sample_data)
    
    # Display results
    print("\n=== Preprocessing Results ===")
    print(f"Original records: {results.preprocessing_stats['original_records']}")
    print(f"Cleaned records: {results.preprocessing_stats['cleaned_records']}")
    print(f"Records removed: {results.preprocessing_stats['records_removed']}")
    print(f"Quality score: {results.quality_metrics['overall_quality']:.1f}%")
    
    print(f"\nData splits:")
    print(f"  Train: {len(results.data_splits.train_data)} records")
    print(f"  Validation: {len(results.data_splits.validation_data)} records")
    print(f"  Test: {len(results.data_splits.test_data)} records")
    
    print(f"\nScaled features: {list(results.scalers.keys())}")
    
    if results.warnings:
        print(f"\nWarnings: {results.warnings}")
    
    if results.errors:
        print(f"\nErrors: {results.errors}")


if __name__ == "__main__":
    main()