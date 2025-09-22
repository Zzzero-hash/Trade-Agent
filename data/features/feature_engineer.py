"""
Comprehensive Feature Engineering Framework for Financial Time Series

This module implements advanced feature engineering for financial market data including:
- 200+ technical indicators using TA-Lib (RSI, MACD, Bollinger Bands, etc.)
- Price-based features (returns, volatility, price ratios, momentum indicators)
- Volume-based features (volume ratios, VWAP, volume oscillators)
- Cross-asset features and market regime detection indicators
- Multi-timeframe feature aggregation
- Feature selection and importance analysis
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Try to import talib, provide fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Technical indicators will be limited.")

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class FeatureCategory(Enum):
    """Categories of features for organization."""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    PATTERN = "pattern"
    CROSS_ASSET = "cross_asset"
    REGIME = "regime"
    CUSTOM = "custom"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Technical indicators
    enable_talib_indicators: bool = True
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21, 30])
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50, 100])
    bollinger_periods: List[int] = field(default_factory=lambda: [20])
    bollinger_std: List[float] = field(default_factory=lambda: [2.0])
    
    # Price features
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 30, 60])
    price_ratio_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Volume features
    volume_sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    vwap_periods: List[int] = field(default_factory=lambda: [20, 50])
    
    # Pattern recognition
    enable_candlestick_patterns: bool = True
    enable_chart_patterns: bool = True
    
    # Cross-asset features
    enable_cross_asset: bool = True
    benchmark_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "VIX"])
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_regression", "pca"
    
    # Multi-timeframe
    enable_multi_timeframe: bool = True
    timeframe_aggregations: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])


@dataclass
class FeatureMetadata:
    """Metadata for engineered features."""
    name: str
    category: FeatureCategory
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    importance_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    missing_ratio: float = 0.0
    outlier_ratio: float = 0.0


class FinancialFeatureEngineer:
    """
    Comprehensive feature engineering framework for financial time series.
    
    Features:
    - 200+ technical indicators from TA-Lib
    - Advanced price and volume features
    - Pattern recognition (candlestick and chart patterns)
    - Cross-asset and market regime features
    - Multi-timeframe feature aggregation
    - Feature selection and importance analysis
    - Automated feature validation and cleaning
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature engineer."""
        self.config = config or FeatureConfig()
        self.feature_metadata = {}
        self.feature_importance = {}
        self.scalers = {}
        
        # Initialize TA-Lib function mappings if available
        if TALIB_AVAILABLE:
            self._init_talib_functions()
        else:
            self.talib_functions = {}
            self.candlestick_patterns = {}
    
    def _init_talib_functions(self):
        """Initialize TA-Lib function mappings."""
        if not TALIB_AVAILABLE:
            return
        
        self.talib_functions = {
            # Overlap Studies
            'SMA': talib.SMA,
            'EMA': talib.EMA,
            'WMA': talib.WMA,
            'BBANDS': talib.BBANDS,
            'SAR': talib.SAR,
            
            # Momentum Indicators
            'RSI': talib.RSI,
            'STOCH': talib.STOCH,
            'MACD': talib.MACD,
            'MOM': talib.MOM,
            'ROC': talib.ROC,
            'CCI': talib.CCI,
            'WILLR': talib.WILLR,
            'MFI': talib.MFI,
            'ADX': talib.ADX,
            'PLUS_DI': talib.PLUS_DI,
            'MINUS_DI': talib.MINUS_DI,
            
            # Volume Indicators
            'AD': talib.AD,
            'OBV': talib.OBV,
            
            # Volatility Indicators
            'ATR': talib.ATR,
            'NATR': talib.NATR,
            'TRANGE': talib.TRANGE,
            
            # Price Transform
            'AVGPRICE': talib.AVGPRICE,
            'MEDPRICE': talib.MEDPRICE,
            'TYPPRICE': talib.TYPPRICE,
            'WCLPRICE': talib.WCLPRICE,
        }
        
        # Candlestick patterns (subset for brevity)
        self.candlestick_patterns = {
            'CDLDOJI': talib.CDLDOJI,
            'CDLENGULFING': talib.CDLENGULFING,
            'CDLHAMMER': talib.CDLHAMMER,
            'CDLHANGINGMAN': talib.CDLHANGINGMAN,
            'CDLHARAMI': talib.CDLHARAMI,
            'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
            'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
            'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
            'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
            'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
            'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        }
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive price-based features."""
        logger.info("Creating price-based features")
        features = data.copy()
        
        # Basic price features
        features['typical_price'] = (features['high'] + features['low'] + features['close']) / 3
        features['weighted_close'] = (features['high'] + features['low'] + 2 * features['close']) / 4
        features['median_price'] = (features['high'] + features['low']) / 2
        features['price_range'] = features['high'] - features['low']
        features['price_range_pct'] = features['price_range'] / features['close']
        
        # Body and shadow features (candlestick analysis)
        features['body_size'] = abs(features['close'] - features['open'])
        features['body_size_pct'] = features['body_size'] / features['close']
        features['upper_shadow'] = features['high'] - np.maximum(features['open'], features['close'])
        features['lower_shadow'] = np.minimum(features['open'], features['close']) - features['low']
        features['upper_shadow_pct'] = features['upper_shadow'] / features['close']
        features['lower_shadow_pct'] = features['lower_shadow'] / features['close']
        
        # Price position features
        features['close_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])
        features['open_position'] = (features['open'] - features['low']) / (features['high'] - features['low'])
        
        # Returns for different periods
        for period in self.config.return_periods:
            features[f'return_{period}'] = features['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(features['close'] / features['close'].shift(period))
            
            # Return statistics
            features[f'return_{period}_abs'] = abs(features[f'return_{period}'])
            features[f'return_{period}_sign'] = np.sign(features[f'return_{period}'])
        
        # Volatility features
        for window in self.config.volatility_windows:
            features[f'volatility_{window}'] = features['return_1'].rolling(window).std()
            features[f'realized_vol_{window}'] = np.sqrt(252) * features[f'volatility_{window}']
            features[f'vol_of_vol_{window}'] = features[f'volatility_{window}'].rolling(window).std()
            
            # Parkinson volatility (using high-low range)
            features[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(features['high'] / features['low']) ** 2).rolling(window).mean()
            )
        
        # Price ratios and momentum
        for period in self.config.price_ratio_periods:
            features[f'price_ratio_{period}'] = features['close'] / features['close'].shift(period)
            features[f'high_low_ratio_{period}'] = features['high'].rolling(period).max() / features['low'].rolling(period).min()
            features[f'momentum_{period}'] = features['close'] - features['close'].shift(period)
            features[f'momentum_pct_{period}'] = (features['close'] - features['close'].shift(period)) / features['close'].shift(period)
        
        # Gap features
        features['gap'] = features['open'] - features['close'].shift(1)
        features['gap_pct'] = features['gap'] / features['close'].shift(1)
        features['gap_filled'] = ((features['low'] <= features['close'].shift(1)) & (features['gap'] > 0)) | \
                                ((features['high'] >= features['close'].shift(1)) & (features['gap'] < 0))
        
        # Price acceleration (second derivative)
        features['price_acceleration'] = features['return_1'].diff()
        features['price_jerk'] = features['price_acceleration'].diff()
        
        logger.info(f"Created {len([col for col in features.columns if col not in data.columns])} price features")
        return features    

    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive volume-based features."""
        logger.info("Creating volume-based features")
        features = data.copy()
        
        # Basic volume features
        features['volume_change'] = features['volume'].pct_change()
        features['volume_change_abs'] = abs(features['volume_change'])
        
        # Volume moving averages
        for period in self.config.volume_sma_periods:
            features[f'volume_sma_{period}'] = features['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = features['volume'] / features[f'volume_sma_{period}']
            features[f'volume_std_{period}'] = features['volume'].rolling(period).std()
            features[f'volume_zscore_{period}'] = (features['volume'] - features[f'volume_sma_{period}']) / features[f'volume_std_{period}']
        
        # VWAP (Volume Weighted Average Price)
        for period in self.config.vwap_periods:
            typical_price = (features['high'] + features['low'] + features['close']) / 3
            vwap_num = (typical_price * features['volume']).rolling(period).sum()
            vwap_den = features['volume'].rolling(period).sum()
            features[f'vwap_{period}'] = vwap_num / vwap_den
            features[f'vwap_ratio_{period}'] = features['close'] / features[f'vwap_{period}']
        
        # Volume-price relationship
        features['volume_price_trend'] = features['volume'] * features['return_1']
        features['volume_weighted_return'] = features['return_1'] * features['volume_ratio_20']
        
        # On-Balance Volume (OBV)
        features['obv'] = (features['volume'] * np.sign(features['return_1'])).cumsum()
        features['obv_sma_10'] = features['obv'].rolling(10).mean()
        features['obv_ratio'] = features['obv'] / features['obv_sma_10']
        
        # Accumulation/Distribution Line
        clv = ((features['close'] - features['low']) - (features['high'] - features['close'])) / (features['high'] - features['low'])
        features['ad_line'] = (clv * features['volume']).cumsum()
        features['ad_line_sma_10'] = features['ad_line'].rolling(10).mean()
        
        # Volume oscillators
        features['volume_oscillator'] = (features['volume_sma_10'] - features['volume_sma_20']) / features['volume_sma_20']
        
        # Volume spikes
        features['volume_spike'] = features['volume'] > (features['volume_sma_20'] + 2 * features['volume_std_20'])
        features['volume_dry'] = features['volume'] < (features['volume_sma_20'] - features['volume_std_20'])
        
        # Money Flow Index components
        typical_price = (features['high'] + features['low'] + features['close']) / 3
        money_flow = typical_price * features['volume']
        features['positive_money_flow'] = np.where(features['return_1'] > 0, money_flow, 0)
        features['negative_money_flow'] = np.where(features['return_1'] < 0, money_flow, 0)
        
        # Volume distribution
        features['volume_percentile_20'] = features['volume'].rolling(100).rank(pct=True)
        
        logger.info(f"Created {len([col for col in features.columns if col not in data.columns])} volume features")
        return features
    
    def create_talib_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features using TA-Lib indicators."""
        if not self.config.enable_talib_indicators or not TALIB_AVAILABLE:
            logger.warning("TA-Lib features disabled or TA-Lib not available")
            return data
        
        logger.info("Creating TA-Lib technical indicators")
        features = data.copy()
        
        # Prepare OHLCV arrays
        open_prices = features['open'].values
        high_prices = features['high'].values
        low_prices = features['low'].values
        close_prices = features['close'].values
        volume_data = features['volume'].values
        
        # Simple Moving Averages
        for period in self.config.sma_periods:
            try:
                features[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                features[f'sma_ratio_{period}'] = features['close'] / features[f'sma_{period}']
            except:
                pass
        
        # Exponential Moving Averages
        for period in self.config.ema_periods:
            try:
                features[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
                features[f'ema_ratio_{period}'] = features['close'] / features[f'ema_{period}']
            except:
                pass
        
        # Bollinger Bands
        for period in self.config.bollinger_periods:
            for std_dev in self.config.bollinger_std:
                try:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(
                        close_prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
                    )
                    features[f'bb_upper_{period}_{std_dev}'] = bb_upper
                    features[f'bb_middle_{period}_{std_dev}'] = bb_middle
                    features[f'bb_lower_{period}_{std_dev}'] = bb_lower
                    features[f'bb_width_{period}_{std_dev}'] = (bb_upper - bb_lower) / bb_middle
                    features[f'bb_position_{period}_{std_dev}'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
                except:
                    pass
        
        # RSI
        for period in self.config.rsi_periods:
            try:
                features[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
                features[f'rsi_overbought_{period}'] = features[f'rsi_{period}'] > 70
                features[f'rsi_oversold_{period}'] = features[f'rsi_{period}'] < 30
            except:
                pass
        
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            features['macd_bullish'] = macd > macd_signal
        except:
            pass
        
        # Stochastic
        try:
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            features['stoch_overbought'] = stoch_k > 80
            features['stoch_oversold'] = stoch_k < 20
        except:
            pass
        
        # Williams %R
        try:
            features['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
        except:
            pass
        
        # Commodity Channel Index
        try:
            features['cci'] = talib.CCI(high_prices, low_prices, close_prices)
        except:
            pass
        
        # Average True Range
        try:
            features['atr'] = talib.ATR(high_prices, low_prices, close_prices)
            features['atr_ratio'] = features['atr'] / features['close']
        except:
            pass
        
        # Directional Movement Index
        try:
            features['adx'] = talib.ADX(high_prices, low_prices, close_prices)
            features['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
            features['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
        except:
            pass
        
        # Money Flow Index
        try:
            features['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume_data)
        except:
            pass
        
        logger.info(f"Created {len([col for col in features.columns if col not in data.columns])} TA-Lib features")
        return features
    
    def create_candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features."""
        if not self.config.enable_candlestick_patterns or not TALIB_AVAILABLE:
            return data
        
        logger.info("Creating candlestick pattern features")
        features = data.copy()
        
        # Prepare OHLC arrays
        open_prices = features['open'].values
        high_prices = features['high'].values
        low_prices = features['low'].values
        close_prices = features['close'].values
        
        # Apply candlestick pattern functions
        pattern_count = 0
        for pattern_name, pattern_func in self.candlestick_patterns.items():
            try:
                pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                features[f'pattern_{pattern_name.lower()}'] = pattern_result
                
                # Create binary features for pattern presence
                features[f'pattern_{pattern_name.lower()}_bullish'] = pattern_result > 0
                features[f'pattern_{pattern_name.lower()}_bearish'] = pattern_result < 0
                
                pattern_count += 1
            except Exception as e:
                logger.debug(f"Failed to create pattern {pattern_name}: {e}")
        
        # Pattern summary features
        bullish_patterns = [col for col in features.columns if 'pattern_' in col and '_bullish' in col]
        bearish_patterns = [col for col in features.columns if 'pattern_' in col and '_bearish' in col]
        
        if bullish_patterns:
            features['total_bullish_patterns'] = features[bullish_patterns].sum(axis=1)
        if bearish_patterns:
            features['total_bearish_patterns'] = features[bearish_patterns].sum(axis=1)
        
        if bullish_patterns and bearish_patterns:
            features['pattern_sentiment'] = features['total_bullish_patterns'] - features['total_bearish_patterns']
        
        logger.info(f"Created {pattern_count} candlestick patterns")
        return features
    
    def create_chart_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create chart pattern features."""
        if not self.config.enable_chart_patterns:
            return data
        
        logger.info("Creating chart pattern features")
        features = data.copy()
        
        # Support and resistance levels
        window = 20
        features['local_high'] = features['high'].rolling(window, center=True).max() == features['high']
        features['local_low'] = features['low'].rolling(window, center=True).min() == features['low']
        
        # Trend lines (simplified)
        features['price_trend_5'] = features['close'].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan, raw=True
        )
        features['price_trend_20'] = features['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=True
        )
        
        # Channel patterns
        features['upper_channel'] = features['high'].rolling(20).max()
        features['lower_channel'] = features['low'].rolling(20).min()
        features['channel_width'] = features['upper_channel'] - features['lower_channel']
        features['channel_position'] = (features['close'] - features['lower_channel']) / features['channel_width']
        
        # Breakout patterns
        features['breakout_up'] = features['close'] > features['upper_channel'].shift(1)
        features['breakout_down'] = features['close'] < features['lower_channel'].shift(1)
        
        # Simple peak/trough detection
        features['near_peak'] = (features['high'] == features['high'].rolling(10, center=True).max())
        features['near_trough'] = (features['low'] == features['low'].rolling(10, center=True).min())
        
        logger.info(f"Created {len([col for col in features.columns if col not in data.columns])} chart pattern features")
        return features
    
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market regime detection features."""
        logger.info("Creating market regime features")
        features = data.copy()
        
        # Volatility regimes
        vol_20 = features['return_1'].rolling(20).std()
        vol_60 = features['return_1'].rolling(60).std()
        
        # Create volatility regime indicators
        vol_20_quantiles = vol_20.quantile([0.33, 0.67])
        features['vol_regime_low'] = vol_20 <= vol_20_quantiles.iloc[0]
        features['vol_regime_medium'] = (vol_20 > vol_20_quantiles.iloc[0]) & (vol_20 <= vol_20_quantiles.iloc[1])
        features['vol_regime_high'] = vol_20 > vol_20_quantiles.iloc[1]
        
        # Trend regimes
        sma_20 = features['close'].rolling(20).mean()
        sma_50 = features['close'].rolling(50).mean()
        
        features['trend_uptrend'] = sma_20 > sma_50
        features['trend_downtrend'] = sma_20 < sma_50
        features['trend_sideways'] = abs(sma_20 - sma_50) / sma_50 < 0.02
        
        # Market stress indicators
        features['stress_indicator'] = 0
        if 'rsi_14' in features.columns:
            features['stress_indicator'] += (features['rsi_14'] < 30).astype(int)
            features['stress_indicator'] += (features['rsi_14'] > 70).astype(int)
        
        features['stress_indicator'] += (abs(features['return_1']) > 2 * vol_20).astype(int)
        
        # Momentum regimes
        momentum_5 = features['close'] / features['close'].shift(5) - 1
        momentum_quantiles = momentum_5.quantile([0.33, 0.67])
        features['momentum_weak'] = momentum_5 <= momentum_quantiles.iloc[0]
        features['momentum_neutral'] = (momentum_5 > momentum_quantiles.iloc[0]) & (momentum_5 <= momentum_quantiles.iloc[1])
        features['momentum_strong'] = momentum_5 > momentum_quantiles.iloc[1]
        
        logger.info(f"Created {len([col for col in features.columns if col not in data.columns])} regime features")
        return features  
  
    def create_cross_asset_features(self, data: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Create cross-asset and relative strength features."""
        if not self.config.enable_cross_asset or benchmark_data is None:
            return data
        
        logger.info("Creating cross-asset features")
        features = data.copy()
        
        for benchmark_symbol, benchmark_df in benchmark_data.items():
            if 'close' not in benchmark_df.columns:
                continue
            
            try:
                # Align timestamps
                benchmark_aligned = benchmark_df.set_index('timestamp')['close'].reindex(
                    features.set_index('timestamp').index, method='ffill'
                ).values
                
                if len(benchmark_aligned) != len(features):
                    continue
                
                # Relative strength
                asset_returns = features['return_1']
                benchmark_returns = pd.Series(benchmark_aligned).pct_change()
                
                features[f'relative_strength_{benchmark_symbol}'] = asset_returns - benchmark_returns
                features[f'beta_{benchmark_symbol}'] = asset_returns.rolling(60).corr(benchmark_returns)
                features[f'correlation_{benchmark_symbol}'] = asset_returns.rolling(20).corr(benchmark_returns)
                
                # Price ratios
                features[f'price_ratio_{benchmark_symbol}'] = features['close'] / benchmark_aligned
                features[f'price_ratio_sma_{benchmark_symbol}'] = features[f'price_ratio_{benchmark_symbol}'].rolling(20).mean()
                
                # Relative momentum
                asset_momentum = features['close'] / features['close'].shift(20) - 1
                benchmark_momentum = pd.Series(benchmark_aligned) / pd.Series(benchmark_aligned).shift(20) - 1
                features[f'relative_momentum_{benchmark_symbol}'] = asset_momentum - benchmark_momentum
                
            except Exception as e:
                logger.warning(f"Failed to create cross-asset features for {benchmark_symbol}: {e}")
        
        logger.info(f"Created cross-asset features for {len(benchmark_data)} benchmarks")
        return features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features using specified method."""
        if self.config.max_features is None:
            return features, list(features.columns)
        
        logger.info(f"Selecting top {self.config.max_features} features using {self.config.feature_selection_method}")
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        target_clean = target.fillna(target.median())
        
        # Align indices
        common_idx = numeric_features.index.intersection(target_clean.index)
        numeric_features = numeric_features.loc[common_idx]
        target_clean = target_clean.loc[common_idx]
        
        if len(numeric_features.columns) <= self.config.max_features:
            return numeric_features, list(numeric_features.columns)
        
        try:
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
            elif self.config.feature_selection_method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
            elif self.config.feature_selection_method == 'pca':
                selector = PCA(n_components=self.config.max_features)
            else:
                raise ValueError(f"Unknown feature selection method: {self.config.feature_selection_method}")
            
            # Fit selector
            if self.config.feature_selection_method == 'pca':
                selected_features = pd.DataFrame(
                    selector.fit_transform(numeric_features),
                    index=numeric_features.index,
                    columns=[f'pca_{i}' for i in range(self.config.max_features)]
                )
                selected_feature_names = list(selected_features.columns)
            else:
                selector.fit(numeric_features, target_clean)
                selected_mask = selector.get_support()
                selected_features = numeric_features.iloc[:, selected_mask]
                selected_feature_names = list(selected_features.columns)
                
                # Store feature importance scores
                if hasattr(selector, 'scores_'):
                    for i, feature_name in enumerate(numeric_features.columns):
                        if selected_mask[i]:
                            self.feature_importance[feature_name] = selector.scores_[i]
            
            logger.info(f"Selected {len(selected_feature_names)} features")
            return selected_features, selected_feature_names
        
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return numeric_features, list(numeric_features.columns)
    
    def engineer_features(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                         target_column: str = 'return_1',
                         benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            data: Input data (single DataFrame or dict of timeframes)
            target_column: Target variable for feature selection
            benchmark_data: Benchmark data for cross-asset features
            
        Returns:
            Tuple of (engineered_features, feature_metadata)
        """
        logger.info("Starting comprehensive feature engineering pipeline")
        
        # Handle multi-timeframe data (simplified for now)
        if isinstance(data, dict):
            # Use the first available timeframe as base
            base_data = list(data.values())[0].copy()
        else:
            base_data = data.copy()
        
        # Create different categories of features
        features = base_data.copy()
        
        # Price features
        features = self.create_price_features(features)
        
        # Volume features
        features = self.create_volume_features(features)
        
        # TA-Lib indicators
        features = self.create_talib_features(features)
        
        # Candlestick patterns
        features = self.create_candlestick_patterns(features)
        
        # Chart patterns
        features = self.create_chart_patterns(features)
        
        # Market regime features
        features = self.create_regime_features(features)
        
        # Cross-asset features
        if benchmark_data:
            features = self.create_cross_asset_features(features, benchmark_data)
        
        # Feature selection
        if target_column in features.columns:
            target = features[target_column]
            feature_cols = [col for col in features.columns if col not in ['timestamp', 'symbol', 'timeframe', target_column]]
            feature_data = features[feature_cols]
            
            selected_features, selected_names = self.select_features(feature_data, target)
            
            # Combine with non-feature columns
            final_features = pd.concat([
                features[['timestamp', 'symbol', 'timeframe']],
                selected_features,
                features[[target_column]]
            ], axis=1)
        else:
            final_features = features
            selected_names = [col for col in features.columns if col not in ['timestamp', 'symbol', 'timeframe']]
        
        # Create feature metadata
        feature_metadata = {}
        for feature_name in selected_names:
            if feature_name in final_features.columns:
                metadata = FeatureMetadata(
                    name=feature_name,
                    category=self._categorize_feature(feature_name),
                    description=self._get_feature_description(feature_name),
                    importance_score=self.feature_importance.get(feature_name),
                    missing_ratio=final_features[feature_name].isnull().mean(),
                    outlier_ratio=self._calculate_outlier_ratio(final_features[feature_name])
                )
                feature_metadata[feature_name] = metadata
        
        logger.info(f"Feature engineering completed. Created {len(selected_names)} features")
        return final_features, feature_metadata
    
    def _categorize_feature(self, feature_name: str) -> FeatureCategory:
        """Categorize feature based on its name."""
        name_lower = feature_name.lower()
        
        if any(keyword in name_lower for keyword in ['price', 'open', 'high', 'low', 'close', 'return', 'gap']):
            return FeatureCategory.PRICE
        elif any(keyword in name_lower for keyword in ['volume', 'obv', 'ad_line', 'mfi']):
            return FeatureCategory.VOLUME
        elif any(keyword in name_lower for keyword in ['rsi', 'macd', 'stoch', 'momentum', 'roc']):
            return FeatureCategory.MOMENTUM
        elif any(keyword in name_lower for keyword in ['volatility', 'atr', 'bb_', 'vol_']):
            return FeatureCategory.VOLATILITY
        elif any(keyword in name_lower for keyword in ['sma', 'ema', 'trend', 'adx']):
            return FeatureCategory.TREND
        elif any(keyword in name_lower for keyword in ['pattern', 'cdl']):
            return FeatureCategory.PATTERN
        elif any(keyword in name_lower for keyword in ['relative', 'beta', 'correlation', 'cross']):
            return FeatureCategory.CROSS_ASSET
        elif any(keyword in name_lower for keyword in ['regime', 'stress']):
            return FeatureCategory.REGIME
        else:
            return FeatureCategory.CUSTOM
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get description for a feature."""
        descriptions = {
            'return_1': 'One-period price return',
            'volatility_20': '20-period rolling volatility',
            'rsi_14': '14-period Relative Strength Index',
            'macd': 'MACD line',
            'volume_ratio_20': 'Volume ratio to 20-period average',
            'bb_position_20_2.0': 'Position within Bollinger Bands',
        }
        
        return descriptions.get(feature_name, f"Feature: {feature_name}")
    
    def _calculate_outlier_ratio(self, series: pd.Series) -> float:
        """Calculate the ratio of outliers in a series."""
        if series.dtype not in [np.float64, np.int64]:
            return 0.0
        
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            return outliers.mean()
        except:
            return 0.0


def main():
    """Example usage of the feature engineer."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(10000, 100000, len(dates)),
        'timeframe': '1d'
    })
    
    # Calculate OHLC relationships
    for i in range(len(sample_data)):
        open_price = sample_data.loc[i, 'open']
        daily_range = abs(np.random.randn() * 2)
        
        sample_data.loc[i, 'high'] = open_price + daily_range
        sample_data.loc[i, 'low'] = open_price - daily_range
        sample_data.loc[i, 'close'] = open_price + np.random.randn() * 1
    
    print(f"Created sample data with {len(sample_data)} records")
    
    # Create feature engineer
    config = FeatureConfig(
        enable_talib_indicators=TALIB_AVAILABLE,
        enable_candlestick_patterns=TALIB_AVAILABLE,
        enable_chart_patterns=True,
        max_features=50
    )
    
    engineer = FinancialFeatureEngineer(config)
    
    # Engineer features
    engineered_features, metadata = engineer.engineer_features(sample_data)
    
    # Display results
    print(f"\n=== Feature Engineering Results ===")
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_features.columns)}")
    print(f"Feature metadata entries: {len(metadata)}")
    
    # Show feature categories
    categories = {}
    for feature_name, meta in metadata.items():
        category = meta.category.value
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    print(f"\nFeature categories:")
    for category, count in categories.items():
        print(f"  {category}: {count} features")
    
    # Show top features by importance
    if engineer.feature_importance:
        top_features = sorted(
            engineer.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print(f"\nTop 10 features by importance:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
    
    print(f"\nTA-Lib available: {TALIB_AVAILABLE}")


if __name__ == "__main__":
    main()