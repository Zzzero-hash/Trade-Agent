"""
Comprehensive Technical Indicators Framework

This module implements 100+ technical indicators covering:
- Momentum indicators
- Volatility indicators  
- Volume indicators
- Price pattern indicators
- Trend indicators
- Oscillators
- Statistical indicators
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    periods: List[int] = None
    smoothing_periods: List[int] = None
    enable_all: bool = True
    custom_params: Dict = None
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = [5, 10, 14, 20, 30, 50, 100, 200]
        if self.smoothing_periods is None:
            self.smoothing_periods = [3, 5, 9, 21]
        if self.custom_params is None:
            self.custom_params = {}


class BaseIndicator(ABC):
    """Base class for all technical indicators."""
    
    def __init__(self, name: str, config: IndicatorConfig):
        self.name = name
        self.config = config
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the indicator values."""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class MomentumIndicators(BaseIndicator):
    """Momentum-based technical indicators."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__("momentum", config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators."""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # RSI (Relative Strength Index)
        for period in self.config.periods:
            features[f'rsi_{period}'] = talib.RSI(data['close'], timeperiod=period)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(data['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        
        # Stochastic Oscillator
        for period in [14, 21]:
            slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'],
                                     fastk_period=period, slowk_period=3, slowd_period=3)
            features[f'stoch_k_{period}'] = slowk
            features[f'stoch_d_{period}'] = slowd
        
        # Williams %R
        for period in self.config.periods:
            features[f'williams_r_{period}'] = talib.WILLR(data['high'], data['low'], 
                                                         data['close'], timeperiod=period)
        
        # Rate of Change (ROC)
        for period in self.config.periods:
            features[f'roc_{period}'] = talib.ROC(data['close'], timeperiod=period)
        
        # Momentum
        for period in self.config.periods:
            features[f'momentum_{period}'] = talib.MOM(data['close'], timeperiod=period)
        
        # Commodity Channel Index (CCI)
        for period in self.config.periods:
            features[f'cci_{period}'] = talib.CCI(data['high'], data['low'], 
                                                data['close'], timeperiod=period)
        
        # Ultimate Oscillator
        features['ultimate_osc'] = talib.ULTOSC(data['high'], data['low'], data['close'])
        
        # Aroon Oscillator
        for period in [14, 25]:
            aroon_down, aroon_up = talib.AROON(data['high'], data['low'], timeperiod=period)
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            features[f'aroon_osc_{period}'] = aroon_up - aroon_down
        
        # Money Flow Index (MFI)
        for period in [14, 21]:
            features[f'mfi_{period}'] = talib.MFI(data['high'], data['low'], 
                                                data['close'], data['volume'], timeperiod=period)
        
        return features


class VolatilityIndicators(BaseIndicator):
    """Volatility-based technical indicators."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__("volatility", config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility indicators."""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'], timeperiod=period)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_middle_{period}'] = bb_middle
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{period}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range (ATR)
        for period in self.config.periods:
            features[f'atr_{period}'] = talib.ATR(data['high'], data['low'], 
                                                data['close'], timeperiod=period)
        
        # True Range
        features['true_range'] = talib.TRANGE(data['high'], data['low'], data['close'])
        
        # Normalized ATR
        for period in self.config.periods:
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
            features[f'natr_{period}'] = talib.NATR(data['high'], data['low'], 
                                                  data['close'], timeperiod=period)
        
        # Keltner Channels
        for period in [20, 50]:
            ema = talib.EMA(data['close'], timeperiod=period)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
            features[f'keltner_upper_{period}'] = ema + (2 * atr)
            features[f'keltner_lower_{period}'] = ema - (2 * atr)
            features[f'keltner_position_{period}'] = (data['close'] - (ema - 2*atr)) / (4*atr)
        
        # Donchian Channels
        for period in [20, 55]:
            features[f'donchian_upper_{period}'] = data['high'].rolling(period).max()
            features[f'donchian_lower_{period}'] = data['low'].rolling(period).min()
            features[f'donchian_middle_{period}'] = (features[f'donchian_upper_{period}'] + 
                                                   features[f'donchian_lower_{period}']) / 2
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = data['close'].pct_change()
            features[f'hist_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson Volatility
        for period in [10, 20, 30]:
            hl_ratio = np.log(data['high'] / data['low'])
            features[f'parkinson_vol_{period}'] = np.sqrt(hl_ratio.rolling(period).mean() / (4 * np.log(2)))
        
        # Garman-Klass Volatility
        for period in [10, 20, 30]:
            hl = np.log(data['high'] / data['low']) ** 2
            cc = np.log(data['close'] / data['close'].shift(1)) ** 2
            features[f'gk_vol_{period}'] = np.sqrt((0.5 * hl - (2*np.log(2)-1) * cc).rolling(period).mean())
        
        return features


class VolumeIndicators(BaseIndicator):
    """Volume-based technical indicators."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__("volume", config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume indicators."""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # On-Balance Volume (OBV)
        features['obv'] = talib.OBV(data['close'], data['volume'])
        
        # Volume Rate of Change
        for period in self.config.periods:
            features[f'volume_roc_{period}'] = data['volume'].pct_change(period)
        
        # Accumulation/Distribution Line
        features['ad_line'] = talib.AD(data['high'], data['low'], data['close'], data['volume'])
        
        # Chaikin Money Flow
        for period in [20, 21]:
            mfv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
            features[f'cmf_{period}'] = mfv.rolling(period).sum() / data['volume'].rolling(period).sum()
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        for period in [20, 50]:
            vwap_num = (typical_price * data['volume']).rolling(period).sum()
            vwap_den = data['volume'].rolling(period).sum()
            features[f'vwap_{period}'] = vwap_num / vwap_den
            features[f'vwap_ratio_{period}'] = data['close'] / features[f'vwap_{period}']
        
        # Price Volume Trend (PVT)
        features['pvt'] = (data['close'].pct_change() * data['volume']).cumsum()
        
        # Ease of Movement
        for period in [14, 20]:
            distance_moved = (data['high'] + data['low']) / 2 - (data['high'].shift(1) + data['low'].shift(1)) / 2
            box_height = data['volume'] / (data['high'] - data['low'])
            emv = distance_moved / box_height
            features[f'emv_{period}'] = emv.rolling(period).mean()
        
        # Volume Oscillator
        short_vol_ma = data['volume'].rolling(12).mean()
        long_vol_ma = data['volume'].rolling(26).mean()
        features['volume_oscillator'] = (short_vol_ma - long_vol_ma) / long_vol_ma * 100
        
        # Negative Volume Index (NVI) and Positive Volume Index (PVI)
        features['nvi'] = 1000  # Starting value
        features['pvi'] = 1000  # Starting value
        
        for i in range(1, len(data)):
            if data['volume'].iloc[i] < data['volume'].iloc[i-1]:
                features['nvi'].iloc[i] = features['nvi'].iloc[i-1] * (1 + data['close'].pct_change().iloc[i])
            else:
                features['nvi'].iloc[i] = features['nvi'].iloc[i-1]
                
            if data['volume'].iloc[i] > data['volume'].iloc[i-1]:
                features['pvi'].iloc[i] = features['pvi'].iloc[i-1] * (1 + data['close'].pct_change().iloc[i])
            else:
                features['pvi'].iloc[i] = features['pvi'].iloc[i-1]
        
        # Volume Moving Averages
        for period in self.config.periods:
            features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
        
        return features


class PricePatternIndicators(BaseIndicator):
    """Price pattern recognition indicators."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__("price_patterns", config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price pattern indicators."""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Candlestick Patterns (using TA-Lib)
        candlestick_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern in candlestick_patterns:
            try:
                func = getattr(talib, pattern)
                features[pattern.lower()] = func(data['open'], data['high'], 
                                                data['low'], data['close'])
            except:
                continue
        
        # Gap Analysis
        features['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
        features['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
        features['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Price Action Patterns
        features['inside_bar'] = ((data['high'] < data['high'].shift(1)) & 
                                (data['low'] > data['low'].shift(1))).astype(int)
        features['outside_bar'] = ((data['high'] > data['high'].shift(1)) & 
                                 (data['low'] < data['low'].shift(1))).astype(int)
        
        # Higher Highs and Lower Lows
        for period in [5, 10, 20]:
            features[f'higher_high_{period}'] = (data['high'] > data['high'].rolling(period).max().shift(1)).astype(int)
            features[f'lower_low_{period}'] = (data['low'] < data['low'].rolling(period).min().shift(1)).astype(int)
        
        # Support and Resistance Levels
        for period in [20, 50]:
            features[f'resistance_{period}'] = data['high'].rolling(period).max()
            features[f'support_{period}'] = data['low'].rolling(period).min()
            features[f'price_position_{period}'] = ((data['close'] - features[f'support_{period}']) / 
                                                   (features[f'resistance_{period}'] - features[f'support_{period}']))
        
        return features


class TrendIndicators(BaseIndicator):
    """Trend identification indicators."""
    
    def __init__(self, config: IndicatorConfig):
        super().__init__("trend", config)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators."""
        self._validate_data(data)
        features = pd.DataFrame(index=data.index)
        
        # Moving Averages
        for period in self.config.periods:
            features[f'sma_{period}'] = talib.SMA(data['close'], timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(data['close'], timeperiod=period)
            features[f'wma_{period}'] = talib.WMA(data['close'], timeperiod=period)
            features[f'dema_{period}'] = talib.DEMA(data['close'], timeperiod=period)
            features[f'tema_{period}'] = talib.TEMA(data['close'], timeperiod=period)
            
            # Price relative to moving averages
            features[f'price_sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
            features[f'price_ema_ratio_{period}'] = data['close'] / features[f'ema_{period}']
        
        # Moving Average Convergence/Divergence
        for fast, slow in [(12, 26), (5, 35), (8, 21)]:
            ema_fast = talib.EMA(data['close'], timeperiod=fast)
            ema_slow = talib.EMA(data['close'], timeperiod=slow)
            features[f'ma_convergence_{fast}_{slow}'] = (ema_fast - ema_slow) / ema_slow
        
        # Parabolic SAR
        features['sar'] = talib.SAR(data['high'], data['low'])
        features['sar_trend'] = (data['close'] > features['sar']).astype(int)
        
        # Average Directional Index (ADX)
        for period in [14, 21]:
            features[f'adx_{period}'] = talib.ADX(data['high'], data['low'], 
                                                data['close'], timeperiod=period)
            features[f'plus_di_{period}'] = talib.PLUS_DI(data['high'], data['low'], 
                                                         data['close'], timeperiod=period)
            features[f'minus_di_{period}'] = talib.MINUS_DI(data['high'], data['low'], 
                                                          data['close'], timeperiod=period)
        
        # Directional Movement Index
        features['dx'] = talib.DX(data['high'], data['low'], data['close'])
        
        # Trend Strength
        for period in [10, 20, 50]:
            slope = data['close'].rolling(period).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features[f'trend_slope_{period}'] = slope
            features[f'trend_strength_{period}'] = abs(slope) / data['close'].rolling(period).std()
        
        # Linear Regression
        for period in [14, 21]:
            features[f'linreg_{period}'] = talib.LINEARREG(data['close'], timeperiod=period)
            features[f'linreg_angle_{period}'] = talib.LINEARREG_ANGLE(data['close'], timeperiod=period)
            features[f'linreg_slope_{period}'] = talib.LINEARREG_SLOPE(data['close'], timeperiod=period)
        
        return features


class TechnicalIndicatorEngine:
    """Main engine for calculating all technical indicators."""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.indicators = {
            'momentum': MomentumIndicators(self.config),
            'volatility': VolatilityIndicators(self.config),
            'volume': VolumeIndicators(self.config),
            'price_patterns': PricePatternIndicators(self.config),
            'trend': TrendIndicators(self.config)
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        all_features = pd.DataFrame(index=data.index)
        
        for indicator_type, indicator in self.indicators.items():
            try:
                features = indicator.calculate(data)
                # Add prefix to avoid column name conflicts
                features.columns = [f"{indicator_type}_{col}" for col in features.columns]
                all_features = pd.concat([all_features, features], axis=1)
            except Exception as e:
                print(f"Error calculating {indicator_type} indicators: {e}")
                continue
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated."""
        # This would return a comprehensive list of all feature names
        # Implementation would depend on the specific configuration
        pass
    
    def calculate_subset(self, data: pd.DataFrame, 
                        indicator_types: List[str]) -> pd.DataFrame:
        """Calculate only specified indicator types."""
        features = pd.DataFrame(index=data.index)
        
        for indicator_type in indicator_types:
            if indicator_type in self.indicators:
                indicator_features = self.indicators[indicator_type].calculate(data)
                indicator_features.columns = [f"{indicator_type}_{col}" 
                                            for col in indicator_features.columns]
                features = pd.concat([features, indicator_features], axis=1)
        
        return features