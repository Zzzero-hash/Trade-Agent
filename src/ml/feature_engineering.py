"""Abstract base classes for feature engineering"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "FeatureTransformer":
        """Fit the transformer to the data"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform the data using fitted parameters"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform the data in one step"""
        return self.fit(data).transform(data)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of the generated features"""
        pass


class FeatureEngineer:
    """Main feature engineering pipeline"""

    def __init__(self):
        self.transformers: List[FeatureTransformer] = []
        self.feature_names: List[str] = []
        self.is_fitted = False

    def add_transformer(self, transformer: FeatureTransformer) -> "FeatureEngineer":
        """Add a feature transformer to the pipeline"""
        self.transformers.append(transformer)
        return self

    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """Fit all transformers to the data"""
        self.feature_names = []

        for transformer in self.transformers:
            transformer.fit(data)
            self.feature_names.extend(transformer.get_feature_names())

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using all fitted transformers"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        features = []
        for transformer in self.transformers:
            feature_array = transformer.transform(data)
            features.append(feature_array)

        if features:
            return np.concatenate(features, axis=1)
        else:
            return np.array([]).reshape(len(data), 0)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform data in one step"""
        return self.fit(data).transform(data)

    def get_feature_names(self) -> List[str]:
        """Get names of all generated features"""
        return self.feature_names.copy()


class TechnicalIndicators(FeatureTransformer):
    """Technical indicators feature transformer with complete indicator set"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("technical_indicators", config)
        default_indicators = [
            "sma_20", "ema_12", "rsi_14", "macd", "bollinger_bands",
            "atr_14", "obv", "vwap"
        ]
        self.indicators = self.config.get("indicators", default_indicators)

    def fit(self, data: pd.DataFrame) -> "TechnicalIndicators":
        """Fit technical indicators (no parameters to learn)"""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate technical indicators"""
        if not self.is_fitted:
            raise ValueError("TechnicalIndicators must be fitted before transform")

        features = []

        # Simple Moving Average
        if "sma_20" in self.indicators:
            sma_20 = data["close"].rolling(window=20).mean()
            features.append(sma_20.values.reshape(-1, 1))

        # Exponential Moving Average
        if "ema_12" in self.indicators:
            ema_12 = data["close"].ewm(span=12).mean()
            features.append(ema_12.values.reshape(-1, 1))

        # RSI
        if "rsi_14" in self.indicators:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.values.reshape(-1, 1))

        # MACD (Moving Average Convergence Divergence)
        if "macd" in self.indicators:
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            features.extend([
                macd_line.values.reshape(-1, 1),
                signal_line.values.reshape(-1, 1),
                histogram.values.reshape(-1, 1)
            ])

        # Bollinger Bands
        if "bollinger_bands" in self.indicators:
            sma_20 = data["close"].rolling(window=20).mean()
            std_20 = data["close"].rolling(window=20).std()
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            bb_width = (upper_band - lower_band) / sma_20
            bb_position = (data["close"] - lower_band) / (upper_band - lower_band)
            features.extend([
                upper_band.values.reshape(-1, 1),
                lower_band.values.reshape(-1, 1),
                bb_width.values.reshape(-1, 1),
                bb_position.values.reshape(-1, 1)
            ])

        # Average True Range (ATR)
        if "atr_14" in self.indicators:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            features.append(atr.values.reshape(-1, 1))

        # On-Balance Volume (OBV)
        if "obv" in self.indicators:
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data["volume"].iloc[0]
            for i in range(1, len(data)):
                if data["close"].iloc[i] > data["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data["volume"].iloc[i]
                elif data["close"].iloc[i] < data["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            features.append(obv.values.reshape(-1, 1))

        # Volume Weighted Average Price (VWAP)
        if "vwap" in self.indicators:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
            features.append(vwap.values.reshape(-1, 1))

        if features:
            result = np.concatenate(features, axis=1)
            # Fill NaN values with 0
            return np.nan_to_num(result, nan=0.0)
        
        return np.array([]).reshape(len(data), 0)

    def get_feature_names(self) -> List[str]:
        """Get feature names for technical indicators"""
        names = []
        if "sma_20" in self.indicators:
            names.append("sma_20")
        if "ema_12" in self.indicators:
            names.append("ema_12")
        if "rsi_14" in self.indicators:
            names.append("rsi_14")
        if "macd" in self.indicators:
            names.extend(["macd_line", "macd_signal", "macd_histogram"])
        if "bollinger_bands" in self.indicators:
            names.extend(["bb_upper", "bb_lower", "bb_width", "bb_position"])
        if "atr_14" in self.indicators:
            names.append("atr_14")
        if "obv" in self.indicators:
            names.append("obv")
        if "vwap" in self.indicators:
            names.append("vwap")
        return names


class WaveletTransform(FeatureTransformer):
    """Wavelet transform for multi-resolution analysis"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("wavelet_transform", config)
        self.wavelet = self.config.get("wavelet", "db4")
        self.levels = self.config.get("levels", 3)
        self.mode = self.config.get("mode", "symmetric")

    def fit(self, data: pd.DataFrame) -> "WaveletTransform":
        """Fit wavelet transform (no parameters to learn)"""
        try:
            import pywt
            self.pywt = pywt
        except ImportError:
            raise ImportError("PyWavelets is required for WaveletTransform. "
                            "Install with: pip install PyWavelets")
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Apply wavelet transform to price data"""
        if not self.is_fitted:
            raise ValueError("WaveletTransform must be fitted before transform")

        features = []
        price_series = data["close"].values

        # Apply wavelet decomposition
        coeffs = self.pywt.wavedec(price_series, self.wavelet, 
                                  level=self.levels, mode=self.mode)

        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            # Statistical features for each level
            if len(coeff) > 0:
                # Pad or truncate to match original length
                if len(coeff) < len(price_series):
                    coeff_padded = np.pad(coeff, 
                                        (0, len(price_series) - len(coeff)), 
                                        mode='constant')
                else:
                    coeff_padded = coeff[:len(price_series)]
                
                features.append(coeff_padded.reshape(-1, 1))

        if features:
            result = np.concatenate(features, axis=1)
            return np.nan_to_num(result, nan=0.0)
        
        return np.array([]).reshape(len(data), 0)

    def get_feature_names(self) -> List[str]:
        """Get feature names for wavelet coefficients"""
        names = []
        for i in range(self.levels + 1):
            if i == 0:
                names.append(f"wavelet_approx_{i}")
            else:
                names.append(f"wavelet_detail_{i}")
        return names


class FourierFeatures(FeatureTransformer):
    """Fourier transform features for frequency domain analysis"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("fourier_features", config)
        self.n_components = self.config.get("n_components", 10)
        self.window_size = self.config.get("window_size", 50)

    def fit(self, data: pd.DataFrame) -> "FourierFeatures":
        """Fit Fourier features (no parameters to learn)"""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Apply Fourier transform to extract frequency features"""
        if not self.is_fitted:
            raise ValueError("FourierFeatures must be fitted before transform")

        price_series = data["close"].values
        
        # Handle empty data
        if len(price_series) == 0:
            return np.array([]).reshape(0, 2 * self.n_components)
        
        features = []

        # Rolling Fourier transform
        for i in range(len(price_series)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = price_series[start_idx:i+1]
            
            if len(window_data) >= 2:
                # Apply FFT
                fft_values = np.fft.fft(window_data)
                
                # Extract magnitude and phase for top components
                magnitudes = np.abs(fft_values)
                phases = np.angle(fft_values)
                
                # Take top n_components
                n_comp = min(self.n_components, len(magnitudes) // 2)
                
                # Features: magnitude and phase of top components
                mag_features = magnitudes[1:n_comp+1]  # Skip DC component
                phase_features = phases[1:n_comp+1]
                
                # Pad if necessary
                if len(mag_features) < self.n_components:
                    mag_features = np.pad(mag_features, 
                                        (0, self.n_components - len(mag_features)))
                    phase_features = np.pad(phase_features, 
                                          (0, self.n_components - len(phase_features)))
                
                row_features = np.concatenate([mag_features, phase_features])
            else:
                # Not enough data, use zeros
                row_features = np.zeros(2 * self.n_components)
            
            features.append(row_features)

        result = np.array(features)
        return np.nan_to_num(result, nan=0.0)

    def get_feature_names(self) -> List[str]:
        """Get feature names for Fourier components"""
        names = []
        for i in range(self.n_components):
            names.append(f"fft_magnitude_{i+1}")
        for i in range(self.n_components):
            names.append(f"fft_phase_{i+1}")
        return names


class FractalFeatures(FeatureTransformer):
    """Fractal dimension and Hurst exponent features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("fractal_features", config)
        self.window_size = self.config.get("window_size", 50)
        self.max_lag = self.config.get("max_lag", 20)

    def fit(self, data: pd.DataFrame) -> "FractalFeatures":
        """Fit fractal features (no parameters to learn)"""
        self.is_fitted = True
        return self

    def _hurst_exponent(self, time_series: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(time_series) < 10:
            return 0.5  # Default value for insufficient data
        
        # Calculate log returns
        returns = np.diff(np.log(time_series + 1e-8))  # Add small value to avoid log(0)
        
        # Range of lags to test
        lags = range(2, min(self.max_lag, len(returns) // 2))
        rs_values = []
        
        for lag in lags:
            # Split returns into chunks
            n_chunks = len(returns) // lag
            if n_chunks == 0:
                continue
                
            chunks = returns[:n_chunks * lag].reshape(n_chunks, lag)
            
            # Calculate R/S for each chunk
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) > 1:
                    # Mean-adjusted series
                    mean_adj = chunk - np.mean(chunk)
                    
                    # Cumulative sum
                    cum_sum = np.cumsum(mean_adj)
                    
                    # Range
                    R = np.max(cum_sum) - np.min(cum_sum)
                    
                    # Standard deviation
                    S = np.std(chunk)
                    
                    # R/S ratio
                    if S > 0:
                        rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression to find Hurst exponent
        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Simple linear regression
        n = len(log_lags)
        sum_x = np.sum(log_lags)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_lags * log_rs)
        sum_x2 = np.sum(log_lags ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5
        
        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Clamp to reasonable range
        return np.clip(hurst, 0.0, 1.0)

    def _fractal_dimension(self, time_series: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        if len(time_series) < 4:
            return 1.0
        
        # Normalize the time series
        normalized = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series) + 1e-8)
        
        # Box sizes (powers of 2)
        box_sizes = [2**i for i in range(1, int(np.log2(len(normalized))) - 1)]
        if not box_sizes:
            return 1.0
        
        counts = []
        for box_size in box_sizes:
            # Count boxes needed to cover the curve
            n_boxes_x = len(normalized) // box_size
            n_boxes_y = int(1.0 / (1.0 / box_size)) if box_size > 0 else 1
            
            if n_boxes_x == 0:
                continue
            
            # Grid-based counting
            boxes_needed = 0
            for i in range(n_boxes_x):
                start_idx = i * box_size
                end_idx = min((i + 1) * box_size, len(normalized))
                segment = normalized[start_idx:end_idx]
                
                if len(segment) > 0:
                    min_val = np.min(segment)
                    max_val = np.max(segment)
                    boxes_in_segment = int((max_val - min_val) * n_boxes_y) + 1
                    boxes_needed += boxes_in_segment
            
            counts.append(boxes_needed)
        
        if len(counts) < 2:
            return 1.0
        
        # Linear regression on log-log plot
        log_sizes = np.log(box_sizes[:len(counts)])
        log_counts = np.log(counts)
        
        # Simple linear regression
        n = len(log_sizes)
        sum_x = np.sum(log_sizes)
        sum_y = np.sum(log_counts)
        sum_xy = np.sum(log_sizes * log_counts)
        sum_x2 = np.sum(log_sizes ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 1.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Fractal dimension is negative slope
        fractal_dim = -slope
        
        # Clamp to reasonable range
        return np.clip(fractal_dim, 1.0, 2.0)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate fractal features using rolling windows"""
        if not self.is_fitted:
            raise ValueError("FractalFeatures must be fitted before transform")

        price_series = data["close"].values
        features = []

        for i in range(len(price_series)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = price_series[start_idx:i+1]
            
            if len(window_data) >= 10:
                hurst = self._hurst_exponent(window_data)
                fractal_dim = self._fractal_dimension(window_data)
            else:
                hurst = 0.5  # Default for insufficient data
                fractal_dim = 1.0
            
            features.append([hurst, fractal_dim])

        result = np.array(features)
        return np.nan_to_num(result, nan=0.0)

    def get_feature_names(self) -> List[str]:
        """Get feature names for fractal features"""
        return ["hurst_exponent", "fractal_dimension"]


class CrossAssetFeatures(FeatureTransformer):
    """Cross-asset correlation and relationship features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("cross_asset_features", config)
        self.window_size = self.config.get("window_size", 30)
        self.correlation_assets = self.config.get("correlation_assets", [])

    def fit(self, data: pd.DataFrame) -> "CrossAssetFeatures":
        """Fit cross-asset features"""
        # Store reference data for correlation calculation
        self.reference_data = data.copy()
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate cross-asset correlation features"""
        if not self.is_fitted:
            raise ValueError("CrossAssetFeatures must be fitted before transform")

        features = []
        
        # If no external assets specified, use internal relationships
        if not self.correlation_assets:
            # Calculate rolling correlations between OHLC components
            for i in range(len(data)):
                start_idx = max(0, i - self.window_size + 1)
                window_data = data.iloc[start_idx:i+1]
                
                if len(window_data) >= 2:
                    # Price-volume correlation
                    price_vol_corr = window_data["close"].corr(window_data["volume"])
                    
                    # High-low range vs volume correlation
                    hl_range = window_data["high"] - window_data["low"]
                    range_vol_corr = hl_range.corr(window_data["volume"])
                    
                    # Returns autocorrelation
                    returns = window_data["close"].pct_change().dropna()
                    if len(returns) >= 2:
                        returns_autocorr = returns.autocorr(lag=1)
                    else:
                        returns_autocorr = 0.0
                    
                    # Volume momentum
                    volume_momentum = (window_data["volume"].iloc[-1] / 
                                     window_data["volume"].mean() - 1)
                    
                    row_features = [
                        price_vol_corr if not np.isnan(price_vol_corr) else 0.0,
                        range_vol_corr if not np.isnan(range_vol_corr) else 0.0,
                        returns_autocorr if not np.isnan(returns_autocorr) else 0.0,
                        volume_momentum if not np.isnan(volume_momentum) else 0.0
                    ]
                else:
                    row_features = [0.0, 0.0, 0.0, 0.0]
                
                features.append(row_features)
        else:
            # Placeholder for external asset correlations
            # In a real implementation, this would fetch data from other assets
            for i in range(len(data)):
                # Mock correlation features
                row_features = [0.0] * len(self.correlation_assets)
                features.append(row_features)

        result = np.array(features)
        return np.nan_to_num(result, nan=0.0)

    def get_feature_names(self) -> List[str]:
        """Get feature names for cross-asset features"""
        if not self.correlation_assets:
            return [
                "price_volume_correlation",
                "range_volume_correlation", 
                "returns_autocorrelation",
                "volume_momentum"
            ]
        else:
            return [f"correlation_{asset}" for asset in self.correlation_assets]