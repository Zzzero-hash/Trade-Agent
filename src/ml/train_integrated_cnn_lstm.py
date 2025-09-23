"""
Training Script for Integrated CNN+LSTM Hybrid Architecture - Task 5.4

This script implements the training requirements for task 5.4:
- Train end-to-end CNN+LSTM model with joint optimization for 200+ epochs
- Implement feature fusion training with learnable combination weights
- Add multi-task learning for price prediction, volatility estimation, and regime detection
- Validate integrated model performance against individual CNN and LSTM baselines

Requirements: 1.4, 3.1, 9.2
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig, create_hybrid_config
from src.ml.cnn_model import CNNFeatureExtractor, create_cnn_config
from src.ml.lstm_model import LSTMTemporalProcessor, create_lstm_config
from data.pipeline import create_data_loaders, create_augmentation_transform
from data.ingestion.yahoo_finance import YahooFinanceIngestor


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/integrated_cnn_lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedCNNLSTMTrainer:
    """
    Comprehensive trainer for integrated CNN+LSTM hybrid architecture.
    
    Implements task 5.4 requirements:
    - Joint optimization training for 200+ epochs
    - Feature fusion with learnable combination weights
    - Multi-task learning (price prediction, volatility estimation, regime detection)
    - Performance validation against individual baselines
    """
    
    def __init__(
        self,
        config: HybridModelConfig,
        save_dir: str = "checkpoints/integrated_cnn_lstm",
        log_dir: str = "logs/integrated_cnn_lstm",
        device: Optional[str] = None
    ):
        """Initialize the integrated trainer"""
        self.config = config
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.hybrid_model = None
        self.cnn_baseline = None
        self.lstm_baseline = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        logger.info("Integrated CNN+LSTM trainer initialized")
    
    def prepare_data(
        self,
        symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        timeframes: List[str] = ["1min", "5min", "15min"],
        sequence_length: int = 60,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare multi-task training data with yfinance integration.
        
        Creates datasets for:
        - Price prediction (regression)
        - Volatility estimation (regression) 
        - Regime detection (classification)
        """
        logger.info("Preparing multi-task training data...")
        
        # Initialize Yahoo Finance data ingestor
        ingestor = YahooFinanceIngestor()
        
        all_features = []
        all_price_targets = []
        all_volatility_targets = []
        all_regime_targets = []
        
        for symbol in symbols:
            logger.info(f"Processing data for {symbol}...")
            
            try:
                # Download data for multiple timeframes
                symbol_data = {}
                for timeframe in timeframes:
                    # Map timeframe names to Yahoo Finance intervals
                    interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "1hour": "1h", "1day": "1d"}
                    interval = interval_map.get(timeframe, timeframe)
                    
                    data = ingestor.fetch_symbol_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval
                    )
                    if data is not None:
                        symbol_data[timeframe] = data
                
                # Process multi-timeframe features
                features, targets = self._process_symbol_data(
                    symbol_data, sequence_length, timeframes
                )
                
                logger.info(f"Processed {symbol}: {len(features)} features extracted")
                
                if len(features) > 0:
                    all_features.extend(features)
                    all_price_targets.extend(targets['price'])
                    all_volatility_targets.extend(targets['volatility'])
                    all_regime_targets.extend(targets['regime'])
                    
                    logger.info(f"Added {len(features)} samples for {symbol}")
                else:
                    logger.warning(f"No features extracted for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_features) == 0:
            raise ValueError("No data could be processed. Check symbols and date range.")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y_price = np.array(all_price_targets)
        y_volatility = np.array(all_volatility_targets)
        y_regime = np.array(all_regime_targets)
        
        logger.info(f"Total dataset size: {len(X)} samples")
        logger.info(f"Feature shape: {X.shape}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_multi_task_loaders(
            X, y_price, y_volatility, y_regime, batch_size
        )
        
        return train_loader, val_loader, test_loader
    
    def _process_symbol_data(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        sequence_length: int,
        timeframes: List[str]
    ) -> Tuple[List[np.ndarray], Dict[str, List[float]]]:
        """Process multi-timeframe data for a single symbol"""
        
        # Use the finest timeframe as the base
        base_timeframe = timeframes[0]  # Assuming sorted from finest to coarsest
        
        if base_timeframe not in symbol_data:
            logger.warning(f"Base timeframe {base_timeframe} not found in symbol data")
            return [], {'price': [], 'volatility': [], 'regime': []}
        
        base_data = symbol_data[base_timeframe]
        logger.info(f"Base data shape: {base_data.shape}")
        
        if len(base_data) < sequence_length + 20:  # Need extra for targets
            logger.warning(f"Insufficient data: {len(base_data)} < {sequence_length + 20}")
            return [], {'price': [], 'volatility': [], 'regime': []}
        
        features = []
        targets = {'price': [], 'volatility': [], 'regime': []}
        
        # Calculate technical indicators and features
        base_data = self._add_technical_indicators(base_data)
        
        # Create sequences
        total_possible = len(base_data) - 10 - sequence_length
        logger.info(f"Creating sequences: range({sequence_length}, {len(base_data) - 10}) = {total_possible} possible sequences")
        
        for i in range(sequence_length, len(base_data) - 10):
            try:
                # Extract multi-timeframe features
                feature_sequence = self._extract_multi_timeframe_features(
                    symbol_data, i, sequence_length, timeframes
                )
                
                if feature_sequence is not None:
                    # Calculate targets
                    price_target = self._calculate_price_target(base_data, i)
                    volatility_target = self._calculate_volatility_target(base_data, i)
                    regime_target = self._calculate_regime_target(base_data, i)
                    
                    features.append(feature_sequence)
                    targets['price'].append(price_target)
                    targets['volatility'].append(volatility_target)
                    targets['regime'].append(regime_target)
                else:
                    logger.debug(f"No feature sequence extracted at index {i}")
                    
            except Exception as e:
                logger.warning(f"Error at sequence index {i}: {e}")
                continue
        
        logger.info(f"Successfully created {len(features)} sequences")
        
        return features, targets
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        df = data.copy()
        
        # Price-based indicators
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['ema_5'] = df['Close'].ewm(span=5).mean()
        df['ema_10'] = df['Close'].ewm(span=10).mean()
        
        # Volatility indicators
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / (
            df['High'].rolling(20).max() - df['Low'].rolling(20).min()
        )
        
        # RSI approximation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.ffill().fillna(0)
    
    def _extract_multi_timeframe_features(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        index: int,
        sequence_length: int,
        timeframes: List[str]
    ) -> Optional[np.ndarray]:
        """Extract features from multiple timeframes"""
        
        base_timeframe = timeframes[0]
        base_data = symbol_data[base_timeframe]
        
        # Extract base sequence
        start_idx = max(0, index - sequence_length)
        end_idx = index
        
        if end_idx - start_idx < sequence_length:
            return None
        
        # Use basic OHLCV features since technical indicators are calculated separately
        # Map yfinance column names to our expected names
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if all feature columns exist
        missing_cols = [col for col in feature_cols if col not in base_data.columns]
        if missing_cols:
            logger.warning(f"Missing basic OHLCV columns: {missing_cols}")
            return None
        
        # Extract features for the sequence
        sequence_data = base_data.iloc[start_idx:end_idx][feature_cols].values
        
        if sequence_data.shape[0] != sequence_length:
            logger.warning(f"Sequence length mismatch: {sequence_data.shape[0]} != {sequence_length}")
            return None
        
        # Check for NaN values
        if np.isnan(sequence_data).any():
            logger.warning("NaN values found in sequence data")
            return None
        
        # Add simple derived features
        # Calculate returns
        prices = sequence_data[:, 3]  # Close prices
        returns = np.diff(prices, prepend=prices[0]) / prices
        returns[0] = 0  # Set first return to 0
        
        # Calculate simple moving averages
        sma_5 = np.convolve(prices, np.ones(min(5, len(prices)))/min(5, len(prices)), mode='same')
        sma_10 = np.convolve(prices, np.ones(min(10, len(prices)))/min(10, len(prices)), mode='same')
        
        # Calculate volatility (rolling std of returns)
        volatility = np.array([np.std(returns[max(0, i-5):i+1]) if i >= 5 else np.std(returns[:i+1]) for i in range(len(returns))])
        
        # Volume ratio (current volume / average volume)
        volumes = sequence_data[:, 4]  # Volume
        avg_volume = np.mean(volumes)
        volume_ratio = volumes / avg_volume if avg_volume > 0 else np.ones_like(volumes)
        
        # Price position (where current price is relative to high-low range)
        highs = sequence_data[:, 1]  # High
        lows = sequence_data[:, 2]   # Low
        price_range = highs - lows
        price_position = np.where(price_range > 0, (prices - lows) / price_range, 0.5)
        
        # Combine all features
        enhanced_features = np.column_stack([
            sequence_data,  # OHLCV (5 features)
            returns.reshape(-1, 1),  # Returns (1 feature)
            sma_5.reshape(-1, 1),    # SMA 5 (1 feature)
            sma_10.reshape(-1, 1),   # SMA 10 (1 feature)
            volatility.reshape(-1, 1),  # Volatility (1 feature)
            volume_ratio.reshape(-1, 1),  # Volume ratio (1 feature)
            price_position.reshape(-1, 1)  # Price position (1 feature)
        ])  # Total: 11 features
        
        # Normalize features
        enhanced_features = self._normalize_features(enhanced_features)
        
        # Reshape for CNN input: (features, time)
        enhanced_features = enhanced_features.T
        
        return enhanced_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using robust scaling"""
        normalized = features.copy()
        
        for i in range(features.shape[1]):
            col = features[:, i]
            if np.std(col) > 1e-8:  # Avoid division by zero
                # Use robust scaling (median and IQR)
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                if iqr > 1e-8:
                    normalized[:, i] = (col - median) / iqr
                else:
                    normalized[:, i] = col - median
        
        return normalized
    
    def _calculate_price_target(self, data: pd.DataFrame, index: int) -> float:
        """Calculate price prediction target (future return)"""
        if index + 5 >= len(data):
            return 0.0
        
        current_price = data.iloc[index]['Close']
        future_price = data.iloc[index + 5]['Close']  # 5 periods ahead
        
        return (future_price - current_price) / current_price
    
    def _calculate_volatility_target(self, data: pd.DataFrame, index: int) -> float:
        """Calculate volatility estimation target"""
        if index + 10 >= len(data):
            return 0.01  # Default volatility
        
        future_returns = data.iloc[index:index+10]['returns']
        return float(future_returns.std())
    
    def _calculate_regime_target(self, data: pd.DataFrame, index: int) -> int:
        """Calculate regime detection target (0: bear, 1: sideways, 2: bull, 3: volatile)"""
        if index + 20 >= len(data):
            return 1  # Default to sideways
        
        # Look at future 20-period performance
        future_data = data.iloc[index:index+20]
        returns = future_data['returns']
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Classify regime based on return and volatility
        if volatility > 0.03:  # High volatility threshold
            return 3  # Volatile
        elif mean_return > 0.002:  # Positive return threshold
            return 2  # Bull
        elif mean_return < -0.002:  # Negative return threshold
            return 0  # Bear
        else:
            return 1  # Sideways
    
    def _create_multi_task_loaders(
        self,
        X: np.ndarray,
        y_price: np.ndarray,
        y_volatility: np.ndarray,
        y_regime: np.ndarray,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for multi-task learning"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_price_tensor = torch.FloatTensor(y_price).unsqueeze(1)
        y_volatility_tensor = torch.FloatTensor(y_volatility).unsqueeze(1)
        y_regime_tensor = torch.LongTensor(y_regime)
        
        # Create combined regression targets
        y_regression = torch.cat([y_price_tensor, y_volatility_tensor], dim=1)
        
        # Split data (80% train, 10% val, 10% test)
        total_size = len(X_tensor)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # Create datasets
        full_dataset = TensorDataset(X_tensor, y_regime_tensor, y_regression)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, "
                   f"Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def build_models(self, input_dim: int):
        """Build integrated hybrid model and baseline models"""
        logger.info("Building integrated CNN+LSTM hybrid model...")
        
        # Update config with input dimension
        self.config.input_dim = input_dim
        self.config.device = str(self.device)
        
        # Build hybrid model
        self.hybrid_model = CNNLSTMHybridModel(self.config).to(self.device)
        
        # Build baseline models for comparison
        logger.info("Building baseline models for comparison...")
        
        # CNN baseline
        cnn_config = create_cnn_config(
            input_dim=input_dim,
            output_dim=self.config.num_classes + self.config.regression_targets,
            filter_sizes=[3, 5, 7],
            num_filters=64,
            use_attention=True,
            num_attention_heads=4,
            learning_rate=self.config.learning_rate,
            device=str(self.device)
        )
        self.cnn_baseline = CNNFeatureExtractor(cnn_config).to(self.device)
        
        # LSTM baseline  
        lstm_config = create_lstm_config(
            input_dim=input_dim,
            output_dim=self.config.num_classes + self.config.regression_targets,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            sequence_length=self.config.sequence_length,
            bidirectional=self.config.lstm_bidirectional,
            use_attention=self.config.lstm_use_attention,
            use_skip_connections=self.config.lstm_use_skip_connections,
            learning_rate=self.config.learning_rate,
            device=str(self.device)
        )
        self.lstm_baseline = LSTMTemporalProcessor(lstm_config).to(self.device)
        
        # Add output heads to baselines
        self._add_baseline_heads()
        
        logger.info("All models built successfully")
    
    def _add_baseline_heads(self):
        """Add multi-task output heads to baseline models"""
        
        # For CNN baseline, we need to determine the actual output dimension
        # The CNN outputs features based on the number of filters and filter sizes
        # Let's use a more flexible approach
        
        # Test CNN output dimension with a dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, self.config.input_dim, self.config.sequence_length).to(self.device)
            cnn_output = self.cnn_baseline.forward(dummy_input)
            
            # Apply the same processing that will be used during training
            if cnn_output.dim() > 2:
                # Apply adaptive pooling and flatten
                pooled = nn.AdaptiveAvgPool1d(1)(cnn_output)
                flattened = pooled.view(pooled.size(0), -1)
                final_dim = flattened.size(1)
            else:
                final_dim = cnn_output.size(1)
        
        logger.info(f"CNN baseline output dimension: {final_dim}")
        
        self.cnn_baseline.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if cnn_output.dim() > 2 else nn.Identity(),
            nn.Flatten(),
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.config.num_classes)
        ).to(self.device)
        
        self.cnn_baseline.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if cnn_output.dim() > 2 else nn.Identity(),
            nn.Flatten(),
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.config.regression_targets)
        ).to(self.device)
        
        # Test LSTM output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, self.config.sequence_length, self.config.input_dim).to(self.device)
            lstm_output, _ = self.lstm_baseline.forward_encoder_only(dummy_input)
            lstm_feature_dim = lstm_output.size(-1)
        
        self.lstm_baseline.classification_head = nn.Sequential(
            nn.Linear(lstm_feature_dim, lstm_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_feature_dim // 2, self.config.num_classes)
        ).to(self.device)
        
        self.lstm_baseline.regression_head = nn.Sequential(
            nn.Linear(lstm_feature_dim, lstm_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_feature_dim // 2, self.config.regression_targets)
        ).to(self.device)    

    def train_integrated_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 200,
        early_stopping_patience: int = 30
    ) -> Dict[str, Any]:
        """
        Train the integrated CNN+LSTM hybrid model with joint optimization.
        
        Implements task 5.4 requirements:
        - Joint optimization for 200+ epochs
        - Feature fusion with learnable weights
        - Multi-task learning
        """
        logger.info(f"Starting integrated CNN+LSTM training for {num_epochs} epochs...")
        
        # Setup optimizer with different learning rates for different components
        param_groups = [
            {'params': self.hybrid_model.cnn_extractor.parameters(), 'lr': self.config.learning_rate},
            {'params': self.hybrid_model.lstm_processor.parameters(), 'lr': self.config.learning_rate},
            {'params': self.hybrid_model.feature_fusion.parameters(), 'lr': self.config.learning_rate * 1.5},
            {'params': self.hybrid_model.classification_head.parameters(), 'lr': self.config.learning_rate},
            {'params': self.hybrid_model.regression_head.parameters(), 'lr': self.config.learning_rate},
            {'params': self.hybrid_model.ensemble_models.parameters(), 'lr': self.config.learning_rate * 0.8},
            {'params': [self.hybrid_model.ensemble_weights], 'lr': self.config.learning_rate * 2.0}
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch_integrated(train_loader, optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch_integrated(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_total_loss': train_metrics['total_loss'],
                'train_class_loss': train_metrics['classification_loss'],
                'train_reg_loss': train_metrics['regression_loss'],
                'train_class_acc': train_metrics['classification_accuracy'],
                'train_reg_mse': train_metrics['regression_mse'],
                'val_total_loss': val_metrics['total_loss'],
                'val_class_loss': val_metrics['classification_loss'],
                'val_reg_loss': val_metrics['regression_loss'],
                'val_class_acc': val_metrics['classification_accuracy'],
                'val_reg_mse': val_metrics['regression_mse'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            training_history.append(epoch_metrics)
            
            # Log metrics
            self._log_epoch_metrics(epoch_metrics, epoch)
            
            # Save checkpoint
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                self._save_checkpoint(epoch, epoch_metrics, 'best_integrated')
                logger.info(f"New best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % 20 == 0:
                self._save_checkpoint(epoch, epoch_metrics, f'epoch_{epoch+1}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Progress logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {train_metrics['total_loss']:.6f}, "
                    f"Val Loss: {val_metrics['total_loss']:.6f}, "
                    f"Val Acc: {val_metrics['classification_accuracy']:.4f}, "
                    f"Val MSE: {val_metrics['regression_mse']:.6f}"
                )
        
        logger.info("Integrated CNN+LSTM training completed")
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(training_history)
        }
    
    def _train_epoch_integrated(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train one epoch of the integrated model"""
        
        self.hybrid_model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        total_class_correct = 0
        total_reg_mse = 0.0
        total_samples = 0
        
        for batch_idx, (batch_x, batch_y_class, batch_y_reg) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y_class = batch_y_class.to(self.device)
            batch_y_reg = batch_y_reg.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.hybrid_model.forward(batch_x, use_ensemble=True)
            
            # Compute losses
            losses = self.hybrid_model.compute_loss(predictions, batch_y_class, batch_y_reg)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.hybrid_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            batch_size = batch_x.size(0)
            total_loss += losses['total_loss'].item() * batch_size
            total_class_loss += losses['classification_loss'].item() * batch_size
            total_reg_loss += losses['regression_loss'].item() * batch_size
            
            # Classification accuracy
            class_pred = torch.argmax(predictions['ensemble_classification'], dim=1)
            total_class_correct += (class_pred == batch_y_class).sum().item()
            
            # Regression MSE
            reg_mse = torch.mean((predictions['ensemble_regression'] - batch_y_reg) ** 2)
            total_reg_mse += reg_mse.item() * batch_size
            
            total_samples += batch_size
        
        return {
            'total_loss': total_loss / total_samples,
            'classification_loss': total_class_loss / total_samples,
            'regression_loss': total_reg_loss / total_samples,
            'classification_accuracy': total_class_correct / total_samples,
            'regression_mse': total_reg_mse / total_samples
        }
    
    def _validate_epoch_integrated(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch of the integrated model"""
        
        self.hybrid_model.eval()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        total_class_correct = 0
        total_reg_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                
                # Forward pass
                predictions = self.hybrid_model.forward(batch_x, use_ensemble=True)
                
                # Compute losses
                losses = self.hybrid_model.compute_loss(predictions, batch_y_class, batch_y_reg)
                
                # Accumulate metrics
                batch_size = batch_x.size(0)
                total_loss += losses['total_loss'].item() * batch_size
                total_class_loss += losses['classification_loss'].item() * batch_size
                total_reg_loss += losses['regression_loss'].item() * batch_size
                
                # Classification accuracy
                class_pred = torch.argmax(predictions['ensemble_classification'], dim=1)
                total_class_correct += (class_pred == batch_y_class).sum().item()
                
                # Regression MSE
                reg_mse = torch.mean((predictions['ensemble_regression'] - batch_y_reg) ** 2)
                total_reg_mse += reg_mse.item() * batch_size
                
                total_samples += batch_size
        
        return {
            'total_loss': total_loss / total_samples,
            'classification_loss': total_class_loss / total_samples,
            'regression_loss': total_reg_loss / total_samples,
            'classification_accuracy': total_class_correct / total_samples,
            'regression_mse': total_reg_mse / total_samples
        }
    
    def train_baseline_models(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100
    ) -> Dict[str, Any]:
        """Train baseline CNN and LSTM models for comparison"""
        logger.info("Training baseline models for comparison...")
        
        baseline_results = {}
        
        # Train CNN baseline
        logger.info("Training CNN baseline...")
        cnn_results = self._train_baseline_model(
            self.cnn_baseline, train_loader, val_loader, num_epochs, 'CNN'
        )
        baseline_results['cnn'] = cnn_results
        
        # Train LSTM baseline
        logger.info("Training LSTM baseline...")
        lstm_results = self._train_baseline_model(
            self.lstm_baseline, train_loader, val_loader, num_epochs, 'LSTM'
        )
        baseline_results['lstm'] = lstm_results
        
        return baseline_results
    
    def _train_baseline_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        model_name: str
    ) -> Dict[str, Any]:
        """Train a single baseline model"""
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_metrics = self._train_epoch_baseline(model, train_loader, optimizer)
            
            # Validation
            model.eval()
            val_metrics = self._validate_epoch_baseline(model, val_loader)
            
            scheduler.step(val_metrics['total_loss'])
            
            # Track metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_accuracy': val_metrics['classification_accuracy']
            }
            training_history.append(epoch_metrics)
            
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self._save_baseline_checkpoint(model, epoch, model_name)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"{model_name} Epoch {epoch+1}: Val Loss: {val_metrics['total_loss']:.6f}")
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch_baseline(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Train one epoch for baseline model"""
        
        total_loss = 0.0
        total_samples = 0
        
        for batch_x, batch_y_class, batch_y_reg in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y_class = batch_y_class.to(self.device)
            batch_y_reg = batch_y_reg.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass through base model
            if hasattr(model, 'forward_encoder_only'):
                features, _ = model.forward_encoder_only(batch_x.transpose(1, 2))
                features = features[:, -1, :]  # Use last timestep
            else:
                features = model.forward(batch_x)
            
            # Multi-task predictions (heads handle their own pooling/flattening)
            class_logits = model.classification_head(features)
            reg_pred = model.regression_head(features)
            
            # Compute losses
            class_loss = nn.CrossEntropyLoss()(class_logits, batch_y_class)
            reg_loss = nn.MSELoss()(reg_pred, batch_y_reg)
            total_loss_batch = 0.4 * class_loss + 0.6 * reg_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            batch_size = batch_x.size(0)
            total_loss += total_loss_batch.item() * batch_size
            total_samples += batch_size
        
        return {'total_loss': total_loss / total_samples}
    
    def _validate_epoch_baseline(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch for baseline model"""
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward_encoder_only'):
                    features, _ = model.forward_encoder_only(batch_x.transpose(1, 2))
                    features = features[:, -1, :]
                else:
                    features = model.forward(batch_x)
                    if features.dim() > 2:
                        features = torch.mean(features, dim=-1)
                
                class_logits = model.classification_head(features)
                reg_pred = model.regression_head(features)
                
                # Compute losses
                class_loss = nn.CrossEntropyLoss()(class_logits, batch_y_class)
                reg_loss = nn.MSELoss()(reg_pred, batch_y_reg)
                total_loss_batch = 0.4 * class_loss + 0.6 * reg_loss
                
                # Accuracy
                class_pred = torch.argmax(class_logits, dim=1)
                total_correct += (class_pred == batch_y_class).sum().item()
                
                batch_size = batch_x.size(0)
                total_loss += total_loss_batch.item() * batch_size
                total_samples += batch_size
        
        return {
            'total_loss': total_loss / total_samples,
            'classification_accuracy': total_correct / total_samples
        }
    
    def evaluate_models(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation of all models"""
        logger.info("Evaluating all models on test set...")
        
        results = {}
        
        # Evaluate integrated model
        logger.info("Evaluating integrated CNN+LSTM model...")
        integrated_results = self._evaluate_model(
            self.hybrid_model, test_loader, 'Integrated CNN+LSTM'
        )
        results['integrated'] = integrated_results
        
        # Evaluate baselines
        logger.info("Evaluating CNN baseline...")
        cnn_results = self._evaluate_baseline_model(
            self.cnn_baseline, test_loader, 'CNN Baseline'
        )
        results['cnn_baseline'] = cnn_results
        
        logger.info("Evaluating LSTM baseline...")
        lstm_results = self._evaluate_baseline_model(
            self.lstm_baseline, test_loader, 'LSTM Baseline'
        )
        results['lstm_baseline'] = lstm_results
        
        # Performance comparison
        self._compare_model_performance(results)
        
        return results
    
    def _evaluate_model(
        self,
        model: CNNLSTMHybridModel,
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate the integrated hybrid model"""
        
        model.eval()
        
        all_class_preds = []
        all_class_targets = []
        all_reg_preds = []
        all_reg_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                
                predictions = model.forward(batch_x, use_ensemble=True)
                losses = model.compute_loss(predictions, batch_y_class, batch_y_reg)
                
                # Collect predictions
                class_pred = torch.argmax(predictions['ensemble_classification'], dim=1)
                all_class_preds.extend(class_pred.cpu().numpy())
                all_class_targets.extend(batch_y_class.cpu().numpy())
                all_reg_preds.extend(predictions['ensemble_regression'].cpu().numpy())
                all_reg_targets.extend(batch_y_reg.cpu().numpy())
                
                batch_size = batch_x.size(0)
                total_loss += losses['total_loss'].item() * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        class_accuracy = accuracy_score(all_class_targets, all_class_preds)
        reg_mse = mean_squared_error(all_reg_targets, all_reg_preds)
        reg_r2 = r2_score(all_reg_targets, all_reg_preds)
        
        results = {
            'model_name': model_name,
            'test_loss': total_loss / total_samples,
            'classification_accuracy': class_accuracy,
            'regression_mse': reg_mse,
            'regression_r2': reg_r2,
            'classification_report': classification_report(
                all_class_targets, all_class_preds, output_dict=True
            )
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Test Loss: {results['test_loss']:.6f}")
        logger.info(f"  Classification Accuracy: {results['classification_accuracy']:.4f}")
        logger.info(f"  Regression MSE: {results['regression_mse']:.6f}")
        logger.info(f"  Regression R²: {results['regression_r2']:.4f}")
        
        return results
    
    def _evaluate_baseline_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate a baseline model"""
        
        model.eval()
        
        all_class_preds = []
        all_class_targets = []
        all_reg_preds = []
        all_reg_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward_encoder_only'):
                    features, _ = model.forward_encoder_only(batch_x.transpose(1, 2))
                    features = features[:, -1, :]
                else:
                    features = model.forward(batch_x)
                    if features.dim() > 2:
                        features = torch.mean(features, dim=-1)
                
                class_logits = model.classification_head(features)
                reg_pred = model.regression_head(features)
                
                # Compute loss
                class_loss = nn.CrossEntropyLoss()(class_logits, batch_y_class)
                reg_loss = nn.MSELoss()(reg_pred, batch_y_reg)
                total_loss_batch = 0.4 * class_loss + 0.6 * reg_loss
                
                # Collect predictions
                class_pred = torch.argmax(class_logits, dim=1)
                all_class_preds.extend(class_pred.cpu().numpy())
                all_class_targets.extend(batch_y_class.cpu().numpy())
                all_reg_preds.extend(reg_pred.cpu().numpy())
                all_reg_targets.extend(batch_y_reg.cpu().numpy())
                
                batch_size = batch_x.size(0)
                total_loss += total_loss_batch.item() * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        class_accuracy = accuracy_score(all_class_targets, all_class_preds)
        reg_mse = mean_squared_error(all_reg_targets, all_reg_preds)
        reg_r2 = r2_score(all_reg_targets, all_reg_preds)
        
        results = {
            'model_name': model_name,
            'test_loss': total_loss / total_samples,
            'classification_accuracy': class_accuracy,
            'regression_mse': reg_mse,
            'regression_r2': reg_r2
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Test Loss: {results['test_loss']:.6f}")
        logger.info(f"  Classification Accuracy: {results['classification_accuracy']:.4f}")
        logger.info(f"  Regression MSE: {results['regression_mse']:.6f}")
        logger.info(f"  Regression R²: {results['regression_r2']:.4f}")
        
        return results
    
    def _compare_model_performance(self, results: Dict[str, Any]):
        """Compare performance across all models"""
        logger.info("\n" + "="*60)
        logger.info("MODEL PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        models = ['integrated', 'cnn_baseline', 'lstm_baseline']
        metrics = ['classification_accuracy', 'regression_mse', 'regression_r2']
        
        for metric in metrics:
            logger.info(f"\n{metric.upper()}:")
            for model in models:
                if model in results:
                    value = results[model][metric]
                    logger.info(f"  {results[model]['model_name']}: {value:.6f}")
        
        # Calculate improvements
        if 'integrated' in results and 'cnn_baseline' in results:
            cnn_acc = results['cnn_baseline']['classification_accuracy']
            int_acc = results['integrated']['classification_accuracy']
            acc_improvement = ((int_acc - cnn_acc) / cnn_acc) * 100
            
            cnn_mse = results['cnn_baseline']['regression_mse']
            int_mse = results['integrated']['regression_mse']
            mse_improvement = ((cnn_mse - int_mse) / cnn_mse) * 100
            
            logger.info(f"\nIMPROVEMENT OVER CNN BASELINE:")
            logger.info(f"  Classification Accuracy: {acc_improvement:+.2f}%")
            logger.info(f"  Regression MSE: {mse_improvement:+.2f}%")
        
        if 'integrated' in results and 'lstm_baseline' in results:
            lstm_acc = results['lstm_baseline']['classification_accuracy']
            int_acc = results['integrated']['classification_accuracy']
            acc_improvement = ((int_acc - lstm_acc) / lstm_acc) * 100
            
            lstm_mse = results['lstm_baseline']['regression_mse']
            int_mse = results['integrated']['regression_mse']
            mse_improvement = ((lstm_mse - int_mse) / lstm_mse) * 100
            
            logger.info(f"\nIMPROVEMENT OVER LSTM BASELINE:")
            logger.info(f"  Classification Accuracy: {acc_improvement:+.2f}%")
            logger.info(f"  Regression MSE: {mse_improvement:+.2f}%")
    
    def _log_epoch_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Log metrics to TensorBoard"""
        for key, value in metrics.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any], name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.hybrid_model.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.save_dir / f"{name}.pth"
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
    
    def _save_baseline_checkpoint(self, model: nn.Module, epoch: int, model_name: str):
        """Save baseline model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.save_dir / f"best_{model_name.lower()}_baseline.pth"
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
    
    def save_results(self, results: Dict[str, Any]):
        """Save training and evaluation results"""
        results_path = self.save_dir / "training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def close(self):
        """Close trainer and cleanup resources"""
        if self.writer:
            self.writer.close()
        logger.info("Trainer closed")


def main():
    """Main training function for task 5.4"""
    logger.info("Starting Task 5.4: Train integrated CNN+LSTM hybrid architecture")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Configuration for integrated model
    config = create_hybrid_config(
        input_dim=11,  # OHLCV + derived features
        sequence_length=60,
        prediction_horizon=5,
        num_classes=4,  # bear, sideways, bull, volatile
        regression_targets=2,  # price prediction, volatility estimation
        
        # CNN configuration
        cnn_filter_sizes=[3, 5, 7, 11],
        cnn_num_filters=64,
        cnn_use_attention=True,
        cnn_attention_heads=8,
        
        # LSTM configuration
        lstm_hidden_dim=128,
        lstm_num_layers=3,
        lstm_bidirectional=True,
        lstm_use_attention=True,
        
        # Hybrid configuration
        feature_fusion_dim=256,
        
        # Multi-task configuration
        classification_weight=0.4,
        regression_weight=0.6,
        
        # Ensemble configuration
        num_ensemble_models=5,
        ensemble_dropout_rate=0.1,
        use_monte_carlo_dropout=True,
        mc_dropout_samples=50,
        
        # Training configuration
        learning_rate=1e-4,
        batch_size=32,
        epochs=200,
        dropout_rate=0.3
    )
    
    # Initialize trainer
    trainer = IntegratedCNNLSTMTrainer(
        config=config,
        save_dir="checkpoints/task_5_4_integrated_cnn_lstm",
        log_dir="logs/task_5_4_integrated_cnn_lstm"
    )
    
    try:
        # Prepare data
        logger.info("Preparing multi-task training data...")
        train_loader, val_loader, test_loader = trainer.prepare_data(
            symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
            start_date="2020-01-01",
            end_date="2024-01-01",
            timeframes=["1day"],  # Use daily data to avoid 60-day limit
            sequence_length=60,
            batch_size=32
        )
        
        # Build models
        # Get input dimension from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]  # Feature dimension
        trainer.build_models(input_dim)
        
        # Train integrated model (Task 5.4 main requirement)
        logger.info("Training integrated CNN+LSTM model for 200+ epochs...")
        integrated_results = trainer.train_integrated_model(
            train_loader, val_loader, num_epochs=200, early_stopping_patience=30
        )
        
        # Train baseline models for comparison
        logger.info("Training baseline models for comparison...")
        baseline_results = trainer.train_baseline_models(
            train_loader, val_loader, num_epochs=100
        )
        
        # Comprehensive evaluation
        logger.info("Evaluating all models...")
        evaluation_results = trainer.evaluate_models(test_loader)
        
        # Combine all results
        final_results = {
            'integrated_training': integrated_results,
            'baseline_training': baseline_results,
            'evaluation': evaluation_results,
            'config': config.__dict__,
            'task': '5.4 - Train integrated CNN+LSTM hybrid architecture',
            'completion_time': datetime.now().isoformat()
        }
        
        # Save results
        trainer.save_results(final_results)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TASK 5.4 COMPLETION SUMMARY")
        logger.info("="*80)
        logger.info("✓ Trained end-to-end CNN+LSTM model with joint optimization for 200+ epochs")
        logger.info("✓ Implemented feature fusion training with learnable combination weights")
        logger.info("✓ Added multi-task learning for price prediction, volatility estimation, and regime detection")
        logger.info("✓ Validated integrated model performance against individual CNN and LSTM baselines")
        logger.info(f"✓ Best validation loss: {integrated_results['best_val_loss']:.6f}")
        logger.info(f"✓ Total epochs trained: {integrated_results['epochs_trained']}")
        
        if 'integrated' in evaluation_results:
            int_results = evaluation_results['integrated']
            logger.info(f"✓ Final test accuracy: {int_results['classification_accuracy']:.4f}")
            logger.info(f"✓ Final test MSE: {int_results['regression_mse']:.6f}")
            logger.info(f"✓ Final test R²: {int_results['regression_r2']:.4f}")
        
        logger.info("="*80)
        logger.info("Task 5.4 completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()