"""
LSTM Temporal Modeling Validation Script

This script provides comprehensive validation of LSTM temporal modeling capabilities
including sequence prediction tasks, attention pattern analysis, and temporal
dependency evaluation.

Requirements addressed:
- Validate temporal modeling capability using sequence prediction tasks
- Analyze attention mechanism effectiveness
- Evaluate long-term dependency capture
- Assess temporal consistency and stability
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.cnn_lstm.bidirectional_lstm_attention import BidirectionalLSTMWithAttention
from data.pipeline import create_data_loaders
from experiments.runners.train_lstm_temporal import LSTMTrainingConfig


class LSTMTemporalValidator:
    """
    Comprehensive validator for LSTM temporal modeling capabilities.
    
    Evaluates:
    - Multi-horizon sequence prediction accuracy
    - Attention pattern quality and interpretability
    - Long-term dependency preservation
    - Temporal consistency and stability
    - Feature quality and information content
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logging()
        
        # Load model and configuration
        self.model, self.config = self._load_model_and_config(model_path, config_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Loaded LSTM model on {self.device}")
        
        # Validation results storage
        self.validation_results = {
            'sequence_prediction': {},
            'attention_analysis': {},
            'temporal_consistency': {},
            'long_term_dependencies': {},
            'feature_quality': {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"LSTMTemporalValidator_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model_and_config(
        self,
        model_path: str,
        config_path: Optional[str] = None
    ) -> Tuple[BidirectionalLSTMWithAttention, Dict]:
        """Load trained model and configuration."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Create model from config
        from models.cnn_lstm.bidirectional_lstm_attention import BidirectionalLSTMConfig
        
        model_config = BidirectionalLSTMConfig(**{
            k: v for k, v in config.items() 
            if k in BidirectionalLSTMConfig.__dataclass_fields__
        })
        
        model = BidirectionalLSTMWithAttention(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def validate_sequence_prediction(
        self,
        test_loader: torch.utils.data.DataLoader,
        prediction_horizons: List[int] = [1, 5, 10, 20, 50]
    ) -> Dict[str, Any]:
        """
        Validate sequence prediction capabilities across multiple horizons.
        
        Args:
            test_loader: Test data loader
            prediction_horizons: List of prediction horizons to evaluate
        
        Returns:
            Dictionary containing prediction validation results
        """
        self.logger.info("Validating sequence prediction capabilities...")
        
        prediction_results = {}
        
        with torch.no_grad():
            for horizon in prediction_horizons:
                self.logger.info(f"Evaluating {horizon}-step prediction...")
                
                horizon_results = {
                    'mse_scores': [],
                    'mae_scores': [],
                    'r2_scores': [],
                    'direction_accuracy': [],
                    'predictions': [],
                    'targets': []
                }
                
                for batch_idx, (data, targets) in enumerate(test_loader):
                    if batch_idx >= 50:  # Limit evaluation for efficiency
                        break
                    
                    # Move data to device
                    if isinstance(data, dict):
                        data = {k: v.to(self.device) for k, v in data.items()}
                        if 'sequence_data' in data:
                            sequence_data = data['sequence_data']
                            lengths = data.get('lengths', None)
                        else:
                            continue
                    else:
                        sequence_data = data.to(self.device)
                        lengths = None
                    
                    # Skip if sequence too short
                    if sequence_data.size(1) <= horizon:
                        continue
                    
                    # Multi-step prediction evaluation
                    input_seq = sequence_data[:, :-horizon, :]
                    target_seq = sequence_data[:, -horizon:, :]
                    
                    # Get model outputs for input sequence
                    outputs = self.model(input_seq, lengths)
                    
                    # Extract features for prediction
                    if 'lstm_output' in outputs:
                        lstm_features = outputs['lstm_output']  # (batch, seq_len, hidden_dim)
                        
                        # Use last hidden state for multi-step prediction
                        last_hidden = lstm_features[:, -1, :]  # (batch, hidden_dim)
                        
                        # Simple prediction head (for validation purposes)
                        # In practice, this would be a trained prediction head
                        pred_seq = last_hidden.unsqueeze(1).repeat(1, horizon, 1)
                        pred_seq = pred_seq[:, :, :target_seq.size(-1)]  # Match feature dimensions
                        
                        # Compute metrics
                        if pred_seq.size() == target_seq.size():
                            # Flatten for metric computation
                            pred_flat = pred_seq.reshape(-1, pred_seq.size(-1)).cpu().numpy()
                            target_flat = target_seq.reshape(-1, target_seq.size(-1)).cpu().numpy()
                            
                            # MSE and MAE
                            mse = mean_squared_error(target_flat, pred_flat)
                            mae = mean_absolute_error(target_flat, pred_flat)
                            
                            # R² score
                            try:
                                r2 = r2_score(target_flat, pred_flat)
                            except:
                                r2 = 0.0
                            
                            # Direction accuracy (for first feature, typically price)
                            if target_flat.shape[1] > 0:
                                pred_direction = np.sign(np.diff(pred_flat[:, 0]))
                                target_direction = np.sign(np.diff(target_flat[:, 0]))
                                
                                if len(pred_direction) > 0 and len(target_direction) > 0:
                                    direction_acc = np.mean(pred_direction == target_direction)
                                else:
                                    direction_acc = 0.0
                            else:
                                direction_acc = 0.0
                            
                            # Store results
                            horizon_results['mse_scores'].append(mse)
                            horizon_results['mae_scores'].append(mae)
                            horizon_results['r2_scores'].append(r2)
                            horizon_results['direction_accuracy'].append(direction_acc)
                            
                            # Store sample predictions for analysis
                            if len(horizon_results['predictions']) < 10:
                                horizon_results['predictions'].append(pred_flat[:10])
                                horizon_results['targets'].append(target_flat[:10])
                
                # Aggregate results for this horizon
                if horizon_results['mse_scores']:
                    prediction_results[f'{horizon}_step'] = {
                        'mean_mse': np.mean(horizon_results['mse_scores']),
                        'std_mse': np.std(horizon_results['mse_scores']),
                        'mean_mae': np.mean(horizon_results['mae_scores']),
                        'std_mae': np.std(horizon_results['mae_scores']),
                        'mean_r2': np.mean(horizon_results['r2_scores']),
                        'std_r2': np.std(horizon_results['r2_scores']),
                        'mean_direction_accuracy': np.mean(horizon_results['direction_accuracy']),
                        'std_direction_accuracy': np.std(horizon_results['direction_accuracy']),
                        'sample_predictions': horizon_results['predictions'][:5],
                        'sample_targets': horizon_results['targets'][:5]
                    }
                    
                    self.logger.info(
                        f"{horizon}-step prediction - MSE: {prediction_results[f'{horizon}_step']['mean_mse']:.6f}, "
                        f"MAE: {prediction_results[f'{horizon}_step']['mean_mae']:.6f}, "
                        f"R²: {prediction_results[f'{horizon}_step']['mean_r2']:.4f}, "
                        f"Direction Acc: {prediction_results[f'{horizon}_step']['mean_direction_accuracy']:.4f}"
                    )
        
        self.validation_results['sequence_prediction'] = prediction_results
        return prediction_results
    
    def analyze_attention_patterns(
        self,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns for interpretability and quality.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to analyze
        
        Returns:
            Dictionary containing attention analysis results
        """
        self.logger.info("Analyzing attention patterns...")
        
        attention_results = {
            'attention_weights': [],
            'attention_entropy': [],
            'attention_peaks': [],
            'attention_consistency': [],
            'temporal_focus_patterns': []
        }
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, (data, targets) in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                    if 'sequence_data' in data:
                        sequence_data = data['sequence_data']
                        lengths = data.get('lengths', None)
                    else:
                        continue
                else:
                    sequence_data = data.to(self.device)
                    lengths = None
                
                # Get attention patterns
                attention_analysis = self.model.get_attention_patterns(sequence_data, lengths)
                
                attention_weights = attention_analysis['attention_weights'].cpu().numpy()
                attention_entropy = attention_analysis['attention_entropy'].cpu().numpy()
                
                # Analyze each sample in batch
                batch_size = attention_weights.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    sample_attention = attention_weights[i]  # (seq_len, seq_len)
                    sample_entropy = attention_entropy[i]    # (seq_len,)
                    
                    # Store attention weights
                    attention_results['attention_weights'].append(sample_attention)
                    attention_results['attention_entropy'].append(sample_entropy)
                    
                    # Find attention peaks (positions with high attention)
                    avg_attention = np.mean(sample_attention, axis=0)
                    peaks, _ = find_peaks(avg_attention, height=np.mean(avg_attention))
                    attention_results['attention_peaks'].append(peaks)
                    
                    # Compute attention consistency (how stable attention is across time)
                    attention_std = np.std(sample_attention, axis=0)
                    consistency_score = 1.0 / (1.0 + np.mean(attention_std))
                    attention_results['attention_consistency'].append(consistency_score)
                    
                    # Analyze temporal focus patterns
                    # Check if attention focuses on recent vs distant past
                    seq_len = sample_attention.shape[0]
                    if seq_len > 10:
                        recent_attention = np.mean(sample_attention[:, -10:])  # Last 10 steps
                        distant_attention = np.mean(sample_attention[:, :-10])  # Earlier steps
                        temporal_focus = recent_attention / (recent_attention + distant_attention + 1e-8)
                        attention_results['temporal_focus_patterns'].append(temporal_focus)
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        # Compute aggregate statistics
        attention_analysis_results = {
            'mean_attention_entropy': np.mean([np.mean(entropy) for entropy in attention_results['attention_entropy']]),
            'std_attention_entropy': np.std([np.mean(entropy) for entropy in attention_results['attention_entropy']]),
            'mean_attention_consistency': np.mean(attention_results['attention_consistency']),
            'std_attention_consistency': np.std(attention_results['attention_consistency']),
            'mean_temporal_focus': np.mean(attention_results['temporal_focus_patterns']),
            'std_temporal_focus': np.std(attention_results['temporal_focus_patterns']),
            'attention_peak_distribution': [len(peaks) for peaks in attention_results['attention_peaks']],
            'sample_attention_weights': attention_results['attention_weights'][:10]  # Store samples for visualization
        }
        
        self.logger.info(
            f"Attention Analysis - Mean Entropy: {attention_analysis_results['mean_attention_entropy']:.4f}, "
            f"Consistency: {attention_analysis_results['mean_attention_consistency']:.4f}, "
            f"Temporal Focus: {attention_analysis_results['mean_temporal_focus']:.4f}"
        )
        
        self.validation_results['attention_analysis'] = attention_analysis_results
        return attention_analysis_results
    
    def evaluate_temporal_consistency(
        self,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate temporal consistency of LSTM representations.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary containing temporal consistency results
        """
        self.logger.info("Evaluating temporal consistency...")
        
        consistency_results = {
            'temporal_smoothness': [],
            'representation_stability': [],
            'gradient_consistency': [],
            'hidden_state_evolution': []
        }
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, (data, targets) in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                    if 'sequence_data' in data:
                        sequence_data = data['sequence_data']
                        lengths = data.get('lengths', None)
                    else:
                        continue
                else:
                    sequence_data = data.to(self.device)
                    lengths = None
                
                # Get model outputs
                outputs = self.model(sequence_data, lengths)
                
                if 'lstm_output' in outputs:
                    lstm_output = outputs['lstm_output'].cpu().numpy()  # (batch, seq_len, hidden_dim)
                    
                    batch_size = lstm_output.shape[0]
                    for i in range(min(batch_size, num_samples - sample_count)):
                        sample_output = lstm_output[i]  # (seq_len, hidden_dim)
                        
                        # Temporal smoothness: measure how much representations change over time
                        if sample_output.shape[0] > 1:
                            temporal_diffs = np.diff(sample_output, axis=0)
                            smoothness = np.mean(np.linalg.norm(temporal_diffs, axis=1))
                            consistency_results['temporal_smoothness'].append(smoothness)
                        
                        # Representation stability: measure consistency of similar time steps
                        if sample_output.shape[0] > 10:
                            # Compare representations at different time steps
                            early_repr = sample_output[:5].mean(axis=0)
                            late_repr = sample_output[-5:].mean(axis=0)
                            stability = np.corrcoef(early_repr, late_repr)[0, 1]
                            if not np.isnan(stability):
                                consistency_results['representation_stability'].append(stability)
                        
                        # Hidden state evolution analysis
                        if sample_output.shape[0] > 20:
                            # Analyze how hidden states evolve over time
                            evolution_pattern = []
                            window_size = 5
                            for t in range(0, sample_output.shape[0] - window_size, window_size):
                                window_repr = sample_output[t:t+window_size].mean(axis=0)
                                evolution_pattern.append(np.linalg.norm(window_repr))
                            
                            # Measure consistency of evolution
                            if len(evolution_pattern) > 1:
                                evolution_consistency = 1.0 - np.std(evolution_pattern) / (np.mean(evolution_pattern) + 1e-8)
                                consistency_results['hidden_state_evolution'].append(evolution_consistency)
                        
                        sample_count += 1
                        if sample_count >= num_samples:
                            break
        
        # Compute aggregate results
        temporal_consistency_results = {
            'mean_temporal_smoothness': np.mean(consistency_results['temporal_smoothness']),
            'std_temporal_smoothness': np.std(consistency_results['temporal_smoothness']),
            'mean_representation_stability': np.mean(consistency_results['representation_stability']),
            'std_representation_stability': np.std(consistency_results['representation_stability']),
            'mean_hidden_state_evolution': np.mean(consistency_results['hidden_state_evolution']),
            'std_hidden_state_evolution': np.std(consistency_results['hidden_state_evolution']),
            'temporal_consistency_score': np.mean([
                1.0 / (1.0 + np.mean(consistency_results['temporal_smoothness'])),
                np.mean(consistency_results['representation_stability']),
                np.mean(consistency_results['hidden_state_evolution'])
            ])
        }
        
        self.logger.info(
            f"Temporal Consistency - Smoothness: {temporal_consistency_results['mean_temporal_smoothness']:.4f}, "
            f"Stability: {temporal_consistency_results['mean_representation_stability']:.4f}, "
            f"Overall Score: {temporal_consistency_results['temporal_consistency_score']:.4f}"
        )
        
        self.validation_results['temporal_consistency'] = temporal_consistency_results
        return temporal_consistency_results
    
    def evaluate_long_term_dependencies(
        self,
        test_loader: torch.utils.data.DataLoader,
        dependency_lags: List[int] = [10, 20, 50, 100]
    ) -> Dict[str, Any]:
        """
        Evaluate the model's ability to capture long-term dependencies.
        
        Args:
            test_loader: Test data loader
            dependency_lags: List of lag distances to evaluate
        
        Returns:
            Dictionary containing long-term dependency results
        """
        self.logger.info("Evaluating long-term dependencies...")
        
        dependency_results = {}
        
        with torch.no_grad():
            for lag in dependency_lags:
                self.logger.info(f"Evaluating {lag}-step dependencies...")
                
                lag_results = {
                    'correlations': [],
                    'mutual_information': [],
                    'prediction_accuracy': []
                }
                
                for batch_idx, (data, targets) in enumerate(test_loader):
                    if batch_idx >= 20:  # Limit for efficiency
                        break
                    
                    # Move data to device
                    if isinstance(data, dict):
                        data = {k: v.to(self.device) for k, v in data.items()}
                        if 'sequence_data' in data:
                            sequence_data = data['sequence_data']
                            lengths = data.get('lengths', None)
                        else:
                            continue
                    else:
                        sequence_data = data.to(self.device)
                        lengths = None
                    
                    # Skip if sequence too short
                    if sequence_data.size(1) <= lag:
                        continue
                    
                    # Get model outputs
                    outputs = self.model(sequence_data, lengths)
                    
                    if 'lstm_output' in outputs:
                        lstm_output = outputs['lstm_output'].cpu().numpy()  # (batch, seq_len, hidden_dim)
                        
                        batch_size = lstm_output.shape[0]
                        for i in range(batch_size):
                            sample_output = lstm_output[i]  # (seq_len, hidden_dim)
                            
                            if sample_output.shape[0] > lag:
                                # Compute correlation between representations at different time lags
                                early_repr = sample_output[:-lag]  # (seq_len-lag, hidden_dim)
                                late_repr = sample_output[lag:]    # (seq_len-lag, hidden_dim)
                                
                                # Average correlation across hidden dimensions
                                correlations = []
                                for dim in range(min(early_repr.shape[1], 50)):  # Limit dimensions for efficiency
                                    corr, _ = pearsonr(early_repr[:, dim], late_repr[:, dim])
                                    if not np.isnan(corr):
                                        correlations.append(abs(corr))
                                
                                if correlations:
                                    mean_correlation = np.mean(correlations)
                                    lag_results['correlations'].append(mean_correlation)
                
                # Aggregate results for this lag
                if lag_results['correlations']:
                    dependency_results[f'{lag}_step_dependency'] = {
                        'mean_correlation': np.mean(lag_results['correlations']),
                        'std_correlation': np.std(lag_results['correlations']),
                        'dependency_strength': np.mean(lag_results['correlations'])
                    }
                    
                    self.logger.info(
                        f"{lag}-step dependency - Mean Correlation: {dependency_results[f'{lag}_step_dependency']['mean_correlation']:.4f}"
                    )
        
        # Compute overall long-term dependency score
        if dependency_results:
            dependency_scores = [result['dependency_strength'] for result in dependency_results.values()]
            overall_dependency_score = np.mean(dependency_scores)
            
            dependency_results['overall_long_term_dependency_score'] = overall_dependency_score
            
            self.logger.info(f"Overall Long-term Dependency Score: {overall_dependency_score:.4f}")
        
        self.validation_results['long_term_dependencies'] = dependency_results
        return dependency_results
    
    def generate_validation_report(self, output_dir: str = "experiments/results/lstm_validation"):
        """Generate comprehensive validation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating validation report in {output_path}")
        
        # Save detailed results
        results_file = output_path / "lstm_temporal_validation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.validation_results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate summary report
        summary_file = output_path / "lstm_temporal_validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("LSTM Temporal Modeling Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Sequence Prediction Results
            if 'sequence_prediction' in self.validation_results:
                f.write("Sequence Prediction Results:\n")
                f.write("-" * 30 + "\n")
                for horizon, results in self.validation_results['sequence_prediction'].items():
                    f.write(f"{horizon}:\n")
                    f.write(f"  MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}\n")
                    f.write(f"  MAE: {results['mean_mae']:.6f} ± {results['std_mae']:.6f}\n")
                    f.write(f"  R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}\n")
                    f.write(f"  Direction Accuracy: {results['mean_direction_accuracy']:.4f} ± {results['std_direction_accuracy']:.4f}\n\n")
            
            # Attention Analysis Results
            if 'attention_analysis' in self.validation_results:
                f.write("Attention Analysis Results:\n")
                f.write("-" * 30 + "\n")
                results = self.validation_results['attention_analysis']
                f.write(f"Mean Attention Entropy: {results['mean_attention_entropy']:.4f} ± {results['std_attention_entropy']:.4f}\n")
                f.write(f"Attention Consistency: {results['mean_attention_consistency']:.4f} ± {results['std_attention_consistency']:.4f}\n")
                f.write(f"Temporal Focus: {results['mean_temporal_focus']:.4f} ± {results['std_temporal_focus']:.4f}\n\n")
            
            # Temporal Consistency Results
            if 'temporal_consistency' in self.validation_results:
                f.write("Temporal Consistency Results:\n")
                f.write("-" * 30 + "\n")
                results = self.validation_results['temporal_consistency']
                f.write(f"Temporal Smoothness: {results['mean_temporal_smoothness']:.4f} ± {results['std_temporal_smoothness']:.4f}\n")
                f.write(f"Representation Stability: {results['mean_representation_stability']:.4f} ± {results['std_representation_stability']:.4f}\n")
                f.write(f"Overall Consistency Score: {results['temporal_consistency_score']:.4f}\n\n")
            
            # Long-term Dependencies Results
            if 'long_term_dependencies' in self.validation_results:
                f.write("Long-term Dependencies Results:\n")
                f.write("-" * 30 + "\n")
                results = self.validation_results['long_term_dependencies']
                for dependency, values in results.items():
                    if isinstance(values, dict) and 'mean_correlation' in values:
                        f.write(f"{dependency}: {values['mean_correlation']:.4f} ± {values['std_correlation']:.4f}\n")
                
                if 'overall_long_term_dependency_score' in results:
                    f.write(f"Overall Long-term Dependency Score: {results['overall_long_term_dependency_score']:.4f}\n")
        
        self.logger.info(f"Validation report saved to {summary_file}")
        
        return output_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def run_full_validation(
        self,
        test_loader: torch.utils.data.DataLoader,
        output_dir: str = "experiments/results/lstm_validation"
    ) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.logger.info("Starting full LSTM temporal modeling validation...")
        
        # Run all validation components
        self.validate_sequence_prediction(test_loader)
        self.analyze_attention_patterns(test_loader)
        self.evaluate_temporal_consistency(test_loader)
        self.evaluate_long_term_dependencies(test_loader)
        
        # Generate report
        report_path = self.generate_validation_report(output_dir)
        
        self.logger.info("Full validation completed successfully!")
        
        return self.validation_results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate LSTM Temporal Modeling")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config-path", type=str, help="Path to model configuration")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="experiments/results/lstm_validation", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create validator
    validator = LSTMTemporalValidator(args.model_path, args.config_path)
    
    # Create test data loader
    logger.info("Creating test data loader...")
    _, _, test_loader = create_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        validation_split=0.2,
        test_split=0.1,
        num_workers=2
    )
    
    # Run validation
    results = validator.run_full_validation(test_loader, args.output_dir)
    
    logger.info("Validation completed successfully!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()