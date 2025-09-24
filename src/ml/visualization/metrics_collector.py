"""
Advanced Metrics Collector
Collects and analyzes detailed training metrics for visualization
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Advanced metrics collection and analysis"""
    
    def __init__(self, save_dir: str = "training_progress"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.trial_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(lambda: defaultdict(list))
        self.hyperparameter_history = {}
        self.optimization_stats = {}
        
        # Files
        self.metrics_file = self.save_dir / "detailed_metrics.json"
        self.stats_file = self.save_dir / "optimization_stats.json"
        
    def record_trial_start(self, trial_number: int, hyperparameters: Dict[str, Any]):
        """Record trial start with hyperparameters"""
        self.hyperparameter_history[trial_number] = {
            'params': hyperparameters,
            'start_time': datetime.now().isoformat(),
            'epochs': []
        }
        
    def record_epoch_metrics(self, trial_number: int, epoch: int, metrics: Dict[str, float]):
        """Record detailed epoch metrics"""
        timestamp = datetime.now().isoformat()
        
        epoch_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.hyperparameter_history[trial_number]['epochs'].append(epoch_data)
        
        # Store for analysis
        for metric_name, value in metrics.items():
            self.epoch_metrics[trial_number][metric_name].append(value)
            
    def record_trial_completion(self, trial_number: int, final_metrics: Dict[str, float], 
                              status: str = "completed"):
        """Record trial completion"""
        if trial_number in self.hyperparameter_history:
            self.hyperparameter_history[trial_number]['end_time'] = datetime.now().isoformat()
            self.hyperparameter_history[trial_number]['final_metrics'] = final_metrics
            self.hyperparameter_history[trial_number]['status'] = status
            
        # Add to trial metrics for analysis
        self.trial_metrics[trial_number] = {
            'final_metrics': final_metrics,
            'status': status,
            'completion_time': datetime.now().isoformat()
        }
        
        # Update optimization statistics
        self._update_optimization_stats()
        
        # Save data
        self._save_metrics()
        
    def get_trial_analysis(self, trial_number: int) -> Dict[str, Any]:
        """Get detailed analysis for a specific trial"""
        if trial_number not in self.hyperparameter_history:
            return {}
            
        trial_data = self.hyperparameter_history[trial_number]
        epochs = trial_data.get('epochs', [])
        
        if not epochs:
            return {'trial_number': trial_number, 'status': 'no_data'}
            
        # Extract metrics over time
        metrics_over_time = defaultdict(list)
        for epoch_data in epochs:
            for metric_name, value in epoch_data['metrics'].items():
                metrics_over_time[metric_name].append(value)
                
        # Calculate statistics
        analysis = {
            'trial_number': trial_number,
            'hyperparameters': trial_data.get('params', {}),
            'total_epochs': len(epochs),
            'metrics_summary': {}
        }
        
        for metric_name, values in metrics_over_time.items():
            if values:
                analysis['metrics_summary'][metric_name] = {
                    'final_value': values[-1],
                    'best_value': max(values) if 'accuracy' in metric_name else min(values),
                    'worst_value': min(values) if 'accuracy' in metric_name else max(values),
                    'mean_value': np.mean(values),
                    'std_value': np.std(values),
                    'improvement': values[-1] - values[0] if len(values) > 1 else 0,
                    'convergence_epoch': self._find_convergence_epoch(values)
                }
                
        return analysis
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get overall optimization summary"""
        if not self.trial_metrics:
            return {'status': 'no_trials'}
            
        completed_trials = [t for t in self.trial_metrics.values() 
                          if t['status'] == 'completed']
        
        if not completed_trials:
            return {'status': 'no_completed_trials'}
            
        # Extract accuracies and times
        accuracies = []
        training_times = []
        
        for trial_data in completed_trials:
            final_metrics = trial_data.get('final_metrics', {})
            if 'accuracy' in final_metrics:
                accuracies.append(final_metrics['accuracy'])
            if 'training_time' in final_metrics:
                training_times.append(final_metrics['training_time'])
                
        summary = {
            'total_trials': len(self.trial_metrics),
            'completed_trials': len(completed_trials),
            'failed_trials': len([t for t in self.trial_metrics.values() 
                                if t['status'] == 'failed']),
            'optimization_progress': self._calculate_optimization_progress()
        }
        
        if accuracies:
            summary['accuracy_stats'] = {
                'best': max(accuracies),
                'worst': min(accuracies),
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'improvement_rate': self._calculate_improvement_rate(accuracies)
            }
            
        if training_times:
            summary['timing_stats'] = {
                'mean_time': np.mean(training_times),
                'total_time': sum(training_times),
                'fastest_trial': min(training_times),
                'slowest_trial': max(training_times)
            }
            
        return summary
        
    def get_hyperparameter_analysis(self) -> Dict[str, Any]:
        """Analyze hyperparameter effectiveness"""
        if not self.hyperparameter_history:
            return {}
            
        # Collect hyperparameter values and corresponding accuracies
        param_performance = defaultdict(list)
        
        for trial_num, trial_data in self.hyperparameter_history.items():
            if trial_data.get('status') == 'completed':
                final_metrics = trial_data.get('final_metrics', {})
                accuracy = final_metrics.get('accuracy')
                
                if accuracy is not None:
                    params = trial_data.get('params', {})
                    for param_name, param_value in params.items():
                        param_performance[param_name].append((param_value, accuracy))
                        
        # Analyze each parameter
        analysis = {}
        for param_name, values in param_performance.items():
            if len(values) > 1:
                param_values, accuracies = zip(*values)
                
                analysis[param_name] = {
                    'correlation': self._calculate_correlation(param_values, accuracies),
                    'best_value': param_values[np.argmax(accuracies)],
                    'worst_value': param_values[np.argmin(accuracies)],
                    'value_range': (min(param_values), max(param_values)),
                    'performance_impact': max(accuracies) - min(accuracies)
                }
                
        return analysis
        
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export data in format suitable for visualization"""
        return {
            'trial_history': [
                {
                    'trial': trial_num,
                    'hyperparameters': data.get('params', {}),
                    'final_metrics': data.get('final_metrics', {}),
                    'status': data.get('status', 'unknown'),
                    'epochs': len(data.get('epochs', [])),
                    'start_time': data.get('start_time'),
                    'end_time': data.get('end_time')
                }
                for trial_num, data in self.hyperparameter_history.items()
            ],
            'optimization_summary': self.get_optimization_summary(),
            'hyperparameter_analysis': self.get_hyperparameter_analysis(),
            'export_time': datetime.now().isoformat()
        }
        
    def _find_convergence_epoch(self, values: List[float], 
                               window_size: int = 5, threshold: float = 0.001) -> Optional[int]:
        """Find epoch where metric converged"""
        if len(values) < window_size * 2:
            return None
            
        for i in range(window_size, len(values) - window_size):
            window_before = values[i-window_size:i]
            window_after = values[i:i+window_size]
            
            if abs(np.mean(window_after) - np.mean(window_before)) < threshold:
                return i
                
        return None
        
    def _calculate_optimization_progress(self) -> Dict[str, float]:
        """Calculate optimization progress metrics"""
        if len(self.trial_metrics) < 2:
            return {}
            
        # Get accuracies in order
        trial_numbers = sorted(self.trial_metrics.keys())
        accuracies = []
        
        for trial_num in trial_numbers:
            final_metrics = self.trial_metrics[trial_num].get('final_metrics', {})
            if 'accuracy' in final_metrics:
                accuracies.append(final_metrics['accuracy'])
                
        if len(accuracies) < 2:
            return {}
            
        # Calculate progress metrics
        best_so_far = []
        current_best = 0
        for acc in accuracies:
            current_best = max(current_best, acc)
            best_so_far.append(current_best)
            
        return {
            'total_improvement': best_so_far[-1] - best_so_far[0],
            'improvement_rate': (best_so_far[-1] - best_so_far[0]) / len(accuracies),
            'plateau_length': self._calculate_plateau_length(best_so_far),
            'convergence_ratio': sum(1 for i in range(1, len(best_so_far)) 
                                   if best_so_far[i] > best_so_far[i-1]) / len(best_so_far)
        }
        
    def _calculate_plateau_length(self, best_so_far: List[float]) -> int:
        """Calculate length of current plateau"""
        if len(best_so_far) < 2:
            return 0
            
        plateau_length = 0
        for i in range(len(best_so_far) - 1, 0, -1):
            if best_so_far[i] == best_so_far[i-1]:
                plateau_length += 1
            else:
                break
                
        return plateau_length
        
    def _calculate_improvement_rate(self, accuracies: List[float]) -> float:
        """Calculate rate of improvement over trials"""
        if len(accuracies) < 2:
            return 0.0
            
        # Simple linear regression slope
        x = np.arange(len(accuracies))
        y = np.array(accuracies)
        
        slope = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
        return slope
        
    def _calculate_correlation(self, param_values: List, accuracies: List[float]) -> float:
        """Calculate correlation between parameter values and accuracy"""
        try:
            # Convert to numeric if possible
            numeric_params = []
            for val in param_values:
                if isinstance(val, (int, float)):
                    numeric_params.append(float(val))
                elif isinstance(val, str):
                    # Try to convert string to number
                    try:
                        numeric_params.append(float(val))
                    except ValueError:
                        # Use hash for categorical values
                        numeric_params.append(hash(val) % 1000)
                else:
                    numeric_params.append(hash(str(val)) % 1000)
                    
            return np.corrcoef(numeric_params, accuracies)[0, 1]
            
        except Exception:
            return 0.0
            
    def _update_optimization_stats(self):
        """Update optimization statistics"""
        self.optimization_stats = {
            'last_update': datetime.now().isoformat(),
            'summary': self.get_optimization_summary(),
            'hyperparameter_analysis': self.get_hyperparameter_analysis()
        }
        
    def _save_metrics(self):
        """Save metrics to files"""
        try:
            # Save detailed metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(self.export_for_visualization(), f, indent=2, default=str)
                
            # Save optimization stats
            with open(self.stats_file, 'w') as f:
                json.dump(self.optimization_stats, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


# Global collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector(save_dir: str = "training_progress") -> MetricsCollector:
    """Get or create global metrics collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector(save_dir)
    return _global_collector