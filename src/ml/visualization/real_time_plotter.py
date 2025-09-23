"""
Real-time Training Visualization System
Provides live plots and dashboards for hyperparameter optimization progress
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
import pandas as pd
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import deque, defaultdict

# Set up matplotlib for interactive use
plt.ion()
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)

class RealTimePlotter:
    """Real-time visualization for training progress"""
    
    def __init__(self, progress_file: str, update_interval: float = 2.0):
        self.progress_file = Path(progress_file)
        self.update_interval = update_interval
        self.running = False
        
        # Data storage
        self.trial_data = defaultdict(list)
        self.epoch_data = defaultdict(lambda: defaultdict(list))
        self.last_update = None
        
        # Plot setup
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Animation
        self.ani = None
        
    def start_plotting(self):
        """Start real-time plotting"""
        self.running = True
        self._setup_plots()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, 
            interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
        
        plt.show()
        
    def stop_plotting(self):
        """Stop real-time plotting"""
        self.running = False
        if self.ani:
            self.ani.event_source.stop()
            
    def _setup_plots(self):
        """Setup the plot layout"""
        self.fig, axes_array = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Real-Time Hyperparameter Optimization Progress', fontsize=16)
        
        # Flatten axes for easier access
        axes_flat = axes_array.flatten()
        
        # Define plot types
        plot_configs = [
            ('trial_progress', 'Trial Progress', 'Trial', 'Best Accuracy'),
            ('loss_curves', 'Loss Curves (Current Trial)', 'Epoch', 'Loss'),
            ('accuracy_curves', 'Accuracy Curves (Current Trial)', 'Epoch', 'Accuracy'),
            ('training_time', 'Training Time per Trial', 'Trial', 'Time (minutes)'),
            ('hyperparameter_dist', 'Hyperparameter Distribution', 'Parameter', 'Value'),
            ('convergence', 'Optimization Convergence', 'Trial', 'Objective Value')
        ]
        
        for i, (key, title, xlabel, ylabel) in enumerate(plot_configs):
            ax = axes_flat[i]
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            self.axes[key] = ax
            
        plt.tight_layout()
        
    def _load_progress_data(self) -> Optional[Dict]:
        """Load progress data from file"""
        try:
            if not self.progress_file.exists():
                return None
                
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                
            # Check if data has been updated
            current_update = data.get('last_update')
            if current_update == self.last_update:
                return None
                
            self.last_update = current_update
            return data
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not load progress data: {e}")
            return None
            
    def _update_plots(self, frame):
        """Update all plots with new data"""
        if not self.running:
            return
            
        data = self._load_progress_data()
        if not data:
            return
            
        # Update data storage
        self._process_new_data(data)
        
        # Update individual plots
        self._update_trial_progress()
        self._update_loss_curves()
        self._update_accuracy_curves()
        self._update_training_time()
        self._update_hyperparameter_dist()
        self._update_convergence()
        
        # Update figure title with current status
        status = data.get('status', 'unknown')
        trial = data.get('trial_number', 'N/A')
        self.fig.suptitle(
            f'Real-Time Optimization Progress - Trial {trial} ({status.upper()})',
            fontsize=16
        )
        
    def _process_new_data(self, data: Dict):
        """Process and store new data"""
        trial_num = data.get('trial_number')
        if trial_num is None:
            return
            
        status = data.get('status', 'unknown')
        
        # Store trial-level data
        if status == 'completed':
            final_metrics = data.get('final_metrics', {})
            if 'accuracy' in final_metrics:
                self.trial_data['trials'].append(trial_num)
                self.trial_data['accuracy'].append(final_metrics['accuracy'])
                self.trial_data['training_time'].append(
                    final_metrics.get('training_time', 0) / 60  # Convert to minutes
                )
                
        # Store epoch-level data for current trial
        if status == 'training':
            epoch = data.get('current_epoch', 0)
            metrics = data.get('last_metrics', {})
            
            if metrics:
                self.epoch_data[trial_num]['epochs'].append(epoch)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.epoch_data[trial_num][key].append(value)
                        
    def _update_trial_progress(self):
        """Update trial progress plot"""
        ax = self.axes['trial_progress']
        ax.clear()
        
        if not self.trial_data['trials']:
            ax.text(0.5, 0.5, 'Waiting for trial data...', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        trials = self.trial_data['trials']
        accuracies = self.trial_data['accuracy']
        
        # Plot individual trials
        ax.scatter(trials, accuracies, alpha=0.6, s=50)
        
        # Plot best accuracy line
        best_accuracies = []
        current_best = 0
        for acc in accuracies:
            current_best = max(current_best, acc)
            best_accuracies.append(current_best)
            
        ax.plot(trials, best_accuracies, 'r-', linewidth=2, label='Best So Far')
        
        ax.set_title('Trial Progress')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _update_loss_curves(self):
        """Update loss curves for current trial"""
        ax = self.axes['loss_curves']
        ax.clear()
        
        # Get current trial data
        current_trials = list(self.epoch_data.keys())
        if not current_trials:
            ax.text(0.5, 0.5, 'Waiting for epoch data...', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Show last few trials
        recent_trials = sorted(current_trials)[-3:]
        
        for i, trial in enumerate(recent_trials):
            trial_data = self.epoch_data[trial]
            if 'epochs' in trial_data and 'loss' in trial_data:
                epochs = trial_data['epochs']
                losses = trial_data['loss']
                if epochs and losses:
                    color = self.colors[i % len(self.colors)]
                    ax.plot(epochs, losses, color=color, 
                           label=f'Trial {trial}', linewidth=2)
                    
        ax.set_title('Loss Curves (Recent Trials)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _update_accuracy_curves(self):
        """Update accuracy curves for current trial"""
        ax = self.axes['accuracy_curves']
        ax.clear()
        
        # Get current trial data
        current_trials = list(self.epoch_data.keys())
        if not current_trials:
            ax.text(0.5, 0.5, 'Waiting for epoch data...', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Show last few trials
        recent_trials = sorted(current_trials)[-3:]
        
        for i, trial in enumerate(recent_trials):
            trial_data = self.epoch_data[trial]
            accuracy_keys = [k for k in trial_data.keys() if 'accuracy' in k.lower()]
            
            for acc_key in accuracy_keys:
                if 'epochs' in trial_data and acc_key in trial_data:
                    epochs = trial_data['epochs']
                    accuracies = trial_data[acc_key]
                    if epochs and accuracies:
                        color = self.colors[i % len(self.colors)]
                        ax.plot(epochs, accuracies, color=color, 
                               label=f'Trial {trial} - {acc_key}', linewidth=2)
                        
        ax.set_title('Accuracy Curves (Recent Trials)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _update_training_time(self):
        """Update training time plot"""
        ax = self.axes['training_time']
        ax.clear()
        
        if not self.trial_data['trials']:
            ax.text(0.5, 0.5, 'Waiting for timing data...', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        trials = self.trial_data['trials']
        times = self.trial_data['training_time']
        
        ax.bar(trials, times, alpha=0.7)
        
        # Add average line
        if times:
            avg_time = np.mean(times)
            ax.axhline(y=avg_time, color='r', linestyle='--', 
                      label=f'Average: {avg_time:.1f}m')
            ax.legend()
            
        ax.set_title('Training Time per Trial')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Time (minutes)')
        ax.grid(True, alpha=0.3)
        
    def _update_hyperparameter_dist(self):
        """Update hyperparameter distribution plot"""
        ax = self.axes['hyperparameter_dist']
        ax.clear()
        
        # This would need hyperparameter data from the progress file
        # For now, show placeholder
        ax.text(0.5, 0.5, 'Hyperparameter distribution\n(requires extended logging)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Hyperparameter Distribution')
        
    def _update_convergence(self):
        """Update convergence plot"""
        ax = self.axes['convergence']
        ax.clear()
        
        if not self.trial_data['trials']:
            ax.text(0.5, 0.5, 'Waiting for convergence data...', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        trials = self.trial_data['trials']
        accuracies = self.trial_data['accuracy']
        
        # Plot convergence
        ax.plot(trials, accuracies, 'bo-', alpha=0.6, markersize=4)
        
        # Add trend line
        if len(trials) > 1:
            z = np.polyfit(trials, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(trials, p(trials), "r--", alpha=0.8, label='Trend')
            ax.legend()
            
        ax.set_title('Optimization Convergence')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.grid(True, alpha=0.3)


class LiveDashboard:
    """Enhanced live dashboard with multiple views"""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.plotter = RealTimePlotter(progress_file)
        
    def start(self):
        """Start the live dashboard"""
        print("🚀 Starting Live Training Dashboard")
        print("=" * 50)
        print(f"📊 Monitoring: {self.progress_file}")
        print("💡 Close the plot window to stop monitoring")
        print("=" * 50)
        
        try:
            self.plotter.start_plotting()
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        finally:
            self.plotter.stop_plotting()


def create_live_dashboard(progress_file: str) -> LiveDashboard:
    """Create a live dashboard for monitoring training"""
    return LiveDashboard(progress_file)


if __name__ == "__main__":
    import sys
    
    # Default progress file locations
    progress_files = [
        "experiments/results/hyperopt_task_5_5/training_progress/progress.json",
        "hyperopt_results_task_5_5_fixed/training_progress/progress.json", 
        "test_all_fixes_results/training_progress/progress.json",
        "test_monitoring_results/training_progress/progress.json",
        "training_progress/progress.json"
    ]
    
    # Use command line argument or find existing file
    if len(sys.argv) > 1:
        progress_file = sys.argv[1]
    else:
        progress_file = None
        for pf in progress_files:
            if Path(pf).exists():
                progress_file = pf
                break
                
        if not progress_file:
            print("❌ No progress file found. Usage:")
            print("python real_time_plotter.py [progress_file.json]")
            print("\nExpected locations:")
            for pf in progress_files:
                print(f"  - {pf}")
            sys.exit(1)
    
    # Start dashboard
    dashboard = create_live_dashboard(progress_file)
    dashboard.start()