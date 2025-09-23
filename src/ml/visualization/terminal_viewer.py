"""
Terminal-based Training Viewer
Lightweight alternative to GUI plotting for monitoring training progress
"""
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TerminalViewer:
    """Terminal-based real-time training viewer"""
    
    def __init__(self, progress_file: str, update_interval: float = 2.0):
        self.progress_file = Path(progress_file)
        self.update_interval = update_interval
        self.running = False
        self.last_update = None
        
        # Data storage for statistics
        self.trial_history = []
        self.current_trial_epochs = []
        
    def start_monitoring(self):
        """Start terminal monitoring"""
        self.running = True
        print("🚀 Live Training Monitor - Terminal View")
        print("=" * 70)
        print(f"📊 Monitoring: {self.progress_file}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 70)
        
        try:
            while self.running:
                self._update_display()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
        finally:
            self.running = False
            
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
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
            
    def _update_display(self):
        """Update terminal display"""
        data = self._load_progress_data()
        if not data:
            return
            
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Update data storage
        self._process_data(data)
        
        # Display current status
        self._display_header()
        self._display_current_trial(data)
        self._display_statistics()
        self._display_recent_trials()
        
    def _process_data(self, data: Dict):
        """Process and store data for statistics"""
        trial_num = data.get('trial_number')
        status = data.get('status', 'unknown')
        
        # Store completed trials
        if status == 'completed' and trial_num is not None:
            final_metrics = data.get('final_metrics', {})
            trial_info = {
                'trial': trial_num,
                'accuracy': final_metrics.get('accuracy', 0),
                'training_time': final_metrics.get('training_time', 0),
                'model_size': final_metrics.get('model_size', 0),
                'completed_at': datetime.now()
            }
            
            # Add if not already present
            if not any(t['trial'] == trial_num for t in self.trial_history):
                self.trial_history.append(trial_info)
                
        # Store current trial epoch data
        if status == 'training':
            epoch = data.get('current_epoch', 0)
            metrics = data.get('last_metrics', {})
            
            epoch_info = {
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            # Keep only recent epochs for current trial (bounded to prevent memory leaks)
            self.current_trial_epochs = [e for e in self.current_trial_epochs 
                                       if e['timestamp'] > datetime.now() - timedelta(minutes=10)]
            self.current_trial_epochs.append(epoch_info)
            
            # Ensure we don't exceed maximum size
            if len(self.current_trial_epochs) > 50:
                self.current_trial_epochs = self.current_trial_epochs[-50:]
            
    def _display_header(self):
        """Display header information"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("🚀 Live Training Monitor")
        print("=" * 70)
        print(f"📅 {now}")
        print()
        
    def _display_current_trial(self, data: Dict):
        """Display current trial information"""
        trial = data.get('trial_number', 'N/A')
        status = data.get('status', 'unknown')
        
        print(f"🔢 Current Trial: {trial}")
        print(f"📊 Status: {status.upper()}")
        
        if status == 'training':
            epoch = data.get('current_epoch', 0)
            epochs_completed = data.get('epochs_completed', 0)
            elapsed = data.get('elapsed_time_seconds', 0)
            
            print(f"⏱️  Epoch: {epoch} (completed: {epochs_completed})")
            print(f"⏰ Elapsed: {self._format_time(elapsed)}")
            
            # Display current metrics
            metrics = data.get('last_metrics', {})
            if metrics:
                print("📈 Current Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                        
        elif status == 'completed':
            total_time = data.get('total_time_seconds', 0)
            epochs_completed = data.get('epochs_completed', 0)
            print(f"✅ Completed in {self._format_time(total_time)}")
            print(f"📊 Total epochs: {epochs_completed}")
            
            final_metrics = data.get('final_metrics', {})
            if final_metrics:
                print("🎯 Final Results:")
                for key, value in final_metrics.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                        
        elif status == 'failed':
            print("❌ Trial failed")
            error = data.get('final_metrics', {}).get('error', 'Unknown error')
            print(f"💥 Error: {error}")
            
        print()
        
    def _display_statistics(self):
        """Display optimization statistics"""
        if not self.trial_history:
            print("📊 Statistics: Waiting for completed trials...")
            print()
            return
            
        print("📊 Optimization Statistics:")
        
        # Basic stats
        total_trials = len(self.trial_history)
        accuracies = [t['accuracy'] for t in self.trial_history]
        times = [t['training_time'] for t in self.trial_history]
        
        best_accuracy = max(accuracies) if accuracies else 0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_time = sum(times) / len(times) if times else 0
        
        print(f"   Total Trials: {total_trials}")
        print(f"   Best Accuracy: {best_accuracy:.4f}")
        print(f"   Average Accuracy: {avg_accuracy:.4f}")
        print(f"   Average Time: {self._format_time(avg_time)}")
        
        # Recent performance
        if len(self.trial_history) >= 5:
            recent_accuracies = accuracies[-5:]
            recent_avg = sum(recent_accuracies) / len(recent_accuracies)
            print(f"   Recent 5 Avg: {recent_avg:.4f}")
            
        print()
        
    def _display_recent_trials(self):
        """Display recent trial results"""
        if not self.trial_history:
            return
            
        print("🏆 Recent Trial Results:")
        
        # Show last 5 trials
        recent_trials = self.trial_history[-5:]
        
        print("   Trial | Accuracy | Time     | Status")
        print("   ------|----------|----------|--------")
        
        for trial in recent_trials:
            trial_num = trial['trial']
            accuracy = trial['accuracy']
            time_str = self._format_time(trial['training_time'])
            
            # Mark best trial
            is_best = accuracy == max(t['accuracy'] for t in self.trial_history)
            status = "🥇 BEST" if is_best else "✅ Done"
            
            print(f"   {trial_num:5d} | {accuracy:8.4f} | {time_str:8s} | {status}")
            
        print()
        
        # Show progress bar for current optimization
        if len(self.trial_history) > 1:
            self._display_progress_bar()
            
    def _display_progress_bar(self):
        """Display a simple progress visualization"""
        accuracies = [t['accuracy'] for t in self.trial_history]
        
        # Simple trend analysis
        if len(accuracies) >= 3:
            recent_trend = accuracies[-3:]
            if recent_trend[-1] > recent_trend[0]:
                trend = "📈 Improving"
            elif recent_trend[-1] < recent_trend[0]:
                trend = "📉 Declining" 
            else:
                trend = "➡️  Stable"
                
            print(f"📊 Trend: {trend}")
            
        # Simple accuracy histogram
        print("📊 Accuracy Distribution:")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist = [0] * (len(bins) - 1)
        
        for acc in accuracies:
            for i in range(len(bins) - 1):
                if bins[i] <= acc < bins[i + 1]:
                    hist[i] += 1
                    break
                    
        max_count = max(hist) if hist else 1
        for i, count in enumerate(hist):
            bar_length = int(20 * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({count})")
            
        print()
        
    def _format_time(self, seconds: float) -> str:
        """Format time in a readable way"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def create_terminal_viewer(progress_file: str) -> TerminalViewer:
    """Create a terminal viewer for monitoring training"""
    return TerminalViewer(progress_file)


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
            print("python terminal_viewer.py [progress_file.json]")
            print("\nExpected locations:")
            for pf in progress_files:
                print(f"  - {pf}")
            sys.exit(1)
    
    # Start viewer
    viewer = create_terminal_viewer(progress_file)
    viewer.start_monitoring()