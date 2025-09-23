"""
Training Progress Monitor - Surgical Addition

This module provides real-time monitoring of training progress without
disrupting existing functionality.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, save_dir: str = "training_progress"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.save_dir / "progress.json"
        self.hyperparams_file = self.save_dir / "hyperparameters.json"
        self.current_trial = None
        self.current_epoch = None
        self.start_time = None
        self.trial_hyperparams = {}
        
    def start_trial(self, trial_number: int, trial_params: Dict[str, Any]):
        """Start monitoring a new trial"""
        self.current_trial = trial_number
        self.current_epoch = 0
        self.start_time = time.time()
        
        # Store hyperparameters for visualization
        self.trial_hyperparams[trial_number] = trial_params
        self._save_hyperparams()
        
        progress = {
            'trial_number': trial_number,
            'trial_params': trial_params,
            'start_time': datetime.now().isoformat(),
            'status': 'started',
            'current_epoch': 0,
            'epochs_completed': 0,
            'last_update': datetime.now().isoformat()
        }
        
        self._save_progress(progress)
        logger.info(f"Started monitoring trial {trial_number}")
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Update progress for current epoch"""
        if self.current_trial is None:
            return
            
        self.current_epoch = epoch
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        progress = {
            'trial_number': self.current_trial,
            'status': 'training',
            'current_epoch': epoch,
            'epochs_completed': epoch + 1,
            'elapsed_time_seconds': elapsed_time,
            'last_metrics': metrics,
            'last_update': datetime.now().isoformat()
        }
        
        self._save_progress(progress)
        
        # Log every 5 epochs to avoid spam
        if epoch % 5 == 0:
            loss_val = metrics.get('total_loss', 'N/A')
            acc_val = metrics.get('classification_accuracy', 'N/A')
            
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
            acc_str = f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else str(acc_val)
            
            logger.info(f"Trial {self.current_trial} - Epoch {epoch}: "
                       f"Loss={loss_str}, Acc={acc_str}")
    
    def finish_trial(self, trial_number: int, final_metrics: Dict[str, float], status: str = "completed"):
        """Finish monitoring current trial"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        progress = {
            'trial_number': trial_number,
            'status': status,
            'epochs_completed': self.current_epoch + 1 if self.current_epoch else 0,
            'total_time_seconds': elapsed_time,
            'final_metrics': final_metrics,
            'completion_time': datetime.now().isoformat()
        }
        
        self._save_progress(progress)
        logger.info(f"Trial {trial_number} {status} in {elapsed_time:.1f}s")
        
        self.current_trial = None
        self.current_epoch = None
        self.start_time = None
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def _save_hyperparams(self):
        """Save hyperparameters to file"""
        try:
            with open(self.hyperparams_file, 'w') as f:
                json.dump(self.trial_hyperparams, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save hyperparameters: {e}")


class LiveProgressViewer:
    """Live progress viewer that runs in background"""
    
    def __init__(self, progress_file: str = "training_progress/progress.json"):
        self.progress_file = Path(progress_file)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the live viewer in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("Live progress viewer started - check training_progress/progress.json")
    
    def stop(self):
        """Stop the live viewer"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        last_update = None
        
        while self.running:
            try:
                if self.progress_file.exists():
                    with open(self.progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    current_update = progress.get('last_update')
                    if current_update != last_update:
                        self._print_progress(progress)
                        last_update = current_update
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                # Silently continue on errors
                time.sleep(5)
    
    def _print_progress(self, progress: Dict[str, Any]):
        """Print current progress"""
        trial = progress.get('trial_number', 'N/A')
        status = progress.get('status', 'unknown')
        epoch = progress.get('current_epoch', 0)
        
        if status == 'training':
            metrics = progress.get('last_metrics', {})
            loss = metrics.get('total_loss', 'N/A')
            acc = metrics.get('classification_accuracy', 'N/A')
            elapsed = progress.get('elapsed_time_seconds', 0)
            
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Trial {trial} | Epoch {epoch} | "
                  f"Loss: {loss_str} | "
                  f"Acc: {acc_str} | "
                  f"Time: {elapsed:.0f}s", end="", flush=True)


# Global monitor instance
_global_monitor: Optional[TrainingMonitor] = None

def get_monitor(save_dir: str = "training_progress") -> TrainingMonitor:
    """Get or create global training monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TrainingMonitor(save_dir)
    return _global_monitor