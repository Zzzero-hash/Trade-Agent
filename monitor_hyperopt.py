#!/usr/bin/env python3
"""
Monitor hyperparameter optimization progress
"""

import os
import time
import json
from pathlib import Path

def monitor_progress():
    """Monitor the hyperparameter optimization progress"""
    
    results_dirs = [
        "experiments/results/hyperopt_task_5_5",
        "hyperopt_results_task_5_5_fixed", 
        "test_hyperopt_fix_results",
        "test_all_fixes_results"
    ]
    
    print("Monitoring hyperparameter optimization progress...")
    print("="*60)
    
    while True:
        found_activity = False
        
        for results_dir in results_dirs:
            results_path = Path(results_dir)
            if results_path.exists():
                print(f"\nResults directory: {results_dir}")
                
                # Check for analysis file
                analysis_file = results_path / "optimization_analysis.json"
                if analysis_file.exists():
                    try:
                        with open(analysis_file, 'r') as f:
                            analysis = json.load(f)
                        
                        summary = analysis.get('optimization_summary', {})
                        total_trials = summary.get('total_trials', 0)
                        completed_trials = summary.get('completed_trials', 0)
                        pruned_trials = summary.get('pruned_trials', 0)
                        failed_trials = summary.get('failed_trials', 0)
                        
                        print(f"  Total trials: {total_trials}")
                        print(f"  Completed: {completed_trials}")
                        print(f"  Pruned: {pruned_trials}")
                        print(f"  Failed: {failed_trials}")
                        
                        if total_trials > 0:
                            found_activity = True
                            
                    except Exception as e:
                        print(f"  Error reading analysis: {e}")
                
                # Check for study file
                study_file = results_path / "optuna_study.pkl"
                if study_file.exists():
                    file_size = study_file.stat().st_size
                    mod_time = study_file.stat().st_mtime
                    print(f"  Study file: {file_size} bytes, modified {time.ctime(mod_time)}")
                    found_activity = True
                
                # Check for log files
                log_files = list(results_path.glob("*.log"))
                if log_files:
                    for log_file in log_files:
                        file_size = log_file.stat().st_size
                        print(f"  Log: {log_file.name} ({file_size} bytes)")
        
        # Check main log files
        log_files = ["hyperopt_fixed.log", "experiments/logs/hyperopt_optimization.log"]
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                file_size = log_path.stat().st_size
                mod_time = log_path.stat().st_mtime
                print(f"\nMain log: {log_file} ({file_size} bytes, modified {time.ctime(mod_time)})")
                found_activity = True
        
        if not found_activity:
            print("No hyperparameter optimization activity detected")
        
        print("\n" + "="*60)
        print("Press Ctrl+C to stop monitoring")
        
        try:
            time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break

if __name__ == "__main__":
    monitor_progress()