#!/usr/bin/env python3
"""
Enhanced Live Training Viewer - Real-time monitoring with multiple visualization options
Run this script in a separate terminal to see live training progress.

Usage:
    python live_viewer_enhanced.py                    # Auto-find progress file, terminal view
    python live_viewer_enhanced.py --terminal         # Terminal-only view
    python live_viewer_enhanced.py --gui              # GUI plots view
    python live_viewer_enhanced.py [progress_file]    # Specific file
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def find_progress_file():
    """Find an existing progress file"""
    progress_files = [
        # Standard training locations
        "experiments/results/hyperopt_task_5_5/training_progress/progress.json",
        "hyperopt_results_task_5_5_fixed/training_progress/progress.json",
        "test_all_fixes_results/training_progress/progress.json",
        "test_monitoring_results/training_progress/progress.json",
        "live_training_session/training_progress/progress.json",
        "training_progress/progress.json",
        # Test and demo locations
        "test_visualization_data/progress.json",
        "benchmark_test_progress/progress.json"
    ]
    
    for pf in progress_files:
        if Path(pf).exists():
            return pf
    return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Live Training Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python live_viewer_enhanced.py                    # Auto-find file, terminal view
    python live_viewer_enhanced.py --gui              # GUI plots view
    python live_viewer_enhanced.py progress.json      # Specific file
        """
    )
    
    parser.add_argument(
        'progress_file', 
        nargs='?', 
        help='Path to progress.json file (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--terminal', 
        action='store_true',
        help='Use terminal-based viewer (default)'
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Use GUI-based viewer with plots'
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='Use web-based dashboard'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for web dashboard (default: 8080)'
    )
    
    parser.add_argument(
        '--update-interval',
        type=float,
        default=2.0,
        help='Update interval in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Determine progress file
    if args.progress_file:
        progress_file = args.progress_file
        if not Path(progress_file).exists():
            print(f"❌ Progress file not found: {progress_file}")
            sys.exit(1)
    else:
        progress_file = find_progress_file()
        if not progress_file:
            print("❌ No progress file found. Usage:")
            print("python live_viewer_enhanced.py [progress_file.json]")
            print("\nExpected locations:")
            expected_files = [
                "experiments/results/hyperopt_task_5_5/training_progress/progress.json",
                "hyperopt_results_task_5_5_fixed/training_progress/progress.json",
                "test_all_fixes_results/training_progress/progress.json",
                "test_monitoring_results/training_progress/progress.json",
                "training_progress/progress.json"
            ]
            for pf in expected_files:
                print(f"  - {pf}")
            sys.exit(1)
    
    # Determine viewer type
    use_gui = args.gui and not args.terminal and not args.web
    use_web = args.web and not args.terminal and not args.gui
    
    print("Starting Enhanced Live Training Viewer")
    print(f"Monitoring: {progress_file}")
    mode = 'Web' if use_web else ('GUI' if use_gui else 'Terminal')
    print(f"Mode: {mode}")
    if use_web:
        print(f"Port: {args.port}")
    print("=" * 60)
    
    try:
        if use_web:
            # Try to import and use web dashboard
            try:
                from src.ml.visualization import create_web_dashboard
                dashboard = create_web_dashboard(progress_file, args.port)
                dashboard.start()
            except ImportError as e:
                print(f"Web dashboard not available: {e}")
                print("Falling back to terminal viewer...")
                use_terminal_viewer(progress_file, args.update_interval)
            except Exception as e:
                print(f"Web dashboard failed: {e}")
                print("Falling back to terminal viewer...")
                use_terminal_viewer(progress_file, args.update_interval)
        elif use_gui:
            # Try to import and use GUI viewer
            try:
                from src.ml.visualization import create_live_dashboard
                dashboard = create_live_dashboard(progress_file)
                dashboard.start()
            except ImportError as e:
                print(f"GUI viewer not available: {e}")
                print("Falling back to terminal viewer...")
                use_terminal_viewer(progress_file, args.update_interval)
            except Exception as e:
                print(f"GUI viewer failed: {e}")
                print("Falling back to terminal viewer...")
                use_terminal_viewer(progress_file, args.update_interval)
        else:
            # Use terminal viewer
            use_terminal_viewer(progress_file, args.update_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Viewer stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def use_terminal_viewer(progress_file: str, update_interval: float):
    """Use the terminal-based viewer"""
    try:
        from src.ml.visualization import create_terminal_viewer
        viewer = create_terminal_viewer(progress_file)
        viewer.update_interval = update_interval
        viewer.start_monitoring()
    except ImportError:
        # Fallback to simple viewer if visualization module not available
        print("Using simple terminal viewer...")
        simple_terminal_viewer(progress_file, update_interval)

def simple_terminal_viewer(progress_file: str, update_interval: float):
    """Simple fallback terminal viewer"""
    import json
    import time
    import os
    from datetime import datetime
    
    last_update = None
    
    print("Simple Terminal Viewer")
    print("=" * 50)
    
    try:
        while True:
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                current_update = progress.get('last_update')
                if current_update != last_update:
                    # Clear screen
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Display progress
                    print("Live Training Progress")
                    print("=" * 50)
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    trial = progress.get('trial_number', 'N/A')
                    status = progress.get('status', 'unknown')
                    print(f"Trial: {trial}")
                    print(f"Status: {status.upper()}")
                    
                    if status == 'training':
                        epoch = progress.get('current_epoch', 0)
                        epochs_completed = progress.get('epochs_completed', 0)
                        elapsed = progress.get('elapsed_time_seconds', 0)
                        print(f"Epoch: {epoch} (completed: {epochs_completed})")
                        print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
                        
                        metrics = progress.get('last_metrics', {})
                        if metrics:
                            print("Current Metrics:")
                            for key, value in metrics.items():
                                if isinstance(value, float):
                                    print(f"   {key}: {value:.4f}")
                                else:
                                    print(f"   {key}: {value}")
                    
                    elif status == 'completed':
                        total_time = progress.get('total_time_seconds', 0)
                        epochs_completed = progress.get('epochs_completed', 0)
                        print(f"Completed in {total_time:.1f}s ({total_time/60:.1f}m)")
                        print(f"Total epochs: {epochs_completed}")
                        
                        final_metrics = progress.get('final_metrics', {})
                        if final_metrics:
                            print("Final Results:")
                            for key, value in final_metrics.items():
                                if isinstance(value, float):
                                    print(f"   {key}: {value:.4f}")
                                else:
                                    print(f"   {key}: {value}")
                    
                    elif status == 'failed':
                        print("Trial failed")
                        error = progress.get('final_metrics', {}).get('error', 'Unknown error')
                        print(f"Error: {error}")
                    
                    print("=" * 50)
                    print("Press Ctrl+C to stop monitoring")
                    
                    last_update = current_update
                
                time.sleep(update_interval)
                
            except (FileNotFoundError, json.JSONDecodeError):
                print("⏳ Waiting for progress data...")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    main()