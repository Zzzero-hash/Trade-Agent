#!/usr/bin/env python3
"""
Find Progress Files
Helps locate and diagnose progress files for visualization
"""
import os
import json
from pathlib import Path
from datetime import datetime

def find_all_progress_files():
    """Find all progress files in the project"""
    print("🔍 Searching for progress files...")
    print("=" * 60)
    
    # Search patterns
    search_patterns = [
        "**/progress.json",
        "**/training_progress/progress.json",
        "**/*progress*.json"
    ]
    
    found_files = []
    
    for pattern in search_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file() and "progress" in file_path.name:
                found_files.append(file_path)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if not found_files:
        print("❌ No progress files found")
        print("\n💡 To create progress files:")
        print("   1. Run: python start_training_with_monitoring.py")
        print("   2. Or run: python test_monitoring.py")
        print("   3. Then monitor with: python live_viewer_enhanced.py")
        return []
    
    print(f"✅ Found {len(found_files)} progress file(s):")
    
    for i, file_path in enumerate(found_files, 1):
        print(f"\n{i}. {file_path}")
        
        # Check file info
        try:
            stat = file_path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"   Size: {size} bytes")
            print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Try to read content
            if size > 0:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    trial = data.get('trial_number', 'N/A')
                    status = data.get('status', 'unknown')
                    
                    print(f"   Trial: {trial}, Status: {status}")
                    
                    if 'last_metrics' in data:
                        metrics = data['last_metrics']
                        if 'loss' in metrics:
                            print(f"   Loss: {metrics['loss']:.4f}")
                        if 'classification_accuracy' in metrics:
                            print(f"   Accuracy: {metrics['classification_accuracy']:.4f}")
                            
                except json.JSONDecodeError:
                    print("   ⚠️  Invalid JSON format")
                except Exception as e:
                    print(f"   ⚠️  Error reading: {e}")
            else:
                print("   ⚠️  Empty file")
                
        except Exception as e:
            print(f"   ❌ Error accessing file: {e}")
    
    return found_files

def check_training_directories():
    """Check for training result directories"""
    print("\n🗂️  Training Result Directories:")
    print("=" * 60)
    
    # Common training directories
    training_dirs = [
        "experiments/results",
        "hyperopt_results",
        "hyperopt_results_task_5_5",
        "hyperopt_results_task_5_5_fixed",
        "test_all_fixes_results",
        "test_monitoring_results",
        "live_training_session"
    ]
    
    found_dirs = []
    
    for dir_name in training_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            found_dirs.append(dir_path)
            
            print(f"✅ {dir_name}")
            
            # Check for progress subdirectory
            progress_dir = dir_path / "training_progress"
            if progress_dir.exists():
                print(f"   📊 Has training_progress/ directory")
                
                # Check for progress.json
                progress_file = progress_dir / "progress.json"
                if progress_file.exists():
                    print(f"   📄 Has progress.json file")
                else:
                    print(f"   ⚠️  Missing progress.json file")
            else:
                print(f"   ⚠️  No training_progress/ directory")
        else:
            print(f"❌ {dir_name} (not found)")
    
    if not found_dirs:
        print("❌ No training directories found")
        print("\n💡 Create training session:")
        print("   python start_training_with_monitoring.py")
    
    return found_dirs

def test_viewer_with_files(progress_files):
    """Test viewer with found files"""
    if not progress_files:
        return
        
    print("\n🧪 Testing Viewer Commands:")
    print("=" * 60)
    
    for i, file_path in enumerate(progress_files, 1):
        print(f"\n{i}. To monitor {file_path}:")
        print(f"   python live_viewer_enhanced.py {file_path}")
        print(f"   python live_viewer_enhanced.py --terminal {file_path}")
        print(f"   python live_viewer_enhanced.py --gui {file_path}")

def show_auto_detection_paths():
    """Show paths that auto-detection searches"""
    print("\n🎯 Auto-Detection Search Paths:")
    print("=" * 60)
    
    search_paths = [
        "experiments/results/hyperopt_task_5_5/training_progress/progress.json",
        "hyperopt_results_task_5_5_fixed/training_progress/progress.json",
        "test_all_fixes_results/training_progress/progress.json",
        "test_monitoring_results/training_progress/progress.json",
        "live_training_session/training_progress/progress.json",
        "training_progress/progress.json"
    ]
    
    for path in search_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")

def main():
    """Main diagnostic function"""
    print("🔧 Progress File Diagnostic Tool")
    print("=" * 60)
    
    # Find progress files
    progress_files = find_all_progress_files()
    
    # Check training directories
    check_training_directories()
    
    # Show auto-detection paths
    show_auto_detection_paths()
    
    # Test viewer commands
    test_viewer_with_files(progress_files)
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    
    if progress_files:
        print(f"✅ Found {len(progress_files)} progress file(s)")
        print("💡 You can now run: python live_viewer_enhanced.py")
    else:
        print("❌ No progress files found")
        print("💡 Start training first:")
        print("   python start_training_with_monitoring.py")
        print("   # Then in another terminal:")
        print("   python live_viewer_enhanced.py")

if __name__ == "__main__":
    main()