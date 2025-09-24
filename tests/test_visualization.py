#!/usr/bin/env python3
"""
Test the visualization system with mock data
"""
import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_progress_data():
    """Create mock progress data for testing"""
    test_dir = Path("test_visualization_data")
    test_dir.mkdir(exist_ok=True)
    
    progress_file = test_dir / "progress.json"
    
    # Mock training data
    mock_trials = [
        {
            'trial_number': 1,
            'status': 'completed',
            'epochs_completed': 50,
            'total_time_seconds': 120.5,
            'final_metrics': {
                'accuracy': 0.7234,
                'training_time': 120.5,
                'model_size': 1024000
            }
        },
        {
            'trial_number': 2,
            'status': 'completed', 
            'epochs_completed': 45,
            'total_time_seconds': 98.2,
            'final_metrics': {
                'accuracy': 0.7891,
                'training_time': 98.2,
                'model_size': 896000
            }
        },
        {
            'trial_number': 3,
            'status': 'training',
            'current_epoch': 25,
            'epochs_completed': 24,
            'elapsed_time_seconds': 67.3,
            'last_metrics': {
                'loss': 0.3456,
                'classification_accuracy': 0.8123,
                'val_loss': 0.3789,
                'val_accuracy': 0.7956
            }
        }
    ]
    
    return progress_file, mock_trials

def simulate_training_progress(progress_file: Path, mock_trials: list):
    """Simulate training progress by updating the progress file"""
    logger.info("🎭 Starting mock training simulation...")
    
    try:
        # Simulate completed trials first
        for trial_data in mock_trials[:2]:
            trial_data['last_update'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(trial_data, f, indent=2)
            
            logger.info(f"📊 Simulated trial {trial_data['trial_number']} completion")
            time.sleep(3)
        
        # Simulate ongoing training
        ongoing_trial = mock_trials[2].copy()
        
        for epoch in range(1, 51):
            # Update epoch data
            ongoing_trial['current_epoch'] = epoch
            ongoing_trial['epochs_completed'] = epoch - 1
            ongoing_trial['elapsed_time_seconds'] = epoch * 2.5
            
            # Simulate improving metrics
            base_loss = 0.5
            base_acc = 0.6
            
            # Add some realistic variation
            import random
            loss_noise = random.uniform(-0.02, 0.01)
            acc_noise = random.uniform(-0.01, 0.02)
            
            ongoing_trial['last_metrics'] = {
                'loss': max(0.1, base_loss - (epoch * 0.008) + loss_noise),
                'classification_accuracy': min(0.95, base_acc + (epoch * 0.006) + acc_noise),
                'val_loss': max(0.1, base_loss - (epoch * 0.007) + loss_noise * 1.2),
                'val_accuracy': min(0.92, base_acc + (epoch * 0.005) + acc_noise * 0.8)
            }
            
            ongoing_trial['last_update'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(ongoing_trial, f, indent=2)
            
            if epoch % 5 == 0:
                logger.info(f"📈 Simulated epoch {epoch}/50")
            
            time.sleep(1)  # Update every second for demo
        
        # Mark trial as completed
        ongoing_trial['status'] = 'completed'
        ongoing_trial['total_time_seconds'] = ongoing_trial['elapsed_time_seconds']
        ongoing_trial['final_metrics'] = {
            'accuracy': ongoing_trial['last_metrics']['classification_accuracy'],
            'training_time': ongoing_trial['total_time_seconds'],
            'model_size': 1152000
        }
        ongoing_trial['last_update'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(ongoing_trial, f, indent=2)
        
        logger.info("✅ Mock training simulation completed")
        
    except Exception as e:
        logger.error(f"❌ Simulation error: {e}")

def test_terminal_viewer():
    """Test the terminal viewer"""
    logger.info("🖥️  Testing terminal viewer...")
    
    try:
        from src.ml.visualization import create_terminal_viewer
        
        progress_file, mock_trials = create_mock_progress_data()
        
        # Start simulation in background
        sim_thread = threading.Thread(
            target=simulate_training_progress,
            args=(progress_file, mock_trials),
            daemon=True
        )
        sim_thread.start()
        
        # Start viewer
        viewer = create_terminal_viewer(str(progress_file))
        viewer.update_interval = 1.0  # Fast updates for demo
        
        print("🚀 Starting terminal viewer test...")
        print("⏰ Will run for 30 seconds, then stop automatically")
        print("Press Ctrl+C to stop early")
        
        # Run for limited time
        start_time = time.time()
        try:
            while time.time() - start_time < 30:
                viewer._update_display()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        logger.info("✅ Terminal viewer test completed")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Terminal viewer import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Terminal viewer test failed: {e}")
        return False

def test_gui_viewer():
    """Test the GUI viewer"""
    logger.info("🎨 Testing GUI viewer...")
    
    try:
        from src.ml.visualization import create_live_dashboard
        
        progress_file, mock_trials = create_mock_progress_data()
        
        # Start simulation in background
        sim_thread = threading.Thread(
            target=simulate_training_progress,
            args=(progress_file, mock_trials),
            daemon=True
        )
        sim_thread.start()
        
        # Start dashboard
        dashboard = create_live_dashboard(str(progress_file))
        
        print("🚀 Starting GUI viewer test...")
        print("🖼️  Close the plot window to stop the test")
        
        dashboard.start()
        
        logger.info("✅ GUI viewer test completed")
        return True
        
    except ImportError as e:
        logger.error(f"❌ GUI viewer import failed: {e}")
        logger.info("💡 This is expected if matplotlib GUI backend is not available")
        return False
    except Exception as e:
        logger.error(f"❌ GUI viewer test failed: {e}")
        return False

def test_enhanced_viewer():
    """Test the enhanced viewer script"""
    logger.info("🔧 Testing enhanced viewer script...")
    
    try:
        progress_file, mock_trials = create_mock_progress_data()
        
        # Start simulation in background
        sim_thread = threading.Thread(
            target=simulate_training_progress,
            args=(progress_file, mock_trials),
            daemon=True
        )
        sim_thread.start()
        
        # Test command line interface
        import subprocess
        
        cmd = [
            sys.executable, 
            "live_viewer_enhanced.py",
            "--terminal",
            str(progress_file)
        ]
        
        logger.info(f"🚀 Running: {' '.join(cmd)}")
        
        # Run for limited time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let it run for a few seconds
        time.sleep(5)
        process.terminate()
        
        stdout, stderr = process.communicate(timeout=5)
        
        if process.returncode in [0, -15]:  # 0 = success, -15 = terminated
            logger.info("✅ Enhanced viewer script test completed")
            return True
        else:
            logger.error(f"❌ Enhanced viewer failed with code {process.returncode}")
            logger.error(f"stderr: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced viewer test failed: {e}")
        return False

def main():
    """Run all visualization tests"""
    logger.info("🧪 Starting visualization system tests...")
    
    results = {}
    
    # Test terminal viewer
    results['terminal'] = test_terminal_viewer()
    
    # Test GUI viewer (may fail if no display)
    results['gui'] = test_gui_viewer()
    
    # Test enhanced viewer script
    results['enhanced'] = test_enhanced_viewer()
    
    # Summary
    logger.info("📊 Test Results Summary:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name.capitalize()} viewer: {status}")
    
    # Overall result
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All visualization tests passed!")
        return True
    elif passed_tests > 0:
        logger.info("⚠️  Some visualization tests passed")
        return True
    else:
        logger.error("💥 All visualization tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)