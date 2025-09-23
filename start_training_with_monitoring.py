#!/usr/bin/env python3
"""
Start Training with Monitoring
Creates a training session that generates progress files for visualization
"""
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_simple_training_session():
    """Create a simple training session with progress monitoring"""
    from src.ml.training_monitor import get_monitor
    
    print("🚀 Starting Training Session with Monitoring")
    print("=" * 60)
    
    # Create monitor
    results_dir = "live_training_session"
    monitor = get_monitor(f"{results_dir}/training_progress")
    
    print(f"📊 Progress files will be created in: {results_dir}/training_progress/")
    print("💡 In another terminal, run: python live_viewer_enhanced.py")
    print("=" * 60)
    
    try:
        # Simulate multiple trials
        for trial in range(1, 6):  # 5 trials
            print(f"\n🔥 Starting Trial {trial}/5")
            
            # Start trial with hyperparameters
            hyperparams = {
                'learning_rate': 0.001 + trial * 0.0001,
                'batch_size': 32 + trial * 8,
                'hidden_size': 128 + trial * 32,
                'optimizer': 'adam',
                'scheduler': 'cosine'
            }
            
            monitor.start_trial(trial, hyperparams)
            
            # Simulate training epochs
            num_epochs = 20 + trial * 5  # Variable epoch count
            for epoch in range(num_epochs):
                # Simulate training progress
                base_loss = 0.8
                base_acc = 0.5
                
                # Add some realistic variation
                import random
                loss_noise = random.uniform(-0.02, 0.01)
                acc_noise = random.uniform(-0.01, 0.02)
                
                # Simulate improving metrics
                current_loss = max(0.1, base_loss - (epoch * 0.02) + loss_noise)
                current_acc = min(0.95, base_acc + (epoch * 0.015) + acc_noise)
                
                metrics = {
                    'loss': current_loss,
                    'classification_accuracy': current_acc,
                    'val_loss': current_loss + 0.05,
                    'val_accuracy': current_acc - 0.02,
                    'learning_rate': hyperparams['learning_rate'] * (0.95 ** epoch)
                }
                
                monitor.update_epoch(epoch, metrics)
                
                # Print progress every 5 epochs
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:2d}: Loss={current_loss:.4f}, Acc={current_acc:.4f}")
                
                # Simulate training time
                time.sleep(0.5)  # 0.5 seconds per epoch
                
            # Complete trial
            final_metrics = {
                'accuracy': metrics['classification_accuracy'],
                'training_time': num_epochs * 0.5,
                'model_size': 1000000 + trial * 100000
            }
            
            monitor.finish_trial(trial, final_metrics)
            print(f"✅ Trial {trial} completed: Accuracy={final_metrics['accuracy']:.4f}")
            
            # Brief pause between trials
            time.sleep(1)
            
        print("\n🎉 Training session completed!")
        print("📊 Progress files are available for viewing")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
def show_monitoring_instructions():
    """Show instructions for monitoring"""
    print("\n" + "=" * 60)
    print("📖 MONITORING INSTRUCTIONS")
    print("=" * 60)
    print("1. Keep this terminal running (training in progress)")
    print("2. Open a NEW terminal window")
    print("3. Run one of these commands:")
    print()
    print("   # Auto-detect progress file")
    print("   python live_viewer_enhanced.py")
    print()
    print("   # Terminal viewer")
    print("   python live_viewer_enhanced.py --terminal")
    print()
    print("   # GUI plots (if available)")
    print("   python live_viewer_enhanced.py --gui")
    print()
    print("   # Web dashboard")
    print("   python live_viewer_enhanced.py --web")
    print()
    print("   # Specific file")
    print("   python live_viewer_enhanced.py live_training_session/training_progress/progress.json")
    print()
    print("=" * 60)

def main():
    """Main function"""
    print("🎯 Training Session Starter")
    print("This script creates a training session that you can monitor in real-time")
    print()
    
    # Show instructions first
    show_monitoring_instructions()
    
    # Ask user if they want to continue
    try:
        response = input("\nStart training session? (y/N): ").strip().lower()
        if response != 'y':
            print("👋 Training cancelled")
            return
            
        # Start training
        create_simple_training_session()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()