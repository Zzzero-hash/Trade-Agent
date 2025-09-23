#!/usr/bin/env python3
"""
Comprehensive Visualization Demo
Demonstrates all visualization features with realistic training simulation
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description="Visualization System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Demo Options:
    --quick         Quick 30-second demo
    --full          Full demo with multiple trials
    --terminal      Terminal viewer only
    --gui           GUI viewer only (requires display)
    --test          Run test suite
        """
    )
    
    parser.add_argument('--quick', action='store_true', help='Quick 30-second demo')
    parser.add_argument('--full', action='store_true', help='Full demo with multiple trials')
    parser.add_argument('--terminal', action='store_true', help='Terminal viewer only')
    parser.add_argument('--gui', action='store_true', help='GUI viewer only')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    
    args = parser.parse_args()
    
    print("🎨 Visualization System Demo")
    print("=" * 50)
    
    if args.test:
        return run_test_suite()
    elif args.quick:
        return run_quick_demo(args.terminal, args.gui)
    elif args.full:
        return run_full_demo(args.terminal, args.gui)
    else:
        return show_menu()

def show_menu():
    """Show interactive menu"""
    print("Choose a demo option:")
    print("1. Quick Demo (30 seconds)")
    print("2. Full Demo (multiple trials)")
    print("3. Terminal Viewer Test")
    print("4. GUI Viewer Test")
    print("5. Run Test Suite")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                return run_quick_demo()
            elif choice == '2':
                return run_full_demo()
            elif choice == '3':
                return run_quick_demo(terminal_only=True)
            elif choice == '4':
                return run_quick_demo(gui_only=True)
            elif choice == '5':
                return run_test_suite()
            elif choice == '6':
                print("👋 Goodbye!")
                return True
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo cancelled by user")
            return True

def run_quick_demo(terminal_only=False, gui_only=False):
    """Run a quick 30-second demo"""
    print("🚀 Starting Quick Demo (30 seconds)")
    
    try:
        # Import test module
        import test_visualization
        
        # Create mock data
        progress_file, mock_trials = test_visualization.create_mock_progress_data()
        
        # Start simulation
        import threading
        sim_thread = threading.Thread(
            target=test_visualization.simulate_training_progress,
            args=(progress_file, mock_trials),
            daemon=True
        )
        sim_thread.start()
        
        # Choose viewer
        if gui_only:
            print("🎨 Starting GUI viewer...")
            return test_gui_viewer_demo(progress_file)
        elif terminal_only:
            print("🖥️  Starting terminal viewer...")
            return test_terminal_viewer_demo(progress_file)
        else:
            # Ask user preference
            print("Choose viewer:")
            print("1. Terminal viewer")
            print("2. GUI viewer")
            
            choice = input("Enter choice (1-2): ").strip()
            
            if choice == '2':
                return test_gui_viewer_demo(progress_file)
            else:
                return test_terminal_viewer_demo(progress_file)
                
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False

def run_full_demo(terminal_only=False, gui_only=False):
    """Run full demo with actual hyperparameter optimization"""
    print("🔥 Starting Full Demo with Real Hyperparameter Optimization")
    print("⚠️  This will run actual ML training - may take several minutes")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Demo cancelled")
        return True
    
    try:
        # Import and run actual hyperopt
        from test_monitoring import test_monitoring
        
        print("🚀 Starting hyperparameter optimization with monitoring...")
        print("💡 Run 'python live_viewer_enhanced.py' in another terminal")
        
        return test_monitoring()
        
    except Exception as e:
        print(f"❌ Full demo failed: {e}")
        return False

def test_terminal_viewer_demo(progress_file):
    """Test terminal viewer with demo data"""
    try:
        from src.ml.visualization import create_terminal_viewer
        
        viewer = create_terminal_viewer(str(progress_file))
        viewer.update_interval = 1.0
        
        print("🖥️  Terminal viewer running for 30 seconds...")
        print("Press Ctrl+C to stop early")
        
        import time
        start_time = time.time()
        
        try:
            while time.time() - start_time < 30:
                viewer._update_display()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        print("\n✅ Terminal viewer demo completed")
        return True
        
    except Exception as e:
        print(f"❌ Terminal viewer demo failed: {e}")
        return False

def test_gui_viewer_demo(progress_file):
    """Test GUI viewer with demo data"""
    try:
        from src.ml.visualization import create_live_dashboard
        
        print("🎨 Starting GUI dashboard...")
        print("🖼️  Close the plot window to stop")
        
        dashboard = create_live_dashboard(str(progress_file))
        dashboard.start()
        
        print("✅ GUI viewer demo completed")
        return True
        
    except ImportError as e:
        print(f"❌ GUI viewer not available: {e}")
        print("💡 Falling back to terminal viewer...")
        return test_terminal_viewer_demo(progress_file)
    except Exception as e:
        print(f"❌ GUI viewer demo failed: {e}")
        return False

def run_test_suite():
    """Run the complete test suite"""
    print("🧪 Running Visualization Test Suite")
    
    try:
        import test_visualization
        return test_visualization.main()
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("=" * 30)
    
    examples = [
        ("Quick terminal demo", "python demo_visualization.py --quick --terminal"),
        ("Quick GUI demo", "python demo_visualization.py --quick --gui"),
        ("Full training demo", "python demo_visualization.py --full"),
        ("Run tests", "python demo_visualization.py --test"),
        ("Monitor existing training", "python live_viewer_enhanced.py"),
        ("Terminal viewer only", "python live_viewer_enhanced.py --terminal"),
        ("GUI viewer only", "python live_viewer_enhanced.py --gui"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print(f"    {cmd}")
        print()

if __name__ == "__main__":
    try:
        success = main()
        
        if not success:
            print("\n📚 For more options, try:")
            show_usage_examples()
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)