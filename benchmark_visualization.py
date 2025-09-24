#!/usr/bin/env python3
"""
Visualization Performance Benchmark
Tests performance and reliability of all visualization components
"""
import sys
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationBenchmark:
    """Benchmark suite for visualization components"""
    
    def __init__(self):
        self.results = {}
        self.test_data_dir = Path("benchmark_test_data")
        
    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        logger.info("🏁 Starting Visualization Performance Benchmark")
        logger.info("=" * 60)
        
        # Test data generation performance
        self.benchmark_data_generation()
        
        # Test terminal viewer performance
        self.benchmark_terminal_viewer()
        
        # Test GUI viewer performance
        self.benchmark_gui_viewer()
        
        # Test web dashboard performance
        self.benchmark_web_dashboard()
        
        # Test monitoring integration performance
        self.benchmark_monitoring_integration()
        
        # Generate report
        self.generate_report()
        
    def benchmark_data_generation(self):
        """Benchmark data generation and file I/O"""
        logger.info("📊 Benchmarking data generation...")
        
        start_time = time.time()
        
        try:
            import test_visualization
            
            # Generate test data
            progress_file, mock_trials = test_visualization.create_mock_progress_data()
            
            # Simulate rapid updates
            update_count = 0
            for i in range(100):  # 100 rapid updates
                mock_data = {
                    'trial_number': 1,
                    'status': 'training',
                    'current_epoch': i,
                    'last_metrics': {
                        'loss': 0.5 - i * 0.005,
                        'accuracy': 0.6 + i * 0.003
                    },
                    'last_update': datetime.now().isoformat()
                }
                
                with open(progress_file, 'w') as f:
                    import json
                    json.dump(mock_data, f)
                    
                update_count += 1
                
            elapsed = time.time() - start_time
            
            self.results['data_generation'] = {
                'updates_per_second': update_count / elapsed,
                'total_time': elapsed,
                'status': 'success'
            }
            
            logger.info(f"✅ Data generation: {update_count/elapsed:.1f} updates/sec")
            
        except Exception as e:
            self.results['data_generation'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Data generation benchmark failed: {e}")
            
    def benchmark_terminal_viewer(self):
        """Benchmark terminal viewer performance"""
        logger.info("🖥️  Benchmarking terminal viewer...")
        
        try:
            from src.ml.visualization import create_terminal_viewer
            
            # Create test data
            import test_visualization
            progress_file, mock_trials = test_visualization.create_mock_progress_data()
            
            # Start simulation
            sim_thread = threading.Thread(
                target=test_visualization.simulate_training_progress,
                args=(progress_file, mock_trials),
                daemon=True
            )
            sim_thread.start()
            
            # Benchmark viewer updates
            viewer = create_terminal_viewer(str(progress_file))
            viewer.update_interval = 0.1  # Fast updates
            
            start_time = time.time()
            update_count = 0
            
            # Run for 10 seconds
            while time.time() - start_time < 10:
                viewer._update_display()
                update_count += 1
                time.sleep(0.1)
                
            elapsed = time.time() - start_time
            
            self.results['terminal_viewer'] = {
                'updates_per_second': update_count / elapsed,
                'total_time': elapsed,
                'status': 'success'
            }
            
            logger.info(f"✅ Terminal viewer: {update_count/elapsed:.1f} updates/sec")
            
        except Exception as e:
            self.results['terminal_viewer'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Terminal viewer benchmark failed: {e}")
            
    def benchmark_gui_viewer(self):
        """Benchmark GUI viewer performance"""
        logger.info("🎨 Benchmarking GUI viewer...")
        
        try:
            from src.ml.visualization import RealTimePlotter
            
            # Create test data
            import test_visualization
            progress_file, mock_trials = test_visualization.create_mock_progress_data()
            
            # Test plotter creation and data processing
            plotter = RealTimePlotter(str(progress_file), update_interval=0.1)
            
            start_time = time.time()
            
            # Simulate data processing without actual plotting
            for i in range(100):
                mock_data = {
                    'trial_number': i % 5 + 1,
                    'status': 'training',
                    'current_epoch': i,
                    'last_metrics': {
                        'loss': 0.5 - i * 0.001,
                        'accuracy': 0.6 + i * 0.002
                    },
                    'last_update': datetime.now().isoformat()
                }
                
                plotter._process_new_data(mock_data)
                
            elapsed = time.time() - start_time
            
            # Avoid division by zero
            if elapsed > 0:
                self.results['gui_viewer'] = {
                    'data_processing_rate': 100 / elapsed,
                    'total_time': elapsed,
                    'status': 'success'
                }
            else:
                self.results['gui_viewer'] = {
                    'data_processing_rate': 0,
                    'total_time': elapsed,
                    'status': 'failed',
                    'error': 'Processing too fast to measure'
                }
            
            if elapsed > 0:
                logger.info(f"✅ GUI viewer: {100/elapsed:.1f} data points/sec")
            else:
                logger.info("✅ GUI viewer: Processing too fast to measure")
            
        except ImportError as e:
            self.results['gui_viewer'] = {
                'status': 'skipped',
                'reason': 'GUI dependencies not available'
            }
            logger.info("⏭️  GUI viewer benchmark skipped (no display)")
            
        except Exception as e:
            self.results['gui_viewer'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ GUI viewer benchmark failed: {e}")
            
    def benchmark_web_dashboard(self):
        """Benchmark web dashboard performance"""
        logger.info("🌐 Benchmarking web dashboard...")
        
        try:
            from src.ml.visualization import WebDashboard
            
            # Create test data
            import test_visualization
            progress_file, mock_trials = test_visualization.create_mock_progress_data()
            
            # Test dashboard data processing
            dashboard = WebDashboard(str(progress_file), port=8081)
            
            start_time = time.time()
            
            # Simulate data updates
            for i in range(50):
                mock_data = {
                    'trial_number': i % 3 + 1,
                    'status': 'completed' if i % 10 == 0 else 'training',
                    'final_metrics': {
                        'accuracy': 0.7 + i * 0.001,
                        'training_time': 120 + i
                    },
                    'last_update': datetime.now().isoformat()
                }
                
                dashboard._update_history(mock_data)
                
            elapsed = time.time() - start_time
            
            # Avoid division by zero
            if elapsed > 0:
                self.results['web_dashboard'] = {
                    'data_processing_rate': 50 / elapsed,
                    'total_time': elapsed,
                    'status': 'success'
                }
            else:
                self.results['web_dashboard'] = {
                    'data_processing_rate': 0,
                    'total_time': elapsed,
                    'status': 'failed',
                    'error': 'Processing too fast to measure'
                }
            
            if elapsed > 0:
                logger.info(f"✅ Web dashboard: {50/elapsed:.1f} updates/sec")
            else:
                logger.info("✅ Web dashboard: Processing too fast to measure")
            
        except Exception as e:
            self.results['web_dashboard'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Web dashboard benchmark failed: {e}")
            
    def benchmark_monitoring_integration(self):
        """Benchmark monitoring integration performance"""
        logger.info("📡 Benchmarking monitoring integration...")
        
        try:
            from src.ml.training_monitor import get_monitor
            from src.ml.visualization.metrics_collector import get_metrics_collector
            
            # Test monitor performance
            monitor = get_monitor("benchmark_test_progress")
            collector = get_metrics_collector("benchmark_test_progress")
            
            start_time = time.time()
            
            # Simulate training with monitoring
            for trial in range(5):
                hyperparams = {
                    'learning_rate': 0.001 + trial * 0.0001,
                    'batch_size': 32 + trial * 8,
                    'hidden_size': 128 + trial * 32
                }
                
                monitor.start_trial(trial, hyperparams)
                collector.record_trial_start(trial, hyperparams)
                
                # Simulate epochs
                for epoch in range(20):
                    metrics = {
                        'loss': 0.5 - epoch * 0.02,
                        'accuracy': 0.6 + epoch * 0.015,
                        'val_loss': 0.55 - epoch * 0.018,
                        'val_accuracy': 0.58 + epoch * 0.012
                    }
                    
                    monitor.update_epoch(epoch, metrics)
                    collector.record_epoch_metrics(trial, epoch, metrics)
                    
                # Complete trial
                final_metrics = {
                    'accuracy': 0.8 + trial * 0.01,
                    'training_time': 120 + trial * 10
                }
                
                monitor.finish_trial(trial, final_metrics)
                collector.record_trial_completion(trial, final_metrics)
                
            elapsed = time.time() - start_time
            
            self.results['monitoring_integration'] = {
                'trials_per_second': 5 / elapsed,
                'total_time': elapsed,
                'status': 'success'
            }
            
            logger.info(f"✅ Monitoring integration: {5/elapsed:.2f} trials/sec")
            
        except Exception as e:
            self.results['monitoring_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Monitoring integration benchmark failed: {e}")
            
    def benchmark_concurrent_viewers(self):
        """Benchmark multiple concurrent viewers"""
        logger.info("🔄 Benchmarking concurrent viewers...")
        
        try:
            # Create test data
            import test_visualization
            progress_file, mock_trials = test_visualization.create_mock_progress_data()
            
            # Start data simulation
            sim_thread = threading.Thread(
                target=test_visualization.simulate_training_progress,
                args=(progress_file, mock_trials),
                daemon=True
            )
            sim_thread.start()
            
            # Start multiple viewers concurrently
            viewers = []
            threads = []
            
            start_time = time.time()
            
            # Create multiple terminal viewers
            for i in range(3):
                from src.ml.visualization import create_terminal_viewer
                viewer = create_terminal_viewer(str(progress_file))
                viewer.update_interval = 0.5
                viewers.append(viewer)
                
                # Start viewer in thread
                thread = threading.Thread(
                    target=self._run_viewer_for_time,
                    args=(viewer, 5),  # Run for 5 seconds
                    daemon=True
                )
                threads.append(thread)
                thread.start()
                
            # Wait for all threads
            for thread in threads:
                thread.join()
                
            elapsed = time.time() - start_time
            
            self.results['concurrent_viewers'] = {
                'concurrent_count': len(viewers),
                'total_time': elapsed,
                'status': 'success'
            }
            
            logger.info(f"✅ Concurrent viewers: {len(viewers)} viewers for {elapsed:.1f}s")
            
        except Exception as e:
            self.results['concurrent_viewers'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Concurrent viewers benchmark failed: {e}")
            
    def _run_viewer_for_time(self, viewer, duration: float):
        """Run a viewer for specified duration"""
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                viewer._update_display()
                time.sleep(viewer.update_interval)
            except Exception:
                break
                
    def generate_report(self):
        """Generate benchmark report"""
        logger.info("📋 Generating benchmark report...")
        
        report = {
            'benchmark_time': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        report_file = Path("visualization_benchmark_report.json")
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2)
            
        # Print summary
        self._print_summary()
        
        logger.info(f"📄 Full report saved to: {report_file}")
        
    def _get_system_info(self):
        """Get system information"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown'
        }
        
    def _generate_summary(self):
        """Generate performance summary"""
        successful_tests = [name for name, result in self.results.items() 
                          if result.get('status') == 'success']
        failed_tests = [name for name, result in self.results.items() 
                       if result.get('status') == 'failed']
        skipped_tests = [name for name, result in self.results.items() 
                        if result.get('status') == 'skipped']
        
        return {
            'total_tests': len(self.results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'skipped_tests': len(skipped_tests),
            'success_rate': len(successful_tests) / len(self.results) if self.results else 0,
            'performance_score': self._calculate_performance_score()
        }
        
    def _calculate_performance_score(self):
        """Calculate overall performance score"""
        scores = []
        
        # Data generation score (target: 100 updates/sec)
        if 'data_generation' in self.results:
            result = self.results['data_generation']
            if result.get('status') == 'success':
                rate = result.get('updates_per_second', 0)
                score = min(100, rate / 100 * 100)  # Normalize to 100
                scores.append(score)
                
        # Terminal viewer score (target: 10 updates/sec)
        if 'terminal_viewer' in self.results:
            result = self.results['terminal_viewer']
            if result.get('status') == 'success':
                rate = result.get('updates_per_second', 0)
                score = min(100, rate / 10 * 100)
                scores.append(score)
                
        # Monitoring integration score (target: 1 trial/sec)
        if 'monitoring_integration' in self.results:
            result = self.results['monitoring_integration']
            if result.get('status') == 'success':
                rate = result.get('trials_per_second', 0)
                score = min(100, rate / 1 * 100)
                scores.append(score)
                
        return sum(scores) / len(scores) if scores else 0
        
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("📊 VISUALIZATION BENCHMARK SUMMARY")
        print("=" * 60)
        
        summary = self._generate_summary()
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Performance Score: {summary['performance_score']:.1f}/100")
        
        print("\n📈 Performance Results:")
        for test_name, result in self.results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                if 'updates_per_second' in result:
                    print(f"  {test_name}: {result['updates_per_second']:.1f} updates/sec")
                elif 'trials_per_second' in result:
                    print(f"  {test_name}: {result['trials_per_second']:.2f} trials/sec")
                elif 'data_processing_rate' in result:
                    print(f"  {test_name}: {result['data_processing_rate']:.1f} data points/sec")
            else:
                print(f"  {test_name}: {status}")
                
        print("=" * 60)


def main():
    """Main benchmark function"""
    benchmark = VisualizationBenchmark()
    
    try:
        benchmark.run_all_benchmarks()
        return True
    except KeyboardInterrupt:
        logger.info("\n👋 Benchmark interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)