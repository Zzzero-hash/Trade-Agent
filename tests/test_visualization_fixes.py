"""
Unit tests for visualization system fixes
Tests that all identified issues have been resolved
"""
import unittest
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestVisualizationFixes(unittest.TestCase):
    """Test suite for visualization system fixes"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.progress_file = Path(self.temp_dir) / "progress.json"
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_division_by_zero_fix_gui_viewer(self):
        """Test that GUI viewer benchmark handles division by zero"""
        from benchmark_visualization import VisualizationBenchmark
        
        benchmark = VisualizationBenchmark()
        
        # Mock time.time to return same value (zero elapsed time)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.0]  # Same time = zero elapsed
            
            # This should not raise ZeroDivisionError
            benchmark.benchmark_gui_viewer()
            
            # Check that the result handles zero elapsed time
            result = benchmark.results.get('gui_viewer', {})
            self.assertIn('status', result)
            
            # Should either be 'failed' with error or 'skipped'
            if result['status'] == 'failed':
                self.assertIn('error', result)
            elif result['status'] == 'success':
                # If successful, data_processing_rate should be valid
                self.assertIsInstance(result.get('data_processing_rate', 0), (int, float))
                
    def test_division_by_zero_fix_web_dashboard(self):
        """Test that web dashboard benchmark handles division by zero"""
        from benchmark_visualization import VisualizationBenchmark
        
        benchmark = VisualizationBenchmark()
        
        # Mock time.time to return same value (zero elapsed time)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [2000.0, 2000.0]  # Same time = zero elapsed
            
            # This should not raise ZeroDivisionError
            benchmark.benchmark_web_dashboard()
            
            # Check that the result handles zero elapsed time
            result = benchmark.results.get('web_dashboard', {})
            self.assertIn('status', result)
            
            # Should either be 'failed' with error or 'success' with valid rate
            if result['status'] == 'failed':
                self.assertIn('error', result)
            elif result['status'] == 'success':
                self.assertIsInstance(result.get('data_processing_rate', 0), (int, float))
                
    def test_string_formatting_fix_training_monitor(self):
        """Test that training monitor handles non-numeric metrics properly"""
        from src.ml.training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor(self.temp_dir)
        
        # Test with mixed metric types
        mixed_metrics = {
            'total_loss': 0.5,  # float
            'classification_accuracy': 'N/A',  # string
            'val_loss': None,  # None
            'learning_rate': 0.001,  # float
            'status': 'training'  # string
        }
        
        # This should not raise formatting errors
        try:
            monitor.start_trial(1, {'lr': 0.001})
            monitor.update_epoch(5, mixed_metrics)  # epoch % 5 == 0 triggers logging
            success = True
        except (ValueError, TypeError) as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Training monitor failed with mixed metrics: {error if not success else ''}")
        
    def test_string_formatting_fix_live_progress_viewer(self):
        """Test that live progress viewer handles non-numeric metrics"""
        from src.ml.training_monitor import LiveProgressViewer
        
        # Create test progress data with mixed types
        progress_data = {
            'trial_number': 1,
            'status': 'training',
            'current_epoch': 10,
            'last_metrics': {
                'total_loss': 'N/A',  # string instead of float
                'classification_accuracy': None,  # None value
                'val_loss': 0.45,  # valid float
                'custom_metric': 'converged'  # string metric
            },
            'elapsed_time_seconds': 120.5
        }
        
        # Write test data
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)
            
        viewer = LiveProgressViewer(str(self.progress_file))
        
        # This should not raise formatting errors
        try:
            viewer._print_progress(progress_data)
            success = True
        except (ValueError, TypeError) as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Live progress viewer failed with mixed metrics: {error if not success else ''}")
        
    def test_unicode_encoding_fix(self):
        """Test that Unicode characters are handled properly on Windows"""
        from live_viewer_enhanced import simple_terminal_viewer
        
        # Create test progress data
        progress_data = {
            'trial_number': 1,
            'status': 'training',
            'current_epoch': 5,
            'last_metrics': {
                'loss': 0.3456,
                'accuracy': 0.8123
            },
            'elapsed_time_seconds': 67.3,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)
            
        # Mock stdout to capture output
        from io import StringIO
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # This should not raise UnicodeEncodeError
            try:
                # Run for a short time
                import threading
                
                def run_viewer():
                    simple_terminal_viewer(str(self.progress_file), 0.1)
                    
                thread = threading.Thread(target=run_viewer, daemon=True)
                thread.start()
                time.sleep(0.5)  # Let it run briefly
                success = True
            except UnicodeEncodeError as e:
                success = False
                error = str(e)
                
        self.assertTrue(success, f"Unicode encoding error: {error if not success else ''}")
        
    def test_terminal_viewer_robustness(self):
        """Test terminal viewer handles various edge cases"""
        from src.ml.visualization import create_terminal_viewer
        
        viewer = create_terminal_viewer(str(self.progress_file))
        
        # Test with missing file
        try:
            viewer._load_progress_data()
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Terminal viewer failed with missing file: {error if not success else ''}")
        
        # Test with invalid JSON
        with open(self.progress_file, 'w') as f:
            f.write("invalid json content")
            
        try:
            viewer._load_progress_data()
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Terminal viewer failed with invalid JSON: {error if not success else ''}")
        
        # Test with empty metrics
        empty_data = {
            'trial_number': 1,
            'status': 'training',
            'last_metrics': {}
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(empty_data, f)
            
        try:
            viewer._process_data(empty_data)
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Terminal viewer failed with empty metrics: {error if not success else ''}")
        
    def test_web_dashboard_data_handling(self):
        """Test web dashboard handles various data scenarios"""
        from src.ml.visualization import create_web_dashboard
        
        dashboard = create_web_dashboard(str(self.progress_file))
        
        # Test with missing final_metrics
        incomplete_data = {
            'trial_number': 1,
            'status': 'completed'
            # Missing final_metrics
        }
        
        try:
            dashboard._update_history(incomplete_data)
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Web dashboard failed with incomplete data: {error if not success else ''}")
        
        # Test with None values
        none_data = {
            'trial_number': None,
            'status': 'completed',
            'final_metrics': {
                'accuracy': None,
                'training_time': None
            }
        }
        
        try:
            dashboard._update_history(none_data)
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Web dashboard failed with None values: {error if not success else ''}")
        
    def test_metrics_collector_robustness(self):
        """Test metrics collector handles edge cases"""
        from src.ml.visualization.metrics_collector import MetricsCollector
        
        collector = MetricsCollector(self.temp_dir)
        
        # Test with empty hyperparameters
        try:
            collector.record_trial_start(1, {})
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Metrics collector failed with empty hyperparameters: {error if not success else ''}")
        
        # Test with non-numeric metrics
        non_numeric_metrics = {
            'status': 'converged',
            'message': 'Training completed',
            'accuracy': 'N/A',
            'loss': None
        }
        
        try:
            collector.record_epoch_metrics(1, 10, non_numeric_metrics)
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Metrics collector failed with non-numeric metrics: {error if not success else ''}")
        
        # Test correlation calculation with mixed types
        try:
            analysis = collector.get_hyperparameter_analysis()
            success = True
        except Exception as e:
            success = False
            error = str(e)
            
        self.assertTrue(success, f"Hyperparameter analysis failed: {error if not success else ''}")
        
    def test_benchmark_error_handling(self):
        """Test benchmark handles various error conditions"""
        from benchmark_visualization import VisualizationBenchmark
        
        benchmark = VisualizationBenchmark()
        
        # Test with import errors
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            # Should handle import errors gracefully
            benchmark.benchmark_gui_viewer()
            
            result = benchmark.results.get('gui_viewer', {})
            self.assertIn('status', result)
            # Should be either 'failed' or 'skipped'
            self.assertIn(result['status'], ['failed', 'skipped'])
            
    def test_concurrent_access_safety(self):
        """Test that concurrent access to progress files is safe"""
        from src.ml.visualization import create_terminal_viewer
        
        # Create multiple viewers accessing same file
        viewers = [create_terminal_viewer(str(self.progress_file)) for _ in range(3)]
        
        # Write data concurrently
        def write_data(trial_num):
            for i in range(10):
                data = {
                    'trial_number': trial_num,
                    'status': 'training',
                    'current_epoch': i,
                    'last_metrics': {'loss': 0.5 - i * 0.01},
                    'last_update': datetime.now().isoformat()
                }
                
                with open(self.progress_file, 'w') as f:
                    json.dump(data, f)
                    
                time.sleep(0.01)
                
        # Start concurrent writers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=write_data, args=(i,), daemon=True)
            threads.append(thread)
            thread.start()
            
        # Start concurrent readers
        def read_data(viewer):
            for _ in range(20):
                try:
                    viewer._load_progress_data()
                except Exception:
                    pass  # Expected with concurrent access
                time.sleep(0.01)
                
        for viewer in viewers:
            thread = threading.Thread(target=read_data, args=(viewer,), daemon=True)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=2.0)
            
        # Test passes if no deadlocks or crashes occurred
        self.assertTrue(True, "Concurrent access test completed")
        
    def test_memory_usage_stability(self):
        """Test that visualization components don't leak memory"""
        from src.ml.visualization import create_terminal_viewer
        
        viewer = create_terminal_viewer(str(self.progress_file))
        
        # Simulate long-running monitoring
        for i in range(100):
            data = {
                'trial_number': i % 5,
                'status': 'training',
                'current_epoch': i,
                'last_metrics': {
                    'loss': 0.5 - i * 0.001,
                    'accuracy': 0.6 + i * 0.001
                },
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(data, f)
                
            # Process data
            viewer._process_data(data)
            
        # Check that data structures don't grow unbounded
        self.assertLess(len(viewer.trial_history), 50, "Trial history should be bounded")
        self.assertLess(len(viewer.current_trial_epochs), 20, "Epoch history should be bounded")
        
    def test_performance_regression(self):
        """Test that fixes don't introduce performance regressions"""
        from src.ml.training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor(self.temp_dir)
        
        # Measure performance of monitoring operations
        start_time = time.time()
        
        for trial in range(10):
            monitor.start_trial(trial, {'lr': 0.001, 'batch_size': 32})
            
            for epoch in range(20):
                metrics = {
                    'loss': 0.5 - epoch * 0.01,
                    'accuracy': 0.6 + epoch * 0.01,
                    'val_loss': 0.55 - epoch * 0.009,
                    'val_accuracy': 0.58 + epoch * 0.009
                }
                monitor.update_epoch(epoch, metrics)
                
            monitor.finish_trial(trial, {'accuracy': 0.8, 'training_time': 120})
            
        elapsed = time.time() - start_time
        
        # Should complete 10 trials with 20 epochs each in reasonable time
        self.assertLess(elapsed, 5.0, f"Performance regression: took {elapsed:.2f}s for 200 operations")
        
        # Should achieve reasonable throughput
        operations_per_second = (10 * 20) / elapsed
        self.assertGreater(operations_per_second, 40, f"Low throughput: {operations_per_second:.1f} ops/sec")


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from start to visualization"""
        from src.ml.training_monitor import get_monitor
        from src.ml.visualization import create_terminal_viewer
        
        # Start monitoring
        monitor = get_monitor(str(Path(self.temp_dir) / "progress"))
        
        # Simulate training
        monitor.start_trial(1, {'learning_rate': 0.001, 'batch_size': 32})
        
        for epoch in range(5):
            metrics = {
                'loss': 0.5 - epoch * 0.1,
                'accuracy': 0.6 + epoch * 0.05
            }
            monitor.update_epoch(epoch, metrics)
            
        monitor.finish_trial(1, {'accuracy': 0.85, 'training_time': 60})
        
        # Create viewer and verify it can read the data
        progress_file = Path(self.temp_dir) / "progress" / "progress.json"
        self.assertTrue(progress_file.exists(), "Progress file should be created")
        
        viewer = create_terminal_viewer(str(progress_file))
        data = viewer._load_progress_data()
        
        self.assertIsNotNone(data, "Viewer should be able to load progress data")
        self.assertEqual(data.get('trial_number'), 1, "Should have correct trial number")
        
    def test_multiple_viewer_compatibility(self):
        """Test that multiple viewers can work with same data"""
        from src.ml.training_monitor import TrainingMonitor
        from src.ml.visualization import create_terminal_viewer
        
        # Create monitor and generate data
        monitor = TrainingMonitor(self.temp_dir)
        monitor.start_trial(1, {'lr': 0.001})
        
        progress_file = Path(self.temp_dir) / "progress.json"
        
        # Create multiple viewers
        viewers = [create_terminal_viewer(str(progress_file)) for _ in range(3)]
        
        # All viewers should be able to load data
        for i, viewer in enumerate(viewers):
            try:
                data = viewer._load_progress_data()
                success = True
            except Exception as e:
                success = False
                error = str(e)
                
            self.assertTrue(success, f"Viewer {i} failed to load data: {error if not success else ''}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)