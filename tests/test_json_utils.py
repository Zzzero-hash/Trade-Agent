"""Unit tests for JSON utilities."""

import pytest
import json
import numpy as np
import tempfile
import os
from pathlib import Path

from src.utils.json_utils import NumpyEncoder, safe_json_dump, convert_numpy_types, safe_metrics_dict


class TestNumpyEncoder:
    """Test NumPy JSON encoder."""
    
    def test_numpy_integer_encoding(self):
        """Test encoding of NumPy integers."""
        data = {'value': np.int32(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['value'] == 42
        assert isinstance(parsed['value'], int)
    
    def test_numpy_float_encoding(self):
        """Test encoding of NumPy floats."""
        data = {'value': np.float64(3.14159)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['value'] - 3.14159) < 1e-6
        assert isinstance(parsed['value'], float)
    
    def test_numpy_array_encoding(self):
        """Test encoding of NumPy arrays."""
        data = {'array': np.array([1, 2, 3, 4])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['array'] == [1, 2, 3, 4]
        assert isinstance(parsed['array'], list)
    
    def test_numpy_bool_encoding(self):
        """Test encoding of NumPy booleans."""
        data = {'flag': np.bool_(True)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['flag'] is True
        assert isinstance(parsed['flag'], bool)
    
    def test_mixed_numpy_types(self):
        """Test encoding of mixed NumPy types."""
        data = {
            'int_val': np.int64(100),
            'float_val': np.float32(2.5),
            'array_val': np.array([1.1, 2.2, 3.3]),
            'bool_val': np.bool_(False),
            'regular_val': 'string'
        }
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        
        assert parsed['int_val'] == 100
        assert abs(parsed['float_val'] - 2.5) < 1e-6
        assert len(parsed['array_val']) == 3
        assert parsed['bool_val'] is False
        assert parsed['regular_val'] == 'string'


class TestSafeJsonDump:
    """Test safe JSON dump functionality."""
    
    def test_safe_json_dump_with_numpy_types(self):
        """Test safe JSON dump handles NumPy types."""
        data = {
            'metrics': {
                'accuracy': np.float64(0.95),
                'epoch': np.int32(10),
                'losses': np.array([0.1, 0.05, 0.02])
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            safe_json_dump(data, temp_path, indent=2)
            
            # Verify file was created and is valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['metrics']['accuracy'] == 0.95
            assert loaded_data['metrics']['epoch'] == 10
            assert len(loaded_data['metrics']['losses']) == 3
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConvertNumpyTypes:
    """Test NumPy type conversion."""
    
    def test_convert_nested_dict(self):
        """Test conversion of nested dictionaries."""
        data = {
            'level1': {
                'level2': {
                    'numpy_int': np.int32(42),
                    'numpy_float': np.float64(3.14)
                }
            }
        }
        
        result = convert_numpy_types(data)
        
        assert isinstance(result['level1']['level2']['numpy_int'], int)
        assert isinstance(result['level1']['level2']['numpy_float'], float)
        assert result['level1']['level2']['numpy_int'] == 42
    
    def test_convert_list_with_numpy(self):
        """Test conversion of lists containing NumPy types."""
        data = [np.int32(1), np.float64(2.5), np.bool_(True)]
        
        result = convert_numpy_types(data)
        
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)
        assert result == [1, 2.5, True]
    
    def test_convert_numpy_array(self):
        """Test conversion of NumPy arrays."""
        data = np.array([[1, 2], [3, 4]])
        
        result = convert_numpy_types(data)
        
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]
    
    def test_preserve_regular_types(self):
        """Test that regular Python types are preserved."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'none': None
        }
        
        result = convert_numpy_types(data)
        
        assert result == data  # Should be identical


class TestSafeMetricsDict:
    """Test safe metrics dictionary conversion."""
    
    def test_training_metrics_conversion(self):
        """Test conversion of typical training metrics."""
        metrics = {
            'epoch': np.int32(10),
            'train_loss': np.float64(0.1234),
            'val_accuracy': np.float32(0.95),
            'learning_rate': 0.001,  # Regular float
            'batch_size': 32,  # Regular int
            'converged': np.bool_(True),
            'loss_history': np.array([0.5, 0.3, 0.1])
        }
        
        result = safe_metrics_dict(metrics)
        
        # Verify all NumPy types are converted
        assert isinstance(result['epoch'], int)
        assert isinstance(result['train_loss'], float)
        assert isinstance(result['val_accuracy'], float)
        assert isinstance(result['converged'], bool)
        assert isinstance(result['loss_history'], list)
        
        # Verify regular types are preserved
        assert isinstance(result['learning_rate'], float)
        assert isinstance(result['batch_size'], int)
        
        # Verify values are correct
        assert result['epoch'] == 10
        assert abs(result['train_loss'] - 0.1234) < 1e-6
        assert result['converged'] is True
        assert len(result['loss_history']) == 3
    
    def test_empty_metrics(self):
        """Test handling of empty metrics."""
        result = safe_metrics_dict({})
        assert result == {}
    
    def test_none_values(self):
        """Test handling of None values in metrics."""
        metrics = {
            'value': None,
            'numpy_val': np.float64(1.0)
        }
        
        result = safe_metrics_dict(metrics)
        
        assert result['value'] is None
        assert isinstance(result['numpy_val'], float)


class TestJsonSerializationIntegration:
    """Integration tests for JSON serialization fixes."""
    
    def test_ppo_evaluation_metrics_serialization(self):
        """Test serialization of PPO evaluation metrics (real-world scenario)."""
        # Simulate metrics that would come from PPO evaluation
        eval_metrics = {
            'timestep': np.int64(10000),
            'mean_reward': np.float64(0.001234),
            'std_reward': np.float64(0.000567),
            'mean_return': np.float32(0.0012),
            'sortino_ratio': np.float64(1.0),
            'mean_drawdown': np.float32(0.0),
            'max_drawdown': np.float64(0.0),
            'timestamp': '2024-01-01T12:00:00'
        }
        
        # This should not raise any JSON serialization errors
        safe_metrics = safe_metrics_dict(eval_metrics)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            safe_json_dump([safe_metrics], temp_path, indent=2)
            
            # Verify the file can be loaded back
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 1
            metrics = loaded_data[0]
            assert metrics['timestep'] == 10000
            assert abs(metrics['mean_reward'] - 0.001234) < 1e-6
            assert metrics['sortino_ratio'] == 1.0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])