"""JSON Utilities for ML Training

Handles serialization of NumPy types and other non-JSON-serializable objects.
"""

import json
import numpy as np
from typing import Any, Dict, List, Union


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def safe_json_dump(data: Any, filepath: str, **kwargs) -> None:
    """Safely dump data to JSON file with NumPy type handling
    
    Args:
        data: Data to serialize
        filepath: Output file path
        **kwargs: Additional arguments for json.dump
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, **kwargs)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy types to Python native types
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def safe_metrics_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metrics dictionary to JSON-safe format
    
    Args:
        metrics: Dictionary containing metrics (may have NumPy types)
        
    Returns:
        JSON-safe metrics dictionary
    """
    return convert_numpy_types(metrics)