"""Decision Audit Trail for CNN+LSTM Hybrid Model

This module provides decision audit trails with complete model version tracking
for regulatory compliance and model performance monitoring.

Requirements: 
12.5 - Decision audit trails with complete model version tracking
12.6 - Uncertainty calibration and confidence score validation
"""

import numpy as np
import torch
import json
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
import os

from .hybrid_model import CNNLSTMHybridModel


@dataclass
class AuditEntry:
    """Audit entry for a single decision."""
    timestamp: datetime
    model_version: str
    model_hash: str
    input_hash: str
    prediction: Dict[str, Any]
    shap_values: Optional[np.ndarray] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, np.ndarray]] = None
    ensemble_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelVersionInfo:
    """Information about a model version."""
    version: str
    hash: str
    timestamp: datetime
    training_data_hash: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


class DecisionAuditor:
    """Decision audit trail with model version tracking."""
    
    def __init__(self, audit_log_path: str = "audit_trail.json"):
        """
        Initialize decision auditor.
        
        Args:
            audit_log_path: Path to audit log file
        """
        self.audit_log_path = audit_log_path
        self.audit_entries: List[AuditEntry] = []
        self.model_versions: Dict[str, ModelVersionInfo] = {}
        
        # Load existing audit log if it exists
        self._load_audit_log()
    
    def _load_audit_log(self) -> None:
        """Load existing audit log from file."""
        if os.path.exists(self.audit_log_path):
            try:
                with open(self.audit_log_path, 'r') as f:
                    data = json.load(f)
                    
                # Load audit entries
                for entry_data in data.get('audit_entries', []):
                    # Convert timestamp strings to datetime objects
                    entry_data['timestamp'] = datetime.fromisoformat(entry_data['timestamp'])
                    if 'shap_values' in entry_data and entry_data['shap_values']:
                        entry_data['shap_values'] = np.array(entry_data['shap_values'])
                    
                    self.audit_entries.append(AuditEntry(**entry_data))
                
                # Load model versions
                for version, version_info in data.get('model_versions', {}).items():
                    version_info['timestamp'] = datetime.fromisoformat(version_info['timestamp'])
                    self.model_versions[version] = ModelVersionInfo(**version_info)
                    
            except Exception as e:
                warnings.warn(f"Could not load audit log: {e}")
    
    def _save_audit_log(self) -> None:
        """Save audit log to file."""
        try:
            # Convert audit entries to serializable format
            audit_entries_serializable = []
            for entry in self.audit_entries:
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry_dict['timestamp'].isoformat()
                if entry_dict['shap_values'] is not None:
                    entry_dict['shap_values'] = entry_dict['shap_values'].tolist()
                audit_entries_serializable.append(entry_dict)
            
            # Convert model versions to serializable format
            model_versions_serializable = {}
            for version, version_info in self.model_versions.items():
                version_dict = asdict(version_info)
                version_dict['timestamp'] = version_dict['timestamp'].isoformat()
                model_versions_serializable[version] = version_dict
            
            # Save to file
            data = {
                'audit_entries': audit_entries_serializable,
                'model_versions': model_versions_serializable,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.audit_log_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Could not save audit log: {e}")
    
    def _hash_model(self, model: CNNLSTMHybridModel) -> str:
        """
        Create hash of model parameters for version tracking.
        
        Args:
            model: Model to hash
            
        Returns:
            Hash string of model parameters
        """
        try:
            # Get model state dict
            state_dict = model.state_dict()
            
            # Create hash of all parameters
            hasher = hashlib.md5()
            for key in sorted(state_dict.keys()):
                param = state_dict[key]
                if isinstance(param, torch.Tensor):
                    # Convert tensor to bytes
                    param_bytes = param.cpu().numpy().tobytes()
                    hasher.update(param_bytes)
                else:
                    # Convert other parameters to string and then to bytes
                    param_str = str(param)
                    hasher.update(param_str.encode('utf-8'))
            
            return hasher.hexdigest()
        except Exception as e:
            warnings.warn(f"Could not hash model: {e}")
            return "unknown"
    
    def _hash_input(self, input_data: np.ndarray) -> str:
        """
        Create hash of input data for tracking.
        
        Args:
            input_data: Input data to hash
            
        Returns:
            Hash string of input data
        """
        try:
            hasher = hashlib.md5()
            hasher.update(input_data.tobytes())
            return hasher.hexdigest()
        except Exception as e:
            warnings.warn(f"Could not hash input data: {e}")
            return "unknown"
    
    def log_decision(
        self,
        model: CNNLSTMHybridModel,
        input_data: np.ndarray,
        prediction: Dict[str, Any],
        shap_values: Optional[np.ndarray] = None,
        attention_weights: Optional[Dict[str, np.ndarray]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, np.ndarray]] = None,
        ensemble_weights: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a decision with complete audit trail.
        
        Args:
            model: Model that made the decision
            input_data: Input data for the decision
            prediction: Model prediction
            shap_values: SHAP values for explanation
            attention_weights: Attention weights from model
            confidence_scores: Confidence scores for prediction
            feature_importance: Feature importance scores
            ensemble_weights: Ensemble model weights
            metadata: Additional metadata
        """
        # Get model information
        model_version = getattr(model, 'version', 'unknown')
        model_hash = self._hash_model(model)
        
        # Get input hash
        input_hash = self._hash_input(input_data)
        
        # Create audit entry
        audit_entry = AuditEntry(
            timestamp=datetime.now(),
            model_version=model_version,
            model_hash=model_hash,
            input_hash=input_hash,
            prediction=prediction,
            shap_values=shap_values,
            attention_weights=attention_weights,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance,
            ensemble_weights=ensemble_weights,
            metadata=metadata
        )
        
        # Add to audit entries
        self.audit_entries.append(audit_entry)
        
        # Save audit log
        self._save_audit_log()
    
    def register_model_version(
        self,
        model: CNNLSTMHybridModel,
        training_data_hash: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Register a model version for tracking.
        
        Args:
            model: Model to register
            training_data_hash: Hash of training data
            hyperparameters: Model hyperparameters
            performance_metrics: Model performance metrics
            
        Returns:
            Model version identifier
        """
        model_version = getattr(model, 'version', f'v{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        model_hash = self._hash_model(model)
        
        # Set version on model if not already set
        if not hasattr(model, 'version'):
            model.version = model_version
        
        # Create model version info
        model_info = ModelVersionInfo(
            version=model_version,
            hash=model_hash,
            timestamp=datetime.now(),
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics
        )
        
        # Register model version
        self.model_versions[model_version] = model_info
        
        # Save audit log
        self._save_audit_log()
        
        return model_version
    
    def track_model_performance(
        self,
        model_version: str,
        performance_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Track performance metrics for a model version over time.
        
        Args:
            model_version: Model version identifier
            performance_metrics: Performance metrics to track
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create a performance tracking entry
        performance_entry = {
            'model_version': model_version,
            'performance_metrics': performance_metrics,
            'timestamp': timestamp.isoformat(),
            'entry_type': 'performance_tracking'
        }
        
        # Add to audit entries for tracking
        audit_entry = AuditEntry(
            timestamp=timestamp,
            model_version=model_version,
            model_hash=self.model_versions.get(model_version, ModelVersionInfo('', '', timestamp)).hash,
            input_hash='performance_tracking',
            prediction=performance_entry,
            metadata={'entry_type': 'performance_tracking'}
        )
        
        self.audit_entries.append(audit_entry)
        self._save_audit_log()
    
    def get_model_performance_history(
        self,
        model_version: str,
        metric_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific model version.
        
        Args:
            model_version: Model version identifier
            metric_name: Optional specific metric to filter by
            
        Returns:
            List of performance entries
        """
        performance_entries = []
        
        for entry in self.audit_entries:
            if (entry.model_version == model_version and 
                entry.metadata and 
                entry.metadata.get('entry_type') == 'performance_tracking'):
                
                if metric_name:
                    # Filter by specific metric
                    if (isinstance(entry.prediction, dict) and 
                        'performance_metrics' in entry.prediction and
                        metric_name in entry.prediction['performance_metrics']):
                        performance_entries.append({
                            'timestamp': entry.timestamp,
                            'metric_value': entry.prediction['performance_metrics'][metric_name],
                            'all_metrics': entry.prediction['performance_metrics']
                        })
                else:
                    # Return all metrics
                    if isinstance(entry.prediction, dict) and 'performance_metrics' in entry.prediction:
                        performance_entries.append({
                            'timestamp': entry.timestamp,
                            'metrics': entry.prediction['performance_metrics']
                        })
        
        return performance_entries
    
    def get_decision_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """
        Get decision history with filtering.
        
        Args:
            model_version: Filter by model version
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of audit entries matching filters
        """
        filtered_entries = self.audit_entries
        
        # Apply model version filter
        if model_version:
            filtered_entries = [
                entry for entry in filtered_entries 
                if entry.model_version == model_version
            ]
        
        # Apply time filters
        if start_time:
            filtered_entries = [
                entry for entry in filtered_entries 
                if entry.timestamp >= start_time
            ]
        
        if end_time:
            filtered_entries = [
                entry for entry in filtered_entries 
                if entry.timestamp <= end_time
            ]
        
        return filtered_entries
    
    def generate_audit_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Args:
            start_time: Start time for report
            end_time: End time for report
            output_path: Optional path to save report
            
        Returns:
            Dictionary with audit report data
        """
        # Get filtered entries
        entries = self.get_decision_history(start_time=start_time, end_time=end_time)
        
        # Calculate statistics
        total_decisions = len(entries)
        
        # Model usage statistics
        model_usage = {}
        for entry in entries:
            model_version = entry.model_version
            if model_version not in model_usage:
                model_usage[model_version] = 0
            model_usage[model_version] += 1
        
        # Decision types
        decision_types = {}
        for entry in entries:
            pred_type = entry.prediction.get('type', 'unknown')
            if pred_type not in decision_types:
                decision_types[pred_type] = 0
            decision_types[pred_type] += 1
        
        # Create report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'period_start': start_time.isoformat() if start_time else None,
            'period_end': end_time.isoformat() if end_time else None,
            'total_decisions': total_decisions,
            'model_usage': model_usage,
            'decision_types': decision_types,
            'model_versions': {k: asdict(v) for k, v in self.model_versions.items()}
        }
        
        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                warnings.warn(f"Could not save audit report: {e}")
        
        return report


def create_decision_auditor(audit_log_path: str = "audit_trail.json") -> DecisionAuditor:
    """
    Factory function to create decision auditor.
    
    Args:
        audit_log_path: Path to audit log file
        
    Returns:
        Configured DecisionAuditor instance
    """
    return DecisionAuditor(audit_log_path=audit_log_path)