"""Model Registry with Versioning and Automated Rollback Capabilities

This module implements a model registry that manages model versions with semantic versioning
and provides automated rollback capabilities based on performance metrics.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
from packaging import version

from src.ml.base_models import ModelConfig
from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ModelStatus(Enum):
    """Status of a model version"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


class RollbackReason(Enum):
    """Reasons for model rollback"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIFT_DETECTED = "drift_detected"
    ERROR_RATE_HIGH = "error_rate_high"
    MANUAL_ROLLBACK = "manual_rollback"


@dataclass
class ModelVersion:
    """Represents a specific version of a model"""
    model_id: str
    version: str
    file_path: str
    config: Dict[str, Any]
    status: ModelStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class ModelDeployment:
    """Represents a deployed model with its configuration"""
    model_id: str
    version: str
    deployment_time: datetime
    config: Dict[str, Any]
    status: str  # active, inactive
    traffic_percentage: float  # 0.0 to 1.0


@dataclass
class RollbackEvent:
    """Represents a model rollback event"""
    model_id: str
    from_version: str
    to_version: str
    reason: RollbackReason
    timestamp: datetime
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None
    description: Optional[str] = None


class SemanticVersion:
    """Semantic versioning helper class"""
    
    @staticmethod
    def parse(version_str: str) -> version.Version:
        """Parse a semantic version string"""
        return version.parse(version_str)
    
    @staticmethod
    def is_valid(version_str: str) -> bool:
        """Check if a version string is valid semantic version"""
        try:
            version.parse(version_str)
            return True
        except version.InvalidVersion:
            return False
    
    @staticmethod
    def next_major(current_version: str) -> str:
        """Get next major version"""
        v = version.parse(current_version)
        return f"{v.major + 1}.0.0"
    
    @staticmethod
    def next_minor(current_version: str) -> str:
        """Get next minor version"""
        v = version.parse(current_version)
        return f"{v.major}.{v.minor + 1}.0"
    
    @staticmethod
    def next_patch(current_version: str) -> str:
        """Get next patch version"""
        v = version.parse(current_version)
        return f"{v.major}.{v.minor}.{v.micro + 1}"


class ModelRegistry:
    """Model registry with versioning and automated rollback capabilities"""
    
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize model registry
        
        Args:
            registry_path: Path to store registry data
        """
        self.registry_path = registry_path
        self.models: Dict[str, List[ModelVersion]] = {}
        self.deployments: Dict[str, ModelDeployment] = {}
        self.rollback_history: List[RollbackEvent] = []
        self.performance_thresholds: Dict[str, float] = {
            "accuracy": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "f1_score": 0.7,
            "error_rate": 0.05,
            "latency_95th_percentile": 100.0  # milliseconds
        }
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_path, exist_ok=True)
        
        # Load existing registry data
        self._load_registry()
        
        logger.info("Model registry initialized")
    
    def _load_registry(self) -> None:
        """Load registry data from disk"""
        registry_file = os.path.join(self.registry_path, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                
                # Load models
                for model_id, versions_data in data.get("models", {}).items():
                    self.models[model_id] = []
                    for version_data in versions_data:
                        # Convert string dates to datetime objects
                        version_data["created_at"] = datetime.fromisoformat(
                            version_data["created_at"]
                        )
                        if version_data.get("deployed_at"):
                            version_data["deployed_at"] = datetime.fromisoformat(
                                version_data["deployed_at"]
                            )
                        
                        model_version = ModelVersion(**version_data)
                        self.models[model_id].append(model_version)
                
                # Load deployments
                for deployment_id, deployment_data in data.get("deployments", {}).items():
                    deployment_data["deployment_time"] = datetime.fromisoformat(
                        deployment_data["deployment_time"]
                    )
                    self.deployments[deployment_id] = ModelDeployment(**deployment_data)
                
                # Load rollback history
                for rollback_data in data.get("rollback_history", []):
                    rollback_data["timestamp"] = datetime.fromisoformat(
                        rollback_data["timestamp"]
                    )
                    rollback_data["reason"] = RollbackReason(rollback_data["reason"])
                    self.rollback_history.append(RollbackEvent(**rollback_data))
                    
                logger.info(f"Loaded registry with {len(self.models)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry data to disk"""
        try:
            registry_file = os.path.join(self.registry_path, "registry.json")
            
            # Convert to serializable format
            data = {
                "models": {},
                "deployments": {},
                "rollback_history": []
            }
            
            # Serialize models
            for model_id, versions in self.models.items():
                data["models"][model_id] = []
                for model_version in versions:
                    version_dict = asdict(model_version)
                    # Convert datetime objects to strings
                    version_dict["created_at"] = model_version.created_at.isoformat()
                    if model_version.deployed_at:
                        version_dict["deployed_at"] = model_version.deployed_at.isoformat()
                    data["models"][model_id].append(version_dict)
            
            # Serialize deployments
            for deployment_id, deployment in self.deployments.items():
                deployment_dict = asdict(deployment)
                deployment_dict["deployment_time"] = deployment.deployment_time.isoformat()
                data["deployments"][deployment_id] = deployment_dict
            
            # Serialize rollback history
            for rollback_event in self.rollback_history:
                rollback_dict = asdict(rollback_event)
                rollback_dict["timestamp"] = rollback_event.timestamp.isoformat()
                rollback_dict["reason"] = rollback_event.reason.value
                data["rollback_history"].append(rollback_dict)
            
            # Write to file
            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model_id: str,
        version: str,
        file_path: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Register a new model version
        
        Args:
            model_id: Unique identifier for the model
            version: Semantic version string (e.g., "1.0.0")
            file_path: Path to the model file
            config: Model configuration
            metadata: Additional metadata
            
        Returns:
            Registered ModelVersion object
            
        Raises:
            ValueError: If version is not valid semantic version
        """
        # Validate version
        if not SemanticVersion.is_valid(version):
            raise ValueError(f"Invalid semantic version: {version}")
        
        # Check if model file exists
        if not os.path.exists(file_path):
            raise ValueError(f"Model file not found: {file_path}")
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            file_path=file_path,
            config=config,
            status=ModelStatus.INACTIVE,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add to registry
        if model_id not in self.models:
            self.models[model_id] = []
        self.models[model_id].append(model_version)
        
        # Sort versions by semantic version
        self.models[model_id].sort(
            key=lambda mv: SemanticVersion.parse(mv.version),
            reverse=True
        )
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {model_id} version {version}")
        return model_version
    
    def get_model_version(
        self, 
        model_id: str, 
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get a specific model version
        
        Args:
            model_id: Model identifier
            version: Specific version (if None, returns latest active version)
            
        Returns:
            ModelVersion object or None if not found
        """
        if model_id not in self.models:
            return None
        
        versions = self.models[model_id]
        
        if version is None:
            # Return latest active version
            for model_version in versions:
                if model_version.status == ModelStatus.ACTIVE:
                    return model_version
            # If no active version, return latest version
            return versions[0] if versions else None
        else:
            # Return specific version
            for model_version in versions:
                if model_version.version == version:
                    return model_version
            return None
    
    def get_all_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of ModelVersion objects
        """
        return self.models.get(model_id, [])
    
    def deploy_model(
        self,
        model_id: str,
        version: str,
        config: Dict[str, Any],
        traffic_percentage: float = 1.0
    ) -> bool:
        """Deploy a model version
        
        Args:
            model_id: Model identifier
            version: Model version to deploy
            config: Deployment configuration
            traffic_percentage: Percentage of traffic to route to this model (0.0 to 1.0)
            
        Returns:
            True if deployment successful, False otherwise
        """
        # Get model version
        model_version = self.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model {model_id} version {version} not found")
            return False
        
        # Update model status
        model_version.status = ModelStatus.ACTIVE
        model_version.deployed_at = datetime.now()
        
        # Create deployment record
        deployment = ModelDeployment(
            model_id=model_id,
            version=version,
            deployment_time=datetime.now(),
            config=config,
            status="active",
            traffic_percentage=traffic_percentage
        )
        
        # Store deployment
        deployment_id = f"{model_id}:{version}"
        self.deployments[deployment_id] = deployment
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deployed model {model_id} version {version} with {traffic_percentage*100}% traffic")
        return True
    
    def undeploy_model(self, model_id: str, version: str) -> bool:
        """Undeploy a model version
        
        Args:
            model_id: Model identifier
            version: Model version to undeploy
            
        Returns:
            True if undeployment successful, False otherwise
        """
        # Get model version
        model_version = self.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model {model_id} version {version} not found")
            return False
        
        # Update model status
        model_version.status = ModelStatus.INACTIVE
        
        # Update deployment record
        deployment_id = f"{model_id}:{version}"
        if deployment_id in self.deployments:
            self.deployments[deployment_id].status = "inactive"
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Undeployed model {model_id} version {version}")
        return True
    
    def get_active_models(self) -> Dict[str, ModelVersion]:
        """Get all active models
        
        Returns:
            Dictionary mapping model_id to active ModelVersion
        """
        active_models = {}
        for model_id, versions in self.models.items():
            for model_version in versions:
                if model_version.status == ModelStatus.ACTIVE:
                    active_models[model_id] = model_version
                    break
        return active_models
    
    def update_performance_metrics(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Update performance metrics for a model version
        
        Args:
            model_id: Model identifier
            version: Model version
            metrics: Performance metrics dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        model_version = self.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model {model_id} version {version} not found")
            return False
        
        # Update metrics
        model_version.performance_metrics.update(metrics)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Updated performance metrics for model {model_id} version {version}")
        return True
    
    def check_rollback_conditions(
        self,
        model_id: str,
        version: str
    ) -> Tuple[bool, Optional[RollbackReason], Optional[str]]:
        """Check if model should be rolled back based on performance metrics
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Tuple of (should_rollback, reason, description)
        """
        model_version = self.get_model_version(model_id, version)
        if not model_version or not model_version.performance_metrics:
            return False, None, None
        
        metrics = model_version.performance_metrics
        
        # Check accuracy threshold
        if "accuracy" in metrics and metrics["accuracy"] < self.performance_thresholds["accuracy"]:
            return (
                True,
                RollbackReason.PERFORMANCE_DEGRADATION,
                f"Accuracy {metrics['accuracy']:.3f} below threshold {self.performance_thresholds['accuracy']}"
            )
        
        # Check precision threshold
        if "precision" in metrics and metrics["precision"] < self.performance_thresholds["precision"]:
            return (
                True,
                RollbackReason.PERFORMANCE_DEGRADATION,
                f"Precision {metrics['precision']:.3f} below threshold {self.performance_thresholds['precision']}"
            )
        
        # Check recall threshold
        if "recall" in metrics and metrics["recall"] < self.performance_thresholds["recall"]:
            return (
                True,
                RollbackReason.PERFORMANCE_DEGRADATION,
                f"Recall {metrics['recall']:.3f} below threshold {self.performance_thresholds['recall']}"
            )
        
        # Check F1 score threshold
        if "f1_score" in metrics and metrics["f1_score"] < self.performance_thresholds["f1_score"]:
            return (
                True,
                RollbackReason.PERFORMANCE_DEGRADATION,
                f"F1 score {metrics['f1_score']:.3f} below threshold {self.performance_thresholds['f1_score']}"
            )
        
        # Check error rate threshold
        if "error_rate" in metrics and metrics["error_rate"] > self.performance_thresholds["error_rate"]:
            return (
                True,
                RollbackReason.ERROR_RATE_HIGH,
                f"Error rate {metrics['error_rate']:.3f} above threshold {self.performance_thresholds['error_rate']}"
            )
        
        # Check latency threshold
        if "latency_95th_percentile" in metrics and metrics["latency_95th_percentile"] > self.performance_thresholds["latency_95th_percentile"]:
            return (
                True,
                RollbackReason.PERFORMANCE_DEGRADATION,
                f"95th percentile latency {metrics['latency_95th_percentile']:.2f}ms above threshold {self.performance_thresholds['latency_95th_percentile']}ms"
            )
        
        return False, None, None
    
    def rollback_model(
        self,
        model_id: str,
        reason: RollbackReason,
        description: Optional[str] = None,
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback to a previous model version
        
        Args:
            model_id: Model identifier
            reason: Reason for rollback
            description: Additional description
            target_version: Specific version to rollback to (if None, rollback to previous version)
            
        Returns:
            True if rollback successful, False otherwise
        """
        # Get current deployed version
        current_deployment = None
        current_version = None
        for deployment_id, deployment in self.deployments.items():
            if deployment.model_id == model_id and deployment.status == "active":
                current_deployment = deployment
                current_version = deployment.version
                break
        
        if not current_deployment or not current_version:
            logger.error(f"No active deployment found for model {model_id}")
            return False
        
        # Determine target version
        if target_version is None:
            # Get previous version
            versions = self.get_all_versions(model_id)
            # Find current version index
            current_index = None
            for i, model_version in enumerate(versions):
                if model_version.version == current_version:
                    current_index = i
                    break
            
            if current_index is None or current_index + 1 >= len(versions):
                logger.error(f"No previous version found for model {model_id}")
                return False
            
            target_version = versions[current_index + 1].version
        
        # Get target model version
        target_model_version = self.get_model_version(model_id, target_version)
        if not target_model_version:
            logger.error(f"Target version {target_version} not found for model {model_id}")
            return False
        
        # Record metrics before rollback
        current_model_version = self.get_model_version(model_id, current_version)
        metrics_before = current_model_version.performance_metrics if current_model_version else None
        
        # Deploy target version
        if not self.deploy_model(
            model_id, 
            target_version, 
            current_deployment.config,
            current_deployment.traffic_percentage
        ):
            logger.error(f"Failed to deploy target version {target_version} for model {model_id}")
            return False
        
        # Undeploy current version
        self.undeploy_model(model_id, current_version)
        
        # Update current version status to rolled back
        if current_model_version:
            current_model_version.status = ModelStatus.ROLLED_BACK
        
        # Record rollback event
        rollback_event = RollbackEvent(
            model_id=model_id,
            from_version=current_version,
            to_version=target_version,
            reason=reason,
            timestamp=datetime.now(),
            metrics_before=metrics_before,
            description=description
        )
        self.rollback_history.append(rollback_event)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Rolled back model {model_id} from version {current_version} to {target_version}")
        return True
    
    def get_rollback_history(self, model_id: Optional[str] = None) -> List[RollbackEvent]:
        """Get rollback history
        
        Args:
            model_id: Filter by model identifier (if None, return all)
            
        Returns:
            List of RollbackEvent objects
        """
        if model_id:
            return [event for event in self.rollback_history if event.model_id == model_id]
        return self.rollback_history
    
    def set_performance_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set performance thresholds for rollback decisions
        
        Args:
            thresholds: Dictionary of threshold values
        """
        self.performance_thresholds.update(thresholds)
        logger.info(f"Updated performance thresholds: {self.performance_thresholds}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model information
        """
        versions = self.get_all_versions(model_id)
        active_version = self.get_model_version(model_id)
        
        return {
            "model_id": model_id,
            "total_versions": len(versions),
            "active_version": active_version.version if active_version else None,
            "versions": [asdict(version) for version in versions],
            "deployments": [
                asdict(deployment) for deployment_id, deployment in self.deployments.items()
                if deployment.model_id == model_id
            ],
            "rollback_history": [
                asdict(event) for event in self.get_rollback_history(model_id)
            ]
        }


# Global registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance
    
    Returns:
        ModelRegistry instance
    """
    global model_registry
    if model_registry is None:
        model_registry = ModelRegistry()
    return model_registry