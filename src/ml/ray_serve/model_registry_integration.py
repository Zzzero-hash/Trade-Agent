"""Ray Serve integration with model registry for versioning and automated rollback

This module provides integration between the model registry and Ray Serve deployments,
enabling versioned model serving with automated rollback capabilities.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from src.ml.model_registry import get_model_registry, ModelStatus, RollbackReason
from src.ml.ray_serve.ab_testing import ab_test_manager
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class RayServeModelRegistryIntegration:
    """Integration between model registry and Ray Serve deployments"""
    
    def __init__(self):
        """Initialize the integration"""
        self.model_registry = get_model_registry()
        self.monitoring_tasks = {}
        self.is_monitoring = False
    
    async def deploy_model_version(
        self,
        model_id: str,
        version: str,
        ray_deployment_name: str,
        ray_bind_args: Dict[str, Any] = None
    ) -> bool:
        """Deploy a specific model version using Ray Serve
        
        Args:
            model_id: Model identifier in registry
            version: Model version to deploy
            ray_deployment_name: Name for Ray Serve deployment
            ray_bind_args: Arguments for Ray Serve bind function
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Get model version from registry
            model_version = self.model_registry.get_model_version(model_id, version)
            if not model_version:
                logger.error(f"Model {model_id} version {version} not found in registry")
                return False
            
            # Import Ray Serve dynamically to avoid import issues
            import ray
            from ray import serve
            
            # Check if Ray is initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Get the deployment class (this would depend on your model type)
            # For now, we'll assume a generic deployment approach
            deployment_cls = self._get_deployment_class(model_version.config.get("model_type"))
            
            if deployment_cls is None:
                logger.error(f"No deployment class found for model type: {model_version.config.get('model_type')}")
                return False
            
            # Create deployment configuration
            bind_args = ray_bind_args or {}
            bind_args.update({
                "model_path": model_version.file_path,
                "model_config": model_version.config
            })
            
            # Deploy model with Ray Serve
            deployment = serve.deployment(
                deployment_cls,
                name=ray_deployment_name,
                num_replicas=2,
                ray_actor_options={"num_cpus": 1}
            )
            
            # Bind deployment with arguments
            bound_deployment = deployment.bind(**bind_args)
            
            # Deploy to Ray Serve
            serve.run(bound_deployment, name=ray_deployment_name)
            
            # Update registry with deployment info
            deployment_config = {
                "ray_deployment_name": ray_deployment_name,
                "deployment_time": datetime.now().isoformat(),
                "bind_args": bind_args
            }
            
            self.model_registry.deploy_model(
                model_id, 
                version, 
                deployment_config,
                traffic_percentage=1.0
            )
            
            logger.info(f"Deployed model {model_id} version {version} to Ray Serve as {ray_deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id} version {version}: {e}")
            return False
    
    def _get_deployment_class(self, model_type: str):
        """Get the appropriate deployment class for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Deployment class or None if not found
        """
        # Import deployment classes dynamically
        try:
            if model_type == "CNNLSTMHybridModel":
                from src.ml.ray_serve.cnn_lstm_deployment import CNNLSTMPredictor
                return CNNLSTMPredictor
            # Add other model types as needed
            else:
                # Default to a generic predictor if model type not specifically handled
                from src.ml.ray_serve.cnn_lstm_deployment import CNNLSTMPredictor
                return CNNLSTMPredictor
        except ImportError as e:
            logger.error(f"Failed to import deployment class for {model_type}: {e}")
            return None
    
    async def start_monitoring(
        self,
        model_id: str,
        version: str,
        check_interval: int = 60 # seconds
    ) -> None:
        """Start monitoring a deployed model for performance degradation
        
        Args:
            model_id: Model identifier
            version: Model version
            check_interval: Interval between checks in seconds
        """
        # Create monitoring task
        task_key = f"{model_id}:{version}"
        
        if task_key in self.monitoring_tasks:
            logger.warning(f"Monitoring already running for {task_key}")
            return
        
        # Start monitoring task
        task = asyncio.create_task(
            self._monitor_model_performance(model_id, version, check_interval)
        )
        self.monitoring_tasks[task_key] = task
        self.is_monitoring = True
        
        logger.info(f"Started monitoring for model {model_id} version {version}")
    
    async def stop_monitoring(self, model_id: str, version: str) -> None:
        """Stop monitoring a deployed model
        
        Args:
            model_id: Model identifier
            version: Model version
        """
        task_key = f"{model_id}:{version}"
        
        if task_key in self.monitoring_tasks:
            task = self.monitoring_tasks[task_key]
            task.cancel()
            del self.monitoring_tasks[task_key]
            
            logger.info(f"Stopped monitoring for model {model_id} version {version}")
        
        if not self.monitoring_tasks:
            self.is_monitoring = False
    
    async def _monitor_model_performance(
        self,
        model_id: str,
        version: str,
        check_interval: int
    ) -> None:
        """Monitor model performance and trigger rollback if needed
        
        Args:
            model_id: Model identifier
            version: Model version
            check_interval: Interval between checks in seconds
        """
        logger.info(f"Starting performance monitoring for {model_id}:{version}")
        
        try:
            while True:
                # Check if we should still be monitoring
                task_key = f"{model_id}:{version}"
                if task_key not in self.monitoring_tasks:
                    break
                
                # Collect performance metrics from Ray Serve
                performance_metrics = await self._collect_performance_metrics(
                    model_id, version
                )
                
                if performance_metrics:
                    # Update registry with metrics
                    self.model_registry.update_performance_metrics(
                        model_id, version, performance_metrics
                    )
                    
                    # Check rollback conditions
                    should_rollback, reason, description = self.model_registry.check_rollback_conditions(
                        model_id, version
                    )
                    
                    if should_rollback:
                        logger.warning(
                            f"Performance degradation detected for {model_id}:{version}. "
                            f"Reason: {reason.value}. Description: {description}"
                        )
                        
                        # Trigger rollback
                        rollback_success = self.model_registry.rollback_model(
                            model_id, reason, description
                        )
                        
                        if rollback_success:
                            logger.info(f"Successfully rolled back {model_id}:{version}")
                            
                            # Stop monitoring for rolled back version
                            await self.stop_monitoring(model_id, version)
                            
                            # Get new active version and start monitoring it
                            active_version = self.model_registry.get_model_version(model_id)
                            if active_version:
                                await self.start_monitoring(
                                    model_id, active_version.version, check_interval
                                )
                        else:
                            logger.error(f"Failed to rollback {model_id}:{version}")
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for {model_id}:{version}")
        except Exception as e:
            logger.error(f"Error in monitoring for {model_id}:{version}: {e}")
    
    async def _collect_performance_metrics(
        self,
        model_id: str,
        version: str
    ) -> Optional[Dict[str, float]]:
        """Collect performance metrics from Ray Serve deployment
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Dictionary of performance metrics or None if collection failed
        """
        try:
            # Import Ray Serve
            import ray
            from ray import serve
            
            # Check if Ray is initialized
            if not ray.is_initialized():
                logger.warning("Ray not initialized, cannot collect metrics")
                return None
            
            # Get deployment name from registry
            deployment_id = f"{model_id}:{version}"
            deployments = self.model_registry.deployments
            
            if deployment_id not in deployments:
                logger.warning(f"No deployment found for {deployment_id}")
                return None
            
            deployment_name = deployments[deployment_id].config.get("ray_deployment_name")
            if not deployment_name:
                logger.warning(f"No Ray deployment name found for {deployment_id}")
                return None
            
            # Get deployment handle
            try:
                handle = serve.get_deployment_handle(deployment_name)
                if handle:
                    # Get stats from deployment
                    stats = await handle.get_stats.remote()
                    
                    if stats:
                        # Extract relevant metrics
                        metrics = {}
                        
                        # Accuracy-like metrics (if available in model output)
                        if "accuracy" in stats:
                            metrics["accuracy"] = stats["accuracy"]
                        
                        # Latency metrics
                        if "avg_processing_time_ms" in stats:
                            metrics["latency_95th_percentile"] = stats["avg_processing_time_ms"] * 2  # Approximation
                        
                        # Error rate (if available)
                        if "request_count" in stats and "error_count" in stats:
                            if stats["request_count"] > 0:
                                metrics["error_rate"] = stats["error_count"] / stats["request_count"]
                        
                        # Add more metrics as needed based on your model's output
                        return metrics
            except Exception as e:
                logger.warning(f"Could not get deployment handle for {deployment_name}: {e}")
            
            # Fallback: Use metrics collector if available
            # This would collect metrics from your monitoring system
            collected_metrics = {}
            
            # Simulate collecting some metrics (in a real implementation, you would
            # collect actual metrics from your monitoring system)
            # For now, we'll return None to indicate no metrics collected
            return None
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics for {model_id}:{version}: {e}")
            return None
    
    async def rollback_to_version(
        self,
        model_id: str,
        target_version: str,
        reason: str = "manual_rollback"
    ) -> bool:
        """Manually rollback to a specific model version
        
        Args:
            model_id: Model identifier
            target_version: Target version to rollback to
            reason: Reason for rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            # Perform rollback in registry
            rollback_reason = RollbackReason.MANUAL_ROLLBACK
            if reason == "performance_degradation":
                rollback_reason = RollbackReason.PERFORMANCE_DEGRADATION
            elif reason == "drift_detected":
                rollback_reason = RollbackReason.DRIFT_DETECTED
            elif reason == "error_rate_high":
                rollback_reason = RollbackReason.ERROR_RATE_HIGH
            
            success = self.model_registry.rollback_model(
                model_id, rollback_reason, f"Manual rollback to version {target_version}", target_version
            )
            
            if success:
                logger.info(f"Manually rolled back {model_id} to version {target_version}")
                
                # Redeploy the target version with Ray Serve
                deployment_info = self.model_registry.deployments.get(f"{model_id}:{target_version}")
                if deployment_info:
                    ray_deployment_name = deployment_info.config.get("ray_deployment_name")
                    if ray_deployment_name:
                        # Stop monitoring for previous version
                        current_version = self.model_registry.get_model_version(model_id)
                        if current_version:
                            await self.stop_monitoring(model_id, current_version.version)
                        
                        # Redeploy target version
                        redeploy_success = await self.deploy_model_version(
                            model_id, target_version, ray_deployment_name
                        )
                        
                        if redeploy_success:
                            # Start monitoring new version
                            await self.start_monitoring(model_id, target_version)
                            return True
                        else:
                            logger.error(f"Failed to redeploy {model_id} version {target_version}")
                            return False
                else:
                    logger.warning(f"No deployment info found for {model_id}:{target_version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during manual rollback of {model_id} to {target_version}: {e}")
            return False
    
    async def get_deployment_status(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get deployment status for a model version
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            Dictionary with deployment status information
        """
        try:
            # Get model version info from registry
            model_version = self.model_registry.get_model_version(model_id, version)
            if not model_version:
                return {"error": f"Model {model_id} version {version} not found"}
            
            # Get deployment info
            deployment_id = f"{model_id}:{version}"
            deployment = self.model_registry.deployments.get(deployment_id)
            
            # Get Ray Serve status if available
            ray_status = "unknown"
            try:
                import ray
                from ray import serve
                
                if ray.is_initialized():
                    deployment_name = deployment.config.get("ray_deployment_name") if deployment else None
                    if deployment_name:
                        # Check if deployment exists
                        try:
                            serve.get_deployment(deployment_name)
                            ray_status = "deployed"
                        except Exception:
                            ray_status = "not_deployed"
            except Exception as e:
                logger.warning(f"Could not check Ray Serve status: {e}")
            
            return {
                "model_id": model_id,
                "version": version,
                "status": model_version.status.value,
                "deployed": deployment is not None,
                "ray_status": ray_status,
                "performance_metrics": model_version.performance_metrics,
                "deployment_info": deployment.config if deployment else None
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status for {model_id}:{version}: {e}")
            return {"error": str(e)}
    
    async def list_deployed_models(self) -> List[Dict[str, Any]]:
        """List all currently deployed models
        
        Returns:
            List of dictionaries with deployment information
        """
        try:
            deployed_models = []
            
            # Get active models from registry
            active_models = self.model_registry.get_active_models()
            
            for model_id, model_version in active_models.items():
                deployment_id = f"{model_id}:{model_version.version}"
                deployment = self.model_registry.deployments.get(deployment_id)
                
                if deployment:
                    deployed_models.append({
                        "model_id": model_id,
                        "version": model_version.version,
                        "deployment_name": deployment.config.get("ray_deployment_name"),
                        "deployment_time": deployment.deployment_time.isoformat(),
                        "traffic_percentage": deployment.traffic_percentage,
                        "performance_metrics": model_version.performance_metrics
                    })
            
            return deployed_models
            
        except Exception as e:
            logger.error(f"Error listing deployed models: {e}")
            return []


# Global integration instance
ray_serve_integration = RayServeModelRegistryIntegration()


def get_ray_serve_integration() -> RayServeModelRegistryIntegration:
    """Get the global Ray Serve integration instance
    
    Returns:
        RayServeModelRegistryIntegration instance
    """
    global ray_serve_integration
    if ray_serve_integration is None:
        ray_serve_integration = RayServeModelRegistryIntegration()
    return ray_serve_integration