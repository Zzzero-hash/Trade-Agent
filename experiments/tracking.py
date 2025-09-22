"""
Experiment Tracking and Model Versioning Utilities

This module provides unified interfaces for experiment tracking using MLflow,
Weights & Biases, and other tracking systems.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

import mlflow
import wandb
import torch
from omegaconf import DictConfig, OmegaConf


class ExperimentTracker:
    """Unified experiment tracking interface"""
    
    def __init__(self, config_path: str = "configs/mlflow_config.yaml"):
        self.config = self._load_config(config_path)
        self.mlflow_client = None
        self.wandb_run = None
        self._setup_mlflow()
        
    def _load_config(self, config_path: str) -> DictConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return OmegaConf.create(config)
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow_config = self.config.mlflow
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    mlflow_config.experiment_name,
                    artifact_location=mlflow_config.artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"MLflow setup warning: {e}")
            experiment_id = "0"
            
        mlflow.set_experiment(experiment_id=experiment_id)
        
        # Enable autologging if configured
        if mlflow_config.autolog.pytorch:
            mlflow.pytorch.autolog(disable=mlflow_config.autolog.disable)
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run"""
        # Start MLflow run
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)
        
        return mlflow_run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to tracking systems"""
        mlflow.log_params(params)
        
        if self.wandb_run:
            wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to tracking systems"""
        mlflow.log_metrics(metrics, step=step)
        
        if self.wandb_run:
            wandb.log(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, model_name: str, 
                  artifacts: Optional[Dict[str, str]] = None):
        """Log model to tracking systems"""
        # Log to MLflow
        mlflow.pytorch.log_model(
            model, 
            model_name,
            registered_model_name=self.config.mlflow.model_registry.registered_model_name
        )
        
        # Log artifacts if provided
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)
    
    def end_run(self):
        """End the current experiment run"""
        mlflow.end_run()
        
        if self.wandb_run:
            wandb.finish()


class ModelVersionManager:
    """Model versioning and registry management"""
    
    def __init__(self, registry_uri: str = "sqlite:///model_registry.db"):
        self.registry_uri = registry_uri
        mlflow.set_registry_uri(registry_uri)
    
    def register_model(self, model_uri: str, model_name: str, 
                      version_tags: Optional[Dict[str, str]] = None) -> str:
        """Register a model version"""
        model_version = mlflow.register_model(model_uri, model_name)
        
        if version_tags:
            for key, value in version_tags.items():
                mlflow.set_model_version_tag(
                    model_name, model_version.version, key, value
                )
        
        return model_version.version
    
    def get_latest_model(self, model_name: str, stage: str = "Production") -> str:
        """Get the latest model version URI"""
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(
            model_name, stages=[stage]
        )[0]
        return f"models:/{model_name}/{latest_version.version}"
    
    def promote_model(self, model_name: str, version: str, 
                     stage: str = "Production"):
        """Promote a model version to a specific stage"""
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )


def setup_wandb(config_path: str = "configs/wandb_config.yaml") -> wandb.sdk.wandb_run.Run:
    """Setup Weights & Biases tracking"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    wandb_config = config['wandb']
    
    run = wandb.init(
        project=wandb_config['project'],
        entity=wandb_config.get('entity'),
        name=wandb_config['run'].get('name'),
        tags=wandb_config['run']['tags'],
        notes=wandb_config['run']['notes']
    )
    
    return run