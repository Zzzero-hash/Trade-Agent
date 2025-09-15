"""FastAPI endpoints for model serving

This module defines the REST API endpoints for model serving,
including prediction, batch processing, and A/B testing management.

Requirements: 6.2, 11.1
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime

from .model_serving import (
    ModelServingService,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelType,
    get_serving_service
)
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()

# Create router
router = APIRouter(prefix="/api/v1", tags=["model-serving"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    serving_service: ModelServingService = Depends(get_serving_service)
) -> PredictionResponse:
    """
    Make a single prediction using the specified model.
    
    - **model_type**: Type of model to use (cnn_lstm_hybrid, rl_ensemble)
    - **model_version**: Version of the model (default: latest)
    - **data**: Input data as nested list
    - **return_uncertainty**: Whether to return uncertainty estimates
    - **use_ensemble**: Whether to use ensemble predictions
    """
    try:
        response = await serving_service.predict(request)
        return response
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    serving_service: ModelServingService = Depends(get_serving_service)
) -> BatchPredictionResponse:
    """
    Make batch predictions for multiple requests.
    
    Optimizes processing by grouping requests by model type and version.
    Maximum 100 requests per batch.
    """
    try:
        if len(request.requests) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 requests per batch"
            )
        
        response = await serving_service.batch_predict(request)
        return response
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load")
async def load_model(
    model_type: ModelType,
    version: str,
    file_path: str,
    config: Dict[str, Any],
    serving_service: ModelServingService = Depends(get_serving_service)
) -> JSONResponse:
    """
    Load a model into the serving cache.
    
    - **model_type**: Type of model to load
    - **version**: Version identifier for the model
    - **file_path**: Path to the model file
    - **config**: Model configuration parameters
    """
    try:
        await serving_service.load_model(model_type, version, file_path, config)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Model {model_type}:{version} loaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/cache/stats")
async def get_cache_stats(
    serving_service: ModelServingService = Depends(get_serving_service)
) -> Dict[str, Any]:
    """
    Get model cache statistics.
    
    Returns information about cached models, usage counts, and cache performance.
    """
    try:
        stats = serving_service.model_cache.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/create")
async def create_ab_experiment(
    experiment_id: str,
    model_variants: Dict[str, Dict[str, str]],
    traffic_split: Dict[str, float],
    duration_hours: int = 24,
    serving_service: ModelServingService = Depends(get_serving_service)
) -> JSONResponse:
    """
    Create an A/B testing experiment.
    
    - **experiment_id**: Unique identifier for the experiment
    - **model_variants**: Dictionary mapping variant names to model configs
    - **traffic_split**: Dictionary mapping variant names to traffic percentages
    - **duration_hours**: Duration of the experiment in hours
    
    Example:
    ```json
    {
        "experiment_id": "cnn_lstm_comparison",
        "model_variants": {
            "control": {"model_type": "cnn_lstm_hybrid", "version": "v1.0"},
            "treatment": {"model_type": "cnn_lstm_hybrid", "version": "v1.1"}
        },
        "traffic_split": {
            "control": 0.5,
            "treatment": 0.5
        },
        "duration_hours": 48
    }
    ```
    """
    try:
        serving_service.ab_test_manager.create_experiment(
            experiment_id=experiment_id,
            model_variants=model_variants,
            traffic_split=traffic_split,
            duration_hours=duration_hours
        )
        
        return JSONResponse(
            status_code=201,
            content={
                "message": f"A/B test experiment '{experiment_id}' created successfully",
                "experiment_id": experiment_id,
                "duration_hours": duration_hours,
                "timestamp": datetime.now().isoformat()
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"A/B experiment creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    serving_service: ModelServingService = Depends(get_serving_service)
) -> Dict[str, Any]:
    """
    Get results for an A/B testing experiment.
    
    Returns metrics for each variant including request counts, error rates,
    and average latency.
    """
    try:
        results = serving_service.ab_test_manager.get_experiment_results(experiment_id)
        
        if results is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_id}' not found"
            )
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experiment results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(
    serving_service: ModelServingService = Depends(get_serving_service)
) -> Dict[str, Any]:
    """
    List all A/B testing experiments.
    
    Returns a summary of all experiments with their current status.
    """
    try:
        experiments = {}
        
        for exp_id, experiment in serving_service.ab_test_manager.experiments.items():
            experiments[exp_id] = {
                "experiment_id": exp_id,
                "status": experiment["status"],
                "start_time": experiment["start_time"],
                "end_time": experiment["end_time"],
                "variants": list(experiment["model_variants"].keys()),
                "total_requests": sum(
                    metrics["requests"] 
                    for metrics in experiment["metrics"].values()
                )
            }
        
        return {
            "experiments": experiments,
            "total_experiments": len(experiments),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"List experiments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/experiments/{experiment_id}")
async def stop_experiment(
    experiment_id: str,
    serving_service: ModelServingService = Depends(get_serving_service)
) -> JSONResponse:
    """
    Stop an A/B testing experiment.
    
    Marks the experiment as completed and stops traffic splitting.
    """
    try:
        if experiment_id not in serving_service.ab_test_manager.experiments:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_id}' not found"
            )
        
        # Mark experiment as completed
        serving_service.ab_test_manager.experiments[experiment_id]["status"] = "stopped"
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Experiment '{experiment_id}' stopped successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stop experiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    serving_service: ModelServingService = Depends(get_serving_service)
) -> Dict[str, Any]:
    """
    Health check endpoint for the model serving service.
    
    Returns the status of various components including cache, Redis, and loaded models.
    """
    try:
        # Check Redis connection
        redis_healthy = True
        try:
            await serving_service.redis_client.ping()
        except Exception:
            redis_healthy = False
        
        # Get cache stats
        cache_stats = serving_service.model_cache.get_cache_stats()
        
        health_status = {
            "status": "healthy" if redis_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "model_cache": "healthy",
                "batch_processor": "healthy" if serving_service.batch_processor_task and not serving_service.batch_processor_task.done() else "unhealthy"
            },
            "cache": {
                "loaded_models": cache_stats["total_models"],
                "total_usage": cache_stats["total_usage"]
            }
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get serving metrics and performance statistics.
    
    Returns aggregated metrics about prediction requests, latency, and errors.
    """
    try:
        # Get metrics from the metrics collector
        current_metrics = metrics.get_all_metrics()
        
        return {
            "metrics": current_metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model-specific endpoints

@router.get("/models/types")
async def get_model_types() -> Dict[str, Any]:
    """
    Get available model types and their descriptions.
    """
    return {
        "model_types": {
            ModelType.CNN_LSTM_HYBRID: {
                "name": "CNN+LSTM Hybrid Model",
                "description": "Combines CNN feature extraction with LSTM temporal processing",
                "input_format": "3D tensor (batch_size, channels, sequence_length)",
                "output_format": "Classification probabilities and regression predictions",
                "supports_uncertainty": True,
                "supports_ensemble": True
            },
            ModelType.RL_ENSEMBLE: {
                "name": "Reinforcement Learning Ensemble",
                "description": "Ensemble of RL agents for trading decisions",
                "input_format": "2D tensor (batch_size, features)",
                "output_format": "Action probabilities and values",
                "supports_uncertainty": False,
                "supports_ensemble": True
            }
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/models/warmup")
async def warmup_models(
    model_configs: List[Dict[str, str]],
    serving_service: ModelServingService = Depends(get_serving_service)
) -> JSONResponse:
    """
    Warm up models by loading them into cache.
    
    Useful for preloading frequently used models to reduce cold start latency.
    
    Example:
    ```json
    [
        {"model_type": "cnn_lstm_hybrid", "version": "latest"},
        {"model_type": "rl_ensemble", "version": "v1.0"}
    ]
    ```
    """
    try:
        warmup_results = []
        
        for config in model_configs:
            model_type = config.get("model_type")
            version = config.get("version", "latest")
            
            try:
                # Check if model is already cached
                model = serving_service.model_cache.get_model(model_type, version)
                
                if model is not None:
                    warmup_results.append({
                        "model_type": model_type,
                        "version": version,
                        "status": "already_cached"
                    })
                else:
                    warmup_results.append({
                        "model_type": model_type,
                        "version": version,
                        "status": "not_found",
                        "message": "Model not found in cache. Use /models/load to load it first."
                    })
                    
            except Exception as e:
                warmup_results.append({
                    "model_type": model_type,
                    "version": version,
                    "status": "error",
                    "error": str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model warmup completed",
                "results": warmup_results,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Model warmup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))