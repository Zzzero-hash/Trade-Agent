"""Model serving infrastructure with FastAPI endpoints

This module provides FastAPI-based model serving endpoints with caching,
batch inference optimization, and A/B testing capabilities.

Requirements: 6.2, 11.1
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import redis
import json
import pickle
import hashlib
from contextlib import asynccontextmanager
import logging

from src.ml.hybrid_model import CNNLSTMHybridModel, HybridModelConfig
from src.models.market_data import MarketData
from src.models.trading_signal import TradingSignal
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector


logger = get_logger(__name__)
metrics = get_metrics_collector()


class ModelType(str, Enum):
    """Available model types"""

    CNN_LSTM_HYBRID = "cnn_lstm_hybrid"
    RL_ENSEMBLE = "rl_ensemble"


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    model_type: ModelType
    model_version: Optional[str] = "latest"
    data: List[List[float]] = Field(..., description="Input data as nested list")
    batch_size: Optional[int] = Field(32, ge=1, le=1000)
    return_uncertainty: bool = Field(True, description="Return uncertainty estimates")
    use_ensemble: bool = Field(True, description="Use ensemble predictions")

    @validator("data")
    def validate_data_shape(cls, v):
        if not v or not all(isinstance(row, list) for row in v):
            raise ValueError("Data must be a non-empty list of lists")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    request_id: str
    model_type: str
    model_version: str
    predictions: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[List[float]] = None
    processing_time_ms: float
    timestamp: datetime
    ab_test_group: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    requests: List[PredictionRequest] = Field(..., max_items=100)
    priority: int = Field(
        1, ge=1, le=5, description="Priority level (1=highest, 5=lowest)"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    batch_id: str
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    results: List[PredictionResponse]
    total_processing_time_ms: float
    timestamp: datetime


@dataclass
class ModelMetadata:
    """Model metadata for serving"""

    model_type: str
    version: str
    file_path: str
    config: Dict[str, Any]
    loaded_at: datetime
    last_used: datetime
    usage_count: int = 0
    performance_metrics: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelCache:
    """Model caching with TTL and LRU eviction"""

    def __init__(self, max_models: int = 10, ttl_hours: int = 24):
        self.max_models = max_models
        self.ttl_hours = ttl_hours
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.access_order: List[str] = []

    def _get_cache_key(self, model_type: str, version: str) -> str:
        """Generate cache key"""
        return f"{model_type}:{version}"

    def get_model(self, model_type: str, version: str) -> Optional[Any]:
        """Get model from cache"""
        cache_key = self._get_cache_key(model_type, version)

        if cache_key not in self.models:
            return None

        # Check TTL
        metadata = self.metadata[cache_key]
        if datetime.now() - metadata.loaded_at > timedelta(hours=self.ttl_hours):
            self._evict_model(cache_key)
            return None

        # Update access order and metadata
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

        metadata.last_used = datetime.now()
        metadata.usage_count += 1

        return self.models[cache_key]

    def put_model(
        self, model_type: str, version: str, model: Any, metadata: ModelMetadata
    ) -> None:
        """Put model in cache"""
        cache_key = self._get_cache_key(model_type, version)

        # Evict if at capacity
        if len(self.models) >= self.max_models and cache_key not in self.models:
            self._evict_lru()

        self.models[cache_key] = model
        self.metadata[cache_key] = metadata

        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def _evict_lru(self) -> None:
        """Evict least recently used model"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            self._evict_model(lru_key)

    def _evict_model(self, cache_key: str) -> None:
        """Evict specific model"""
        if cache_key in self.models:
            del self.models[cache_key]
            del self.metadata[cache_key]
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_usage = sum(meta.usage_count for meta in self.metadata.values())

        return {
            "total_models": len(self.models),
            "max_capacity": self.max_models,
            "total_usage": total_usage,
            "models": [
                {
                    "key": key,
                    "model_type": meta.model_type,
                    "version": meta.version,
                    "loaded_at": meta.loaded_at.isoformat(),
                    "last_used": meta.last_used.isoformat(),
                    "usage_count": meta.usage_count,
                }
                for key, meta in self.metadata.items()
            ],
        }


class ABTestManager:
    """A/B testing framework for model comparison"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def create_experiment(
        self,
        experiment_id: str,
        model_variants: Dict[
            str, Dict[str, str]
        ],  # variant_name -> {model_type, version}
        traffic_split: Dict[str, float],  # variant_name -> percentage
        duration_hours: int = 24,
    ) -> None:
        """Create A/B test experiment"""

        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")

        experiment = {
            "experiment_id": experiment_id,
            "model_variants": model_variants,
            "traffic_split": traffic_split,
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
            "status": "active",
            "metrics": {
                variant: {"requests": 0, "errors": 0, "total_latency": 0.0}
                for variant in model_variants.keys()
            },
        }

        self.experiments[experiment_id] = experiment

        # Store in Redis for persistence
        self.redis.setex(
            f"experiment:{experiment_id}",
            int(timedelta(hours=duration_hours + 1).total_seconds()),
            json.dumps(experiment, default=str),
        )

        logger.info(f"Created A/B test experiment: {experiment_id}")

    def get_variant_for_request(
        self, experiment_id: str, request_id: str
    ) -> Optional[str]:
        """Get variant assignment for request"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        # Check if experiment is still active
        end_time = datetime.fromisoformat(experiment["end_time"])
        if datetime.now() > end_time:
            experiment["status"] = "completed"
            return None

        # Use request ID hash for consistent assignment
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0

        # Determine variant based on traffic split
        cumulative_split = 0.0
        for variant, split in experiment["traffic_split"].items():
            cumulative_split += split
            if normalized_hash <= cumulative_split:
                return variant

        # Fallback to first variant
        return list(experiment["traffic_split"].keys())[0]

    def record_metrics(
        self, experiment_id: str, variant: str, latency_ms: float, error: bool = False
    ) -> None:
        """Record experiment metrics"""
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]["metrics"][variant]
        metrics["requests"] += 1
        metrics["total_latency"] += latency_ms

        if error:
            metrics["errors"] += 1

        # Update Redis
        self.redis.setex(
            f"experiment:{experiment_id}",
            86400,  # 24 hours
            json.dumps(self.experiments[experiment_id], default=str),
        )

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]
        results = {
            "experiment_id": experiment_id,
            "status": experiment["status"],
            "start_time": experiment["start_time"],
            "end_time": experiment["end_time"],
            "variants": {},
        }

        for variant, metrics in experiment["metrics"].items():
            avg_latency = (
                metrics["total_latency"] / metrics["requests"]
                if metrics["requests"] > 0
                else 0.0
            )
            error_rate = (
                metrics["errors"] / metrics["requests"]
                if metrics["requests"] > 0
                else 0.0
            )

            results["variants"][variant] = {
                "requests": metrics["requests"],
                "errors": metrics["errors"],
                "error_rate": error_rate,
                "avg_latency_ms": avg_latency,
                "model_config": experiment["model_variants"][variant],
            }

        return results


class ModelServingService:
    """Main model serving service"""

    def __init__(self):
        self.settings = get_settings()
        self.model_cache = ModelCache(max_models=10, ttl_hours=24)
        self.redis_client = None
        self.ab_test_manager = None
        self.batch_queue = asyncio.Queue()
        self.batch_processor_task = None

    async def initialize(self) -> None:
        """Initialize the serving service"""
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.settings.redis.host,
            port=self.settings.redis.port,
            db=self.settings.redis.db,
            password=self.settings.redis.password,
            decode_responses=True,
        )

        # Initialize A/B test manager
        self.ab_test_manager = ABTestManager(self.redis_client)

        # Start batch processor
        self.batch_processor_task = asyncio.create_task(self._batch_processor())

        logger.info("Model serving service initialized")

    async def shutdown(self) -> None:
        """Shutdown the serving service"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            await self.redis_client.aclose()

        logger.info("Model serving service shutdown")

    async def load_model(
        self, model_type: str, version: str, file_path: str, config: Dict[str, Any]
    ) -> None:
        """Load model into cache"""
        try:
            if model_type == ModelType.CNN_LSTM_HYBRID:
                # Load CNN+LSTM hybrid model
                model_config = HybridModelConfig(**config)
                model = CNNLSTMHybridModel(model_config)
                model.load_model(file_path)
                model.eval()

            elif model_type == ModelType.RL_ENSEMBLE:
                # Load RL ensemble model - placeholder implementation
                # In production, this would load the actual RL ensemble
                model = type(
                    "MockRLEnsemble",
                    (),
                    {
                        "predict": lambda self, obs, return_confidence=False: (
                            np.array([0.5, 0.3, 0.2]),
                            0.8 if return_confidence else None,
                        )
                    },
                )()

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Create metadata
            metadata = ModelMetadata(
                model_type=model_type,
                version=version,
                file_path=file_path,
                config=config,
                loaded_at=datetime.now(),
                last_used=datetime.now(),
            )

            # Cache the model
            self.model_cache.put_model(model_type, version, model, metadata)

            logger.info(f"Loaded model: {model_type}:{version}")

        except Exception as e:
            logger.error(f"Failed to load model {model_type}:{version}: {e}")
            raise

    async def predict(
        self, request: PredictionRequest, request_id: Optional[str] = None
    ) -> PredictionResponse:
        """Make prediction with caching and A/B testing"""

        if request_id is None:
            request_id = str(uuid.uuid4())

        start_time = time.time()

        try:
            # A/B testing - determine model variant using Ray Serve AB Test Manager
            ab_test_group = None
            model_type = request.model_type
            model_version = request.model_version or "latest"

            # Import the global ab_test_manager from ray_serve module
            from src.ml.ray_serve.ab_testing import ab_test_manager as ray_ab_manager
            
            # Check for active experiments in Ray Serve AB Test Manager
            for exp_id, experiment in ray_ab_manager.experiments.items():
                if experiment.status == "active":
                    variant = ray_ab_manager.get_variant_for_request(exp_id, request_id)
                    if variant and variant in experiment.variants:
                        ab_test_group = f"{exp_id}:{variant}"
                        # Use the model path from the variant config as the version
                        variant_config = experiment.variants[variant]
                        model_version = variant_config.model_path
                        break

            # Get model from cache
            model = self.model_cache.get_model(model_type, model_version)
            if model is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_type}:{model_version} not found in cache",
                )

            # Prepare input data
            input_data = np.array(request.data, dtype=np.float32)

            # Make prediction
            if model_type == ModelType.CNN_LSTM_HYBRID:
                predictions = model.predict(
                    input_data,
                    return_uncertainty=request.return_uncertainty,
                    use_ensemble=request.use_ensemble,
                )

                # Extract confidence scores
                confidence_scores = None
                if "classification_probs" in predictions:
                    confidence_scores = np.max(
                        predictions["classification_probs"], axis=1
                    ).tolist()

                uncertainty = None
                if (
                    request.return_uncertainty
                    and "regression_uncertainty" in predictions
                ):
                    uncertainty = {
                        "regression_uncertainty": predictions[
                            "regression_uncertainty"
                        ].tolist(),
                        "ensemble_weights": predictions.get(
                            "ensemble_weights", []
                        ).tolist()
                        if predictions.get("ensemble_weights") is not None
                        else None,
                    }

            elif model_type == ModelType.RL_ENSEMBLE:
                # RL ensemble prediction
                predictions = {}
                confidence_scores = []
                uncertainty = None

                for i, obs in enumerate(input_data):
                    action, confidence = model.predict(obs, return_confidence=True)
                    predictions[f"action_{i}"] = (
                        action.tolist() if hasattr(action, "tolist") else action
                    )
                    confidence_scores.append(
                        float(confidence) if confidence is not None else 0.0
                    )

            processing_time = (time.time() - start_time) * 1000

            # Record A/B test metrics
            if ab_test_group:
                exp_id, variant = ab_test_group.split(":", 1)
                from src.ml.ray_serve.ab_testing import ab_test_manager as ray_ab_manager
                
                # Calculate confidence score for A/B testing
                avg_confidence = None
                if confidence_scores:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                
                ray_ab_manager.record_metrics(
                    experiment_id=exp_id,
                    variant_name=variant,
                    latency_ms=processing_time,
                    processing_time_ms=processing_time,
                    confidence_score=avg_confidence
                )

            # Record metrics
            metrics.increment_counter(
                "model_predictions_total", {"model_type": model_type}
            )
            metrics.record_histogram(
                "prediction_latency_ms", processing_time, {"model_type": model_type}
            )

            return PredictionResponse(
                request_id=request_id,
                model_type=model_type,
                model_version=model_version,
                predictions=predictions,
                uncertainty=uncertainty,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                ab_test_group=ab_test_group,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            # Record error metrics
            if ab_test_group:
                exp_id, variant = ab_test_group.split(":", 1)
                from src.ml.ray_serve.ab_testing import ab_test_manager as ray_ab_manager
                ray_ab_manager.record_metrics(
                    experiment_id=exp_id,
                    variant_name=variant,
                    latency_ms=processing_time,
                    processing_time_ms=processing_time,
                    error=True
                )

            metrics.increment_counter(
                "model_prediction_errors_total", {"model_type": model_type}
            )

            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def batch_predict(
        self, request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        """Process batch predictions with optimization"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()

        # Group requests by model type for batch optimization
        grouped_requests = {}
        for i, req in enumerate(request.requests):
            key = f"{req.model_type}:{req.model_version or 'latest'}"
            if key not in grouped_requests:
                grouped_requests[key] = []
            grouped_requests[key].append((i, req))

        results = [None] * len(request.requests)
        successful = 0
        failed = 0

        # Process each group
        for model_key, group_requests in grouped_requests.items():
            try:
                # Batch process requests for the same model
                batch_data = []
                request_indices = []

                for idx, req in group_requests:
                    batch_data.extend(req.data)
                    request_indices.append(idx)

                # Create batch request
                batch_req = PredictionRequest(
                    model_type=group_requests[0][1].model_type,
                    model_version=group_requests[0][1].model_version,
                    data=batch_data,
                    batch_size=min(len(batch_data), 128),  # Optimize batch size
                    return_uncertainty=any(
                        req.return_uncertainty for _, req in group_requests
                    ),
                    use_ensemble=any(req.use_ensemble for _, req in group_requests),
                )

                # Make batch prediction
                batch_response = await self.predict(
                    batch_req, f"{batch_id}_batch_{model_key}"
                )

                # Split results back to individual requests
                for i, (original_idx, original_req) in enumerate(group_requests):
                    # Extract individual result from batch
                    individual_response = PredictionResponse(
                        request_id=f"{batch_id}_{original_idx}",
                        model_type=batch_response.model_type,
                        model_version=batch_response.model_version,
                        predictions=batch_response.predictions,  # Simplified - would need proper splitting
                        uncertainty=batch_response.uncertainty,
                        confidence_scores=batch_response.confidence_scores,
                        processing_time_ms=batch_response.processing_time_ms
                        / len(group_requests),
                        timestamp=batch_response.timestamp,
                        ab_test_group=batch_response.ab_test_group,
                    )

                    results[original_idx] = individual_response
                    successful += 1

            except Exception as e:
                logger.error(f"Batch processing failed for {model_key}: {e}")

                # Mark all requests in this group as failed
                for original_idx, _ in group_requests:
                    failed += 1

        total_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            batch_id=batch_id,
            total_requests=len(request.requests),
            successful_predictions=successful,
            failed_predictions=failed,
            results=[r for r in results if r is not None],
            total_processing_time_ms=total_time,
            timestamp=datetime.now(),
        )

    async def _batch_processor(self) -> None:
        """Background batch processor for queued requests"""
        while True:
            try:
                # Wait for batch requests
                batch_requests = []

                # Collect requests for up to 100ms or until we have 10 requests
                deadline = time.time() + 0.1  # 100ms

                while time.time() < deadline and len(batch_requests) < 10:
                    try:
                        request = await asyncio.wait_for(
                            self.batch_queue.get(), timeout=deadline - time.time()
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break

                if batch_requests:
                    # Process batch
                    batch_request = BatchPredictionRequest(requests=batch_requests)
                    await self.batch_predict(batch_request)

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)


# Global service instance
serving_service = ModelServingService()


# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await serving_service.initialize()
    yield
    # Shutdown
    await serving_service.shutdown()


def create_app() -> FastAPI:
    """Create FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title="AI Trading Platform - Model Serving API",
        description="Model serving infrastructure with caching and A/B testing",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# Dependency for getting the serving service
async def get_serving_service() -> ModelServingService:
    """Get the serving service instance"""
    return serving_service
