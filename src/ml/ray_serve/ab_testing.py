"""A/B testing framework for Ray Serve deployments with traffic splitting and statistical significance testing.

This module provides an A/B testing framework that integrates with Ray Serve deployments,
allowing for traffic splitting between different model versions and statistical significance
testing of performance differences.
"""

import asyncio
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats
import logging

from src.utils.logging import get_logger
from src.utils.monitoring import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class VariantStatus(Enum):
    """Status of an A/B test variant."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"


@dataclass
class VariantConfig:
    """Configuration for a model variant in A/B testing."""
    name: str
    model_path: str
    weight: float  # Traffic allocation weight (0.0 to 1.0)
    status: VariantStatus = VariantStatus.ACTIVE
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VariantMetrics:
    """Metrics for a specific variant."""
    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    successful_predictions: int = 0
    confidence_scores: List[float] = None
    prediction_latencies: List[float] = None

    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.prediction_latencies is None:
            self.prediction_latencies = []

    def add_request(self, latency_ms: float, processing_time_ms: float, 
                   confidence_score: Optional[float] = None, error: bool = False):
        """Add metrics for a request."""
        self.requests += 1
        self.total_latency_ms += latency_ms
        self.total_processing_time_ms += processing_time_ms
        self.prediction_latencies.append(latency_ms)  # Always track latency
        
        if error:
            self.errors += 1
        else:
            self.successful_predictions += 1
            if confidence_score is not None:
                self.confidence_scores.append(confidence_score)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_latency_ms / self.requests if self.requests > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        return self.errors / self.requests if self.requests > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "successful_predictions": self.successful_predictions,
            "avg_confidence": self.avg_confidence,
            "confidence_scores_count": len(self.confidence_scores),
            "prediction_latencies_count": len(self.prediction_latencies)
        }


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ABTestExperiment:
    """A/B test experiment for comparing model variants."""

    def __init__(self, experiment_id: str, variants: List[VariantConfig], 
                 duration_hours: int = 24, confidence_level: float = 0.95):
        """Initialize A/B test experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            variants: List of variant configurations
            duration_hours: Duration of the experiment in hours
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.experiment_id = experiment_id
        self.variants = {variant.name: variant for variant in variants}
        self.variant_metrics = {variant.name: VariantMetrics() for variant in variants}
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        self.confidence_level = confidence_level
        self.status = "active"  # active, completed, stopped
        self.created_at = datetime.now()
        
        # Validate traffic weights sum to 1.0
        total_weight = sum(variant.weight for variant in variants)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")

    def get_variant_for_request(self, request_id: str) -> Optional[str]:
        """Get variant assignment for a request based on traffic splitting.
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            Name of the assigned variant, or None if experiment is not active
        """
        if self.status != "active" or datetime.now() > self.end_time:
            self.status = "completed"
            return None

        # Use request ID hash for consistent assignment
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0

        # Determine variant based on traffic weights
        cumulative_weight = 0.0
        for variant_name, variant in self.variants.items():
            if variant.status == VariantStatus.ACTIVE:
                cumulative_weight += variant.weight
                if normalized_hash <= cumulative_weight:
                    return variant_name

        # Fallback to first active variant
        for variant_name, variant in self.variants.items():
            if variant.status == VariantStatus.ACTIVE:
                return variant_name
        
        return None

    def record_metrics(self, variant_name: str, latency_ms: float, 
                      processing_time_ms: float, confidence_score: Optional[float] = None,
                      error: bool = False) -> None:
        """Record metrics for a request.
        
        Args:
            variant_name: Name of the variant
            latency_ms: Request latency in milliseconds
            processing_time_ms: Processing time in milliseconds
            confidence_score: Confidence score of the prediction (optional)
            error: Whether the request resulted in an error
        """
        if variant_name in self.variant_metrics:
            self.variant_metrics[variant_name].add_request(
                latency_ms, processing_time_ms, confidence_score, error
            )

    def get_results(self) -> Dict[str, Any]:
        """Get experiment results.
        
        Returns:
            Dictionary containing experiment results and metrics
        """
        results = {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": (self.end_time - self.start_time).total_seconds() / 3600,
            "confidence_level": self.confidence_level,
            "variants": {}
        }

        for variant_name, metrics in self.variant_metrics.items():
            variant_config = self.variants[variant_name]
            results["variants"][variant_name] = {
                "config": {
                    "name": variant_config.name,
                    "model_path": variant_config.model_path,
                    "weight": variant_config.weight,
                    "status": variant_config.status.value,
                    "metadata": variant_config.metadata
                },
                "metrics": metrics.to_dict()
            }

        return results

    def perform_statistical_tests(self) -> List[StatisticalTestResult]:
        """Perform statistical significance tests on variant metrics.
        
        Returns:
            List of statistical test results
        """
        results = []
        
        # Get active variants with sufficient data
        active_variants = [
            (name, metrics) for name, metrics in self.variant_metrics.items()
            if self.variants[name].status == VariantStatus.ACTIVE and metrics.requests > 30
        ]
        
        if len(active_variants) < 2:
            return results

        # Compare each pair of variants
        for i in range(len(active_variants)):
            for j in range(i + 1, len(active_variants)):
                variant_a_name, metrics_a = active_variants[i]
                variant_b_name, metrics_b = active_variants[j]
                
                # Perform t-test on latency
                if metrics_a.prediction_latencies and metrics_b.prediction_latencies:
                    t_stat, p_value = stats.ttest_ind(
                        metrics_a.prediction_latencies, 
                        metrics_b.prediction_latencies,
                        equal_var=False  # Welch's t-test
                    )
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(metrics_a.prediction_latencies) - 1) * np.var(metrics_a.prediction_latencies) +
                         (len(metrics_b.prediction_latencies) - 1) * np.var(metrics_b.prediction_latencies)) /
                        (len(metrics_a.prediction_latencies) + len(metrics_b.prediction_latencies) - 2)
                    )
                    
                    effect_size = (np.mean(metrics_a.prediction_latencies) - np.mean(metrics_b.prediction_latencies)) / pooled_std
                    
                    test_result = StatisticalTestResult(
                        test_name=f"latency_ttest_{variant_a_name}_vs_{variant_b_name}",
                        statistic=t_stat,
                        p_value=p_value,
                        significant=bool(p_value < (1 - self.confidence_level)),
                        confidence_level=self.confidence_level,
                        effect_size=effect_size,
                        description=f"T-test comparing latency between {variant_a_name} and {variant_b_name}"
                    )
                    results.append(test_result)
                
                # Perform z-test on error rates
                if metrics_a.requests > 0 and metrics_b.requests > 0:
                    p1 = metrics_a.error_rate
                    p2 = metrics_b.error_rate
                    n1 = metrics_a.requests
                    n2 = metrics_b.requests
                    
                    if n1 > 0 and n2 > 0:
                        # Pooled error rate
                        p_pooled = (metrics_a.errors + metrics_b.errors) / (n1 + n2)
                        
                        # Standard error
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                        
                        # Z-statistic
                        if se > 0:
                            z_stat = (p1 - p2) / se
                            
                            # P-value (two-tailed)
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                            
                            test_result = StatisticalTestResult(
                                test_name=f"error_rate_ztest_{variant_a_name}_vs_{variant_b_name}",
                                statistic=z_stat,
                                p_value=p_value,
                                significant=bool(p_value < (1 - self.confidence_level)),
                                confidence_level=self.confidence_level,
                                description=f"Z-test comparing error rates between {variant_a_name} and {variant_b_name}"
                            )
                            results.append(test_result)

        return results

    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of the experiment.
        
        Returns:
            Dictionary containing statistical summary
        """
        tests = self.perform_statistical_tests()
        
        summary = {
            "experiment_id": self.experiment_id,
            "total_tests": len(tests),
            "significant_tests": len([t for t in tests if t.significant]),
            "confidence_level": self.confidence_level,
            "tests": [t.to_dict() for t in tests]
        }
        
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "variants": {name: {
                "name": variant.name,
                "model_path": variant.model_path,
                "weight": variant.weight,
                "status": variant.status.value,
                "metadata": variant.metadata
            } for name, variant in self.variants.items()},
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "confidence_level": self.confidence_level,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }


class RayServeABTestManager:
    """A/B test manager for Ray Serve deployments."""

    def __init__(self):
        """Initialize A/B test manager."""
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.logger = get_logger(__name__)
        self.safety_controls = {
            "max_error_rate": 0.1,  # 10% max error rate
            "min_confidence_score": 0.7,  # Minimum confidence threshold
            "max_latency_ms": 1000,  # Maximum acceptable latency
            "min_sample_size": 100,  # Minimum samples for statistical significance
        }

    def create_experiment(self, experiment_id: str, variants: List[VariantConfig],
                         duration_hours: int = 24, confidence_level: float = 0.95) -> ABTestExperiment:
        """Create a new A/B test experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            variants: List of variant configurations
            duration_hours: Duration of the experiment in hours
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Created ABTestExperiment instance
        """
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            variants=variants,
            duration_hours=duration_hours,
            confidence_level=confidence_level
        )
        
        self.experiments[experiment_id] = experiment
        self.logger.info(f"Created A/B test experiment: {experiment_id}")
        
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """Get an experiment by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            ABTestExperiment instance or None if not found
        """
        return self.experiments.get(experiment_id)

    def get_variant_for_request(self, experiment_id: str, request_id: str) -> Optional[str]:
        """Get variant assignment for a request.
        
        Args:
            experiment_id: Experiment identifier
            request_id: Request identifier
            
        Returns:
            Name of the assigned variant or None
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
            
        return experiment.get_variant_for_request(request_id)

    def record_metrics(self, experiment_id: str, variant_name: str, latency_ms: float,
                      processing_time_ms: float, confidence_score: Optional[float] = None,
                      error: bool = False) -> None:
        """Record metrics for a request.
        
        Args:
            experiment_id: Experiment identifier
            variant_name: Name of the variant
            latency_ms: Request latency in milliseconds
            processing_time_ms: Processing time in milliseconds
            confidence_score: Confidence score of the prediction (optional)
            error: Whether the request resulted in an error
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.record_metrics(
                variant_name, latency_ms, processing_time_ms, confidence_score, error
            )

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing experiment results or None if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
            
        return experiment.get_results()

    def get_statistical_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get statistical summary for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing statistical summary or None if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
            
        return experiment.get_statistical_summary()

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment information
        """
        return [exp.to_dict() for exp in self.experiments.values()]

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if experiment was stopped, False if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        experiment.status = "stopped"
        self.logger.info(f"Stopped A/B test experiment: {experiment_id}")
        return True

    def check_safety_controls(self, experiment_id: str) -> Dict[str, Any]:
        """Check safety controls for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing safety check results
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        safety_results = {
            "experiment_id": experiment_id,
            "safety_violations": [],
            "recommendations": [],
            "should_stop": False
        }

        for variant_name, metrics in experiment.variant_metrics.items():
            if metrics.requests < 10:  # Skip variants with insufficient data
                continue

            # Check error rate
            if metrics.error_rate > self.safety_controls["max_error_rate"]:
                safety_results["safety_violations"].append({
                    "variant": variant_name,
                    "violation": "high_error_rate",
                    "value": metrics.error_rate,
                    "threshold": self.safety_controls["max_error_rate"]
                })
                safety_results["should_stop"] = True

            # Check average confidence
            if metrics.avg_confidence < self.safety_controls["min_confidence_score"]:
                safety_results["safety_violations"].append({
                    "variant": variant_name,
                    "violation": "low_confidence",
                    "value": metrics.avg_confidence,
                    "threshold": self.safety_controls["min_confidence_score"]
                })

            # Check latency
            if metrics.avg_latency_ms > self.safety_controls["max_latency_ms"]:
                safety_results["safety_violations"].append({
                    "variant": variant_name,
                    "violation": "high_latency",
                    "value": metrics.avg_latency_ms,
                    "threshold": self.safety_controls["max_latency_ms"]
                })

        # Generate recommendations
        if safety_results["safety_violations"]:
            if safety_results["should_stop"]:
                safety_results["recommendations"].append("Stop experiment immediately due to safety violations")
            else:
                safety_results["recommendations"].append("Monitor experiment closely and consider reducing traffic to problematic variants")

        return safety_results

    def get_winner_recommendation(self, experiment_id: str) -> Dict[str, Any]:
        """Get winner recommendation based on risk-adjusted metrics.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing winner recommendation
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        # Perform statistical tests
        statistical_tests = experiment.perform_statistical_tests()
        
        # Calculate risk-adjusted scores for each variant
        variant_scores = {}
        
        for variant_name, metrics in experiment.variant_metrics.items():
            if metrics.requests < self.safety_controls["min_sample_size"]:
                continue

            # Calculate composite score (lower is better)
            # Factors: error rate, latency, confidence (inverted)
            error_penalty = metrics.error_rate * 100  # Scale error rate
            latency_penalty = metrics.avg_latency_ms / 100  # Scale latency
            confidence_bonus = (1 - metrics.avg_confidence) * 50  # Invert confidence
            
            composite_score = error_penalty + latency_penalty + confidence_bonus
            
            variant_scores[variant_name] = {
                "composite_score": composite_score,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_confidence": metrics.avg_confidence,
                "requests": metrics.requests,
                "statistical_significance": []
            }

        # Add statistical significance information
        for test in statistical_tests:
            if "vs" in test.test_name:
                variants = test.test_name.split("_")[-1].split("_vs_")
                if len(variants) == 2:
                    variant_a, variant_b = variants
                    if variant_a in variant_scores:
                        variant_scores[variant_a]["statistical_significance"].append({
                            "compared_to": variant_b,
                            "test_type": test.test_name.split("_")[0],
                            "significant": test.significant,
                            "p_value": test.p_value,
                            "effect_size": test.effect_size
                        })

        # Determine winner
        if not variant_scores:
            return {
                "experiment_id": experiment_id,
                "winner": None,
                "reason": "Insufficient data for winner determination",
                "variant_scores": {}
            }

        # Find variant with lowest composite score (best performance)
        winner = min(variant_scores.keys(), key=lambda v: variant_scores[v]["composite_score"])
        
        # Check if winner is statistically significant
        winner_significant = any(
            sig["significant"] for sig in variant_scores[winner]["statistical_significance"]
        )

        return {
            "experiment_id": experiment_id,
            "winner": winner,
            "statistically_significant": winner_significant,
            "confidence_level": experiment.confidence_level,
            "variant_scores": variant_scores,
            "recommendation": f"Variant '{winner}' shows best risk-adjusted performance" if winner_significant 
                           else f"Variant '{winner}' shows best performance but not statistically significant"
        }

    def implement_gradual_rollout(self, experiment_id: str, winning_variant: str, 
                                 rollout_steps: List[float] = None) -> Dict[str, Any]:
        """Implement gradual rollout of winning variant.
        
        Args:
            experiment_id: Experiment identifier
            winning_variant: Name of the winning variant
            rollout_steps: List of traffic percentages for gradual rollout
            
        Returns:
            Dictionary containing rollout plan
        """
        if rollout_steps is None:
            rollout_steps = [0.1, 0.25, 0.5, 0.75, 1.0]  # 10%, 25%, 50%, 75%, 100%

        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        if winning_variant not in experiment.variants:
            return {"error": f"Variant '{winning_variant}' not found in experiment"}

        # Create rollout plan
        rollout_plan = {
            "experiment_id": experiment_id,
            "winning_variant": winning_variant,
            "rollout_steps": [],
            "current_step": 0,
            "status": "planned"
        }

        current_winner_weight = experiment.variants[winning_variant].weight
        
        for i, target_percentage in enumerate(rollout_steps):
            step = {
                "step": i + 1,
                "target_percentage": target_percentage,
                "winner_weight": target_percentage,
                "other_weights": {},
                "safety_checks": True,
                "duration_hours": 2,  # Each step runs for 2 hours
                "completed": False
            }

            # Calculate weights for other variants
            remaining_weight = 1.0 - target_percentage
            other_variants = [v for v in experiment.variants.keys() if v != winning_variant]
            
            if other_variants and remaining_weight > 0:
                weight_per_other = remaining_weight / len(other_variants)
                for variant in other_variants:
                    step["other_weights"][variant] = weight_per_other

            rollout_plan["rollout_steps"].append(step)

        return rollout_plan

    def execute_rollout_step(self, rollout_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the next step in a gradual rollout.
        
        Args:
            rollout_plan: Rollout plan from implement_gradual_rollout
            
        Returns:
            Dictionary containing execution results
        """
        if rollout_plan["current_step"] >= len(rollout_plan["rollout_steps"]):
            return {"error": "All rollout steps completed"}

        current_step = rollout_plan["rollout_steps"][rollout_plan["current_step"]]
        experiment_id = rollout_plan["experiment_id"]
        winning_variant = rollout_plan["winning_variant"]

        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        # Check safety controls before proceeding
        safety_check = self.check_safety_controls(experiment_id)
        if safety_check["should_stop"]:
            return {
                "error": "Rollout stopped due to safety violations",
                "safety_violations": safety_check["safety_violations"]
            }

        # Update variant weights
        experiment.variants[winning_variant].weight = current_step["winner_weight"]
        
        for variant_name, weight in current_step["other_weights"].items():
            if variant_name in experiment.variants:
                experiment.variants[variant_name].weight = weight

        # Mark step as completed
        current_step["completed"] = True
        rollout_plan["current_step"] += 1

        self.logger.info(f"Executed rollout step {current_step['step']} for experiment {experiment_id}")

        return {
            "experiment_id": experiment_id,
            "step_executed": current_step["step"],
            "new_weights": {v.name: v.weight for v in experiment.variants.values()},
            "next_step": rollout_plan["current_step"] + 1 if rollout_plan["current_step"] < len(rollout_plan["rollout_steps"]) else None
        }

    async def cleanup_completed_experiments(self) -> int:
        """Clean up completed experiments.
        
        Returns:
            Number of experiments cleaned up
        """
        now = datetime.now()
        expired_experiments = [
            exp_id for exp_id, exp in self.experiments.items()
            if exp.end_time < now or exp.status == "completed"
        ]
        
        for exp_id in expired_experiments:
            del self.experiments[exp_id]
            
        if expired_experiments:
            self.logger.info(f"Cleaned up {len(expired_experiments)} completed experiments")
            
        return len(expired_experiments)


# Global instance
ab_test_manager = RayServeABTestManager()


def initialize_ab_testing() -> RayServeABTestManager:
    """Initialize and return the global A/B test manager.
    
    Returns:
        RayServeABTestManager instance
    """
    global ab_test_manager
    if ab_test_manager is None:
        ab_test_manager = RayServeABTestManager()
    return ab_test_manager


# Example usage functions
def create_sample_experiment() -> ABTestExperiment:
    """Create a sample A/B test experiment for demonstration.
    
    Returns:
        Created ABTestExperiment instance
    """
    variants = [
        VariantConfig(
            name="control",
            model_path="models/control_model_v1.pth",
            weight=0.5,
            metadata={"description": "Current production model"}
        ),
        VariantConfig(
            name="variant_a",
            model_path="models/experimental_model_v2.pth",
            weight=0.3,
            metadata={"description": "Experimental model with new features"}
        ),
        VariantConfig(
            name="variant_b",
            model_path="models/experimental_model_v3.pth",
            weight=0.2,
            metadata={"description": "Experimental model with different architecture"}
        )
    ]
    
    return ab_test_manager.create_experiment(
        experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
        variants=variants,
        duration_hours=48,
        confidence_level=0.95
    )