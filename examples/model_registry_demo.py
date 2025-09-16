"""Demo script for model registry with versioning and automated rollback capabilities

This script demonstrates how to use the model registry to:
1. Register model versions
2. Deploy models with Ray Serve
3. Monitor model performance
4. Automatically rollback when performance degrades
"""

import asyncio
import os
import tempfile
from datetime import datetime

from src.ml.model_registry import get_model_registry, ModelStatus, RollbackReason
from src.ml.ray_serve.model_registry_integration import get_ray_serve_integration


async def demo_model_registry():
    """Demonstrate model registry functionality"""
    print("=== Model Registry Demo ===")
    
    # Get registry and integration instances
    registry = get_model_registry()
    integration = get_ray_serve_integration()
    
    # Create a temporary directory for model files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # 1. Register model versions
        print("\n1. Registering model versions...")
        
        # Create sample model files
        model_v1_path = os.path.join(temp_dir, "model_v1.pth")
        model_v2_path = os.path.join(temp_dir, "model_v2.pth")
        
        # Create dummy model files
        with open(model_v1_path, "w") as f:
            f.write("dummy model v1 content")
        
        with open(model_v2_path, "w") as f:
            f.write("dummy model v2 content")
        
        # Register first version
        model_v1 = registry.register_model(
            model_id="trading_model",
            version="1.0.0",
            file_path=model_v1_path,
            config={
                "model_type": "CNNLSTMHybridModel",
                "input_dim": 10,
                "output_dim": 3,
                "sequence_length": 60
            },
            metadata={
                "description": "Initial trading model",
                "created_by": "demo_script",
                "created_at": datetime.now().isoformat()
            }
        )
        print(f"   Registered model v1: {model_v1.model_id}:{model_v1.version}")
        
        # Register second version
        model_v2 = registry.register_model(
            model_id="trading_model",
            version="1.1.0",
            file_path=model_v2_path,
            config={
                "model_type": "CNNLSTMHybridModel",
                "input_dim": 10,
                "output_dim": 3,
                "sequence_length": 60,
                "enhanced_features": True
            },
            metadata={
                "description": "Enhanced trading model with new features",
                "created_by": "demo_script",
                "created_at": datetime.now().isoformat()
            }
        )
        print(f"   Registered model v2: {model_v2.model_id}:{model_v2.version}")
        
        # 2. Deploy model
        print("\n2. Deploying model version 1.1.0...")
        
        # Deploy the newer version
        deploy_success = await integration.deploy_model_version(
            model_id="trading_model",
            version="1.1.0",
            ray_deployment_name="trading_model_deployment",
            ray_bind_args={"device": "cpu"}
        )
        
        if deploy_success:
            print("   Model deployed successfully")
        else:
            print("   Model deployment failed")
            return
        
        # 3. Update performance metrics
        print("\n3. Updating performance metrics...")
        
        # Simulate good performance metrics
        good_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "error_rate": 0.02,
            "latency_95th_percentile": 50.0
        }
        
        registry.update_performance_metrics("trading_model", "1.1.0", good_metrics)
        print("   Performance metrics updated")
        
        # Check rollback conditions (should not trigger rollback)
        should_rollback, reason, description = registry.check_rollback_conditions(
            "trading_model", "1.1.0"
        )
        
        if should_rollback:
            print(f"   Rollback needed: {reason.value} - {description}")
        else:
            print("   No rollback needed - performance is good")
        
        # 4. Simulate performance degradation
        print("\n4. Simulating performance degradation...")
        
        # Update with poor performance metrics
        poor_metrics = {
            "accuracy": 0.60,  # Below threshold
            "precision": 0.58,
            "recall": 0.62,
            "f1_score": 0.60,
            "error_rate": 0.08,  # Above threshold
            "latency_95th_percentile": 150.0 # Above threshold
        }
        
        registry.update_performance_metrics("trading_model", "1.1.0", poor_metrics)
        print("   Poor performance metrics updated")
        
        # Check rollback conditions (should trigger rollback)
        should_rollback, reason, description = registry.check_rollback_conditions(
            "trading_model", "1.1.0"
        )
        
        if should_rollback:
            print(f"   Rollback needed: {reason.value} - {description}")
            
            # Perform rollback
            print("\n5. Performing automatic rollback...")
            rollback_success = registry.rollback_model(
                model_id="trading_model",
                reason=reason,
                description=description
            )
            
            if rollback_success:
                print("   Rollback successful")
                
                # Show rollback history
                rollback_history = registry.get_rollback_history("trading_model")
                if rollback_history:
                    latest_rollback = rollback_history[-1]
                    print(f"   Rolled back from {latest_rollback.from_version} to {latest_rollback.to_version}")
                    print(f"   Reason: {latest_rollback.reason.value}")
            else:
                print("   Rollback failed")
        else:
            print("   No rollback needed")
        
        # 6. Show model information
        print("\n6. Model information:")
        model_info = registry.get_model_info("trading_model")
        print(f"   Model ID: {model_info['model_id']}")
        print(f"   Total versions: {model_info['total_versions']}")
        print(f"   Active version: {model_info['active_version']}")
        
        # Show all versions
        print("   All versions:")
        for version_info in model_info['versions']:
            status = version_info['status']
            version = version_info['version']
            print(f"     {version} - {status}")
        
        # Show rollback history
        print("   Rollback history:")
        for rollback_event in model_info['rollback_history']:
            print(f"     {rollback_event['from_version']} -> {rollback_event['to_version']} "
                  f"({rollback_event['reason']})")
        
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")


def main():
    """Main function"""
    print("Model Registry with Versioning and Automated Rollback Demo")
    print("=" * 60)
    
    # Run the demo
    asyncio.run(demo_model_registry())
    
    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    main()