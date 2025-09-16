"""
Demonstration of model interpretability and explainability features.

This script showcases how to use the interpretability tools to understand the
predictions of the CNN+LSTM hybrid model.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

import numpy as np
import torch

from src.ml.hybrid_model import create_hybrid_model_config, CNNLSTMHybridModel
from src.ml.shap_explainer import create_shap_explainer
from src.ml.attention_visualizer import create_attention_visualizer
from src.ml.feature_importance_analyzer import create_feature_importance_analyzer

def demonstrate_interpretability():
    """Demonstrates the model interpretability features."""
    print("--- Demonstrating Model Interpretability Features ---")

    # 1. Load a pre-trained model (or create a mock one for demo)
    print("\n1. Loading pre-trained model...")
    config = create_hybrid_model_config(input_dim=10)
    model = CNNLSTMHybridModel(config)
    model.is_trained = True # In a real scenario, you would load a trained model

    # 2. Create interpretability tools
    print("\n2. Creating interpretability tools...")
    shap_explainer = create_shap_explainer(model)
    attention_visualizer = create_attention_visualizer()
    feature_importance_analyzer = create_feature_importance_analyzer(model)

    # 3. Generate a sample prediction
    print("\n3. Generating a sample prediction...")
    sample_data = np.random.randn(1, 10, 50) # (batch, features, sequence_length)
    prediction = model.predict(sample_data)

    print(f"   Prediction: {prediction}")

    # 4. Explain the prediction
    print("\n4. Explaining the prediction...")

    # 4.1. SHAP Explanations
    print("\n   4.1. SHAP Explanations...")
    background_data = np.random.randn(10, 10, 50)
    shap_values = shap_explainer.compute_shap_values(background_data, sample_data)
    feature_importance = shap_explainer.get_feature_importance(shap_values)

    print(f"      Feature Importance (from SHAP): {feature_importance['importance_by_feature']}")
    # shap_explainer.visualize_shap_summary(shap_values['shap_values'])

    # 4.2. Attention Visualization
    print("\n   4.2. Attention Visualization...")
    attention_weights = shap_explainer.get_attention_weights(sample_data)
    # attention_visualizer.visualize_cnn_attention(attention_weights['cnn_attention'])

    # 4.3. Feature Importance Analysis
    print("\n   4.3. Feature Importance Analysis...")
    # Permutation importance requires a dataset, so we'll use the background data for demonstration
    perm_importance = feature_importance_analyzer.compute_permutation_importance(
        X=background_data,
        y_class=np.random.randint(0, 3, 10),
        y_reg=np.random.randn(10, 1)
    )
    print(f"      Permutation Importance: {perm_importance['importances_mean']}")

    print("\n--- Interpretability Demonstration Complete ---")

if __name__ == "__main__":
    demonstrate_interpretability()
