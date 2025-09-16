# Model Interpretability and Explainability Documentation

This document provides comprehensive documentation for the model interpretability and explainability features implemented for the CNN+LSTM hybrid model.

## Table of Contents

1. [Overview](#overview)
2. [SHAP Explainer](#shap-explainer)
3. [Attention Visualization](#attention-visualization)
4. [Feature Importance Analysis](#feature-importance-analysis)
5. [Integrated Gradients](#integrated-gradients)
6. [Decision Audit Trails](#decision-audit-trails)
7. [Uncertainty Calibration](#uncertainty-calibration)
8. [API Reference](#api-reference)
9. [Usage Examples](#usage-examples)
10. [Testing](#testing)

## Overview

The AI trading platform implements comprehensive model interpretability and explainability features to meet regulatory requirements and provide transparency in trading decisions. These features include:

- SHAP (SHapley Additive exPlanations) for local model interpretability
- Attention mechanism visualization for CNN and LSTM components
- Feature importance analysis with permutation importance
- Integrated gradients for feature attribution
- Decision audit trails with complete model version tracking
- Uncertainty calibration and confidence score validation

All features are designed to comply with requirement 12: "Model Interpretability and Explainability".

## SHAP Explainer

The SHAP explainer provides feature importance scores for individual predictions using SHapley Additive exPlanations.

### Features

- KernelSHAP and DeepSHAP support
- Feature importance extraction
- Summary and force plot visualizations
- Attention weight extraction
- Complete prediction explanations

### Usage

```python
from src.ml.shap_explainer import create_shap_explainer

# Create explainer
explainer = create_shap_explainer(model)

# Compute SHAP values
shap_result = explainer.compute_shap_values(
    background_data, 
    test_data, 
    explainer_type="kernel"
)

# Get feature importance
importance = explainer.get_feature_importance(shap_result, feature_names)

# Visualize results
explainer.visualize_shap_summary(
    shap_result['shap_values'],
    feature_names=feature_names
)
```

## Attention Visualization

Attention visualization provides insights into which parts of the input data the model focuses on during decision making.

### Features

- CNN attention visualization
- LSTM attention visualization
- Cross-attention between CNN and LSTM features
- Heatmap visualizations

### Usage

```python
from src.ml.attention_visualizer import create_attention_visualizer

# Create visualizer
visualizer = create_attention_visualizer()

# Visualize CNN attention
visualizer.visualize_cnn_attention(
    attention_weights,
    feature_names=feature_names,
    sequence_labels=sequence_labels
)

# Visualize LSTM attention
visualizer.visualize_lstm_attention(
    lstm_attention_weights,
    sequence_labels=sequence_labels
)
```

## Feature Importance Analysis

Feature importance analysis provides comprehensive understanding of which features contribute most to model predictions.

### Features

- Permutation importance calculation
- Support for multiple scoring metrics
- Feature importance visualization
- Comparison between different importance methods

### Usage

```python
from src.ml.feature_importance_analyzer import create_feature_importance_analyzer

# Create analyzer
analyzer = create_feature_importance_analyzer(model)

# Compute permutation importance
importance = analyzer.compute_permutation_importance(
    X, y_class, y_reg,
    scoring="accuracy",
    n_repeats=10
)
```

## Integrated Gradients

Integrated gradients provide path-based attribution for deep learning models, showing how input features contribute to predictions.

### Features

- Integrated gradients computation using Captum
- Noise tunnel for smoothing attributions
- Attribution visualization

### Usage

```python
# Compute integrated gradients
attributions, delta = analyzer.compute_integrated_gradients(
    input_data,
    baseline=None,
    steps=50
)

# Compute with noise tunnel
attributions, delta = analyzer.compute_integrated_gradients_with_noise(
    input_data,
    baseline=None,
    steps=50,
    nt_samples=10
)
```

## Decision Audit Trails

Decision audit trails provide complete tracking of model decisions with version control for regulatory compliance.

### Features

- Complete decision history with timestamps
- Model version tracking with hash-based identification
- Input data hashing for traceability
- Comprehensive audit reports

### Usage

```python
from src.ml.decision_auditor import create_decision_auditor

# Create auditor
auditor = create_decision_auditor("audit_trail.json")

# Log a decision
auditor.log_decision(
    model, input_data, prediction,
    shap_values=shap_values,
    attention_weights=attention_weights,
    confidence_scores=confidence_scores
)

# Register model version
auditor.register_model_version(
    model,
    training_data_hash="hash123",
    hyperparameters=hyperparameters,
    performance_metrics=performance_metrics
)

# Generate audit report
report = auditor.generate_audit_report()
```

## Uncertainty Calibration

Uncertainty calibration ensures that model confidence scores accurately reflect prediction reliability.

### Features

- Platt scaling for probability calibration
- Isotonic regression for non-parametric calibration
- Calibration quality validation with metrics
- Reliability diagram visualization

### Usage

```python
from src.ml.uncertainty_calibrator import create_uncertainty_calibrator

# Create calibrator
calibrator = create_uncertainty_calibrator(model)

# Calibrate using Platt scaling
calibrator.calibrate_uncertainty_platt(X_val, y_class_val, y_reg_val)

# Calibrate using isotonic regression
calibrator.calibrate_uncertainty_isotonic(X_val, y_class_val, y_reg_val)

# Validate calibration
metrics = calibrator.validate_calibration(X_test, y_class_test, y_reg_test)

# Visualize reliability
calibrator.visualize_reliability_diagram(X_test, y_class_test)
```

## API Reference

### SHAPExplainer

#### `create_shap_explainer(model, cache_size=100)`
Factory function to create SHAP explainer.

#### `compute_shap_values(background_data, test_data, explainer_type="kernel", cache_result=True)`
Compute SHAP values for test data.

#### `get_feature_importance(shap_result, feature_names=None)`
Extract feature importance from SHAP values.

#### `visualize_shap_summary(shap_values, feature_names=None, max_display=20, show=True, save_path=None)`
Create SHAP summary plot.

#### `visualize_shap_force(shap_values, expected_value, feature_names=None, matplotlib=True, show=True, save_path=None)`
Create SHAP force plot.

#### `get_attention_weights(input_data)`
Extract attention weights from model.

#### `explain_prediction(background_data, test_sample, feature_names=None)`
Provide complete explanation for single prediction.

### AttentionVisualizer

#### `create_attention_visualizer()`
Factory function to create attention visualizer.

#### `visualize_cnn_attention(attention_weights, feature_names=None, sequence_labels=None, title="CNN Attention Weights", save_path=None, show=True)`
Create heatmap visualization of CNN attention weights.

#### `visualize_lstm_attention(attention_weights, sequence_labels=None, title="LSTM Attention Weights", save_path=None, show=True)`
Create heatmap visualization of LSTM attention weights.

#### `visualize_cross_attention(cross_attention_weights, row_labels=None, col_labels=None, title="Cross-Attention Weights", save_path=None, show=True)`
Create heatmap visualization of cross-attention weights.

### FeatureImportanceAnalyzer

#### `create_feature_importance_analyzer(model)`
Factory function to create feature importance analyzer.

#### `compute_permutation_importance(X, y_class, y_reg, scoring="accuracy", n_repeats=10, random_state=None)`
Compute permutation importance for features.

#### `compute_integrated_gradients(input_data, baseline=None, steps=50, method="riemann_trapezoid", return_convergence_delta=False)`
Compute integrated gradients using Captum.

#### `compute_integrated_gradients_with_noise(input_data, baseline=None, steps=50, nt_samples=10, stdevs=0.01, return_convergence_delta=False)`
Compute integrated gradients with noise tunnel for smoothing.

### DecisionAuditor

#### `create_decision_auditor(audit_log_path="audit_trail.json")`
Factory function to create decision auditor.

#### `log_decision(model, input_data, prediction, shap_values=None, attention_weights=None, confidence_scores=None, feature_importance=None, ensemble_weights=None, metadata=None)`
Log a decision with complete audit trail.

#### `register_model_version(model, training_data_hash=None, hyperparameters=None, performance_metrics=None)`
Register a model version for tracking.

#### `get_decision_history(model_version=None, start_time=None, end_time=None)`
Get decision history with filtering.

#### `generate_audit_report(start_time=None, end_time=None, output_path=None)`
Generate comprehensive audit report.

### UncertaintyCalibrator

#### `create_uncertainty_calibrator(model)`
Factory function to create uncertainty calibrator.

#### `calibrate_uncertainty_platt(X_val, y_class_val, y_reg_val)`
Calibrate uncertainty using Platt scaling.

#### `calibrate_uncertainty_isotonic(X_val, y_class_val, y_reg_val)`
Calibrate uncertainty using isotonic regression.

#### `apply_calibration(predictions)`
Apply calibration to model predictions.

#### `validate_calibration(X_test, y_class_test, y_reg_test, n_bins=10)`
Validate calibration quality using various metrics.

#### `visualize_reliability_diagram(X_test, y_class_test, n_bins=10, title="Reliability Diagram", save_path=None, show=True)`
Create reliability diagram visualization.

## Usage Examples

### Complete Prediction Explanation

```python
# Create all interpretability components
explainer = create_shap_explainer(model)
visualizer = create_attention_visualizer()
analyzer = create_feature_importance_analyzer(model)
auditor = create_decision_auditor()
calibrator = create_uncertainty_calibrator(model)

# Get prediction with uncertainty
prediction = model.predict(test_data, return_uncertainty=True)

# Explain prediction
explanation = explainer.explain_prediction(
    background_data, 
    test_data, 
    feature_names=feature_names
)

# Log decision for audit trail
auditor.log_decision(
    model, test_data, prediction,
    shap_values=explanation['shap_values']['shap_values'],
    attention_weights=explanation['attention_weights'],
    confidence_scores={'classification': 0.95, 'regression': 0.87}
)

# Visualize attention weights
attention_weights = explanation['attention_weights']
if 'cnn_attention' in attention_weights:
    visualizer.visualize_cnn_attention(
        attention_weights['cnn_attention'],
        feature_names=feature_names
    )

# Compute permutation importance
importance = analyzer.compute_permutation_importance(
    X_val, y_class_val, y_reg_val,
    scoring="accuracy"
)

# Calibrate uncertainty
calibrator.calibrate_uncertainty_platt(X_val, y_class_val, y_reg_val)
calibrated_prediction = calibrator.apply_calibration(prediction)

# Validate calibration
metrics = calibrator.validate_calibration(X_test, y_class_test, y_reg_test)
print(f"Brier score improvement: {metrics['brier_score_improvement']}")
```

### Model Monitoring and Compliance

```python
# Generate compliance report
report = auditor.generate_audit_report(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Save audit report
report_path = f"audit_report_{datetime.now().strftime('%Y%m%d')}.json"
auditor.generate_audit_report(output_path=report_path)

# Check model version usage
model_usage = report['model_usage']
for version, count in model_usage.items():
    print(f"Model {version}: {count} decisions")
```

## Testing

The interpretability modules include comprehensive test suites to ensure functionality and reliability.

### Running Tests

```bash
# Run interpretability tests
python -m src.ml.test_interpretability
```

### Test Coverage

- SHAP explainer functionality
- Attention visualization functions
- Feature importance analysis
- Integrated gradients computation
- Decision audit trail logging
- Uncertainty calibration
- Error handling and edge cases

### Continuous Integration

Tests are integrated into the CI/CD pipeline to ensure that all interpretability features work correctly with each code change.