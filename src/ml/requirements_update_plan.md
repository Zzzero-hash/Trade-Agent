# Requirements Update Plan

## Dependencies to Add

Add the following dependencies to requirements.txt:

```
# Model Interpretability and Explainability
shap>=0.42.0
matplotlib>=3.5.0
seaborn>=0.11.0
captum>=0.6.0
```

## Reasoning for Each Dependency

### SHAP (shap>=0.42.0)
- Provides state-of-the-art model interpretability using SHapley Additive exPlanations
- Supports multiple explanation methods (KernelSHAP, DeepSHAP, TreeSHAP)
- Integrates well with PyTorch models
- Required for requirement 12.4: "WHEN model explanations are requested THEN the system SHALL use attention mechanisms to highlight influential data points"

### Matplotlib (matplotlib>=3.5.0)
- Standard Python plotting library
- Required for creating visualizations of attention weights, feature importance, and SHAP values
- Used by SHAP library for its built-in visualization functions

### Seaborn (seaborn>=0.11.0)
- Statistical data visualization library built on matplotlib
- Provides more attractive and informative statistical graphics
- Useful for creating heatmaps of attention weights and correlation matrices

### Captum (captum>=0.6.0)
- Model interpretability library for PyTorch
- Provides implementations of integrated gradients and other attribution methods
- Specifically designed for deep learning models like our CNN+LSTM hybrid
- Required for implementing integrated gradients feature importance analysis

## Installation Commands

```bash
# Install new dependencies
pip install shap>=0.42.0 matplotlib>=3.5.0 seaborn>=0.11.0 captum>=0.6.0

# Or update requirements.txt and install all dependencies
pip install -r requirements.txt
```

## Docker Update

The Dockerfile will automatically pick up these new dependencies when the updated requirements.txt is used in the build process.

## Version Compatibility

These versions are chosen to ensure:
1. Compatibility with existing PyTorch 1.9.0+ installation
2. Access to latest features and bug fixes
3. Stability in production environments
4. Compatibility with Python 3.11 (as specified in Dockerfile)