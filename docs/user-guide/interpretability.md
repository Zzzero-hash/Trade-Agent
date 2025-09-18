# AI Model Interpretability and Explainability Guide

This guide helps you understand how the AI Trading Platform's machine learning models make decisions and how to interpret their outputs.

## Overview

The AI Trading Platform provides comprehensive interpretability tools to help you understand:

- **Why** a trading signal was generated
- **Which features** influenced the decision most
- **How confident** the model is in its predictions
- **What patterns** the model learned from historical data

## Key Concepts

### Interpretability vs. Explainability

- **Interpretability**: Understanding how a model works internally
- **Explainability**: Understanding why a model made a specific decision

### Types of Explanations

1. **Global Explanations**: How the model behaves overall
2. **Local Explanations**: Why the model made a specific prediction
3. **Feature Importance**: Which inputs matter most
4. **Attention Visualization**: What the model focuses on

## Available Explanation Methods

### 1. SHAP (SHapley Additive exPlanations)

SHAP provides the most comprehensive explanations for individual predictions.

#### Accessing SHAP Explanations

```python
from src.ml.shap_explainer import SHAPExplainer

# Initialize explainer with your trained model
explainer = SHAPExplainer(model_path="checkpoints/best_model.pth")

# Get explanation for a specific prediction
explanation = explainer.explain_prediction(
    symbol="AAPL",
    timestamp="2024-01-15T10:30:00Z",
    return_format="detailed"
)

print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.2%}")
print("\nTop contributing features:")
for feature, impact in explanation['feature_impacts'][:5]:
    print(f"  {feature}: {impact:+.3f}")
```

#### Via API

```bash
# Get SHAP explanation for a signal
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/signals/SIGNAL_ID/explanation"
```

#### Understanding SHAP Values

- **Positive values**: Push prediction toward "Buy"
- **Negative values**: Push prediction toward "Sell"
- **Magnitude**: Strength of the feature's influence

**Example SHAP Output:**
```json
{
  "prediction": "buy",
  "confidence": 0.78,
  "base_value": 0.02,
  "shap_values": {
    "rsi_14": 0.15,           // Strong buy signal
    "macd_signal": 0.08,      // Moderate buy signal
    "volume_ratio": -0.03,    // Slight sell signal
    "price_momentum": 0.12,   // Strong buy signal
    "volatility": -0.05       // Moderate sell signal
  },
  "feature_importance_rank": [
    "rsi_14", "price_momentum", "macd_signal", "volatility", "volume_ratio"
  ]
}
```

### 2. Attention Visualization

For CNN+LSTM models, attention mechanisms show what time periods and features the model focuses on.

#### Accessing Attention Maps

```python
from src.ml.attention_visualizer import AttentionVisualizer

visualizer = AttentionVisualizer(model_path="checkpoints/hybrid_model.pth")

# Generate attention visualization
attention_data = visualizer.visualize_attention(
    symbol="AAPL",
    sequence_length=60,  # Last 60 time steps
    save_plot=True,
    plot_path="attention_AAPL.png"
)

# Access attention weights
temporal_attention = attention_data['temporal_attention']  # Time focus
feature_attention = attention_data['feature_attention']    # Feature focus
```

#### Interpreting Attention Maps

- **Temporal Attention**: Shows which time periods (recent vs. distant past) the model considers most important
- **Feature Attention**: Shows which technical indicators or price features the model focuses on
- **Cross-Attention**: Shows how CNN spatial features interact with LSTM temporal features

### 3. Feature Importance Analysis

Global feature importance shows which features matter most across all predictions.

#### Permutation Importance

```python
from src.ml.feature_importance_analyzer import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model_path="checkpoints/best_model.pth")

# Calculate permutation importance
importance_scores = analyzer.permutation_importance(
    test_data=test_dataset,
    n_repeats=10,
    scoring_metric="sharpe_ratio"
)

# Display results
for feature, score in importance_scores.items():
    print(f"{feature}: {score:.4f} ± {score.std():.4f}")
```

#### Integrated Gradients

```python
# Calculate integrated gradients
ig_scores = analyzer.integrated_gradients(
    baseline="zero",  # or "random", "mean"
    n_steps=50
)

# Visualize feature importance
analyzer.plot_feature_importance(
    importance_scores=ig_scores,
    save_path="feature_importance.png"
)
```

### 4. Uncertainty Quantification

Understanding model confidence helps assess prediction reliability.

#### Monte Carlo Dropout

```python
from src.ml.uncertainty_calibrator import UncertaintyCalibrator

calibrator = UncertaintyCalibrator(model_path="checkpoints/hybrid_model.pth")

# Get prediction with uncertainty
result = calibrator.predict_with_uncertainty(
    symbol="AAPL",
    n_samples=100,  # Number of MC dropout samples
    return_distribution=True
)

print(f"Prediction: {result['mean_prediction']}")
print(f"Uncertainty: {result['uncertainty']:.3f}")
print(f"Confidence Interval: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

#### Calibration Assessment

```python
# Check if model confidence is well-calibrated
calibration_results = calibrator.assess_calibration(
    test_data=test_dataset,
    n_bins=10
)

print(f"Expected Calibration Error: {calibration_results['ece']:.3f}")
print(f"Reliability Diagram saved to: {calibration_results['plot_path']}")
```

## Practical Usage Examples

### Example 1: Analyzing a Buy Signal

```python
# Get a recent buy signal
signal_id = "abc123-def456-ghi789"

# Fetch detailed explanation
explanation = explainer.explain_signal(signal_id)

print(f"Signal: {explanation['signal_type']} {explanation['symbol']}")
print(f"Confidence: {explanation['confidence']:.1%}")
print(f"Price Target: ${explanation['price_target']:.2f}")

# Top positive factors (supporting the buy decision)
print("\n🟢 Factors supporting BUY:")
for factor in explanation['positive_factors']:
    print(f"  • {factor['name']}: {factor['impact']:+.3f} ({factor['description']})")

# Top negative factors (opposing the buy decision)
print("\n🔴 Factors opposing BUY:")
for factor in explanation['negative_factors']:
    print(f"  • {factor['name']}: {factor['impact']:+.3f} ({factor['description']})")

# Model attention focus
print(f"\n🎯 Model focused on: {explanation['attention_summary']}")
print(f"📊 Key time period: {explanation['key_timeframe']}")
```

**Sample Output:**
```
Signal: BUY AAPL
Confidence: 78.5%
Price Target: $152.30

🟢 Factors supporting BUY:
  • RSI(14): +0.152 (Oversold condition, potential reversal)
  • MACD Signal: +0.089 (Bullish crossover detected)
  • Price Momentum: +0.124 (Strong upward momentum)
  • Volume Profile: +0.067 (Above-average buying volume)

🔴 Factors opposing BUY:
  • Market Volatility: -0.045 (High volatility increases risk)
  • Sector Correlation: -0.023 (Tech sector showing weakness)

🎯 Model focused on: Recent price action and momentum indicators
📊 Key time period: Last 5 trading days (high attention weight)
```

### Example 2: Understanding Model Uncertainty

```python
# Analyze uncertainty for different market conditions
symbols = ["AAPL", "TSLA", "SPY", "BTC-USD"]

for symbol in symbols:
    prediction = calibrator.predict_with_uncertainty(symbol, n_samples=100)
    
    print(f"\n{symbol}:")
    print(f"  Prediction: {prediction['signal']} (confidence: {prediction['confidence']:.1%})")
    print(f"  Uncertainty: {prediction['uncertainty']:.3f}")
    
    # Interpret uncertainty level
    if prediction['uncertainty'] < 0.1:
        uncertainty_level = "Low (High confidence)"
    elif prediction['uncertainty'] < 0.2:
        uncertainty_level = "Medium (Moderate confidence)"
    else:
        uncertainty_level = "High (Low confidence)"
    
    print(f"  Uncertainty Level: {uncertainty_level}")
    
    # Risk assessment based on uncertainty
    if prediction['uncertainty'] > 0.25:
        print(f"  ⚠️  Warning: High uncertainty - consider reducing position size")
```

### Example 3: Comparing Model Decisions

```python
# Compare explanations for similar stocks
stocks = ["AAPL", "MSFT", "GOOGL"]
explanations = {}

for stock in stocks:
    explanations[stock] = explainer.explain_prediction(stock)

# Find common patterns
common_factors = explainer.find_common_factors(explanations)
print("Common decision factors across tech stocks:")
for factor, avg_impact in common_factors.items():
    print(f"  {factor}: {avg_impact:+.3f}")

# Identify unique factors
for stock in stocks:
    unique_factors = explainer.find_unique_factors(
        explanations[stock], 
        reference_explanations=[explanations[s] for s in stocks if s != stock]
    )
    print(f"\nUnique factors for {stock}:")
    for factor, impact in unique_factors.items():
        print(f"  {factor}: {impact:+.3f}")
```

## Dashboard Integration

### Real-Time Explanations

The web dashboard provides interactive explanations for all signals:

1. **Signal Card**: Click on any signal to see its explanation
2. **Feature Impact Chart**: Visual representation of SHAP values
3. **Attention Heatmap**: Shows temporal and feature attention
4. **Confidence Meter**: Displays prediction uncertainty

### Explanation Widgets

#### SHAP Waterfall Chart
Shows how each feature contributes to the final prediction:

```javascript
// Frontend component for SHAP visualization
import { SHAPWaterfallChart } from '@/components/explanations';

<SHAPWaterfallChart
  signalId="abc123"
  showTop={10}
  interactive={true}
  onFeatureClick={handleFeatureClick}
/>
```

#### Attention Heatmap
Visualizes model attention across time and features:

```javascript
import { AttentionHeatmap } from '@/components/explanations';

<AttentionHeatmap
  modelType="cnn_lstm"
  symbol="AAPL"
  timeRange="1h"
  showFeatureNames={true}
/>
```

#### Uncertainty Gauge
Displays prediction confidence:

```javascript
import { UncertaintyGauge } from '@/components/explanations';

<UncertaintyGauge
  confidence={0.785}
  uncertainty={0.142}
  showCalibration={true}
  threshold={0.7}
/>
```

## Best Practices

### 1. Interpreting SHAP Values

**Do:**
- Focus on the magnitude and direction of SHAP values
- Consider the base value (model's average prediction)
- Look for consistent patterns across similar predictions
- Use SHAP values to validate your domain knowledge

**Don't:**
- Treat SHAP values as absolute truth
- Ignore the base value when interpreting individual features
- Over-interpret small SHAP values (noise)
- Use SHAP values for causal inference

### 2. Using Attention Visualizations

**Do:**
- Look for attention patterns that align with market events
- Check if attention focuses on relevant time periods
- Validate attention against known market dynamics
- Use attention to debug model behavior

**Don't:**
- Assume high attention always means high importance
- Ignore the overall attention distribution
- Over-interpret attention on individual time steps
- Use attention as the sole explanation method

### 3. Assessing Model Uncertainty

**Do:**
- Use uncertainty to adjust position sizes
- Be more cautious with high-uncertainty predictions
- Monitor uncertainty trends over time
- Combine uncertainty with other risk metrics

**Don't:**
- Ignore high-uncertainty signals completely
- Assume low uncertainty guarantees success
- Use uncertainty as the only decision factor
- Forget to recalibrate uncertainty periodically

### 4. Combining Multiple Explanation Methods

**Do:**
- Use SHAP for feature-level explanations
- Use attention for temporal pattern analysis
- Use uncertainty for risk assessment
- Cross-validate explanations across methods

**Don't:**
- Rely on a single explanation method
- Ignore contradictions between methods
- Over-complicate the interpretation process
- Forget the business context

## Advanced Features

### Custom Explanation Pipelines

```python
from src.ml.explanation_pipeline import ExplanationPipeline

# Create custom explanation pipeline
pipeline = ExplanationPipeline([
    'shap_values',
    'attention_weights', 
    'uncertainty_quantification',
    'feature_importance',
    'counterfactual_analysis'
])

# Generate comprehensive explanation
comprehensive_explanation = pipeline.explain(
    model=model,
    input_data=sample_data,
    explanation_depth='detailed'
)
```

### Counterfactual Analysis

```python
from src.ml.counterfactual_explainer import CounterfactualExplainer

cf_explainer = CounterfactualExplainer(model)

# Find minimal changes needed to flip prediction
counterfactuals = cf_explainer.generate_counterfactuals(
    original_input=sample_data,
    target_class='sell',  # What would make this a sell signal?
    max_changes=3
)

print("To change this BUY signal to SELL, you would need:")
for change in counterfactuals['minimal_changes']:
    print(f"  • {change['feature']}: {change['original']:.3f} → {change['required']:.3f}")
```

### Explanation Auditing

```python
from src.ml.explanation_auditor import ExplanationAuditor

auditor = ExplanationAuditor()

# Audit explanation quality
audit_results = auditor.audit_explanations(
    model=model,
    test_data=test_dataset,
    explanation_methods=['shap', 'attention', 'uncertainty']
)

print(f"Explanation Consistency Score: {audit_results['consistency_score']:.3f}")
print(f"Explanation Stability Score: {audit_results['stability_score']:.3f}")
print(f"Human Agreement Score: {audit_results['human_agreement']:.3f}")
```

## Troubleshooting Explanations

### Common Issues

#### Issue: SHAP Values Don't Make Sense

**Possible Causes:**
- Model overfitting to noise
- Feature preprocessing issues
- Incorrect baseline selection

**Solutions:**
```python
# Check feature preprocessing
print("Feature statistics:")
print(features.describe())

# Try different baselines
for baseline in ['zero', 'mean', 'median']:
    shap_values = explainer.explain_prediction(
        sample_data, 
        baseline=baseline
    )
    print(f"Baseline {baseline}: {shap_values.sum():.3f}")

# Validate with simpler model
from sklearn.linear_model import LogisticRegression
simple_model = LogisticRegression()
simple_model.fit(X_train, y_train)
print("Linear model coefficients:", simple_model.coef_)
```

#### Issue: Attention Weights Are Uniform

**Possible Causes:**
- Model not learning meaningful patterns
- Insufficient training data
- Attention mechanism not properly configured

**Solutions:**
```python
# Check attention layer configuration
print("Attention layer parameters:")
for name, param in model.attention.named_parameters():
    print(f"{name}: {param.shape}")

# Visualize attention during training
attention_history = trainer.get_attention_history()
plot_attention_evolution(attention_history)

# Try different attention mechanisms
model.attention = MultiHeadAttention(
    embed_dim=256,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
```

#### Issue: High Uncertainty for All Predictions

**Possible Causes:**
- Model not well-calibrated
- Insufficient training data
- High inherent market uncertainty

**Solutions:**
```python
# Recalibrate uncertainty estimates
calibrator.recalibrate(
    validation_data=val_dataset,
    method='temperature_scaling'
)

# Check calibration curve
calibration_plot = calibrator.plot_calibration_curve(test_dataset)

# Adjust uncertainty threshold
optimal_threshold = calibrator.find_optimal_threshold(
    validation_data=val_dataset,
    metric='expected_calibration_error'
)
```

## Integration with Trading Decisions

### Risk-Adjusted Position Sizing

```python
def calculate_position_size(signal, explanation, base_size=0.05):
    """Adjust position size based on explanation confidence"""
    
    # Base position size
    position_size = base_size
    
    # Adjust for SHAP value magnitude
    shap_magnitude = sum(abs(v) for v in explanation['shap_values'].values())
    confidence_multiplier = min(shap_magnitude / 0.5, 2.0)  # Cap at 2x
    
    # Adjust for uncertainty
    uncertainty_penalty = 1 - explanation['uncertainty']
    
    # Adjust for attention consistency
    attention_bonus = 1 + (explanation['attention_consistency'] - 0.5)
    
    # Final position size
    final_size = position_size * confidence_multiplier * uncertainty_penalty * attention_bonus
    
    return max(0.01, min(final_size, 0.20))  # Between 1% and 20%

# Example usage
signal = get_latest_signal("AAPL")
explanation = explainer.explain_signal(signal['id'])
position_size = calculate_position_size(signal, explanation)

print(f"Recommended position size: {position_size:.1%}")
```

### Explanation-Based Alerts

```python
def check_explanation_quality(explanation, thresholds):
    """Generate alerts based on explanation quality"""
    
    alerts = []
    
    # Check for conflicting signals
    positive_impact = sum(v for v in explanation['shap_values'].values() if v > 0)
    negative_impact = sum(v for v in explanation['shap_values'].values() if v < 0)
    
    if abs(positive_impact + negative_impact) < thresholds['min_net_impact']:
        alerts.append({
            'type': 'conflicting_signals',
            'message': 'Positive and negative factors nearly cancel out',
            'severity': 'warning'
        })
    
    # Check for high uncertainty
    if explanation['uncertainty'] > thresholds['max_uncertainty']:
        alerts.append({
            'type': 'high_uncertainty',
            'message': f"High prediction uncertainty: {explanation['uncertainty']:.1%}",
            'severity': 'warning'
        })
    
    # Check for unusual attention patterns
    if explanation['attention_entropy'] > thresholds['max_attention_entropy']:
        alerts.append({
            'type': 'scattered_attention',
            'message': 'Model attention is scattered across many features',
            'severity': 'info'
        })
    
    return alerts

# Set up monitoring
thresholds = {
    'min_net_impact': 0.1,
    'max_uncertainty': 0.3,
    'max_attention_entropy': 2.5
}

for signal in recent_signals:
    explanation = explainer.explain_signal(signal['id'])
    alerts = check_explanation_quality(explanation, thresholds)
    
    if alerts:
        print(f"⚠️  Alerts for {signal['symbol']} signal:")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
```

## Next Steps

1. **Explore Interactive Dashboards**: Use the web interface to explore explanations visually
2. **Set Up Monitoring**: Configure alerts for explanation quality issues
3. **Customize Explanations**: Adapt explanation methods to your specific use cases
4. **Validate with Domain Knowledge**: Cross-check explanations with your market expertise
5. **Continuous Learning**: Use explanations to improve model training and feature engineering

For more advanced topics, see:
- [Model Training Guide](../ml/training-guide.md)
- [API Documentation](../api/README.md)
- [Risk Management Guide](./risk-management.md)