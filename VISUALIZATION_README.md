# 🎨 Real-Time Training Visualization System

A comprehensive visualization system for monitoring hyperparameter optimization and ML training progress in real-time.

## 🚀 Quick Start

### Option 1: Interactive Demo
```bash
python demo_visualization.py
```

### Option 2: Monitor Existing Training
```bash
# Start hyperparameter optimization
python test_monitoring.py

# In another terminal, start live viewer
python live_viewer_enhanced.py
```

### Option 3: Quick Test
```bash
python test_visualization.py
```

## 📊 Features

### 🖥️ Terminal Viewer
- **Real-time progress updates** - Live trial and epoch progress
- **Statistics dashboard** - Best accuracy, averages, trends
- **Recent trials table** - Last 5 trials with performance
- **Progress visualization** - ASCII charts and trend indicators
- **Lightweight** - Works in any terminal, no GUI required

### 🎨 GUI Dashboard
- **Multi-panel plots** - 6 different visualization panels
- **Live updating charts** - Real-time loss and accuracy curves
- **Trial progress tracking** - Best accuracy over time
- **Training time analysis** - Time per trial visualization
- **Convergence monitoring** - Optimization progress trends
- **Interactive plots** - Zoom, pan, and explore data

### 📈 Monitoring Features
- **Non-intrusive** - Doesn't affect training performance
- **Background monitoring** - Runs in separate thread
- **Error handling** - Graceful failure recovery
- **Multiple formats** - JSON progress files + hyperparameter logs
- **Auto-detection** - Finds progress files automatically

## 🛠️ Usage Examples

### Basic Monitoring
```bash
# Auto-detect progress file and use terminal viewer
python live_viewer_enhanced.py

# Specify progress file
python live_viewer_enhanced.py path/to/progress.json

# Use GUI viewer
python live_viewer_enhanced.py --gui

# Use terminal viewer with custom update interval
python live_viewer_enhanced.py --terminal --update-interval 1.0
```

### Demo and Testing
```bash
# Quick 30-second demo
python demo_visualization.py --quick

# Full demo with real training
python demo_visualization.py --full

# Test all components
python demo_visualization.py --test
```

### Integration with Training
```python
from src.ml.training_monitor import get_monitor

# In your training code
monitor = get_monitor("my_experiment/progress")
monitor.start_trial(trial_number, hyperparameters)

# During training loop
for epoch in range(num_epochs):
    # ... training code ...
    monitor.update_epoch(epoch, metrics)

# After training
monitor.finish_trial(trial_number, final_metrics)
```

## 📁 File Structure

```
src/ml/visualization/
├── __init__.py                 # Module exports
├── real_time_plotter.py        # GUI dashboard with matplotlib
└── terminal_viewer.py          # Terminal-based viewer

# Standalone scripts
├── live_viewer_enhanced.py     # Main viewer script
├── demo_visualization.py       # Interactive demo
└── test_visualization.py       # Test suite
```

## 🎯 Visualization Panels

### Terminal Viewer Displays:
1. **Current Trial Status** - Trial number, status, epoch progress
2. **Real-time Metrics** - Loss, accuracy, training time
3. **Optimization Statistics** - Best/average accuracy, trial count
4. **Recent Results Table** - Last 5 trials with performance
5. **Trend Analysis** - Improving/declining/stable indicators
6. **Accuracy Distribution** - Histogram of trial accuracies

### GUI Dashboard Panels:
1. **Trial Progress** - Accuracy over trials with best-so-far line
2. **Loss Curves** - Training loss for recent trials
3. **Accuracy Curves** - Training accuracy for recent trials  
4. **Training Time** - Time per trial with average line
5. **Hyperparameter Distribution** - Parameter value distributions
6. **Convergence Plot** - Optimization convergence with trend

## 🔧 Configuration

### Progress File Format
```json
{
  "trial_number": 42,
  "status": "training",
  "current_epoch": 25,
  "epochs_completed": 24,
  "elapsed_time_seconds": 67.3,
  "last_metrics": {
    "loss": 0.3456,
    "classification_accuracy": 0.8123,
    "val_loss": 0.3789,
    "val_accuracy": 0.7956
  },
  "last_update": "2025-01-23T10:30:45"
}
```

### Expected File Locations
The system automatically searches for progress files in:
- `experiments/results/*/training_progress/progress.json`
- `hyperopt_results_*/training_progress/progress.json`
- `test_*_results/training_progress/progress.json`
- `training_progress/progress.json`

## 🚨 Troubleshooting

### GUI Viewer Issues
```bash
# If GUI fails, use terminal viewer
python live_viewer_enhanced.py --terminal

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

### No Progress File Found
```bash
# Check if training is running
ls -la */training_progress/

# Create test data
python test_visualization.py
```

### Import Errors
```bash
# Install missing dependencies
pip install matplotlib seaborn

# Check Python path
python -c "import sys; print(sys.path)"
```

## 🎉 Examples in Action

### Terminal Viewer Output:
```
🚀 Live Training Monitor
======================================================================
📅 2025-01-23 10:30:45

🔢 Current Trial: 42
📊 Status: TRAINING
⏱️  Epoch: 25 (completed: 24)
⏰ Elapsed: 67.3s (1.1m)
📈 Current Metrics:
   loss: 0.3456
   classification_accuracy: 0.8123
   val_loss: 0.3789
   val_accuracy: 0.7956

📊 Optimization Statistics:
   Total Trials: 42
   Best Accuracy: 0.8934
   Average Accuracy: 0.7456
   Average Time: 2.3m
   Recent 5 Avg: 0.8123

🏆 Recent Trial Results:
   Trial | Accuracy | Time     | Status
   ------|----------|----------|--------
      38 | 0.7234   | 2.1m     | ✅ Done
      39 | 0.8456   | 1.8m     | ✅ Done
      40 | 0.8934   | 2.5m     | 🥇 BEST
      41 | 0.8123   | 2.0m     | ✅ Done
      42 | 0.8123   | 1.1m     | 🔄 Running

📊 Trend: 📈 Improving
```

### GUI Dashboard:
- Real-time updating plots with multiple panels
- Interactive charts you can zoom and explore
- Color-coded trial progress
- Trend lines and statistical overlays

## 🤝 Contributing

The visualization system is designed to be:
- **Modular** - Easy to add new viewers
- **Extensible** - Simple to add new metrics
- **Non-intrusive** - Doesn't affect training performance
- **Robust** - Handles errors gracefully

Feel free to extend with new visualization types or improve existing ones!