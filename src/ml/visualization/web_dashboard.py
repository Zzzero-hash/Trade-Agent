"""
Web-based Real-time Training Dashboard
Provides a browser-based interface for monitoring training progress
"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class WebDashboard:
    """Web-based dashboard using simple HTML/JavaScript"""
    
    def __init__(self, progress_file: str, port: int = 8080):
        self.progress_file = Path(progress_file)
        self.port = port
        self.running = False
        self.server_thread = None
        
        # Data storage
        self.current_data = {}
        self.trial_history = []
        
    def start(self):
        """Start the web dashboard"""
        self.running = True
        
        # Start data monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_data, daemon=True)
        monitor_thread.start()
        
        # Start web server
        self._start_web_server()
        
    def stop(self):
        """Stop the web dashboard"""
        self.running = False
        
    def _monitor_data(self):
        """Monitor progress data in background"""
        last_update = None
        
        while self.running:
            try:
                if self.progress_file.exists():
                    with open(self.progress_file, 'r') as f:
                        data = json.load(f)
                    
                    current_update = data.get('last_update')
                    if current_update != last_update:
                        self.current_data = data
                        self._update_history(data)
                        last_update = current_update
                
                time.sleep(2)
                
            except Exception as e:
                logger.debug(f"Data monitoring error: {e}")
                time.sleep(5)
                
    def _update_history(self, data: Dict):
        """Update trial history"""
        trial_num = data.get('trial_number')
        status = data.get('status')
        
        if status == 'completed' and trial_num is not None:
            final_metrics = data.get('final_metrics', {})
            
            # Check if already recorded
            if not any(t.get('trial') == trial_num for t in self.trial_history):
                self.trial_history.append({
                    'trial': trial_num,
                    'accuracy': final_metrics.get('accuracy', 0),
                    'training_time': final_metrics.get('training_time', 0),
                    'completed_at': datetime.now().isoformat()
                })
                
    def _start_web_server(self):
        """Start simple web server"""
        try:
            import http.server
            import socketserver
            from urllib.parse import urlparse, parse_qs
            
            class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, dashboard=None, **kwargs):
                    self.dashboard = dashboard
                    super().__init__(*args, **kwargs)
                    
                def do_GET(self):
                    if self.path == '/':
                        self._serve_dashboard()
                    elif self.path == '/api/data':
                        self._serve_data()
                    elif self.path == '/api/history':
                        self._serve_history()
                    else:
                        super().do_GET()
                        
                def _serve_dashboard(self):
                    html = self.dashboard._generate_html()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode())
                    
                def _serve_data(self):
                    data = json.dumps(self.dashboard.current_data)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data.encode())
                    
                def _serve_history(self):
                    data = json.dumps(self.dashboard.trial_history)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data.encode())
            
            # Create handler with dashboard reference
            handler = lambda *args, **kwargs: DashboardHandler(*args, dashboard=self, **kwargs)
            
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                print(f"🌐 Web dashboard running at http://localhost:{self.port}")
                print("Press Ctrl+C to stop")
                httpd.serve_forever()
                
        except Exception as e:
            logger.error(f"Web server failed: {e}")
            print(f"❌ Could not start web server on port {self.port}")
            
    def _generate_html(self) -> str:
        """Generate HTML dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 24px; font-weight: bold; color: #3498db; }
        .label { font-size: 14px; color: #7f8c8d; margin-bottom: 5px; }
        .status { padding: 5px 10px; border-radius: 4px; font-weight: bold; }
        .status.training { background: #f39c12; color: white; }
        .status.completed { background: #27ae60; color: white; }
        .status.failed { background: #e74c3c; color: white; }
        .progress-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #3498db; transition: width 0.3s; }
        .history-table { width: 100%; border-collapse: collapse; }
        .history-table th, .history-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .history-table th { background: #34495e; color: white; }
        .best-trial { background: #d5f4e6; }
        #chart { width: 100%; height: 300px; border: 1px solid #ddd; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Real-Time Training Dashboard</h1>
            <p>Live monitoring of hyperparameter optimization progress</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="label">Current Trial</div>
                <div class="metric" id="trial-number">-</div>
                <div class="status" id="trial-status">Unknown</div>
            </div>
            
            <div class="card">
                <div class="label">Current Epoch</div>
                <div class="metric" id="current-epoch">-</div>
                <div id="epoch-progress"></div>
            </div>
            
            <div class="card">
                <div class="label">Training Time</div>
                <div class="metric" id="training-time">-</div>
            </div>
            
            <div class="card">
                <div class="label">Best Accuracy</div>
                <div class="metric" id="best-accuracy">-</div>
            </div>
        </div>
        
        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h3>Current Metrics</h3>
                <div id="current-metrics">No data available</div>
            </div>
            
            <div class="card">
                <h3>Progress Chart</h3>
                <canvas id="chart"></canvas>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <h3>Trial History</h3>
            <table class="history-table" id="history-table">
                <thead>
                    <tr>
                        <th>Trial</th>
                        <th>Accuracy</th>
                        <th>Time (min)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="history-body">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let chart;
        let chartData = {
            labels: [],
            datasets: [{
                label: 'Accuracy',
                data: [],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.1
            }]
        };

        function initChart() {
            const ctx = document.getElementById('chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }

        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    // Update current trial info
                    document.getElementById('trial-number').textContent = data.trial_number || '-';
                    
                    const statusEl = document.getElementById('trial-status');
                    const status = data.status || 'unknown';
                    statusEl.textContent = status.toUpperCase();
                    statusEl.className = 'status ' + status;
                    
                    document.getElementById('current-epoch').textContent = data.current_epoch || '-';
                    
                    const elapsed = data.elapsed_time_seconds || 0;
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = Math.floor(elapsed % 60);
                    document.getElementById('training-time').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    
                    // Update metrics
                    const metricsEl = document.getElementById('current-metrics');
                    if (data.last_metrics) {
                        let metricsHtml = '';
                        for (const [key, value] of Object.entries(data.last_metrics)) {
                            const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                            metricsHtml += `<div><strong>${key}:</strong> ${displayValue}</div>`;
                        }
                        metricsEl.innerHTML = metricsHtml;
                    } else {
                        metricsEl.innerHTML = 'No metrics available';
                    }
                })
                .catch(error => console.error('Error fetching data:', error));
            
            // Update history
            fetch('/api/history')
                .then(response => response.json())
                .then(history => {
                    updateHistory(history);
                    updateChart(history);
                })
                .catch(error => console.error('Error fetching history:', error));
        }

        function updateHistory(history) {
            const tbody = document.getElementById('history-body');
            const bestAccuracy = Math.max(...history.map(t => t.accuracy || 0));
            
            // Update best accuracy display
            document.getElementById('best-accuracy').textContent = bestAccuracy.toFixed(4);
            
            tbody.innerHTML = '';
            history.slice(-10).forEach(trial => {
                const row = tbody.insertRow();
                const isBest = trial.accuracy === bestAccuracy;
                if (isBest) row.className = 'best-trial';
                
                row.insertCell(0).textContent = trial.trial;
                row.insertCell(1).textContent = trial.accuracy.toFixed(4);
                row.insertCell(2).textContent = (trial.training_time / 60).toFixed(1);
                row.insertCell(3).textContent = isBest ? '🏆 BEST' : '✅ Done';
            });
        }

        function updateChart(history) {
            chartData.labels = history.map(t => `Trial ${t.trial}`);
            chartData.datasets[0].data = history.map(t => t.accuracy);
            chart.update();
        }

        // Initialize
        initChart();
        updateDashboard();
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
        """


def create_web_dashboard(progress_file: str, port: int = 8080) -> WebDashboard:
    """Create a web-based dashboard"""
    return WebDashboard(progress_file, port)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Default progress file locations
    progress_files = [
        "experiments/results/hyperopt_task_5_5/training_progress/progress.json",
        "hyperopt_results_task_5_5_fixed/training_progress/progress.json",
        "test_all_fixes_results/training_progress/progress.json",
        "test_monitoring_results/training_progress/progress.json",
        "training_progress/progress.json"
    ]
    
    # Use command line argument or find existing file
    if len(sys.argv) > 1:
        progress_file = sys.argv[1]
    else:
        progress_file = None
        for pf in progress_files:
            if Path(pf).exists():
                progress_file = pf
                break
                
        if not progress_file:
            print("❌ No progress file found. Usage:")
            print("python web_dashboard.py [progress_file.json]")
            sys.exit(1)
    
    # Start dashboard
    dashboard = create_web_dashboard(progress_file)
    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\n👋 Web dashboard stopped")
        dashboard.stop()