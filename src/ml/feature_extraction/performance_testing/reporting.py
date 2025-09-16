"""Performance reporting for feature extraction tests.

This module provides reporting capabilities for feature extraction performance
tests, including HTML and JSON report generation.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
import numpy as np

from .metrics import PerformanceReport

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Base class for performance reporting."""
    
    def __init__(self):
        """Initialize performance reporter."""
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self,
        reports: List[PerformanceReport],
        output_path: Optional[str] = None
    ) -> str:
        """Generate performance report.
        
        Args:
            reports: List of performance reports
            output_path: Path to save report (optional)
            
        Returns:
            Generated report as string
        """
        raise NotImplementedError("Subclasses must implement generate_report method")


class JSONReportGenerator(PerformanceReporter):
    """JSON report generator for performance tests."""
    
    def generate_report(
        self,
        reports: List[PerformanceReport],
        output_path: Optional[str] = None
    ) -> str:
        """Generate JSON performance report.
        
        Args:
            reports: List of performance reports
            output_path: Path to save report (optional)
            
        Returns:
            JSON report as string
        """
        # Convert reports to serializable format
        report_data = []
        for report in reports:
            report_dict = asdict(report)
            # Convert datetime to string
            report_dict['timestamp'] = report_dict['timestamp'].isoformat()
            report_data.append(report_dict)
        
        # Create summary statistics
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_tests': len(reports),
            'tests_meeting_requirements': sum(1 for r in reports if r.meets_latency_requirement and r.meets_throughput_requirement),
            'average_success_rate': float(np.mean([r.success_rate for r in reports])) if reports else 0.0,
            'average_latency_ms': float(np.mean([r.latency_stats.avg_ms for r in reports])) if reports else 0.0,
            'average_throughput_rps': float(np.mean([r.throughput_rps for r in reports])) if reports else 0.0
        }
        
        # Create complete report
        full_report = {
            'summary': summary,
            'reports': report_data
        }
        
        # Convert to JSON
        json_report = json.dumps(full_report, indent=2)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(json_report)
                self.logger.info(f"JSON report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save JSON report: {e}")
        
        return json_report


class HTMLReportGenerator(PerformanceReporter):
    """HTML report generator for performance tests."""
    
    def generate_report(
        self,
        reports: List[PerformanceReport],
        output_path: Optional[str] = None
    ) -> str:
        """Generate HTML performance report.
        
        Args:
            reports: List of performance reports
            output_path: Path to save report (optional)
            
        Returns:
            HTML report as string
        """
        # Generate HTML content
        html_content = self._generate_html_content(reports)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                self.logger.info(f"HTML report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save HTML report: {e}")
        
        return html_content
    
    def _generate_html_content(self, reports: List[PerformanceReport]) -> str:
        """Generate HTML content from reports.
        
        Args:
            reports: List of performance reports
            
        Returns:
            HTML content as string
        """
        # Calculate summary statistics
        total_tests = len(reports)
        tests_meeting_requirements = sum(1 for r in reports if r.meets_latency_requirement and r.meets_throughput_requirement)
        avg_success_rate = float(np.mean([r.success_rate for r in reports])) if reports else 0.0
        avg_latency = float(np.mean([r.latency_stats.avg_ms for r in reports])) if reports else 0.0
        avg_throughput = float(np.mean([r.throughput_rps for r in reports])) if reports else 0.0
        
        # Generate test table rows
        test_rows = ""
        for i, report in enumerate(reports, 1):
            meets_req = "✅" if (report.meets_latency_requirement and report.meets_throughput_requirement) else "❌"
            test_rows += f"""
                <tr>
                    <td>{i}</td>
                    <td>{report.test_name}</td>
                    <td>{report.test_type}</td>
                    <td>{report.total_requests}</td>
                    <td>{report.success_rate:.2%}</td>
                    <td>{report.latency_stats.avg_ms:.2f}</td>
                    <td>{report.latency_stats.p95_ms:.2f}</td>
                    <td>{report.throughput_rps:.2f}</td>
                    <td>{report.resource_stats.peak_memory_mb:.1f}</td>
                    <td>{meets_req}</td>
                </tr>
            """
        
        # Create HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Extraction Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #33;
            text-align: center;
        }}
        .summary {{
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pass {{
            color: green;
        }}
        .fail {{
            color: red;
        }}
        .timestamp {{
            text-align: right;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Feature Extraction Performance Report</h1>
        <div class="timestamp">Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div>Total Tests</div>
                    <div class="summary-value">{total_tests}</div>
                </div>
                <div class="summary-item">
                    <div>Tests Meeting Requirements</div>
                    <div class="summary-value">{tests_meeting_requirements}</div>
                </div>
                <div class="summary-item">
                    <div>Average Success Rate</div>
                    <div class="summary-value">{avg_success_rate:.1%}</div>
                </div>
                <div class="summary-item">
                    <div>Average Latency (ms)</div>
                    <div class="summary-value">{avg_latency:.2f}</div>
                </div>
                <div class="summary-item">
                    <div>Average Throughput (RPS)</div>
                    <div class="summary-value">{avg_throughput:.2f}</div>
                </div>
            </div>
        </div>
        
        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Requests</th>
                    <th>Success Rate</th>
                    <th>Avg Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>Throughput (RPS)</th>
                    <th>Peak Memory (MB)</th>
                    <th>Meets Requirements</th>
                </tr>
            </thead>
            <tbody>
                {test_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        return html_template.strip()


def generate_comparison_report(
    reports: List[PerformanceReport],
    output_paths: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Generate comparison report in multiple formats.
    
    Args:
        reports: List of performance reports
        output_paths: Dictionary mapping format to output path (optional)
        
    Returns:
        Dictionary mapping format to generated report content
    """
    reports_dict = {}
    
    # Generate JSON report
    json_generator = JSONReportGenerator()
    json_report = json_generator.generate_report(reports)
    reports_dict['json'] = json_report
    
    if output_paths and 'json' in output_paths:
        json_generator.generate_report(reports, output_paths['json'])
    
    # Generate HTML report
    html_generator = HTMLReportGenerator()
    html_report = html_generator.generate_report(reports)
    reports_dict['html'] = html_report
    
    if output_paths and 'html' in output_paths:
        html_generator.generate_report(reports, output_paths['html'])
    
    return reports_dict