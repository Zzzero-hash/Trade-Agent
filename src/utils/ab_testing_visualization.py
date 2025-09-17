"""Visualization utilities for A/B testing experiments and results.

This module provides utilities for creating visualizations of A/B testing
experiments, including performance comparisons, statistical test results,
and rollout progress tracking.

Requirements: 6.2, 11.1
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.figure import Figure
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from src.ml.ray_serve.ab_testing import ABTestExperiment, StatisticalTestResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ABTestVisualizer:
    """Visualizer for A/B testing experiments and results."""

    def __init__(self, style: str = "whitegrid"):
        """Initialize the visualizer.
        
        Args:
            style: Seaborn style for plots
        """
        self.style = style
        if PLOTTING_AVAILABLE:
            sns.set_style(style)
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 10

    def create_performance_comparison_chart(self, experiment: ABTestExperiment) -> Optional[str]:
        """Create a performance comparison chart for experiment variants.
        
        Args:
            experiment: ABTestExperiment instance
            
        Returns:
            Base64 encoded PNG image or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None

        # Prepare data
        variants_data = []
        for variant_name, metrics in experiment.variant_metrics.items():
            if metrics.requests > 0:
                variants_data.append({
                    'variant': variant_name,
                    'requests': metrics.requests,
                    'error_rate': metrics.error_rate * 100,  # Convert to percentage
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'avg_confidence': metrics.avg_confidence * 100,  # Convert to percentage
                    'weight': experiment.variants[variant_name].weight * 100
                })

        if not variants_data:
            return None

        df = pd.DataFrame(variants_data)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'A/B Test Performance Comparison - {experiment.experiment_id}', fontsize=16)

        # 1. Error Rate Comparison
        sns.barplot(data=df, x='variant', y='error_rate', ax=axes[0, 0])
        axes[0, 0].set_title('Error Rate by Variant (%)')
        axes[0, 0].set_ylabel('Error Rate (%)')
        
        # Add value labels on bars
        for i, v in enumerate(df['error_rate']):
            axes[0, 0].text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')

        # 2. Average Latency Comparison
        sns.barplot(data=df, x='variant', y='avg_latency_ms', ax=axes[0, 1])
        axes[0, 1].set_title('Average Latency by Variant (ms)')
        axes[0, 1].set_ylabel('Latency (ms)')
        
        for i, v in enumerate(df['avg_latency_ms']):
            axes[0, 1].text(i, v + 1, f'{v:.1f}ms', ha='center', va='bottom')

        # 3. Confidence Score Comparison
        sns.barplot(data=df, x='variant', y='avg_confidence', ax=axes[1, 0])
        axes[1, 0].set_title('Average Confidence Score by Variant (%)')
        axes[1, 0].set_ylabel('Confidence Score (%)')
        
        for i, v in enumerate(df['avg_confidence']):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

        # 4. Request Volume and Traffic Weight
        x = np.arange(len(df))
        width = 0.35

        axes[1, 1].bar(x - width/2, df['requests'], width, label='Requests', alpha=0.8)
        axes[1, 1].bar(x + width/2, df['weight'] * max(df['requests']) / 100, width, 
                      label='Traffic Weight (scaled)', alpha=0.8)
        
        axes[1, 1].set_title('Request Volume vs Traffic Weight')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df['variant'])
        axes[1, 1].legend()

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64

    def create_statistical_significance_chart(self, statistical_tests: List[StatisticalTestResult]) -> Optional[str]:
        """Create a chart showing statistical significance test results.
        
        Args:
            statistical_tests: List of statistical test results
            
        Returns:
            Base64 encoded PNG image or None if plotting not available
        """
        if not PLOTTING_AVAILABLE or not statistical_tests:
            return None

        # Prepare data
        test_data = []
        for test in statistical_tests:
            test_data.append({
                'test_name': test.test_name.replace('_', ' ').title(),
                'p_value': test.p_value,
                'significant': test.significant,
                'effect_size': test.effect_size or 0,
                'statistic': test.statistic
            })

        df = pd.DataFrame(test_data)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Statistical Significance Test Results', fontsize=16)

        # 1. P-values with significance threshold
        colors = ['red' if sig else 'blue' for sig in df['significant']]
        bars = axes[0].bar(range(len(df)), df['p_value'], color=colors, alpha=0.7)
        
        # Add significance threshold line
        axes[0].axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
        
        axes[0].set_title('P-values by Test')
        axes[0].set_ylabel('P-value')
        axes[0].set_xlabel('Test')
        axes[0].set_xticks(range(len(df)))
        axes[0].set_xticklabels(df['test_name'], rotation=45, ha='right')
        axes[0].legend()
        
        # Add value labels
        for i, (bar, p_val, sig) in enumerate(zip(bars, df['p_value'], df['significant'])):
            label = f'{p_val:.4f}{"*" if sig else ""}'
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        label, ha='center', va='bottom', fontsize=8)

        # 2. Effect sizes
        effect_data = df[df['effect_size'] != 0]
        if not effect_data.empty:
            bars = axes[1].bar(range(len(effect_data)), effect_data['effect_size'], 
                              color='green', alpha=0.7)
            axes[1].set_title('Effect Sizes')
            axes[1].set_ylabel('Effect Size (Cohen\'s d)')
            axes[1].set_xlabel('Test')
            axes[1].set_xticks(range(len(effect_data)))
            axes[1].set_xticklabels(effect_data['test_name'], rotation=45, ha='right')
            
            # Add interpretation lines
            axes[1].axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
            axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium effect')
            axes[1].axhline(y=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large effect')
            axes[1].legend()
            
            # Add value labels
            for bar, effect in zip(bars, effect_data['effect_size']):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{effect:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[1].text(0.5, 0.5, 'No effect size data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Effect Sizes - No Data')

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64

    def create_rollout_progress_chart(self, rollout_plan: Dict[str, Any]) -> Optional[str]:
        """Create a chart showing gradual rollout progress.
        
        Args:
            rollout_plan: Rollout plan dictionary
            
        Returns:
            Base64 encoded PNG image or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            return None

        steps = rollout_plan.get('rollout_steps', [])
        if not steps:
            return None

        # Prepare data
        step_numbers = [step['step'] for step in steps]
        target_percentages = [step['target_percentage'] * 100 for step in steps]
        completed = [step['completed'] for step in steps]
        current_step = rollout_plan.get('current_step', 0)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create step chart
        colors = ['green' if comp else 'orange' if i < current_step else 'lightgray' 
                 for i, comp in enumerate(completed)]
        
        bars = ax.bar(step_numbers, target_percentages, color=colors, alpha=0.8)

        # Add current step indicator
        if current_step > 0 and current_step <= len(steps):
            ax.axvline(x=current_step, color='red', linestyle='--', linewidth=2, 
                      label=f'Current Step: {current_step}')

        ax.set_title(f'Gradual Rollout Progress - {rollout_plan.get("winning_variant", "Unknown")}')
        ax.set_xlabel('Rollout Step')
        ax.set_ylabel('Traffic Percentage (%)')
        ax.set_xticks(step_numbers)
        ax.legend()

        # Add value labels and status
        for i, (bar, percentage, comp) in enumerate(zip(bars, target_percentages, completed)):
            status = "✓" if comp else "⏳" if i < current_step else "⏸"
            label = f'{percentage:.1f}%\n{status}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   label, ha='center', va='bottom', fontsize=10)

        # Add legend for status symbols
        legend_text = "✓ Completed  ⏳ In Progress  ⏸ Pending"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64

    def create_experiment_timeline_chart(self, experiments: List[Dict[str, Any]]) -> Optional[str]:
        """Create a timeline chart showing multiple experiments.
        
        Args:
            experiments: List of experiment dictionaries
            
        Returns:
            Base64 encoded PNG image or None if plotting not available
        """
        if not PLOTTING_AVAILABLE or not experiments:
            return None

        # Prepare data
        timeline_data = []
        for exp in experiments:
            start_time = datetime.fromisoformat(exp['start_time'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(exp['end_time'].replace('Z', '+00:00'))
            
            timeline_data.append({
                'experiment_id': exp['experiment_id'],
                'start_time': start_time,
                'end_time': end_time,
                'status': exp['status'],
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            })

        # Sort by start time
        timeline_data.sort(key=lambda x: x['start_time'])

        fig, ax = plt.subplots(figsize=(15, max(6, len(timeline_data) * 0.5)))

        # Color mapping for status
        status_colors = {
            'active': 'green',
            'completed': 'blue',
            'stopped': 'red',
            'planned': 'orange'
        }

        # Create timeline bars
        for i, exp_data in enumerate(timeline_data):
            start = exp_data['start_time']
            end = exp_data['end_time']
            status = exp_data['status']
            
            # If experiment is active, show progress
            if status == 'active':
                now = datetime.now()
                if now < end:
                    # Show completed portion and remaining portion
                    completed_end = min(now, end)
                    ax.barh(i, (completed_end - start).total_seconds() / 3600, 
                           left=0, height=0.6, color=status_colors[status], alpha=0.8)
                    if now < end:
                        ax.barh(i, (end - completed_end).total_seconds() / 3600,
                               left=(completed_end - start).total_seconds() / 3600,
                               height=0.6, color=status_colors[status], alpha=0.3)
                else:
                    ax.barh(i, exp_data['duration_hours'], left=0, height=0.6, 
                           color=status_colors[status], alpha=0.8)
            else:
                ax.barh(i, exp_data['duration_hours'], left=0, height=0.6, 
                       color=status_colors[status], alpha=0.8)

            # Add experiment ID label
            ax.text(-1, i, exp_data['experiment_id'], ha='right', va='center', fontsize=9)

        ax.set_xlabel('Duration (hours)')
        ax.set_title('Experiment Timeline')
        ax.set_yticks(range(len(timeline_data)))
        ax.set_yticklabels([])  # Remove y-tick labels since we have custom labels
        ax.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=status.title()) 
                          for status, color in status_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64

    def create_variant_metrics_heatmap(self, experiment: ABTestExperiment) -> Optional[str]:
        """Create a heatmap showing variant performance metrics.
        
        Args:
            experiment: ABTestExperiment instance
            
        Returns:
            Base64 encoded PNG image or None if plotting not available
        """
        if not PLOTTING_AVAILABLE:
            return None

        # Prepare data
        metrics_data = []
        metric_names = ['Error Rate (%)', 'Avg Latency (ms)', 'Avg Confidence (%)', 'Requests']
        
        for variant_name, metrics in experiment.variant_metrics.items():
            if metrics.requests > 0:
                metrics_data.append([
                    metrics.error_rate * 100,
                    metrics.avg_latency_ms,
                    metrics.avg_confidence * 100,
                    metrics.requests
                ])
            else:
                metrics_data.append([0, 0, 0, 0])

        if not metrics_data:
            return None

        variant_names = list(experiment.variant_metrics.keys())
        
        # Normalize data for better visualization (except requests)
        df = pd.DataFrame(metrics_data, index=variant_names, columns=metric_names)
        
        # Create normalized version for heatmap (requests column separately)
        df_norm = df.copy()
        for col in ['Error Rate (%)', 'Avg Latency (ms)', 'Avg Confidence (%)']:
            if df[col].max() > 0:
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Normalize requests separately
        if df['Requests'].max() > 0:
            df_norm['Requests'] = df['Requests'] / df['Requests'].max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Heatmap of normalized metrics
        sns.heatmap(df_norm, annot=False, cmap='RdYlGn_r', ax=ax1, cbar_kws={'label': 'Normalized Score'})
        ax1.set_title('Variant Performance Heatmap (Normalized)')
        ax1.set_ylabel('Variants')

        # Actual values table
        ax2.axis('tight')
        ax2.axis('off')
        table_data = []
        for variant in variant_names:
            row = [variant]
            row.extend([f'{df.loc[variant, col]:.2f}' for col in metric_names])
            table_data.append(row)

        table = ax2.table(cellText=table_data,
                         colLabels=['Variant'] + metric_names,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax2.set_title('Actual Metric Values')

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64

    def generate_experiment_report(self, experiment: ABTestExperiment, 
                                 statistical_tests: List[StatisticalTestResult],
                                 winner_recommendation: Dict[str, Any],
                                 rollout_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive experiment report with visualizations.
        
        Args:
            experiment: ABTestExperiment instance
            statistical_tests: List of statistical test results
            winner_recommendation: Winner recommendation dictionary
            rollout_plan: Optional rollout plan dictionary
            
        Returns:
            Dictionary containing report data and visualizations
        """
        report = {
            'experiment_id': experiment.experiment_id,
            'generated_at': datetime.now().isoformat(),
            'experiment_summary': {
                'status': experiment.status,
                'start_time': experiment.start_time.isoformat(),
                'end_time': experiment.end_time.isoformat(),
                'duration_hours': (experiment.end_time - experiment.start_time).total_seconds() / 3600,
                'confidence_level': experiment.confidence_level,
                'total_requests': sum(m.requests for m in experiment.variant_metrics.values()),
                'total_errors': sum(m.errors for m in experiment.variant_metrics.values())
            },
            'variant_summary': {},
            'statistical_analysis': {
                'total_tests': len(statistical_tests),
                'significant_tests': len([t for t in statistical_tests if t.significant]),
                'tests': [t.to_dict() for t in statistical_tests]
            },
            'winner_recommendation': winner_recommendation,
            'visualizations': {},
            'rollout_plan': rollout_plan
        }

        # Add variant summaries
        for variant_name, metrics in experiment.variant_metrics.items():
            variant_config = experiment.variants[variant_name]
            report['variant_summary'][variant_name] = {
                'config': {
                    'model_path': variant_config.model_path,
                    'weight': variant_config.weight,
                    'status': variant_config.status.value
                },
                'metrics': metrics.to_dict()
            }

        # Generate visualizations
        if PLOTTING_AVAILABLE:
            try:
                report['visualizations']['performance_comparison'] = self.create_performance_comparison_chart(experiment)
                report['visualizations']['statistical_significance'] = self.create_statistical_significance_chart(statistical_tests)
                report['visualizations']['metrics_heatmap'] = self.create_variant_metrics_heatmap(experiment)
                
                if rollout_plan:
                    report['visualizations']['rollout_progress'] = self.create_rollout_progress_chart(rollout_plan)
                    
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                report['visualizations']['error'] = str(e)
        else:
            report['visualizations']['error'] = "Plotting libraries not available"

        return report


# Utility functions for web integration

def create_html_report(report_data: Dict[str, Any]) -> str:
    """Create an HTML report from report data.
    
    Args:
        report_data: Report data from generate_experiment_report
        
    Returns:
        HTML string
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A/B Test Experiment Report - {experiment_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            .winner {{ background-color: #d4edda; padding: 10px; border-radius: 5px; }}
            .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>A/B Test Experiment Report</h1>
            <h2>Experiment ID: {experiment_id}</h2>
            <p>Generated at: {generated_at}</p>
        </div>
        
        <div class="section">
            <h3>Experiment Summary</h3>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Duration:</strong> {duration_hours:.1f} hours</p>
            <p><strong>Total Requests:</strong> {total_requests:,}</p>
            <p><strong>Total Errors:</strong> {total_errors:,}</p>
            <p><strong>Confidence Level:</strong> {confidence_level:.1%}</p>
        </div>
        
        {winner_section}
        
        {visualizations_section}
        
        <div class="section">
            <h3>Statistical Analysis</h3>
            <p><strong>Total Tests:</strong> {total_tests}</p>
            <p><strong>Significant Tests:</strong> {significant_tests}</p>
        </div>
        
    </body>
    </html>
    """
    
    # Prepare winner section
    winner_rec = report_data.get('winner_recommendation', {})
    if winner_rec.get('winner'):
        winner_section = f"""
        <div class="section winner">
            <h3>Winner Recommendation</h3>
            <p><strong>Winning Variant:</strong> {winner_rec['winner']}</p>
            <p><strong>Statistically Significant:</strong> {'Yes' if winner_rec.get('statistically_significant') else 'No'}</p>
            <p><strong>Recommendation:</strong> {winner_rec.get('recommendation', 'N/A')}</p>
        </div>
        """
    else:
        winner_section = """
        <div class="section warning">
            <h3>Winner Recommendation</h3>
            <p>No clear winner determined. More data may be needed.</p>
        </div>
        """
    
    # Prepare visualizations section
    visualizations = report_data.get('visualizations', {})
    viz_html = ""
    for viz_name, viz_data in visualizations.items():
        if viz_data and viz_name != 'error':
            viz_html += f"""
            <div class="visualization">
                <h4>{viz_name.replace('_', ' ').title()}</h4>
                <img src="data:image/png;base64,{viz_data}" alt="{viz_name}" style="max-width: 100%;">
            </div>
            """
    
    if not viz_html and visualizations.get('error'):
        viz_html = f"""
        <div class="section warning">
            <h3>Visualizations</h3>
            <p>Visualizations could not be generated: {visualizations['error']}</p>
        </div>
        """
    
    visualizations_section = f"""
    <div class="section">
        <h3>Visualizations</h3>
        {viz_html}
    </div>
    """ if viz_html else ""
    
    # Format the HTML
    summary = report_data['experiment_summary']
    return html_template.format(
        experiment_id=report_data['experiment_id'],
        generated_at=report_data['generated_at'],
        status=summary['status'],
        duration_hours=summary['duration_hours'],
        total_requests=summary['total_requests'],
        total_errors=summary['total_errors'],
        confidence_level=summary['confidence_level'],
        total_tests=report_data['statistical_analysis']['total_tests'],
        significant_tests=report_data['statistical_analysis']['significant_tests'],
        winner_section=winner_section,
        visualizations_section=visualizations_section
    )


# Global visualizer instance
visualizer = ABTestVisualizer()