"""
MLFlow Experiment Visualization and Comparison Tools

This module provides visualization and comparison tools for MLFlow experiments,
enabling model performance analysis, hyperparameter comparison, and automated reporting.

Requirements addressed:
- 7.3: Model performance visualization utilities
- 7.4: Experiment comparison and analysis
- Automated reporting functions for experiment results
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFlowExperimentVisualizer:
    """
    Provides visualization and comparison tools for MLFlow experiments.
    """
    
    def __init__(self, experiment_name: Optional[str] = None, 
               tracking_uri: Optional[str] = None):
        """
        Initialize the MLFlow experiment visualizer.
        
        Args:
            experiment_name: Optional experiment name to focus on
            tracking_uri: Optional MLFlow tracking URI
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.experiment_id = None
        
        # Set experiment if name provided
        if experiment_name:
            self.set_experiment(experiment_name)
        
        logger.info(f"MLFlow experiment visualizer initialized")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    def set_experiment(self, experiment_name: str):
        """
        Set the experiment to visualize.
        
        Args:
            experiment_name: Name of the experiment
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            self.experiment_name = experiment_name
            self.experiment_id = experiment.experiment_id
            logger.info(f"Set experiment: {experiment_name} (ID: {self.experiment_id})")
        else:
            logger.warning(f"Experiment not found: {experiment_name}")
            self.experiment_id = None
    
    def get_experiment_runs(self, experiment_name: Optional[str] = None, 
                          filter_string: Optional[str] = None) -> pd.DataFrame:
        """
        Get runs for an experiment as a DataFrame.
        
        Args:
            experiment_name: Optional experiment name (uses current if None)
            filter_string: Optional filter string for runs
            
        Returns:
            DataFrame with run information
        """
        # Use provided experiment name or current one
        if experiment_name:
            self.set_experiment(experiment_name)
        
        if not self.experiment_id:
            logger.error("No experiment set. Use set_experiment() first.")
            return pd.DataFrame()
        
        # Get runs
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"]
        )
        
        if not runs:
            logger.warning(f"No runs found for experiment: {self.experiment_name}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        runs_data = []
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'end_time': pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                run_data[f"param.{key}"] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                run_data[f"metric.{key}"] = value
            
            runs_data.append(run_data)
        
        df = pd.DataFrame(runs_data)
        
        # Calculate duration
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        
        logger.info(f"Retrieved {len(df)} runs for experiment: {self.experiment_name}")
        return df
    
    def plot_metric_comparison(self, metric_name: str, runs_df: Optional[pd.DataFrame] = None,
                             top_n: int = 10, ascending: bool = False,
                             output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of a specific metric across runs.
        
        Args:
            metric_name: Name of the metric to compare
            runs_df: Optional DataFrame with run data (fetched if None)
            top_n: Number of top runs to include
            ascending: Whether to sort in ascending order
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get runs data if not provided
        if runs_df is None:
            runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            logger.warning("No runs data available for plotting")
            return None
        
        # Check if metric exists
        metric_col = f"metric.{metric_name}"
        if metric_col not in runs_df.columns:
            logger.warning(f"Metric '{metric_name}' not found in runs data")
            return None
        
        # Sort and filter top N runs
        sorted_runs = runs_df.sort_values(metric_col, ascending=ascending).head(top_n)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        ax = sns.barplot(x='run_name', y=metric_col, data=sorted_runs)
        
        # Add value labels on bars
        for i, v in enumerate(sorted_runs[metric_col]):
            ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        
        # Set labels and title
        plt.title(f"Comparison of {metric_name} across top {top_n} runs", fontsize=14)
        plt.xlabel("Run Name", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        return plt.gcf()
    
    def plot_parameter_impact(self, param_name: str, metric_name: str,
                            runs_df: Optional[pd.DataFrame] = None,
                            output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the impact of a parameter on a metric.
        
        Args:
            param_name: Name of the parameter to analyze
            metric_name: Name of the metric to analyze
            runs_df: Optional DataFrame with run data (fetched if None)
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get runs data if not provided
        if runs_df is None:
            runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            logger.warning("No runs data available for plotting")
            return None
        
        # Check if parameter and metric exist
        param_col = f"param.{param_name}"
        metric_col = f"metric.{metric_name}"
        
        if param_col not in runs_df.columns:
            logger.warning(f"Parameter '{param_name}' not found in runs data")
            return None
        
        if metric_col not in runs_df.columns:
            logger.warning(f"Metric '{metric_name}' not found in runs data")
            return None
        
        # Convert parameter to numeric if possible
        try:
            runs_df[param_col] = pd.to_numeric(runs_df[param_col])
            numeric_param = True
        except:
            numeric_param = False
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if numeric_param:
            # Scatter plot for numeric parameters
            sns.scatterplot(x=param_col, y=metric_col, data=runs_df, s=100, alpha=0.7)
            
            # Add trend line
            sns.regplot(x=param_col, y=metric_col, data=runs_df, scatter=False, 
                      line_kws={"color": "red", "alpha": 0.7, "lw": 2})
        else:
            # Box plot for categorical parameters
            sns.boxplot(x=param_col, y=metric_col, data=runs_df)
            
            # Add individual points
            sns.stripplot(x=param_col, y=metric_col, data=runs_df, 
                        size=8, jitter=True, alpha=0.6, color='black')
        
        # Set labels and title
        plt.title(f"Impact of {param_name} on {metric_name}", fontsize=14)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(rotation=45 if not numeric_param else 0)
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        return plt.gcf()
    
    def plot_metric_history(self, run_id: str, metric_names: List[str],
                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the history of metrics for a specific run.
        
        Args:
            run_id: Run ID to analyze
            metric_names: List of metrics to plot
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Get run
            run = self.client.get_run(run_id)
            
            if not run:
                logger.warning(f"Run not found: {run_id}")
                return None
            
            # Create figure with subplots
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)), 
                                   sharex=True)
            
            # Handle single metric case
            if len(metric_names) == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric_name in enumerate(metric_names):
                # Get metric history
                history = self.client.get_metric_history(run_id, metric_name)
                
                if not history:
                    logger.warning(f"No history found for metric: {metric_name}")
                    continue
                
                # Extract values and steps
                steps = [h.step for h in history]
                values = [h.value for h in history]
                timestamps = [pd.to_datetime(h.timestamp, unit='ms') for h in history]
                
                # Plot metric
                axes[i].plot(steps, values, marker='o', linestyle='-', linewidth=2, markersize=6)
                axes[i].set_title(f"{metric_name} History", fontsize=12)
                axes[i].set_ylabel(metric_name, fontsize=10)
                axes[i].grid(True, linestyle='--', alpha=0.7)
                
                # Add value annotations
                for j, (step, value) in enumerate(zip(steps, values)):
                    if j % max(1, len(steps) // 10) == 0:  # Annotate every 10th point or less
                        axes[i].annotate(f"{value:.4f}", (step, value), 
                                      textcoords="offset points", 
                                      xytext=(0, 10), 
                                      ha='center')
            
            # Set common x-axis label
            axes[-1].set_xlabel("Step", fontsize=12)
            
            # Add run information
            run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
            plt.suptitle(f"Metric History for Run: {run_name} ({run_id})", fontsize=14)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save if output path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to: {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting metric history: {str(e)}")
            return None
    
    def create_parallel_coordinates_plot(self, params: List[str], metrics: List[str],
                                       runs_df: Optional[pd.DataFrame] = None,
                                       output_path: Optional[str] = None) -> go.Figure:
        """
        Create a parallel coordinates plot for hyperparameter analysis.
        
        Args:
            params: List of parameters to include
            metrics: List of metrics to include
            runs_df: Optional DataFrame with run data (fetched if None)
            output_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        # Get runs data if not provided
        if runs_df is None:
            runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            logger.warning("No runs data available for plotting")
            return None
        
        # Prepare columns
        param_cols = [f"param.{p}" for p in params]
        metric_cols = [f"metric.{m}" for m in metrics]
        
        # Check if all columns exist
        missing_cols = [col for col in param_cols + metric_cols if col not in runs_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Filter out missing columns
            param_cols = [col for col in param_cols if col in runs_df.columns]
            metric_cols = [col for col in metric_cols if col in runs_df.columns]
        
        if not param_cols or not metric_cols:
            logger.error("No valid parameters or metrics to plot")
            return None
        
        # Prepare data for plotting
        plot_df = runs_df[['run_name'] + param_cols + metric_cols].copy()
        
        # Convert parameters to numeric if possible
        for col in param_cols:
            try:
                plot_df[col] = pd.to_numeric(plot_df[col])
            except:
                pass
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            plot_df,
            color=metric_cols[0],  # Color by first metric
            labels={col: col.split('.')[-1] for col in param_cols + metric_cols},
            title=f"Parameter and Metric Relationships for {self.experiment_name}",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Update layout
        fig.update_layout(
            font=dict(size=12),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Plot saved to: {output_path}")
        
        return fig
    
    def create_correlation_heatmap(self, params: List[str], metrics: List[str],
                                 runs_df: Optional[pd.DataFrame] = None,
                                 output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation heatmap between parameters and metrics.
        
        Args:
            params: List of parameters to include
            metrics: List of metrics to include
            runs_df: Optional DataFrame with run data (fetched if None)
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get runs data if not provided
        if runs_df is None:
            runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            logger.warning("No runs data available for plotting")
            return None
        
        # Prepare columns
        param_cols = [f"param.{p}" for p in params]
        metric_cols = [f"metric.{m}" for m in metrics]
        
        # Check if all columns exist
        all_cols = param_cols + metric_cols
        missing_cols = [col for col in all_cols if col not in runs_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Filter out missing columns
            all_cols = [col for col in all_cols if col in runs_df.columns]
        
        if not all_cols:
            logger.error("No valid parameters or metrics to plot")
            return None
        
        # Prepare data for correlation
        corr_df = runs_df[all_cols].copy()
        
        # Convert parameters to numeric if possible
        for col in param_cols:
            if col in corr_df.columns:
                try:
                    corr_df[col] = pd.to_numeric(corr_df[col])
                except:
                    # Drop non-numeric columns
                    corr_df.drop(columns=[col], inplace=True)
                    logger.warning(f"Dropped non-numeric column: {col}")
        
        if corr_df.empty or corr_df.shape[1] < 2:
            logger.error("Insufficient numeric data for correlation analysis")
            return None
        
        # Calculate correlation
        corr_matrix = corr_df.corr()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                  mask=mask, vmin=-1, vmax=1, center=0,
                  square=True, linewidths=.5, cbar_kws={"shrink": .8})
        
        # Set title
        plt.title(f"Parameter-Metric Correlation for {self.experiment_name}", fontsize=14)
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        return plt.gcf()
    
    def generate_experiment_report(self, output_dir: str, 
                                 top_n_runs: int = 5,
                                 metrics_to_include: Optional[List[str]] = None,
                                 params_to_include: Optional[List[str]] = None):
        """
        Generate a comprehensive experiment report with visualizations.
        
        Args:
            output_dir: Directory to save the report
            top_n_runs: Number of top runs to highlight
            metrics_to_include: List of metrics to include (all if None)
            params_to_include: List of parameters to include (all if None)
        """
        logger.info(f"Generating experiment report for {self.experiment_name}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get runs data
        runs_df = self.get_experiment_runs()
        
        if runs_df.empty:
            logger.error("No runs data available for report generation")
            return
        
        # Determine metrics and parameters to include
        metric_cols = [col for col in runs_df.columns if col.startswith('metric.')]
        param_cols = [col for col in runs_df.columns if col.startswith('param.')]
        
        metrics = [col.split('.')[-1] for col in metric_cols]
        params = [col.split('.')[-1] for col in param_cols]
        
        if metrics_to_include:
            metrics = [m for m in metrics if m in metrics_to_include]
        
        if params_to_include:
            params = [p for p in params if p in params_to_include]
        
        # Create HTML report
        html_path = os.path.join(output_dir, f"{self.experiment_name}_report.html")
        
        with open(html_path, 'w') as f:
            # Write header
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLFlow Experiment Report: {self.experiment_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-good {{ color: green; font-weight: bold; }}
                    .metric-bad {{ color: red; }}
                    .container {{ margin-bottom: 30px; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>MLFlow Experiment Report: {self.experiment_name}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Tracking URI: {mlflow.get_tracking_uri()}</p>
                <p>Total Runs: {len(runs_df)}</p>
            """)
            
            # Summary of top runs
            f.write(f"""
                <div class="container">
                    <h2>Top Runs Summary</h2>
            """)
            
            # Create tables for each important metric
            for metric in metrics:
                metric_col = f"metric.{metric}"
                
                # Determine if higher or lower is better (assume higher is better)
                # This could be made configurable
                ascending = False
                
                # Sort runs by metric
                sorted_runs = runs_df.sort_values(metric_col, ascending=ascending).head(top_n_runs)
                
                f.write(f"""
                    <h3>Top {top_n_runs} Runs by {metric}</h3>
                    <table>
                        <tr>
                            <th>Run Name</th>
                            <th>Status</th>
                            <th>{metric}</th>
                            <th>Duration</th>
                            <th>Start Time</th>
                """)
                
                # Add parameter columns
                for param in params[:5]:  # Limit to first 5 parameters
                    f.write(f"<th>{param}</th>")
                
                f.write("</tr>")
                
                # Add rows
                for _, row in sorted_runs.iterrows():
                    f.write("<tr>")
                    f.write(f"<td>{row['run_name']}</td>")
                    f.write(f"<td>{row['status']}</td>")
                    f.write(f"<td class='metric-good'>{row[metric_col]:.4f}</td>")
                    
                    # Duration
                    if 'duration' in row and not pd.isna(row['duration']):
                        duration_mins = row['duration'] / 60
                        f.write(f"<td>{duration_mins:.1f} min</td>")
                    else:
                        f.write("<td>N/A</td>")
                    
                    # Start time
                    if 'start_time' in row and not pd.isna(row['start_time']):
                        f.write(f"<td>{row['start_time'].strftime('%Y-%m-%d %H:%M')}</td>")
                    else:
                        f.write("<td>N/A</td>")
                    
                    # Parameters
                    for param in params[:5]:
                        param_col = f"param.{param}"
                        if param_col in row:
                            f.write(f"<td>{row[param_col]}</td>")
                        else:
                            f.write("<td>N/A</td>")
                    
                    f.write("</tr>")
                
                f.write("</table>")
                
                # Generate and include metric comparison plot
                plot_path = os.path.join(output_dir, f"{metric}_comparison.png")
                self.plot_metric_comparison(metric, runs_df, top_n=top_n_runs, 
                                          ascending=ascending, output_path=plot_path)
                
                f.write(f"""
                    <img src="{os.path.basename(plot_path)}" alt="{metric} comparison">
                """)
            
            f.write("</div>")
            
            # Parameter impact analysis
            f.write(f"""
                <div class="container">
                    <h2>Parameter Impact Analysis</h2>
            """)
            
            # Generate parameter impact plots for each metric
            for metric in metrics:
                f.write(f"<h3>Impact on {metric}</h3>")
                
                for param in params[:3]:  # Limit to first 3 parameters
                    plot_path = os.path.join(output_dir, f"{param}_impact_on_{metric}.png")
                    self.plot_parameter_impact(param, metric, runs_df, output_path=plot_path)
                    
                    f.write(f"""
                        <img src="{os.path.basename(plot_path)}" alt="{param} impact on {metric}">
                    """)
            
            f.write("</div>")
            
            # Correlation analysis
            f.write(f"""
                <div class="container">
                    <h2>Correlation Analysis</h2>
            """)
            
            # Generate correlation heatmap
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            self.create_correlation_heatmap(params, metrics, runs_df, output_path=heatmap_path)
            
            f.write(f"""
                <img src="{os.path.basename(heatmap_path)}" alt="Correlation Heatmap">
            """)
            
            f.write("</div>")
            
            # Best run details
            f.write(f"""
                <div class="container">
                    <h2>Best Run Details</h2>
            """)
            
            # Get best run based on primary metric (first in the list)
            if metrics:
                primary_metric = metrics[0]
                metric_col = f"metric.{primary_metric}"
                best_run_idx = runs_df[metric_col].idxmax()
                best_run = runs_df.loc[best_run_idx]
                
                f.write(f"""
                    <h3>Best Run: {best_run['run_name']}</h3>
                    <p>Run ID: {best_run['run_id']}</p>
                    <p>Status: {best_run['status']}</p>
                    <p>{primary_metric}: {best_run[metric_col]:.4f}</p>
                    <p>Start Time: {best_run['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                """)
                
                if 'duration' in best_run and not pd.isna(best_run['duration']):
                    f.write(f"<p>Duration: {best_run['duration'] / 60:.1f} minutes</p>")
                
                # Parameters table
                f.write(f"""
                    <h4>Parameters</h4>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                """)
                
                for param in params:
                    param_col = f"param.{param}"
                    if param_col in best_run:
                        f.write(f"""
                            <tr>
                                <td>{param}</td>
                                <td>{best_run[param_col]}</td>
                            </tr>
                        """)
                
                f.write("</table>")
                
                # Metrics table
                f.write(f"""
                    <h4>Metrics</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                """)
                
                for metric in metrics:
                    metric_col = f"metric.{metric}"
                    if metric_col in best_run:
                        f.write(f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{best_run[metric_col]:.4f}</td>
                            </tr>
                        """)
                
                f.write("</table>")
                
                # Generate metric history plot for best run
                history_path = os.path.join(output_dir, "best_run_metrics.png")
                self.plot_metric_history(best_run['run_id'], metrics[:3], output_path=history_path)
                
                f.write(f"""
                    <h4>Metric History</h4>
                    <img src="{os.path.basename(history_path)}" alt="Best Run Metric History">
                """)
            
            f.write("</div>")
            
            # Footer
            f.write(f"""
                <div class="container">
                    <h2>Report Information</h2>
                    <p>Generated by MLFlowExperimentVisualizer</p>
                    <p>MLFlow Version: {mlflow.__version__}</p>
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Experiment report generated: {html_path}")
        return html_path


def compare_runs(run_ids: List[str], metrics: List[str], 
               tracking_uri: Optional[str] = None) -> plt.Figure:
    """
    Compare specific runs across selected metrics.
    
    Args:
        run_ids: List of run IDs to compare
        metrics: List of metrics to compare
        tracking_uri: Optional MLFlow tracking URI
        
    Returns:
        Matplotlib figure with comparison plots
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    
    # Get run data
    runs_data = []
    run_names = []
    
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            run_name = run.data.tags.get('mlflow.runName', run_id[:8])
            run_names.append(run_name)
            
            run_metrics = {}
            for metric in metrics:
                if metric in run.data.metrics:
                    run_metrics[metric] = run.data.metrics[metric]
                else:
                    run_metrics[metric] = None
            
            runs_data.append(run_metrics)
            
        except Exception as e:
            logger.warning(f"Error getting run {run_id}: {str(e)}")
    
    if not runs_data:
        logger.error("No valid runs found for comparison")
        return None
    
    # Create figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        metric_values = [run_data.get(metric) for run_data in runs_data]
        valid_indices = [j for j, v in enumerate(metric_values) if v is not None]
        
        if not valid_indices:
            axes[i].text(0.5, 0.5, f"No data for {metric}", 
                       ha='center', va='center', fontsize=12)
            continue
        
        valid_values = [metric_values[j] for j in valid_indices]
        valid_names = [run_names[j] for j in valid_indices]
        
        # Create bar plot
        bars = axes[i].bar(valid_names, valid_values)
        
        # Add value labels
        for bar, value in zip(bars, valid_values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
        
        # Set title and labels
        axes[i].set_title(f"{metric}", fontsize=12)
        axes[i].set_ylabel("Value", fontsize=10)
        axes[i].set_xticklabels(valid_names, rotation=45, ha='right')
    
    plt.suptitle("Run Comparison", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def find_best_runs(experiment_name: str, metric_name: str, top_n: int = 5,
                 ascending: bool = False, tracking_uri: Optional[str] = None) -> pd.DataFrame:
    """
    Find the best runs for an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to sort by
        top_n: Number of top runs to return
        ascending: Whether to sort in ascending order (lower is better)
        tracking_uri: Optional MLFlow tracking URI
        
    Returns:
        DataFrame with top runs
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.error(f"Experiment not found: {experiment_name}")
        return pd.DataFrame()
    
    # Search for runs
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=top_n
    )
    
    if not runs:
        logger.warning(f"No runs found for experiment: {experiment_name}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    runs_data = []
    
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
            'status': run.info.status,
            'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
            'end_time': pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None,
            metric_name: run.data.metrics.get(metric_name)
        }
        
        # Add key parameters
        for key, value in run.data.params.items():
            run_data[f"param.{key}"] = value
        
        # Add other metrics
        for key, value in run.data.metrics.items():
            if key != metric_name:
                run_data[f"metric.{key}"] = value
        
        runs_data.append(run_data)
    
    return pd.DataFrame(runs_data)


if __name__ == "__main__":
    # Example usage
    visualizer = MLFlowExperimentVisualizer("yolov11-drone-detection")
    
    # Get experiment runs
    runs_df = visualizer.get_experiment_runs()
    
    if not runs_df.empty:
        # Generate plots
        visualizer.plot_metric_comparison("mAP50", runs_df, output_path="map50_comparison.png")
        visualizer.plot_parameter_impact("learning_rate", "mAP50", runs_df, 
                                       output_path="lr_impact.png")
        
        # Generate report
        visualizer.generate_experiment_report("experiment_report")