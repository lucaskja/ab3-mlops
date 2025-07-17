#!/usr/bin/env python3
"""
MLFlow Integration Test Script

This script tests the MLFlow integration with SageMaker by creating an experiment,
logging metrics and parameters, and verifying the tracking server connection.

Usage:
    python scripts/training/test_mlflow_integration.py --experiment test-experiment
    python scripts/training/test_mlflow_integration.py --list-experiments
    python scripts/training/test_mlflow_integration.py --list-runs --experiment yolov11-drone-detection
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from pipeline.mlflow_integration import MLFlowSageMakerIntegration
from pipeline.mlflow_visualization import MLFlowExperimentVisualizer, find_best_runs
from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test MLFlow integration with SageMaker')
    
    # Actions
    parser.add_argument('--test-connection', action='store_true',
                       help='Test connection to MLFlow tracking server')
    parser.add_argument('--create-experiment', action='store_true',
                       help='Create a new MLFlow experiment')
    parser.add_argument('--log-metrics', action='store_true',
                       help='Log sample metrics to MLFlow')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all MLFlow experiments')
    parser.add_argument('--list-runs', action='store_true',
                       help='List runs for an experiment')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate experiment report')
    
    # Parameters
    parser.add_argument('--experiment', type=str, default='test-mlflow-integration',
                       help='MLFlow experiment name')
    parser.add_argument('--run-name', type=str, default=f'test-run-{int(time.time())}',
                       help='MLFlow run name')
    parser.add_argument('--run-id', type=str,
                       help='MLFlow run ID for specific operations')
    parser.add_argument('--output-dir', type=str, default='mlflow-reports',
                       help='Output directory for reports')
    
    # AWS configuration
    parser.add_argument('--aws-profile', type=str, default='ab',
                       help='AWS profile to use')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
    return parser.parse_args()


def test_connection(mlflow_integration):
    """Test connection to MLFlow tracking server."""
    logger.info("Testing connection to MLFlow tracking server...")
    
    try:
        # Get tracking URI
        tracking_uri = mlflow_integration.client.tracking_uri
        logger.info(f"MLFlow tracking URI: {tracking_uri}")
        
        # List experiments to test connection
        experiments = mlflow_integration.list_experiments()
        
        if experiments:
            logger.info(f"Connection successful! Found {len(experiments)} experiments.")
            logger.info(f"First experiment: {experiments[0]['name']} (ID: {experiments[0]['experiment_id']})")
        else:
            logger.info("Connection successful! No experiments found.")
        
        return True
        
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False


def create_experiment(mlflow_integration, experiment_name):
    """Create a new MLFlow experiment."""
    logger.info(f"Creating MLFlow experiment: {experiment_name}")
    
    try:
        experiment_id = mlflow_integration.create_experiment(experiment_name)
        logger.info(f"Experiment created with ID: {experiment_id}")
        return experiment_id
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        return None


def log_sample_metrics(mlflow_integration, experiment_name, run_name):
    """Log sample metrics to MLFlow."""
    logger.info(f"Logging sample metrics to experiment: {experiment_name}, run: {run_name}")
    
    try:
        # Start run
        with mlflow_integration.start_run(experiment_name, run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started run with ID: {run_id}")
            
            # Log parameters
            params = {
                'model_variant': 'yolov11n',
                'learning_rate': 0.01,
                'batch_size': 16,
                'epochs': 100,
                'image_size': 640,
                'test_run': True
            }
            mlflow_integration.log_parameters(params)
            logger.info(f"Logged {len(params)} parameters")
            
            # Generate sample metrics
            epochs = 10
            train_loss = np.random.rand(epochs) * 0.5
            val_loss = np.random.rand(epochs) * 0.7
            map50 = np.linspace(0.5, 0.85, epochs) + np.random.rand(epochs) * 0.1
            
            # Log metrics for each epoch
            for epoch in range(epochs):
                metrics = {
                    'train_loss': train_loss[epoch],
                    'val_loss': val_loss[epoch],
                    'mAP50': map50[epoch],
                    'precision': 0.7 + epoch * 0.02 + np.random.rand() * 0.05,
                    'recall': 0.65 + epoch * 0.03 + np.random.rand() * 0.05
                }
                mlflow_integration.log_metrics(metrics, step=epoch)
                logger.info(f"Logged metrics for epoch {epoch}")
                
                # Simulate training time
                time.sleep(0.2)
            
            # Create and log a sample plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(epochs), train_loss, label='Train Loss')
            plt.plot(range(epochs), val_loss, label='Val Loss')
            plt.plot(range(epochs), map50, label='mAP@0.5')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Training Metrics')
            plt.legend()
            plt.grid(True)
            
            # Save plot to temp file
            plot_path = 'temp_training_plot.png'
            plt.savefig(plot_path)
            plt.close()
            
            # Log plot as artifact
            mlflow_integration.log_artifact(plot_path, 'plots')
            logger.info(f"Logged training plot as artifact")
            
            # Clean up temp file
            os.remove(plot_path)
            
            logger.info(f"Sample metrics logging completed for run: {run_id}")
            return run_id
            
    except Exception as e:
        logger.error(f"Failed to log sample metrics: {str(e)}")
        return None


def list_experiments(mlflow_integration):
    """List all MLFlow experiments."""
    logger.info("Listing MLFlow experiments...")
    
    try:
        experiments = mlflow_integration.list_experiments()
        
        if not experiments:
            logger.info("No experiments found")
            return
        
        # Print experiment summary
        print("\n" + "="*80)
        print("MLFLOW EXPERIMENTS")
        print("="*80)
        print(f"{'ID':<10} {'Name':<40} {'Artifact Location':<30}")
        print("-"*80)
        
        for exp in experiments:
            exp_id = exp['experiment_id']
            name = exp['name']
            artifact_location = exp['artifact_location']
            print(f"{exp_id:<10} {name:<40} {artifact_location:<30}")
        
        print("="*80)
        print(f"Total experiments: {len(experiments)}")
        
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")


def list_runs(mlflow_integration, experiment_name):
    """List runs for an experiment."""
    logger.info(f"Listing runs for experiment: {experiment_name}")
    
    try:
        runs = mlflow_integration.list_runs(experiment_name)
        
        if not runs:
            logger.info(f"No runs found for experiment: {experiment_name}")
            return
        
        # Print run summary
        print("\n" + "="*100)
        print(f"MLFLOW RUNS FOR EXPERIMENT: {experiment_name}")
        print("="*100)
        print(f"{'Run ID':<36} {'Run Name':<30} {'Status':<10} {'Start Time':<20} {'Duration':<10}")
        print("-"*100)
        
        for run in runs:
            run_id = run['run_id']
            run_name = run.get('run_name', 'Unnamed')
            status = run['status']
            start_time = run['start_time'].strftime('%Y-%m-%d %H:%M:%S') if 'start_time' in run else 'N/A'
            
            # Calculate duration if available
            duration = 'N/A'
            if 'start_time' in run and 'end_time' in run and run['end_time'] is not None:
                duration_secs = (run['end_time'] - run['start_time']).total_seconds()
                duration = f"{duration_secs:.1f}s"
            
            print(f"{run_id:<36} {run_name:<30} {status:<10} {start_time:<20} {duration:<10}")
        
        print("="*100)
        print(f"Total runs: {len(runs)}")
        
        # Print metrics for the latest run
        latest_run = runs[0]
        print("\nLatest Run Metrics:")
        for key, value in latest_run.items():
            if key.startswith('metrics.'):
                metric_name = key.replace('metrics.', '')
                print(f"  {metric_name}: {value}")
        
    except Exception as e:
        logger.error(f"Error listing runs: {str(e)}")


def generate_experiment_report(experiment_name, output_dir):
    """Generate experiment report using the visualizer."""
    logger.info(f"Generating report for experiment: {experiment_name}")
    
    try:
        # Initialize visualizer
        visualizer = MLFlowExperimentVisualizer(experiment_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        report_path = visualizer.generate_experiment_report(output_dir)
        
        if report_path:
            logger.info(f"Report generated successfully: {report_path}")
            print(f"\nReport generated: {report_path}")
        else:
            logger.warning("Failed to generate report")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Starting MLFlow integration test...")
    
    try:
        # Initialize MLFlow integration
        mlflow_integration = MLFlowSageMakerIntegration(
            aws_profile=args.aws_profile,
            region=args.region
        )
        
        # Execute requested action
        if args.test_connection:
            test_connection(mlflow_integration)
            
        elif args.create_experiment:
            create_experiment(mlflow_integration, args.experiment)
            
        elif args.log_metrics:
            log_sample_metrics(mlflow_integration, args.experiment, args.run_name)
            
        elif args.list_experiments:
            list_experiments(mlflow_integration)
            
        elif args.list_runs:
            list_runs(mlflow_integration, args.experiment)
            
        elif args.generate_report:
            generate_experiment_report(args.experiment, args.output_dir)
            
        else:
            logger.info("No action specified. Running full test suite...")
            
            # Test connection
            if test_connection(mlflow_integration):
                # Create experiment
                experiment_id = create_experiment(mlflow_integration, args.experiment)
                
                if experiment_id:
                    # Log sample metrics
                    run_id = log_sample_metrics(mlflow_integration, args.experiment, args.run_name)
                    
                    if run_id:
                        # List runs
                        list_runs(mlflow_integration, args.experiment)
                        
                        # Generate report
                        generate_experiment_report(args.experiment, args.output_dir)
        
        logger.info("MLFlow integration test completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)