"""
SageMaker Clarify Integration for Model Explainability

This module provides functionality for integrating SageMaker Clarify with model
monitoring for bias detection and explainability.

Requirements addressed:
- 5.4: Generate monitoring reports accessible through SageMaker Clarify
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import boto3
import sagemaker
from sagemaker import get_execution_role, clarify
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig
)
from sagemaker.model_monitor import (
    ClarifyMonitoringConfig,
    ModelMonitor,
    DefaultModelMonitor,
    ExplainabilityMonitoringConfig
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ClarifyCheckStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import ClarifyCheckConfig

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClarifyManager:
    """
    Manages SageMaker Clarify configurations and operations for model explainability.
    """
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the Clarify manager.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.sagemaker_client = self.session.client('sagemaker', region_name=region)
        self.s3_client = self.session.client('s3', region_name=region)
        self.cloudwatch_client = self.session.client('cloudwatch', region_name=region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.session,
            sagemaker_client=self.sagemaker_client
        )
        
        # Get project configuration
        self.project_config = get_config()
        self.execution_role = self.project_config['iam']['roles']['sagemaker_execution']['arn']
        
        # Set default S3 bucket
        self.default_bucket = self.sagemaker_session.default_bucket()
        
        logger.info(f"Clarify manager initialized for region: {region}")
        logger.info(f"Using execution role: {self.execution_role}")
        logger.info(f"Using default S3 bucket: {self.default_bucket}")
    
    def create_clarify_processor(
        self,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        volume_size_in_gb: int = 30,
        max_runtime_in_seconds: int = 3600
    ) -> clarify.SageMakerClarifyProcessor:
        """
        Create a SageMaker Clarify processor.
        
        Args:
            instance_type: Instance type for processing
            instance_count: Number of instances
            volume_size_in_gb: Volume size in GB
            max_runtime_in_seconds: Maximum runtime in seconds
            
        Returns:
            Configured SageMakerClarifyProcessor
        """
        logger.info(f"Creating Clarify processor with {instance_type} instance")
        
        clarify_processor = clarify.SageMakerClarifyProcessor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=self.sagemaker_session,
            max_runtime_in_seconds=max_runtime_in_seconds,
            volume_size_in_gb=volume_size_in_gb
        )
        
        logger.info(f"Clarify processor created with {instance_count} {instance_type} instances")
        return clarify_processor
    
    def run_bias_analysis(
        self,
        dataset_path: str,
        model_name: str,
        model_endpoint_name: str,
        target_column: str,
        sensitive_columns: List[str],
        output_path: Optional[str] = None,
        headers: Optional[List[str]] = None,
        label_values_or_threshold: Optional[Union[List[str], float]] = None,
        facet_values_or_threshold: Optional[Union[List[str], float]] = None,
        positive_label: Optional[Union[str, int, float]] = None,
        probability_threshold: float = 0.5,
        content_type: str = "text/csv",
        accept_type: str = "application/json",
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        wait: bool = True
    ) -> str:
        """
        Run bias analysis using SageMaker Clarify.
        
        Args:
            dataset_path: S3 path to the dataset
            model_name: Name of the model
            model_endpoint_name: Name of the model endpoint
            target_column: Name of the target column
            sensitive_columns: List of sensitive column names
            output_path: S3 path for output (optional)
            headers: List of column headers
            label_values_or_threshold: Label values or threshold for binary classification
            facet_values_or_threshold: Facet values or threshold for sensitive attributes
            positive_label: Positive label value for classification
            probability_threshold: Probability threshold for binary classification
            content_type: Content type of the dataset
            accept_type: Accept type for model response
            instance_type: Instance type for processing
            instance_count: Number of instances
            wait: Whether to wait for the job to complete
            
        Returns:
            S3 path to the bias analysis results
        """
        logger.info(f"Running bias analysis for model: {model_name}")
        
        # Set default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"s3://{self.default_bucket}/clarify/bias/{model_name}/{timestamp}"
        
        # Create Clarify processor
        clarify_processor = self.create_clarify_processor(
            instance_type=instance_type,
            instance_count=instance_count
        )
        
        # Create bias config
        bias_config = BiasConfig(
            label_values_or_threshold=label_values_or_threshold,
            facet_values_or_threshold=facet_values_or_threshold,
            label_name=target_column,
            facet_name=sensitive_columns,
            positive_label_value=positive_label
        )
        
        # Create model config
        model_config = ModelConfig(
            model_name=model_name,
            instance_type=instance_type,
            instance_count=instance_count,
            content_type=content_type,
            accept_type=accept_type,
            endpoint_name=model_endpoint_name
        )
        
        # Create data config
        data_config = DataConfig(
            s3_data_input_path=dataset_path,
            s3_output_path=output_path,
            label=target_column,
            headers=headers,
            dataset_type="text/csv"
        )
        
        # Run bias analysis
        clarify_processor.run_bias_analysis(
            data_config=data_config,
            bias_config=bias_config,
            model_config=model_config,
            wait=wait
        )
        
        logger.info(f"Bias analysis job submitted for model: {model_name}")
        logger.info(f"Results will be available at: {output_path}")
        
        return output_path
    
    def run_explainability_analysis(
        self,
        dataset_path: str,
        model_name: str,
        model_endpoint_name: str,
        features: List[str],
        target_column: str,
        output_path: Optional[str] = None,
        num_samples: int = 100,
        content_type: str = "text/csv",
        accept_type: str = "application/json",
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        wait: bool = True
    ) -> str:
        """
        Run explainability analysis using SageMaker Clarify.
        
        Args:
            dataset_path: S3 path to the dataset
            model_name: Name of the model
            model_endpoint_name: Name of the model endpoint
            features: List of feature names
            target_column: Name of the target column
            output_path: S3 path for output (optional)
            num_samples: Number of samples for SHAP
            content_type: Content type of the dataset
            accept_type: Accept type for model response
            instance_type: Instance type for processing
            instance_count: Number of instances
            wait: Whether to wait for the job to complete
            
        Returns:
            S3 path to the explainability analysis results
        """
        logger.info(f"Running explainability analysis for model: {model_name}")
        
        # Set default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"s3://{self.default_bucket}/clarify/explainability/{model_name}/{timestamp}"
        
        # Create Clarify processor
        clarify_processor = self.create_clarify_processor(
            instance_type=instance_type,
            instance_count=instance_count
        )
        
        # Create SHAP config
        shap_config = SHAPConfig(
            num_samples=num_samples,
            seed=42
        )
        
        # Create model config
        model_config = ModelConfig(
            model_name=model_name,
            instance_type=instance_type,
            instance_count=instance_count,
            content_type=content_type,
            accept_type=accept_type,
            endpoint_name=model_endpoint_name
        )
        
        # Create data config
        data_config = DataConfig(
            s3_data_input_path=dataset_path,
            s3_output_path=output_path,
            label=target_column,
            headers=features,
            dataset_type="text/csv"
        )
        
        # Run explainability analysis
        clarify_processor.run_explainability(
            data_config=data_config,
            model_config=model_config,
            explainability_config=shap_config,
            wait=wait
        )
        
        logger.info(f"Explainability analysis job submitted for model: {model_name}")
        logger.info(f"Results will be available at: {output_path}")
        
        return output_path
    
    def create_clarify_monitoring_config(
        self,
        baseline_dataset: str,
        target_column: str,
        features: List[str],
        sensitive_columns: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_runtime_in_seconds: int = 3600
    ) -> Tuple[str, ExplainabilityMonitoringConfig, Optional[BiasConfig]]:
        """
        Create a Clarify monitoring configuration.
        
        Args:
            baseline_dataset: S3 path to the baseline dataset
            target_column: Name of the target column
            features: List of feature names
            sensitive_columns: List of sensitive column names (optional)
            output_path: S3 path for output (optional)
            instance_type: Instance type for processing
            instance_count: Number of instances
            max_runtime_in_seconds: Maximum runtime in seconds
            
        Returns:
            Tuple of (baseline_job_output_path, explainability_monitoring_config, bias_config)
        """
        logger.info(f"Creating Clarify monitoring configuration")
        
        # Set default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"s3://{self.default_bucket}/clarify/baselines/{timestamp}"
        
        # Create Clarify processor
        clarify_processor = self.create_clarify_processor(
            instance_type=instance_type,
            instance_count=instance_count,
            max_runtime_in_seconds=max_runtime_in_seconds
        )
        
        # Create SHAP config
        shap_config = SHAPConfig(
            num_samples=100,
            seed=42
        )
        
        # Create bias config if sensitive columns are provided
        bias_config = None
        if sensitive_columns:
            bias_config = BiasConfig(
                label_name=target_column,
                facet_name=sensitive_columns
            )
        
        # Create data config
        data_config = DataConfig(
            s3_data_input_path=baseline_dataset,
            s3_output_path=output_path,
            label=target_column,
            headers=features,
            dataset_type="text/csv"
        )
        
        # Run baseline job for explainability
        clarify_processor.run_explainability(
            data_config=data_config,
            explainability_config=shap_config,
            wait=True
        )
        
        # Create explainability monitoring config
        explainability_monitoring_config = ExplainabilityMonitoringConfig(
            explainability_config=shap_config,
            features=features
        )
        
        logger.info(f"Clarify monitoring configuration created")
        logger.info(f"Baseline job output path: {output_path}")
        
        return output_path, explainability_monitoring_config, bias_config
    
    def setup_clarify_monitoring(
        self,
        endpoint_name: str,
        baseline_dataset: str,
        target_column: str,
        features: List[str],
        sensitive_columns: Optional[List[str]] = None,
        schedule_expression: str = "cron(0 0 ? * * *)",  # Daily
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        enable_cloudwatch_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Set up Clarify monitoring for an endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            baseline_dataset: S3 path to the baseline dataset
            target_column: Name of the target column
            features: List of feature names
            sensitive_columns: List of sensitive column names (optional)
            schedule_expression: Schedule expression for monitoring
            instance_type: Instance type for monitoring
            instance_count: Number of instances
            enable_cloudwatch_metrics: Whether to enable CloudWatch metrics
            
        Returns:
            Monitoring configuration details
        """
        logger.info(f"Setting up Clarify monitoring for endpoint: {endpoint_name}")
        
        # Create Clarify monitoring config
        baseline_job_output_path, explainability_config, bias_config = self.create_clarify_monitoring_config(
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns,
            instance_type=instance_type,
            instance_count=instance_count
        )
        
        # Create output path for monitoring results
        monitoring_output_path = f"s3://{self.default_bucket}/clarify/monitoring/{endpoint_name}"
        
        # Create model monitor
        monitor = ModelMonitor(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=30,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create monitoring schedule
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{endpoint_name}-clarify-monitor",
            endpoint_input=endpoint_name,
            output_s3_uri=monitoring_output_path,
            statistics=f"{baseline_job_output_path}/statistics.json",
            constraints=f"{baseline_job_output_path}/constraints.json",
            schedule_cron_expression=schedule_expression,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            monitoring_job_definition_name=f"{endpoint_name}-clarify-job-definition",
            explainability_config=explainability_config
        )
        
        # Create bias monitoring if sensitive columns are provided
        bias_monitor = None
        if bias_config:
            bias_monitor = ModelMonitor(
                role=self.execution_role,
                instance_count=instance_count,
                instance_type=instance_type,
                volume_size_in_gb=30,
                max_runtime_in_seconds=3600,
                sagemaker_session=self.sagemaker_session
            )
            
            bias_monitor.create_monitoring_schedule(
                monitor_schedule_name=f"{endpoint_name}-bias-monitor",
                endpoint_input=endpoint_name,
                output_s3_uri=f"{monitoring_output_path}/bias",
                statistics=f"{baseline_job_output_path}/statistics.json",
                constraints=f"{baseline_job_output_path}/constraints.json",
                schedule_cron_expression=schedule_expression,
                enable_cloudwatch_metrics=enable_cloudwatch_metrics,
                monitoring_job_definition_name=f"{endpoint_name}-bias-job-definition",
                bias_config=bias_config
            )
        
        logger.info(f"Clarify monitoring set up for endpoint: {endpoint_name}")
        logger.info(f"Monitoring schedule: {schedule_expression}")
        logger.info(f"Monitoring output path: {monitoring_output_path}")
        
        return {
            "endpoint_name": endpoint_name,
            "baseline_job_output_path": baseline_job_output_path,
            "monitoring_output_path": monitoring_output_path,
            "explainability_monitor": {
                "schedule_name": f"{endpoint_name}-clarify-monitor",
                "schedule_expression": schedule_expression
            },
            "bias_monitor": {
                "enabled": bias_config is not None,
                "schedule_name": f"{endpoint_name}-bias-monitor" if bias_config else None
            }
        }
    
    def create_clarify_pipeline_step(
        self,
        step_name: str,
        model_name: str,
        dataset_path: str,
        target_column: str,
        features: List[str],
        sensitive_columns: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        skip_check: bool = False
    ) -> ClarifyCheckStep:
        """
        Create a Clarify check step for a SageMaker pipeline.
        
        Args:
            step_name: Name of the step
            model_name: Name of the model
            dataset_path: S3 path to the dataset
            target_column: Name of the target column
            features: List of feature names
            sensitive_columns: List of sensitive column names (optional)
            output_path: S3 path for output (optional)
            instance_type: Instance type for processing
            instance_count: Number of instances
            skip_check: Whether to skip the check
            
        Returns:
            Configured ClarifyCheckStep
        """
        logger.info(f"Creating Clarify check step: {step_name}")
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"s3://{self.default_bucket}/clarify/pipeline/{model_name}"
        
        # Create SHAP config
        shap_config = SHAPConfig(
            num_samples=100,
            seed=42
        )
        
        # Create bias config if sensitive columns are provided
        bias_config = None
        if sensitive_columns:
            bias_config = BiasConfig(
                label_name=target_column,
                facet_name=sensitive_columns
            )
        
        # Create clarify check config
        clarify_check_config = ClarifyCheckConfig(
            data_config=DataConfig(
                s3_data_input_path=dataset_path,
                s3_output_path=output_path,
                label=target_column,
                headers=features,
                dataset_type="text/csv"
            ),
            model_config=ModelConfig(
                model_name=model_name,
                instance_type=instance_type,
                instance_count=instance_count,
                content_type="text/csv",
                accept_type="application/json"
            ),
            explainability_config=shap_config,
            bias_config=bias_config
        )
        
        # Create check job config
        check_job_config = CheckJobConfig(
            role=self.execution_role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=30,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.sagemaker_session
        )
        
        # Create clarify check step
        clarify_check_step = ClarifyCheckStep(
            name=step_name,
            clarify_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=skip_check,
            register_new_baseline=True,
            model_package_group_name=f"{model_name}-group"
        )
        
        logger.info(f"Clarify check step created: {step_name}")
        return clarify_check_step
    
    def analyze_clarify_results(
        self,
        clarify_output_path: str
    ) -> Dict[str, Any]:
        """
        Analyze Clarify results from an output path.
        
        Args:
            clarify_output_path: S3 path to Clarify output
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing Clarify results from: {clarify_output_path}")
        
        try:
            # Download analysis.json
            analysis_path = f"{clarify_output_path}/analysis.json"
            local_path = "/tmp/clarify_analysis.json"
            
            # Parse S3 URI
            bucket_name = clarify_output_path.replace("s3://", "").split("/")[0]
            key_prefix = "/".join(clarify_output_path.replace("s3://", "").split("/")[1:])
            
            self.s3_client.download_file(
                Bucket=bucket_name,
                Key=f"{key_prefix}/analysis.json",
                Filename=local_path
            )
            
            # Parse analysis file
            with open(local_path, 'r') as f:
                analysis_data = json.load(f)
            
            # Extract key insights
            insights = {
                "bias_metrics": {},
                "feature_importance": {},
                "analysis_time": analysis_data.get("analysis_time", ""),
                "version": analysis_data.get("version", "")
            }
            
            # Extract bias metrics if available
            if "bias_metrics" in analysis_data:
                for metric in analysis_data["bias_metrics"]:
                    metric_name = metric.get("name", "unknown")
                    metric_value = metric.get("value", 0)
                    insights["bias_metrics"][metric_name] = metric_value
            
            # Extract feature importance if available
            if "explanations" in analysis_data:
                explanations = analysis_data["explanations"]
                if "global_shap_values" in explanations:
                    global_shap = explanations["global_shap_values"]
                    for feature, value in global_shap.items():
                        insights["feature_importance"][feature] = value
            
            logger.info(f"Analysis completed for Clarify results")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing Clarify results: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def generate_explainability_report(
        self,
        clarify_output_path: str,
        report_path: Optional[str] = None,
        model_name: str = "model"
    ) -> str:
        """
        Generate an explainability report from Clarify results.
        
        Args:
            clarify_output_path: S3 path to Clarify output
            report_path: Path to save the report (optional)
            model_name: Name of the model
            
        Returns:
            Path to the generated report
        """
        logger.info(f"Generating explainability report for: {model_name}")
        
        try:
            # Set default report path if not provided
            if not report_path:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                report_path = f"s3://{self.default_bucket}/clarify/reports/{model_name}/{timestamp}/report.html"
            
            # Analyze Clarify results
            analysis_results = self.analyze_clarify_results(clarify_output_path)
            
            # Generate HTML report
            html_content = self._generate_html_report(analysis_results, model_name)
            
            # Upload report to S3
            # Parse S3 URI
            bucket_name = report_path.replace("s3://", "").split("/")[0]
            key_prefix = "/".join(report_path.replace("s3://", "").split("/")[1:])
            
            with open("/tmp/clarify_report.html", 'w') as f:
                f.write(html_content)
            
            self.s3_client.upload_file(
                Filename="/tmp/clarify_report.html",
                Bucket=bucket_name,
                Key=key_prefix
            )
            
            logger.info(f"Explainability report generated at: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating explainability report: {str(e)}")
            return f"Error: {str(e)}"
    
    def _generate_html_report(
        self,
        analysis_results: Dict[str, Any],
        model_name: str
    ) -> str:
        """
        Generate an HTML report from analysis results.
        
        Args:
            analysis_results: Analysis results
            model_name: Name of the model
            
        Returns:
            HTML content as string
        """
        # Extract data for the report
        bias_metrics = analysis_results.get("bias_metrics", {})
        feature_importance = analysis_results.get("feature_importance", {})
        
        # Sort feature importance by value
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explainability Report - {model_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #0066cc;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .card {{
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    border-bottom: 1px solid #eee;
                    padding: 10px 0;
                }}
                .metric-name {{
                    font-weight: bold;
                }}
                .feature-bar {{
                    height: 25px;
                    background-color: #0066cc;
                    margin: 5px 0;
                }}
                .feature-row {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .feature-name {{
                    width: 200px;
                }}
                .feature-value {{
                    width: 80px;
                    text-align: right;
                }}
                .feature-bar-container {{
                    flex-grow: 1;
                    background-color: #eee;
                    margin: 0 10px;
                }}
                .positive {{
                    background-color: #28a745;
                }}
                .negative {{
                    background-color: #dc3545;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Explainability Report</h1>
                <h2>{model_name}</h2>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h3>Feature Importance</h3>
                    <p>The chart below shows the SHAP values for each feature, indicating their contribution to the model's predictions.</p>
        """
        
        # Add feature importance bars
        if sorted_features:
            # Find the maximum absolute value for scaling
            max_abs_value = max([abs(value) for _, value in sorted_features])
            
            for feature, value in sorted_features:
                # Calculate width percentage (scaled to max value)
                width_pct = (abs(value) / max_abs_value) * 100 if max_abs_value > 0 else 0
                bar_class = "positive" if value >= 0 else "negative"
                
                html_content += f"""
                    <div class="feature-row">
                        <div class="feature-name">{feature}</div>
                        <div class="feature-bar-container">
                            <div class="feature-bar {bar_class}" style="width: {width_pct}%"></div>
                        </div>
                        <div class="feature-value">{value:.4f}</div>
                    </div>
                """
        else:
            html_content += "<p>No feature importance data available.</p>"
        
        # Add bias metrics
        html_content += """
                </div>
                
                <div class="card">
                    <h3>Bias Metrics</h3>
                    <p>The following metrics indicate potential bias in the model predictions.</p>
        """
        
        if bias_metrics:
            for metric_name, metric_value in bias_metrics.items():
                html_content += f"""
                    <div class="metric">
                        <div class="metric-name">{metric_name}</div>
                        <div class="metric-value">{metric_value:.4f}</div>
                    </div>
                """
        else:
            html_content += "<p>No bias metrics available.</p>"
        
        # Close HTML
        html_content += """
                </div>
                
                <div class="card">
                    <h3>Interpretation Guide</h3>
                    <p><strong>Feature Importance:</strong> Bars show the SHAP values for each feature. Positive values (green) indicate features that push predictions higher, while negative values (red) push predictions lower.</p>
                    <p><strong>Bias Metrics:</strong> These metrics help identify if the model treats different groups fairly. Values closer to zero generally indicate less bias.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def create_monitoring_dashboard(
        self,
        endpoint_name: str,
        clarify_output_path: str,
        dashboard_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a monitoring dashboard for Clarify results.
        
        Args:
            endpoint_name: Name of the endpoint
            clarify_output_path: S3 path to Clarify output
            dashboard_name: Name of the dashboard (optional)
            
        Returns:
            Dashboard configuration
        """
        logger.info(f"Creating monitoring dashboard for endpoint: {endpoint_name}")
        
        # Set default dashboard name if not provided
        if not dashboard_name:
            dashboard_name = f"{endpoint_name}-monitoring-dashboard"
        
        try:
            # Analyze Clarify results
            analysis_results = self.analyze_clarify_results(clarify_output_path)
            
            # Generate report
            report_path = self.generate_explainability_report(
                clarify_output_path=clarify_output_path,
                model_name=endpoint_name
            )
            
            # Create CloudWatch dashboard
            dashboard_body = {
                "widgets": [
                    {
                        "type": "text",
                        "x": 0,
                        "y": 0,
                        "width": 24,
                        "height": 2,
                        "properties": {
                            "markdown": f"# Model Monitoring Dashboard - {endpoint_name}\n"
                                       f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    },
                    {
                        "type": "text",
                        "x": 0,
                        "y": 2,
                        "width": 24,
                        "height": 2,
                        "properties": {
                            "markdown": f"## Explainability Report\n"
                                       f"[View Full Report]({report_path})"
                        }
                    }
                ]
            }
            
            # Add feature importance widget
            feature_importance = analysis_results.get("feature_importance", {})
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]  # Top 10 features
                
                # Create feature importance widget
                feature_widget = {
                    "type": "metric",
                    "x": 0,
                    "y": 4,
                    "width": 12,
                    "height": 8,
                    "properties": {
                        "title": "Feature Importance",
                        "view": "bar",
                        "metrics": [
                            ["Custom/FeatureImportance", feature, "Endpoint", endpoint_name]
                            for feature, _ in sorted_features
                        ],
                        "region": self.region,
                        "period": 300,
                        "stat": "Average"
                    }
                }
                
                dashboard_body["widgets"].append(feature_widget)
                
                # Publish feature importance metrics to CloudWatch
                for feature, value in sorted_features:
                    self.cloudwatch_client.put_metric_data(
                        Namespace="Custom/FeatureImportance",
                        MetricData=[
                            {
                                "MetricName": feature,
                                "Dimensions": [
                                    {
                                        "Name": "Endpoint",
                                        "Value": endpoint_name
                                    }
                                ],
                                "Value": value,
                                "Unit": "None"
                            }
                        ]
                    )
            
            # Add bias metrics widget
            bias_metrics = analysis_results.get("bias_metrics", {})
            if bias_metrics:
                # Create bias metrics widget
                bias_widget = {
                    "type": "metric",
                    "x": 12,
                    "y": 4,
                    "width": 12,
                    "height": 8,
                    "properties": {
                        "title": "Bias Metrics",
                        "view": "bar",
                        "metrics": [
                            ["Custom/BiasMetrics", metric, "Endpoint", endpoint_name]
                            for metric in bias_metrics.keys()
                        ],
                        "region": self.region,
                        "period": 300,
                        "stat": "Average"
                    }
                }
                
                dashboard_body["widgets"].append(bias_widget)
                
                # Publish bias metrics to CloudWatch
                for metric, value in bias_metrics.items():
                    self.cloudwatch_client.put_metric_data(
                        Namespace="Custom/BiasMetrics",
                        MetricData=[
                            {
                                "MetricName": metric,
                                "Dimensions": [
                                    {
                                        "Name": "Endpoint",
                                        "Value": endpoint_name
                                    }
                                ],
                                "Value": value,
                                "Unit": "None"
                            }
                        ]
                    )
            
            # Add data drift widget
            data_drift_widget = {
                "type": "metric",
                "x": 0,
                "y": 12,
                "width": 24,
                "height": 6,
                "properties": {
                    "title": "Data Drift Violations",
                    "view": "timeSeries",
                    "metrics": [
                        ["aws/sagemaker/Endpoints/data-metrics", "feature_baseline_drift_check_violations", 
                         "Endpoint", endpoint_name, "MonitoringSchedule", f"{endpoint_name}-data-quality-monitor"]
                    ],
                    "region": self.region,
                    "period": 3600,
                    "stat": "Maximum",
                    "annotations": {
                        "horizontal": [
                            {
                                "value": 0,
                                "label": "Threshold",
                                "color": "#ff0000"
                            }
                        ]
                    }
                }
            }
            
            dashboard_body["widgets"].append(data_drift_widget)
            
            # Create or update dashboard
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info(f"Monitoring dashboard created: {dashboard_name}")
            
            return {
                "dashboard_name": dashboard_name,
                "report_path": report_path,
                "feature_importance_count": len(feature_importance),
                "bias_metrics_count": len(bias_metrics),
                "region": self.region
            }
            
        except Exception as e:
            logger.error(f"Error creating monitoring dashboard: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }


def setup_clarify_for_endpoint(
    endpoint_name: str,
    baseline_dataset: str,
    target_column: str,
    features: List[str],
    sensitive_columns: Optional[List[str]] = None,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Set up SageMaker Clarify for an endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        target_column: Name of the target column
        features: List of feature names
        sensitive_columns: List of sensitive column names (optional)
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Configuration details
    """
    logger.info(f"Setting up SageMaker Clarify for endpoint: {endpoint_name}")
    
    try:
        # Create Clarify manager
        clarify_manager = ClarifyManager(
            aws_profile=aws_profile,
            region=region
        )
        
        # Run explainability analysis
        explainability_output_path = clarify_manager.run_explainability_analysis(
            dataset_path=baseline_dataset,
            model_name=endpoint_name,
            model_endpoint_name=endpoint_name,
            features=features,
            target_column=target_column,
            wait=True
        )
        
        # Run bias analysis if sensitive columns are provided
        bias_output_path = None
        if sensitive_columns:
            bias_output_path = clarify_manager.run_bias_analysis(
                dataset_path=baseline_dataset,
                model_name=endpoint_name,
                model_endpoint_name=endpoint_name,
                target_column=target_column,
                sensitive_columns=sensitive_columns,
                wait=True
            )
        
        # Set up Clarify monitoring
        monitoring_config = clarify_manager.setup_clarify_monitoring(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns
        )
        
        # Generate explainability report
        report_path = clarify_manager.generate_explainability_report(
            clarify_output_path=explainability_output_path,
            model_name=endpoint_name
        )
        
        # Create monitoring dashboard
        dashboard_config = clarify_manager.create_monitoring_dashboard(
            endpoint_name=endpoint_name,
            clarify_output_path=explainability_output_path
        )
        
        # Return configuration details
        return {
            "endpoint_name": endpoint_name,
            "explainability": {
                "output_path": explainability_output_path,
                "report_path": report_path
            },
            "bias": {
                "enabled": bias_output_path is not None,
                "output_path": bias_output_path
            },
            "monitoring": monitoring_config,
            "dashboard": dashboard_config
        }
        
    except Exception as e:
        logger.error(f"Error setting up SageMaker Clarify for endpoint: {str(e)}")
        raise