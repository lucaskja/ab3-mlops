"""
SageMaker Model Monitor Integration with Pipeline

This module provides integration between SageMaker Pipelines and Model Monitor,
allowing automatic configuration of monitoring when a model is deployed.

Requirements addressed:
- 5.1: Automatic configuration of SageMaker Model Monitor when a model is deployed
- 5.2: Capture input data and predictions for monitoring
- 5.3: Trigger alerts through EventBridge when data drift is detected
- 5.4: Generate monitoring reports accessible through SageMaker Clarify
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
import boto3
import sagemaker
from sagemaker.lambda_helper import Lambda

from src.pipeline.model_monitor import ModelMonitoringManager, create_model_monitor_for_endpoint
from src.pipeline.clarify_integration import ClarifyManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_monitoring_lambda_step(
    lambda_function_name: str,
    endpoint_name: str,
    baseline_dataset: str,
    email_notifications: Optional[List[str]] = None,
    aws_profile: str = "ab",
    region: str = "us-east-1",
    role: Optional[str] = None,
    enable_model_quality: bool = False,
    problem_type: Optional[str] = None,
    ground_truth_attribute: Optional[str] = None,
    inference_attribute: Optional[str] = None
) -> Lambda:
    """
    Create a Lambda step for setting up model monitoring in a SageMaker pipeline.
    
    Args:
        lambda_function_name: Name of the Lambda function
        endpoint_name: Name of the endpoint to monitor
        baseline_dataset: S3 path to baseline dataset
        email_notifications: List of email addresses for notifications
        aws_profile: AWS profile to use
        region: AWS region
        role: IAM role for Lambda execution
        enable_model_quality: Whether to enable model quality monitoring
        problem_type: Problem type for model quality monitoring
        ground_truth_attribute: Name of ground truth attribute
        inference_attribute: Name of inference attribute
        
    Returns:
        Lambda step for SageMaker pipeline
    """
    logger.info(f"Creating Lambda step for model monitoring: {lambda_function_name}")
    
    # Create Lambda function code
    lambda_code = f"""
import json
import boto3
import time
import os
import sys

def lambda_handler(event, context):
    # Import required modules
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.model_monitor import (
        DataCaptureConfig,
        DefaultModelMonitor,
        ModelQualityMonitor,
        CronExpressionGenerator
    )
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    
    # Get endpoint name from event or use default
    endpoint_name = event.get('endpoint_name', '{endpoint_name}')
    baseline_dataset = event.get('baseline_dataset', '{baseline_dataset}')
    
    # Initialize AWS clients
    session = boto3.Session(region_name='{region}')
    sagemaker_client = session.client('sagemaker')
    s3_client = session.client('s3')
    cloudwatch_client = session.client('cloudwatch')
    events_client = session.client('events')
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session(
        boto_session=session,
        sagemaker_client=sagemaker_client
    )
    
    # Get execution role
    role = '{role}'
    
    # Create data capture config
    capture_prefix = f"data-capture/{{endpoint_name}}"
    default_bucket = sagemaker_session.default_bucket()
    
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100.0,
        destination_s3_uri=f"s3://{{default_bucket}}/{{capture_prefix}}",
        capture_options=["Input", "Output"],
        csv_content_types=["text/csv"],
        json_content_types=["application/json"]
    )
    
    # Update endpoint with data capture
    try:
        # Get endpoint configuration
        endpoint_config_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_config_response['EndpointConfigName']
        
        # Get production variants
        endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        production_variants = endpoint_config['ProductionVariants']
        
        # Create new endpoint config with data capture
        new_config_name = f"{{endpoint_config_name}}-with-capture-{{int(time.time())}}"
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=production_variants,
            DataCaptureConfig={{
                'EnableCapture': True,
                'InitialSamplingPercentage': 100,
                'DestinationS3Uri': data_capture_config.destination_s3_uri,
                'CaptureOptions': [
                    {{'CaptureMode': 'Input'}},
                    {{'CaptureMode': 'Output'}}
                ],
                'CaptureContentTypeHeader': {{
                    'CsvContentTypes': ['text/csv'],
                    'JsonContentTypes': ['application/json']
                }}
            }}
        )
        
        # Update endpoint with new config
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name
        )
        
        print(f"Endpoint {{endpoint_name}} updated with data capture configuration")
        print(f"Data capture destination: {{data_capture_config.destination_s3_uri}}")
    except Exception as e:
        print(f"Error updating endpoint with data capture: {{str(e)}}")
        return {{'statusCode': 500, 'body': json.dumps({{'error': str(e)}})}}
    
    # Create baseline constraints
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"s3://{{default_bucket}}/model-monitor/baselines/{{timestamp}}"
        
        # Create default model monitor
        model_monitor = DefaultModelMonitor(
            role=role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=sagemaker_session
        )
        
        # Suggest baseline constraints and statistics
        model_monitor.suggest_baseline(
            baseline_dataset=baseline_dataset,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_path,
            wait=True
        )
        
        # Get paths to generated files
        constraints_path = f"{{output_path}}/constraints.json"
        statistics_path = f"{{output_path}}/statistics.json"
        
        print(f"Baseline constraints created at: {{constraints_path}}")
        print(f"Baseline statistics created at: {{statistics_path}}")
        
        # Create monitoring schedule
        monitoring_output_path = f"s3://{{default_bucket}}/model-monitor/results/{{endpoint_name}}"
        
        model_monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{{endpoint_name}}-data-quality-monitor",
            endpoint_input=endpoint_name,
            record_preprocessor_script=None,
            post_analytics_processor_script=None,
            output_s3_uri=monitoring_output_path,
            statistics=statistics_path,
            constraints=constraints_path,
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            enable_cloudwatch_metrics=True
        )
        
        print(f"Data quality monitor created for endpoint: {{endpoint_name}}")
        
        # Create SNS topic for alerts
        sns_client = session.client('sns')
        topic_name = f"{{endpoint_name}}-monitor-alerts"
        response = sns_client.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        
        # Add email subscriptions if provided
        email_notifications = {email_notifications or []}
        for email in email_notifications:
            sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            print(f"Subscribed email to SNS topic: {{email}}")
        
        # Create CloudWatch alarm for violations
        alarm_name = f"{{endpoint_name}}-constraint-violations-alarm"
        cloudwatch_client.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='feature_baseline_drift_check_violations',
            Namespace='aws/sagemaker/Endpoints/data-metrics',
            Period=3600,
            Statistic='Maximum',
            Threshold=0,
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=f'Alarm for data drift violations on endpoint {{endpoint_name}}',
            Dimensions=[
                {{
                    'Name': 'Endpoint',
                    'Value': endpoint_name
                }},
                {{
                    'Name': 'MonitoringSchedule',
                    'Value': f"{{endpoint_name}}-data-quality-monitor"
                }}
            ]
        )
        
        print(f"CloudWatch alarm created: {{alarm_name}}")
        
        # Create EventBridge rule for drift detection
        rule_name = f"{{endpoint_name}}-drift-detection-rule"
        pattern = {{
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Monitor Violation"],
            "resources": [f"arn:aws:sagemaker:{region}:{{context.invoked_function_arn.split(':')[4]}}:endpoint/{{endpoint_name}}"],
            "detail": {{
                "MonitoringScheduleName": [f"{{endpoint_name}}-data-quality-monitor"],
                "ViolationsFound": [True]
            }}
        }}
        
        events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(pattern),
            State="ENABLED",
            Description=f"Drift detection rule for SageMaker endpoint {{endpoint_name}}"
        )
        
        # Add SNS target to rule
        events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {{
                    "Id": f"{{rule_name}}-sns-target",
                    "Arn": topic_arn
                }}
            ]
        )
        
        print(f"EventBridge rule created: {{rule_name}}")
        
        # Create model quality monitor if enabled
        enable_model_quality = {str(enable_model_quality).lower()}
        if enable_model_quality == 'true':
            problem_type = '{problem_type or ""}'
            ground_truth_attribute = '{ground_truth_attribute or ""}'
            inference_attribute = '{inference_attribute or ""}'
            
            if problem_type and ground_truth_attribute and inference_attribute:
                # Create model quality monitor
                quality_monitor = ModelQualityMonitor(
                    role=role,
                    instance_count=1,
                    instance_type='ml.m5.xlarge',
                    volume_size_in_gb=20,
                    max_runtime_in_seconds=1800,
                    sagemaker_session=sagemaker_session
                )
                
                # Create baseline for model quality
                baseline_job_name = f"{{endpoint_name}}-quality-baseline-{{int(time.time())}}"
                quality_output_path = f"s3://{{default_bucket}}/model-monitor/quality-baselines/{{endpoint_name}}"
                
                quality_monitor.suggest_baseline(
                    baseline_dataset=baseline_dataset,
                    dataset_format=DatasetFormat.csv(header=True),
                    problem_type=problem_type,
                    inference_attribute=inference_attribute,
                    ground_truth_attribute=ground_truth_attribute,
                    output_s3_uri=quality_output_path,
                    wait=True,
                    job_name=baseline_job_name
                )
                
                # Create monitoring schedule
                quality_monitoring_output_path = f"s3://{{default_bucket}}/model-monitor/quality-results/{{endpoint_name}}"
                
                quality_monitor.create_monitoring_schedule(
                    monitor_schedule_name=f"{{endpoint_name}}-model-quality-monitor",
                    endpoint_input=endpoint_name,
                    record_preprocessor_script=None,
                    post_analytics_processor_script=None,
                    output_s3_uri=quality_monitoring_output_path,
                    problem_type=problem_type,
                    ground_truth_input=None,
                    inference_attribute=inference_attribute,
                    ground_truth_attribute=ground_truth_attribute,
                    schedule_cron_expression=CronExpressionGenerator.daily(),
                    enable_cloudwatch_metrics=True
                )
                
                print(f"Model quality monitor created for endpoint: {{endpoint_name}}")
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'endpoint_name': endpoint_name,
                'data_capture_config': {{
                    'destination_s3_uri': data_capture_config.destination_s3_uri,
                    'sampling_percentage': 100.0
                }},
                'baseline_constraints_path': constraints_path,
                'data_quality_monitor': {{
                    'schedule_name': f"{{endpoint_name}}-data-quality-monitor",
                    'schedule_expression': 'hourly'
                }},
                'model_quality_monitor': {{
                    'enabled': enable_model_quality,
                    'schedule_name': f"{{endpoint_name}}-model-quality-monitor" if enable_model_quality == 'true' else None
                }},
                'alerts': {{
                    'sns_topic_arn': topic_arn,
                    'alarm_name': alarm_name,
                    'rule_name': rule_name
                }}
            }})
        }}
    except Exception as e:
        print(f"Error setting up model monitoring: {{str(e)}}")
        return {{'statusCode': 500, 'body': json.dumps({{'error': str(e)}})}}
    """
    
    # Create Lambda function
    lambda_helper = Lambda(
        function_name=lambda_function_name,
        execution_role_arn=role,
        script=lambda_code,
        handler="index.lambda_handler",
        timeout=900,  # 15 minutes
        memory_size=512
    )
    
    logger.info(f"Lambda step created for model monitoring: {lambda_function_name}")
    return lambda_helper


def add_monitoring_to_pipeline_endpoint(
    pipeline_name: str,
    endpoint_name: str,
    baseline_dataset: str,
    aws_profile: str = "ab",
    region: str = "us-east-1",
    email_notifications: Optional[List[str]] = None
) -> None:
    """
    Add model monitoring to an existing SageMaker pipeline endpoint.
    
    Args:
        pipeline_name: Name of the pipeline
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        aws_profile: AWS profile to use
        region: AWS region
        email_notifications: List of email addresses for notifications
    """
    logger.info(f"Adding model monitoring to pipeline endpoint: {endpoint_name}")
    
    try:
        # Create model monitoring
        monitoring_config = create_model_monitor_for_endpoint(
            endpoint_name=endpoint_name,
            baseline_dataset=baseline_dataset,
            aws_profile=aws_profile,
            region=region,
            email_notifications=email_notifications
        )
        
        logger.info(f"Model monitoring added to pipeline endpoint: {endpoint_name}")
        logger.info(f"Monitoring configuration: {json.dumps(monitoring_config, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error adding model monitoring to pipeline endpoint: {str(e)}")
        raise


def add_clarify_to_endpoint(
    endpoint_name: str,
    baseline_dataset: str,
    target_column: str,
    features: List[str],
    sensitive_columns: Optional[List[str]] = None,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Add SageMaker Clarify explainability to an endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        target_column: Name of the target column
        features: List of feature names
        sensitive_columns: List of sensitive column names (optional)
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Clarify configuration details
    """
    logger.info(f"Adding SageMaker Clarify to endpoint: {endpoint_name}")
    
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
        clarify_config = {
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
        
        logger.info(f"SageMaker Clarify added to endpoint: {endpoint_name}")
        logger.info(f"Clarify configuration: {json.dumps(clarify_config, indent=2)}")
        
        return clarify_config
        
    except Exception as e:
        logger.error(f"Error adding SageMaker Clarify to endpoint: {str(e)}")
        raise


def create_clarify_lambda_step(
    lambda_function_name: str,
    endpoint_name: str,
    baseline_dataset: str,
    target_column: str,
    features: List[str],
    sensitive_columns: Optional[List[str]] = None,
    aws_profile: str = "ab",
    region: str = "us-east-1",
    role: Optional[str] = None
) -> Lambda:
    """
    Create a Lambda step for setting up SageMaker Clarify in a SageMaker pipeline.
    
    Args:
        lambda_function_name: Name of the Lambda function
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        target_column: Name of the target column
        features: List of feature names
        sensitive_columns: List of sensitive column names (optional)
        aws_profile: AWS profile to use
        region: AWS region
        role: IAM role for Lambda execution
        
    Returns:
        Lambda step for SageMaker pipeline
    """
    logger.info(f"Creating Lambda step for SageMaker Clarify: {lambda_function_name}")
    
    # Convert features and sensitive columns to strings for Lambda code
    features_str = json.dumps(features)
    sensitive_columns_str = json.dumps(sensitive_columns) if sensitive_columns else "None"
    
    # Create Lambda function code
    lambda_code = f"""
import json
import boto3
import time
import os
import sys

def lambda_handler(event, context):
    # Import required modules
    import boto3
    import sagemaker
    from sagemaker import get_execution_role, clarify
    from sagemaker.clarify import (
        BiasConfig,
        DataConfig,
        ModelConfig,
        SHAPConfig
    )
    from sagemaker.model_monitor import (
        ClarifyMonitoringConfig,
        ModelMonitor,
        ExplainabilityMonitoringConfig
    )
    
    # Get endpoint name from event or use default
    endpoint_name = event.get('endpoint_name', '{endpoint_name}')
    baseline_dataset = event.get('baseline_dataset', '{baseline_dataset}')
    target_column = event.get('target_column', '{target_column}')
    features = event.get('features', {features_str})
    sensitive_columns = event.get('sensitive_columns', {sensitive_columns_str})
    
    # Initialize AWS clients
    session = boto3.Session(region_name='{region}')
    sagemaker_client = session.client('sagemaker')
    s3_client = session.client('s3')
    cloudwatch_client = session.client('cloudwatch')
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session(
        boto_session=session,
        sagemaker_client=sagemaker_client
    )
    
    # Get execution role
    role = '{role}'
    
    # Get default S3 bucket
    default_bucket = sagemaker_session.default_bucket()
    
    try:
        # Create Clarify processor
        clarify_processor = clarify.SageMakerClarifyProcessor(
            role=role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            sagemaker_session=sagemaker_session
        )
        
        # Create timestamp for output paths
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Run explainability analysis
        explainability_output_path = f"s3://{{default_bucket}}/clarify/explainability/{{endpoint_name}}/{{timestamp}}"
        
        # Create SHAP config
        shap_config = SHAPConfig(
            num_samples=100,
            seed=42
        )
        
        # Create model config
        model_config = ModelConfig(
            model_name=endpoint_name,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            content_type='text/csv',
            accept_type='application/json',
            endpoint_name=endpoint_name
        )
        
        # Create data config
        data_config = DataConfig(
            s3_data_input_path=baseline_dataset,
            s3_output_path=explainability_output_path,
            label=target_column,
            headers=features,
            dataset_type='text/csv'
        )
        
        # Run explainability analysis
        clarify_processor.run_explainability(
            data_config=data_config,
            model_config=model_config,
            explainability_config=shap_config,
            wait=True
        )
        
        print(f"Explainability analysis completed for endpoint: {{endpoint_name}}")
        print(f"Results available at: {{explainability_output_path}}")
        
        # Run bias analysis if sensitive columns are provided
        bias_output_path = None
        if sensitive_columns:
            bias_output_path = f"s3://{{default_bucket}}/clarify/bias/{{endpoint_name}}/{{timestamp}}"
            
            # Create bias config
            bias_config = BiasConfig(
                label_name=target_column,
                facet_name=sensitive_columns
            )
            
            # Run bias analysis
            clarify_processor.run_bias_analysis(
                data_config=data_config,
                model_config=model_config,
                bias_config=bias_config,
                wait=True
            )
            
            print(f"Bias analysis completed for endpoint: {{endpoint_name}}")
            print(f"Results available at: {{bias_output_path}}")
        
        # Set up Clarify monitoring
        monitoring_output_path = f"s3://{{default_bucket}}/clarify/monitoring/{{endpoint_name}}"
        
        # Create explainability monitoring config
        explainability_monitoring_config = ExplainabilityMonitoringConfig(
            explainability_config=shap_config,
            features=features
        )
        
        # Create model monitor
        monitor = ModelMonitor(
            role=role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=30,
            max_runtime_in_seconds=3600,
            sagemaker_session=sagemaker_session
        )
        
        # Create monitoring schedule
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{{endpoint_name}}-clarify-monitor",
            endpoint_input=endpoint_name,
            output_s3_uri=monitoring_output_path,
            statistics=f"{{explainability_output_path}}/statistics.json",
            constraints=f"{{explainability_output_path}}/constraints.json",
            schedule_cron_expression="cron(0 0 ? * * *)",  # Daily
            enable_cloudwatch_metrics=True,
            monitoring_job_definition_name=f"{{endpoint_name}}-clarify-job-definition",
            explainability_config=explainability_monitoring_config
        )
        
        print(f"Clarify monitoring set up for endpoint: {{endpoint_name}}")
        
        # Generate HTML report
        report_path = f"s3://{{default_bucket}}/clarify/reports/{{endpoint_name}}/{{timestamp}}/report.html"
        
        # Download analysis.json
        local_analysis_path = "/tmp/clarify_analysis.json"
        s3_client.download_file(
            Bucket=explainability_output_path.replace("s3://", "").split("/")[0],
            Key="/".join(explainability_output_path.replace("s3://", "").split("/")[1:]) + "/analysis.json",
            Filename=local_analysis_path
        )
        
        # Parse analysis file
        with open(local_analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Extract feature importance
        feature_importance = {{}}
        if "explanations" in analysis_data:
            explanations = analysis_data["explanations"]
            if "global_shap_values" in explanations:
                global_shap = explanations["global_shap_values"]
                for feature, value in global_shap.items():
                    feature_importance[feature] = value
        
        # Create CloudWatch dashboard
        dashboard_name = f"{{endpoint_name}}-clarify-dashboard"
        
        # Create dashboard widgets
        widgets = [
            {{
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 24,
                "height": 2,
                "properties": {{
                    "markdown": f"# Model Explainability Dashboard - {{endpoint_name}}\\nLast updated: {{time.strftime('%Y-%m-%d %H:%M:%S')}}"
                }}
            }},
            {{
                "type": "text",
                "x": 0,
                "y": 2,
                "width": 24,
                "height": 2,
                "properties": {{
                    "markdown": f"## Explainability Report\\n[View Full Report]({{report_path}})"
                }}
            }}
        ]
        
        # Add feature importance metrics to CloudWatch
        for feature, value in feature_importance.items():
            cloudwatch_client.put_metric_data(
                Namespace="Custom/FeatureImportance",
                MetricData=[
                    {{
                        "MetricName": feature,
                        "Dimensions": [
                            {{
                                "Name": "Endpoint",
                                "Value": endpoint_name
                            }}
                        ],
                        "Value": value,
                        "Unit": "None"
                    }}
                ]
            )
        
        # Add feature importance widget
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]  # Top 10 features
            
            feature_widget = {{
                "type": "metric",
                "x": 0,
                "y": 4,
                "width": 24,
                "height": 8,
                "properties": {{
                    "title": "Feature Importance",
                    "view": "bar",
                    "metrics": [
                        ["Custom/FeatureImportance", feature, "Endpoint", endpoint_name]
                        for feature, _ in sorted_features
                    ],
                    "region": "{region}",
                    "period": 300,
                    "stat": "Average"
                }}
            }}
            
            widgets.append(feature_widget)
        
        # Add data drift widget
        data_drift_widget = {{
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 24,
            "height": 6,
            "properties": {{
                "title": "Data Drift Violations",
                "view": "timeSeries",
                "metrics": [
                    ["aws/sagemaker/Endpoints/data-metrics", "feature_baseline_drift_check_violations", 
                     "Endpoint", endpoint_name, "MonitoringSchedule", f"{{endpoint_name}}-data-quality-monitor"]
                ],
                "region": "{region}",
                "period": 3600,
                "stat": "Maximum",
                "annotations": {{
                    "horizontal": [
                        {{
                            "value": 0,
                            "label": "Threshold",
                            "color": "#ff0000"
                        }}
                    ]
                }}
            }}
        }}
        
        widgets.append(data_drift_widget)
        
        # Create or update dashboard
        cloudwatch_client.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps({{"widgets": widgets}})
        )
        
        print(f"CloudWatch dashboard created: {{dashboard_name}}")
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'endpoint_name': endpoint_name,
                'explainability': {{
                    'output_path': explainability_output_path,
                    'report_path': report_path
                }},
                'bias': {{
                    'enabled': sensitive_columns is not None,
                    'output_path': bias_output_path
                }},
                'monitoring': {{
                    'schedule_name': f"{{endpoint_name}}-clarify-monitor"
                }},
                'dashboard': {{
                    'name': dashboard_name
                }}
            }})
        }}
    except Exception as e:
        print(f"Error setting up SageMaker Clarify: {{str(e)}}")
        return {{'statusCode': 500, 'body': json.dumps({{'error': str(e)}})}}
    """
    
    # Create Lambda function
    lambda_helper = Lambda(
        function_name=lambda_function_name,
        execution_role_arn=role,
        script=lambda_code,
        handler="index.lambda_handler",
        timeout=900,  # 15 minutes
        memory_size=512
    )
    
    logger.info(f"Lambda step created for SageMaker Clarify: {lambda_function_name}")
    return lambda_helper


def add_clarify_to_pipeline(
    pipeline_name: str,
    endpoint_name: str,
    baseline_dataset: str,
    target_column: str,
    features: List[str],
    sensitive_columns: Optional[List[str]] = None,
    aws_profile: str = "ab",
    region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Add SageMaker Clarify to a SageMaker pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        endpoint_name: Name of the endpoint
        baseline_dataset: S3 path to baseline dataset
        target_column: Name of the target column
        features: List of feature names
        sensitive_columns: List of sensitive column names (optional)
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        Clarify pipeline step configuration
    """
    logger.info(f"Adding SageMaker Clarify to pipeline: {pipeline_name}")
    
    try:
        # Create Clarify manager
        clarify_manager = ClarifyManager(
            aws_profile=aws_profile,
            region=region
        )
        
        # Create Clarify pipeline step
        clarify_step = clarify_manager.create_clarify_pipeline_step(
            step_name=f"{pipeline_name}-clarify-step",
            model_name=endpoint_name,
            dataset_path=baseline_dataset,
            target_column=target_column,
            features=features,
            sensitive_columns=sensitive_columns
        )
        
        # Return step configuration
        step_config = {
            "pipeline_name": pipeline_name,
            "step_name": clarify_step.name,
            "model_name": endpoint_name,
            "baseline_dataset": baseline_dataset
        }
        
        logger.info(f"SageMaker Clarify added to pipeline: {pipeline_name}")
        logger.info(f"Clarify step configuration: {json.dumps(step_config, indent=2)}")
        
        return step_config
        
    except Exception as e:
        logger.error(f"Error adding SageMaker Clarify to pipeline: {str(e)}")
        raise