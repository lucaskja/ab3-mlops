"""
EventBridge Integration for SageMaker Pipeline and Model Monitoring Alerts

This module provides integration with AWS EventBridge for alerting and event-driven
workflows in the MLOps pipeline, including model drift detection, pipeline failures,
and automated retraining triggers.

Requirements addressed:
- 2.4: Send notifications through EventBridge when a pipeline fails
- 5.3: Trigger alerts through EventBridge when data drift is detected
"""

import os
import json
import logging
import boto3
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Import project configuration
from configs.project_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventBridgeIntegration:
    """
    Manages EventBridge integration for SageMaker Pipeline and Model Monitoring alerts.
    
    This class provides functionality for:
    - Creating EventBridge rules for pipeline and monitoring events
    - Publishing custom events to EventBridge
    - Setting up Lambda targets for event processing
    - Configuring SNS notifications for alerts
    """
    
    # Event source name for custom events
    CUSTOM_EVENT_SOURCE = "com.mlops.sagemaker.demo"
    
    # Event detail types
    EVENT_TYPES = {
        "PIPELINE_FAILURE": "PipelineExecutionFailure",
        "PIPELINE_SUCCESS": "PipelineExecutionSuccess",
        "MODEL_DRIFT_DETECTED": "ModelDriftDetected",
        "DATA_QUALITY_VIOLATION": "DataQualityViolation",
        "MODEL_QUALITY_VIOLATION": "ModelQualityViolation",
        "RETRAINING_TRIGGERED": "RetrainingTriggered",
        "COST_THRESHOLD_EXCEEDED": "CostThresholdExceeded"
    }
    
    def __init__(self, aws_profile: str = "ab", region: str = "us-east-1"):
        """
        Initialize the EventBridge integration.
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
        """
        self.aws_profile = aws_profile
        self.region = region
        
        # Initialize AWS clients
        self.session = boto3.Session(profile_name=aws_profile)
        self.events_client = self.session.client('events', region_name=region)
        self.lambda_client = self.session.client('lambda', region_name=region)
        self.sns_client = self.session.client('sns', region_name=region)
        
        # Get project configuration
        self.project_config = get_config()
        self.project_name = self.project_config['project']['name']
        
        # Set default event bus
        self.event_bus_name = "default"
        
        logger.info(f"EventBridge integration initialized for region: {region}")
    
    def create_rule_for_pipeline_failures(
        self,
        pipeline_name: str,
        target_lambda_arn: Optional[str] = None,
        target_sns_arn: Optional[str] = None,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for SageMaker Pipeline failures.
        
        Args:
            pipeline_name: Name of the SageMaker Pipeline
            target_lambda_arn: ARN of the Lambda function to trigger (optional)
            target_sns_arn: ARN of the SNS topic to notify (optional)
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-pipeline-failure-{pipeline_name}"
        
        # Create rule pattern for pipeline failures
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
            "detail": {
                "pipelineExecutionDisplayName": [{"prefix": pipeline_name}],
                "currentPipelineExecutionStatus": ["Failed"]
            }
        }
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Detect failures in SageMaker Pipeline: {pipeline_name}"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add targets
        targets = []
        
        if target_lambda_arn:
            targets.append({
                "Id": f"{rule_name}-lambda",
                "Arn": target_lambda_arn
            })
            
            # Add Lambda permission
            try:
                self.lambda_client.add_permission(
                    FunctionName=target_lambda_arn,
                    StatementId=f"{rule_name}-permission",
                    Action="lambda:InvokeFunction",
                    Principal="events.amazonaws.com",
                    SourceArn=rule_arn
                )
                logger.info(f"Added permission for EventBridge to invoke Lambda: {target_lambda_arn}")
            except self.lambda_client.exceptions.ResourceConflictException:
                logger.info(f"Permission already exists for Lambda: {target_lambda_arn}")
        
        if target_sns_arn:
            targets.append({
                "Id": f"{rule_name}-sns",
                "Arn": target_sns_arn
            })
        
        if targets:
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=targets
            )
            logger.info(f"Added {len(targets)} targets to rule: {rule_name}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "targets": targets
        }
    
    def create_rule_for_model_drift(
        self,
        endpoint_name: str,
        target_lambda_arn: Optional[str] = None,
        target_sns_arn: Optional[str] = None,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for model drift detection.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            target_lambda_arn: ARN of the Lambda function to trigger (optional)
            target_sns_arn: ARN of the SNS topic to notify (optional)
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-model-drift-{endpoint_name}"
        
        # Create rule pattern for model drift
        event_pattern = {
            "source": ["aws.sagemaker"],
            "detail-type": ["SageMaker Model Monitor Scheduled Rule Status Change"],
            "detail": {
                "monitoringScheduleName": [{"prefix": f"{endpoint_name}-"}],
                "monitoringExecutionStatus": ["Completed"],
                "monitoringScheduleStatus": ["Scheduled"]
            }
        }
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Detect model drift for SageMaker endpoint: {endpoint_name}"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add targets
        targets = []
        
        if target_lambda_arn:
            targets.append({
                "Id": f"{rule_name}-lambda",
                "Arn": target_lambda_arn,
                "InputTransformer": {
                    "InputPathsMap": {
                        "endpoint": "$.detail.monitoringScheduleName",
                        "status": "$.detail.monitoringExecutionStatus"
                    },
                    "InputTemplate": json.dumps({
                        "endpoint": "<endpoint>",
                        "status": "<status>",
                        "message": "Model drift detected for endpoint <endpoint>"
                    })
                }
            })
            
            # Add Lambda permission
            try:
                self.lambda_client.add_permission(
                    FunctionName=target_lambda_arn,
                    StatementId=f"{rule_name}-permission",
                    Action="lambda:InvokeFunction",
                    Principal="events.amazonaws.com",
                    SourceArn=rule_arn
                )
                logger.info(f"Added permission for EventBridge to invoke Lambda: {target_lambda_arn}")
            except self.lambda_client.exceptions.ResourceConflictException:
                logger.info(f"Permission already exists for Lambda: {target_lambda_arn}")
        
        if target_sns_arn:
            targets.append({
                "Id": f"{rule_name}-sns",
                "Arn": target_sns_arn,
                "InputTransformer": {
                    "InputPathsMap": {
                        "endpoint": "$.detail.monitoringScheduleName",
                        "status": "$.detail.monitoringExecutionStatus"
                    },
                    "InputTemplate": "Model drift detected for endpoint <endpoint> with status <status>"
                }
            })
        
        if targets:
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=targets
            )
            logger.info(f"Added {len(targets)} targets to rule: {rule_name}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "targets": targets
        }
    
    def create_sns_topic_for_alerts(
        self,
        topic_name: Optional[str] = None,
        email_subscriptions: Optional[List[str]] = None
    ) -> str:
        """
        Create an SNS topic for alerts.
        
        Args:
            topic_name: Name of the topic (optional)
            email_subscriptions: List of email addresses to subscribe (optional)
            
        Returns:
            SNS topic ARN
        """
        # Generate topic name if not provided
        if not topic_name:
            topic_name = f"{self.project_name}-alerts"
        
        # Create SNS topic
        response = self.sns_client.create_topic(Name=topic_name)
        topic_arn = response["TopicArn"]
        
        logger.info(f"Created SNS topic: {topic_name} (ARN: {topic_arn})")
        
        # Add email subscriptions
        if email_subscriptions:
            for email in email_subscriptions:
                self.sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol="email",
                    Endpoint=email
                )
                logger.info(f"Subscribed email to SNS topic: {email}")
        
        return topic_arn
    
    def publish_custom_event(
        self,
        detail_type: str,
        detail: Dict[str, Any],
        resources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publish a custom event to EventBridge.
        
        Args:
            detail_type: Type of event (use EVENT_TYPES constants)
            detail: Event details
            resources: List of related AWS resources (optional)
            
        Returns:
            Response from EventBridge
        """
        # Validate detail type
        if detail_type not in self.EVENT_TYPES.values():
            logger.warning(f"Unknown event type: {detail_type}. Using as-is.")
        
        # Create event entry
        event_entry = {
            "Source": self.CUSTOM_EVENT_SOURCE,
            "DetailType": detail_type,
            "Detail": json.dumps(detail),
            "EventBusName": self.event_bus_name
        }
        
        if resources:
            event_entry["Resources"] = resources
        
        # Put event
        response = self.events_client.put_events(
            Entries=[event_entry]
        )
        
        logger.info(f"Published custom event: {detail_type}")
        return response
    
    def create_rule_for_custom_events(
        self,
        detail_type: str,
        target_lambda_arn: Optional[str] = None,
        target_sns_arn: Optional[str] = None,
        rule_name: Optional[str] = None,
        event_pattern_detail: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule for custom events.
        
        Args:
            detail_type: Type of event to match (use EVENT_TYPES constants)
            target_lambda_arn: ARN of the Lambda function to trigger (optional)
            target_sns_arn: ARN of the SNS topic to notify (optional)
            rule_name: Name for the rule (optional)
            event_pattern_detail: Additional detail filters for the event pattern (optional)
            
        Returns:
            Rule details
        """
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-{detail_type.lower().replace('_', '-')}"
        
        # Create rule pattern for custom events
        event_pattern = {
            "source": [self.CUSTOM_EVENT_SOURCE],
            "detail-type": [detail_type]
        }
        
        if event_pattern_detail:
            event_pattern["detail"] = event_pattern_detail
        
        # Create rule
        response = self.events_client.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Rule for custom event: {detail_type}"
        )
        
        rule_arn = response["RuleArn"]
        logger.info(f"Created EventBridge rule: {rule_name} (ARN: {rule_arn})")
        
        # Add targets
        targets = []
        
        if target_lambda_arn:
            targets.append({
                "Id": f"{rule_name}-lambda",
                "Arn": target_lambda_arn
            })
            
            # Add Lambda permission
            try:
                self.lambda_client.add_permission(
                    FunctionName=target_lambda_arn,
                    StatementId=f"{rule_name}-permission",
                    Action="lambda:InvokeFunction",
                    Principal="events.amazonaws.com",
                    SourceArn=rule_arn
                )
                logger.info(f"Added permission for EventBridge to invoke Lambda: {target_lambda_arn}")
            except self.lambda_client.exceptions.ResourceConflictException:
                logger.info(f"Permission already exists for Lambda: {target_lambda_arn}")
        
        if target_sns_arn:
            targets.append({
                "Id": f"{rule_name}-sns",
                "Arn": target_sns_arn
            })
        
        if targets:
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=targets
            )
            logger.info(f"Added {len(targets)} targets to rule: {rule_name}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "event_pattern": event_pattern,
            "targets": targets
        }
    
    def create_retraining_trigger(
        self,
        endpoint_name: str,
        pipeline_name: str,
        drift_threshold: float = 0.1,
        target_lambda_arn: Optional[str] = None,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an EventBridge rule to trigger model retraining when drift is detected.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            pipeline_name: Name of the SageMaker Pipeline to trigger
            drift_threshold: Threshold for drift detection
            target_lambda_arn: ARN of the Lambda function to trigger (optional)
            rule_name: Name for the rule (optional)
            
        Returns:
            Rule details
        """
        # Generate rule name if not provided
        if not rule_name:
            rule_name = f"{self.project_name}-retraining-trigger-{endpoint_name}"
        
        # Create Lambda function if not provided
        if not target_lambda_arn:
            # This would typically be implemented separately
            logger.warning("No Lambda ARN provided for retraining trigger")
            return {"error": "Lambda ARN required for retraining trigger"}
        
        # Create rule for model drift
        rule_details = self.create_rule_for_model_drift(
            endpoint_name=endpoint_name,
            target_lambda_arn=target_lambda_arn,
            rule_name=rule_name
        )
        
        logger.info(f"Created retraining trigger for endpoint: {endpoint_name}")
        logger.info(f"Pipeline to trigger: {pipeline_name}")
        logger.info(f"Drift threshold: {drift_threshold}")
        
        return {
            "rule_details": rule_details,
            "endpoint_name": endpoint_name,
            "pipeline_name": pipeline_name,
            "drift_threshold": drift_threshold
        }
    
    def list_rules(self, name_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List EventBridge rules.
        
        Args:
            name_prefix: Prefix to filter rules by name (optional)
            
        Returns:
            List of rules
        """
        kwargs = {}
        if name_prefix:
            kwargs["NamePrefix"] = name_prefix
        
        response = self.events_client.list_rules(**kwargs)
        rules = response.get("Rules", [])
        
        logger.info(f"Found {len(rules)} EventBridge rules")
        return rules
    
    def delete_rule(self, rule_name: str) -> Dict[str, Any]:
        """
        Delete an EventBridge rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            Response from EventBridge
        """
        # List targets for the rule
        targets_response = self.events_client.list_targets_by_rule(Rule=rule_name)
        target_ids = [t["Id"] for t in targets_response.get("Targets", [])]
        
        # Remove targets if any
        if target_ids:
            self.events_client.remove_targets(
                Rule=rule_name,
                Ids=target_ids
            )
            logger.info(f"Removed {len(target_ids)} targets from rule: {rule_name}")
        
        # Delete rule
        response = self.events_client.delete_rule(Name=rule_name)
        logger.info(f"Deleted EventBridge rule: {rule_name}")
        
        return response


# Helper functions
def get_event_bridge_integration(aws_profile: str = "ab", region: str = "us-east-1") -> EventBridgeIntegration:
    """
    Get an EventBridge integration instance.
    
    Args:
        aws_profile: AWS profile to use
        region: AWS region
        
    Returns:
        EventBridgeIntegration instance
    """
    return EventBridgeIntegration(aws_profile=aws_profile, region=region)


def create_pipeline_failure_alert(
    pipeline_name: str,
    email_notifications: List[str],
    aws_profile: str = "ab"
) -> Dict[str, Any]:
    """
    Create an alert for pipeline failures.
    
    Args:
        pipeline_name: Name of the SageMaker Pipeline
        email_notifications: List of email addresses to notify
        aws_profile: AWS profile to use
        
    Returns:
        Alert configuration details
    """
    event_bridge = get_event_bridge_integration(aws_profile=aws_profile)
    
    # Create SNS topic
    topic_arn = event_bridge.create_sns_topic_for_alerts(
        topic_name=f"pipeline-failure-{pipeline_name}",
        email_subscriptions=email_notifications
    )
    
    # Create rule
    rule_details = event_bridge.create_rule_for_pipeline_failures(
        pipeline_name=pipeline_name,
        target_sns_arn=topic_arn
    )
    
    return {
        "pipeline_name": pipeline_name,
        "sns_topic_arn": topic_arn,
        "rule_details": rule_details
    }


def create_model_drift_alert(
    endpoint_name: str,
    email_notifications: List[str],
    aws_profile: str = "ab"
) -> Dict[str, Any]:
    """
    Create an alert for model drift.
    
    Args:
        endpoint_name: Name of the SageMaker endpoint
        email_notifications: List of email addresses to notify
        aws_profile: AWS profile to use
        
    Returns:
        Alert configuration details
    """
    event_bridge = get_event_bridge_integration(aws_profile=aws_profile)
    
    # Create SNS topic
    topic_arn = event_bridge.create_sns_topic_for_alerts(
        topic_name=f"model-drift-{endpoint_name}",
        email_subscriptions=email_notifications
    )
    
    # Create rule
    rule_details = event_bridge.create_rule_for_model_drift(
        endpoint_name=endpoint_name,
        target_sns_arn=topic_arn
    )
    
    return {
        "endpoint_name": endpoint_name,
        "sns_topic_arn": topic_arn,
        "rule_details": rule_details
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="EventBridge Integration for MLOps")
    parser.add_argument("--action", choices=["pipeline-alert", "drift-alert", "list-rules"], 
                      required=True, help="Action to perform")
    parser.add_argument("--name", help="Pipeline or endpoint name")
    parser.add_argument("--email", action="append", help="Email address for notifications")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    
    args = parser.parse_args()
    
    if args.action == "pipeline-alert":
        if not args.name:
            print("Error: --name (pipeline name) is required for pipeline-alert")
            exit(1)
        if not args.email:
            print("Error: --email is required for pipeline-alert")
            exit(1)
        
        result = create_pipeline_failure_alert(
            pipeline_name=args.name,
            email_notifications=args.email,
            aws_profile=args.profile
        )
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "drift-alert":
        if not args.name:
            print("Error: --name (endpoint name) is required for drift-alert")
            exit(1)
        if not args.email:
            print("Error: --email is required for drift-alert")
            exit(1)
        
        result = create_model_drift_alert(
            endpoint_name=args.name,
            email_notifications=args.email,
            aws_profile=args.profile
        )
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "list-rules":
        event_bridge = get_event_bridge_integration(aws_profile=args.profile)
        rules = event_bridge.list_rules(name_prefix=args.name)
        print(json.dumps(rules, indent=2, default=str))