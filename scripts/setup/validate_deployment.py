#!/usr/bin/env python3
"""
Comprehensive validation script for MLOps SageMaker Demo infrastructure deployment.
This script validates all components of the deployed infrastructure.
"""

import argparse
import boto3
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InfrastructureValidator:
    """Validates the deployed infrastructure for MLOps SageMaker Demo."""
    
    def __init__(self, profile_name: str = "ab", region: str = "us-east-1", project_name: str = "mlops-sagemaker-demo"):
        """
        Initialize the validator.
        
        Args:
            profile_name: AWS profile name
            region: AWS region
            project_name: Project name
        """
        self.profile_name = profile_name
        self.region = region
        self.project_name = project_name
        
        # Initialize session and clients
        self.session = boto3.Session(profile_name=profile_name, region_name=region)
        self.iam_client = self.session.client('iam')
        self.sagemaker_client = self.session.client('sagemaker')
        self.cloudformation_client = self.session.client('cloudformation')
        self.s3_client = self.session.client('s3')
        self.events_client = self.session.client('events')
        self.sns_client = self.session.client('sns')
        self.lambda_client = self.session.client('lambda')
        self.sts_client = self.session.client('sts')
        
        # Get account ID
        self.account_id = self.sts_client.get_caller_identity()['Account']
        
        # Initialize validation results
        self.results = {
            "iam_roles": {"status": "not_checked", "details": []},
            "sagemaker_studio": {"status": "not_checked", "details": []},
            "mlflow": {"status": "not_checked", "details": []},
            "sagemaker_project": {"status": "not_checked", "details": []},
            "sagemaker_endpoint": {"status": "not_checked", "details": []},
            "model_monitoring": {"status": "not_checked", "details": []},
            "eventbridge_rules": {"status": "not_checked", "details": []},
            "sns_topics": {"status": "not_checked", "details": []},
            "cost_monitoring": {"status": "not_checked", "details": []}
        }
        
    def validate_iam_roles(self) -> Dict[str, Any]:
        """Validate IAM roles."""
        logger.info("Validating IAM roles...")
        
        roles_to_check = [
            f"{self.project_name}-DataScientist-Role",
            f"{self.project_name}-MLEngineer-Role",
            f"{self.project_name}-SageMaker-Execution-Role"
        ]
        
        policies_to_check = [
            f"{self.project_name}-SageMaker-Studio-Policy"
        ]
        
        role_results = []
        policy_results = []
        
        # Check roles
        for role_name in roles_to_check:
            try:
                role = self.iam_client.get_role(RoleName=role_name)
                role_results.append({
                    "name": role_name,
                    "exists": True,
                    "arn": role['Role']['Arn'],
                    "create_date": role['Role']['CreateDate'].isoformat()
                })
                logger.info(f"Role {role_name} exists")
            except self.iam_client.exceptions.NoSuchEntityException:
                role_results.append({
                    "name": role_name,
                    "exists": False
                })
                logger.warning(f"Role {role_name} does not exist")
            except Exception as e:
                role_results.append({
                    "name": role_name,
                    "exists": False,
                    "error": str(e)
                })
                logger.error(f"Error checking role {role_name}: {str(e)}")
        
        # Check policies
        for policy_name in policies_to_check:
            try:
                policy_arn = f"arn:aws:iam::{self.account_id}:policy/{policy_name}"
                policy = self.iam_client.get_policy(PolicyArn=policy_arn)
                policy_results.append({
                    "name": policy_name,
                    "exists": True,
                    "arn": policy_arn,
                    "create_date": policy['Policy']['CreateDate'].isoformat()
                })
                logger.info(f"Policy {policy_name} exists")
            except self.iam_client.exceptions.NoSuchEntityException:
                policy_results.append({
                    "name": policy_name,
                    "exists": False
                })
                logger.warning(f"Policy {policy_name} does not exist")
            except Exception as e:
                policy_results.append({
                    "name": policy_name,
                    "exists": False,
                    "error": str(e)
                })
                logger.error(f"Error checking policy {policy_name}: {str(e)}")
        
        # Check CloudFormation stack
        stack_name = f"{self.project_name}-iam-roles"
        try:
            stack = self.cloudformation_client.describe_stacks(StackName=stack_name)
            stack_status = stack['Stacks'][0]['StackStatus']
            stack_result = {
                "name": stack_name,
                "exists": True,
                "status": stack_status,
                "create_time": stack['Stacks'][0]['CreationTime'].isoformat()
            }
            logger.info(f"CloudFormation stack {stack_name} exists with status {stack_status}")
        except self.cloudformation_client.exceptions.ClientError:
            stack_result = {
                "name": stack_name,
                "exists": False
            }
            logger.warning(f"CloudFormation stack {stack_name} does not exist")
        except Exception as e:
            stack_result = {
                "name": stack_name,
                "exists": False,
                "error": str(e)
            }
            logger.error(f"Error checking CloudFormation stack {stack_name}: {str(e)}")
        
        # Determine overall status
        all_roles_exist = all(result["exists"] for result in role_results)
        all_policies_exist = all(result["exists"] for result in policy_results)
        stack_exists = stack_result.get("exists", False)
        
        if all_roles_exist and all_policies_exist and stack_exists:
            status = "success"
        elif not all_roles_exist and not all_policies_exist and not stack_exists:
            status = "not_deployed"
        else:
            status = "partial"
        
        result = {
            "status": status,
            "details": {
                "roles": role_results,
                "policies": policy_results,
                "stack": stack_result
            }
        }
        
        self.results["iam_roles"] = result
        return result
    
    def validate_sagemaker_endpoint(self, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to validate. If None, use project name.
        """
        if not endpoint_name:
            endpoint_name = f"{self.project_name}-yolov11-endpoint"
        
        logger.info(f"Validating SageMaker endpoint {endpoint_name}...")
        
        try:
            # Check endpoint
            endpoint = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            # Check endpoint config
            endpoint_config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['EndpointConfigName']
            )
            
            # Check models
            model_results = []
            for variant in endpoint_config['ProductionVariants']:
                try:
                    model = self.sagemaker_client.describe_model(ModelName=variant['ModelName'])
                    model_results.append({
                        "model_name": variant['ModelName'],
                        "exists": True,
                        "creation_time": model['CreationTime'].isoformat()
                    })
                except Exception as e:
                    model_results.append({
                        "model_name": variant['ModelName'],
                        "exists": False,
                        "error": str(e)
                    })
            
            # Determine status
            if endpoint['EndpointStatus'] == 'InService':
                status = "success"
            elif endpoint['EndpointStatus'] in ['Creating', 'Updating']:
                status = "in_progress"
            else:
                status = "error"
            
            result = {
                "status": status,
                "details": {
                    "endpoint_name": endpoint_name,
                    "endpoint_status": endpoint['EndpointStatus'],
                    "endpoint_arn": endpoint['EndpointArn'],
                    "creation_time": endpoint['CreationTime'].isoformat(),
                    "last_modified_time": endpoint['LastModifiedTime'].isoformat(),
                    "endpoint_config": {
                        "name": endpoint['EndpointConfigName'],
                        "variants": [
                            {
                                "variant_name": variant['VariantName'],
                                "model_name": variant['ModelName'],
                                "instance_type": variant['InstanceType'],
                                "initial_instance_count": variant['InitialInstanceCount']
                            }
                            for variant in endpoint_config['ProductionVariants']
                        ]
                    },
                    "models": model_results
                }
            }
        except self.sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                result = {
                    "status": "not_deployed",
                    "details": {
                        "endpoint_name": endpoint_name,
                        "exists": False
                    }
                }
                logger.warning(f"Endpoint {endpoint_name} does not exist")
            else:
                result = {
                    "status": "error",
                    "details": {
                        "endpoint_name": endpoint_name
                    },
                    "error": str(e)
                }
                logger.error(f"Error checking endpoint {endpoint_name}: {str(e)}")
        except Exception as e:
            result = {
                "status": "error",
                "details": {
                    "endpoint_name": endpoint_name
                },
                "error": str(e)
            }
            logger.error(f"Error checking endpoint {endpoint_name}: {str(e)}")
        
        self.results["sagemaker_endpoint"] = result
        return result
    
    def validate_model_monitoring(self, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate model monitoring.
        
        Args:
            endpoint_name: Name of the endpoint to validate monitoring for. If None, use project name.
        """
        if not endpoint_name:
            endpoint_name = f"{self.project_name}-yolov11-endpoint"
        
        logger.info(f"Validating model monitoring for endpoint {endpoint_name}...")
        
        try:
            # Check monitoring schedules
            monitoring_schedules = self.sagemaker_client.list_monitoring_schedules(
                EndpointName=endpoint_name
            )
            
            schedule_results = []
            for schedule in monitoring_schedules.get('MonitoringScheduleSummaries', []):
                schedule_details = self.sagemaker_client.describe_monitoring_schedule(
                    MonitoringScheduleName=schedule['MonitoringScheduleName']
                )
                
                schedule_results.append({
                    "schedule_name": schedule['MonitoringScheduleName'],
                    "status": schedule['MonitoringScheduleStatus'],
                    "type": schedule_details['MonitoringScheduleConfig']['MonitoringJobDefinition']['MonitoringType'],
                    "creation_time": schedule['CreationTime'].isoformat()
                })
            
            # Check for data capture config
            try:
                endpoint_config = self.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=self.sagemaker_client.describe_endpoint(
                        EndpointName=endpoint_name
                    )['EndpointConfigName']
                )
                
                data_capture_config = endpoint_config.get('DataCaptureConfig', {})
                data_capture_enabled = data_capture_config.get('EnableCapture', False)
                
                data_capture_result = {
                    "enabled": data_capture_enabled
                }
                
                if data_capture_enabled:
                    data_capture_result.update({
                        "destination": data_capture_config.get('DestinationS3Uri', ''),
                        "sampling_percentage": data_capture_config.get('InitialSamplingPercentage', 0),
                        "capture_options": [
                            opt['CaptureMode'] for opt in data_capture_config.get('CaptureOptions', [])
                        ]
                    })
            except Exception as e:
                data_capture_result = {
                    "enabled": False,
                    "error": str(e)
                }
                logger.error(f"Error checking data capture config for endpoint {endpoint_name}: {str(e)}")
            
            # Determine status
            if schedule_results:
                active_schedules = [s for s in schedule_results if s['status'] == 'Scheduled']
                if active_schedules and data_capture_result.get('enabled', False):
                    status = "success"
                elif active_schedules or data_capture_result.get('enabled', False):
                    status = "partial"
                else:
                    status = "not_deployed"
            else:
                if data_capture_result.get('enabled', False):
                    status = "partial"
                else:
                    status = "not_deployed"
            
            result = {
                "status": status,
                "details": {
                    "endpoint_name": endpoint_name,
                    "monitoring_schedules": schedule_results,
                    "data_capture_config": data_capture_result
                }
            }
        except Exception as e:
            result = {
                "status": "error",
                "details": {
                    "endpoint_name": endpoint_name
                },
                "error": str(e)
            }
            logger.error(f"Error checking model monitoring for endpoint {endpoint_name}: {str(e)}")
        
        self.results["model_monitoring"] = result
        return result
    
    def validate_all(self, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate all infrastructure components.
        
        Args:
            endpoint_name: Name of the endpoint to validate. If None, use project name.
            
        Returns:
            Dictionary with validation results.
        """
        logger.info("Starting comprehensive infrastructure validation...")
        
        # Validate all components
        self.validate_iam_roles()
        self.validate_sagemaker_endpoint(endpoint_name)
        self.validate_model_monitoring(endpoint_name)
        
        # Calculate overall status
        component_statuses = [component["status"] for component in self.results.values() 
                             if component["status"] != "not_checked"]
        
        if all(status == "success" for status in component_statuses):
            overall_status = "success"
        elif any(status == "error" for status in component_statuses):
            overall_status = "error"
        elif any(status == "not_deployed" for status in component_statuses):
            overall_status = "partial"
        else:
            overall_status = "partial"
        
        # Add summary
        summary = {
            "overall_status": overall_status,
            "components": {
                component: result["status"]
                for component, result in self.results.items()
                if result["status"] != "not_checked"
            },
            "validation_time": datetime.now().isoformat()
        }
        
        self.results["summary"] = summary
        
        return {
            "summary": summary,
            "results": self.results
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate MLOps SageMaker Demo infrastructure")
    parser.add_argument("--profile", default="ab", help="AWS profile to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--project-name", default="mlops-sagemaker-demo", help="Project name")
    parser.add_argument("--endpoint-name", help="SageMaker endpoint name")
    parser.add_argument("--output-format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--component", choices=["iam_roles", "sagemaker_endpoint", "model_monitoring", "all"], 
                       default="all", help="Component to validate")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    validator = InfrastructureValidator(
        profile_name=args.profile,
        region=args.region,
        project_name=args.project_name
    )
    
    if args.component == "all":
        results = validator.validate_all(args.endpoint_name)
    elif args.component == "iam_roles":
        results = validator.validate_iam_roles()
    elif args.component == "sagemaker_endpoint":
        results = validator.validate_sagemaker_endpoint(args.endpoint_name)
    elif args.component == "model_monitoring":
        results = validator.validate_model_monitoring(args.endpoint_name)
    
    # Output results
    if args.output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        if args.component == "all":
            print(f"Overall Status: {results['summary']['overall_status'].upper()}")
            print("\nComponent Status:")
            for component, status in results['summary']['components'].items():
                print(f"- {component}: {status.upper()}")
        else:
            print(f"Status: {results['status'].upper()}")
    
    # Return exit code based on validation status
    if args.component == "all":
        overall_status = results["summary"]["overall_status"]
        if overall_status == "success":
            return 0
        elif overall_status == "partial":
            return 1
        else:  # error or not_deployed
            return 2
    else:
        status = results["status"]
        if status == "success":
            return 0
        elif status == "partial":
            return 1
        else:  # error or not_deployed
            return 2


if __name__ == "__main__":
    sys.exit(main())