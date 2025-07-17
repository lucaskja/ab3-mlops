#!/usr/bin/env python3
"""
MLFlow SageMaker Integration Setup Script

This script sets up MLFlow tracking server on SageMaker, configures the tracking URI,
and initializes the experiment tracking environment.

Usage:
    python scripts/setup/setup_mlflow_sagemaker.py --create-server
    python scripts/setup/setup_mlflow_sagemaker.py --update-tracking-uri
"""

import os
import sys
import argparse
import logging
import json
import boto3
import sagemaker
from sagemaker.mlflow import MlflowServer
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from configs.project_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup MLFlow tracking server on SageMaker"
    )

    # Server management
    parser.add_argument(
        "--create-server",
        action="store_true",
        help="Create MLFlow tracking server on SageMaker",
    )
    parser.add_argument(
        "--update-tracking-uri",
        action="store_true",
        help="Update MLFlow tracking URI in project config",
    )
    parser.add_argument(
        "--delete-server", action="store_true", help="Delete MLFlow tracking server"
    )

    # Server configuration
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.large",
        help="SageMaker instance type for MLFlow server",
    )
    parser.add_argument(
        "--artifact-bucket", type=str, help="S3 bucket for MLFlow artifacts"
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="mlflow-artifacts",
        help="S3 prefix for MLFlow artifacts",
    )

    # AWS configuration
    parser.add_argument(
        "--aws-profile", type=str, default="ab", help="AWS profile to use"
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")

    return parser.parse_args()


def setup_aws_session(aws_profile, region):
    """Setup AWS session with specified profile and region."""
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    return session


def create_mlflow_server(args, project_config):
    """Create MLFlow tracking server on SageMaker."""
    logger.info("Creating MLFlow tracking server on SageMaker...")

    try:
        # Setup AWS session
        session = setup_aws_session(args.aws_profile, args.region)

        # Get SageMaker execution role
        execution_role = project_config["iam"]["roles"]["sagemaker_execution"]["arn"]

        # Get or create artifact bucket
        artifact_bucket = args.artifact_bucket
        if not artifact_bucket:
            artifact_bucket = project_config["mlflow"].get("artifact_bucket")
            if not artifact_bucket:
                artifact_bucket = (
                    f"{project_config['project']['name']}-mlflow-artifacts"
                )

        # Create SageMaker session
        sagemaker_session = sagemaker.Session(boto_session=session)

        # Create MLFlow server
        mlflow_server = MlflowServer(
            role=execution_role,
            instance_type=args.instance_type,
            instance_count=1,
            s3_bucket=artifact_bucket,
            s3_prefix=args.artifact_prefix,
            sagemaker_session=sagemaker_session,
        )

        # Deploy server
        mlflow_server.deploy()

        # Get tracking URI
        tracking_uri = mlflow_server.tracking_uri

        logger.info(f"MLFlow tracking server deployed successfully!")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"Artifact location: s3://{artifact_bucket}/{args.artifact_prefix}")

        # Update project config
        update_project_config(tracking_uri, artifact_bucket, args.artifact_prefix)

        return {
            "tracking_uri": tracking_uri,
            "artifact_bucket": artifact_bucket,
            "artifact_prefix": args.artifact_prefix,
        }

    except Exception as e:
        logger.error(f"Failed to create MLFlow server: {str(e)}")
        raise


def update_project_config(tracking_uri, artifact_bucket, artifact_prefix):
    """Update project config with MLFlow tracking URI and artifact location."""
    logger.info("Updating project config with MLFlow tracking information...")

    try:
        # Update environment variable for immediate use
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        # Update project_config.py
        config_path = os.path.join(project_root, "configs", "project_config.py")

        with open(config_path, "r") as f:
            content = f.read()

        # Update tracking URI
        if "MLFLOW_TRACKING_URI = " in content:
            content = content.replace(
                'MLFLOW_TRACKING_URI = "http://localhost:5000"',
                f'MLFLOW_TRACKING_URI = "{tracking_uri}"',
            )

        # Update artifact bucket
        if "MLFLOW_ARTIFACT_BUCKET = " in content:
            content = content.replace(
                f'MLFLOW_ARTIFACT_BUCKET = "{artifact_bucket}"',
                f'MLFLOW_ARTIFACT_BUCKET = "{artifact_bucket}"',
            )

        # Write updated content
        with open(config_path, "w") as f:
            f.write(content)

        logger.info(f"Project config updated with tracking URI: {tracking_uri}")

    except Exception as e:
        logger.error(f"Failed to update project config: {str(e)}")
        raise


def delete_mlflow_server(args, project_config):
    """Delete MLFlow tracking server from SageMaker."""
    logger.info("Deleting MLFlow tracking server from SageMaker...")

    try:
        # Setup AWS session
        session = setup_aws_session(args.aws_profile, args.region)

        # Get SageMaker client
        sagemaker_client = session.client("sagemaker")

        # Get endpoint name from project config
        tracking_uri = project_config["mlflow"]["tracking_uri"]

        # Extract endpoint name from tracking URI
        if "sagemaker" in tracking_uri and "amazonaws.com" in tracking_uri:
            parts = tracking_uri.split(".")
            for part in parts:
                if part.startswith("sagemaker-"):
                    endpoint_name = part.replace("sagemaker-", "")
                    break
            else:
                logger.error("Could not extract endpoint name from tracking URI")
                return
        else:
            logger.error("Tracking URI is not a SageMaker endpoint")
            return

        # Delete endpoint
        logger.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

        # Delete endpoint configuration
        logger.info(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

        # Delete model
        logger.info(f"Deleting model: {endpoint_name}")
        sagemaker_client.delete_model(ModelName=endpoint_name)

        logger.info("MLFlow tracking server deleted successfully!")

        # Reset tracking URI in project config
        update_project_config(
            "http://localhost:5000",
            project_config["mlflow"]["artifact_bucket"],
            args.artifact_prefix,
        )

    except Exception as e:
        logger.error(f"Failed to delete MLFlow server: {str(e)}")
        raise


def update_tracking_uri_only(args, project_config):
    """Update MLFlow tracking URI in project config without creating a server."""
    logger.info("Updating MLFlow tracking URI in project config...")

    try:
        # Get tracking URI from environment or args
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        if not tracking_uri:
            logger.error("MLFLOW_TRACKING_URI environment variable not set")
            logger.info("Please set MLFLOW_TRACKING_URI or create a server first")
            return

        # Update project config
        update_project_config(
            tracking_uri,
            project_config["mlflow"]["artifact_bucket"],
            args.artifact_prefix,
        )

        logger.info(f"Project config updated with tracking URI: {tracking_uri}")

    except Exception as e:
        logger.error(f"Failed to update tracking URI: {str(e)}")
        raise


def main():
    """Main function."""
    args = parse_arguments()

    logger.info("Starting MLFlow SageMaker integration setup...")

    try:
        # Get project config
        project_config = get_config()

        # Execute requested action
        if args.create_server:
            create_mlflow_server(args, project_config)

        elif args.update_tracking_uri:
            update_tracking_uri_only(args, project_config)

        elif args.delete_server:
            delete_mlflow_server(args, project_config)

        else:
            logger.error(
                "No action specified. Use --create-server, --update-tracking-uri, or --delete-server"
            )
            return 1

        logger.info("MLFlow SageMaker integration setup completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
