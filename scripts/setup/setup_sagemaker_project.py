#!/usr/bin/env python3
"""
Script to set up a SageMaker Project for CI/CD.
"""

import argparse
import json
import boto3
import logging
import os
import zipfile
import tempfile
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up SageMaker Project for CI/CD')
    parser.add_argument('--profile', type=str, default='ab', help='AWS profile to use')
    parser.add_argument('--project-name', type=str, default='mlops-sagemaker-demo', help='Name of the SageMaker project')
    parser.add_argument('--project-description', type=str, default='MLOps SageMaker Demo for YOLOv11 object detection', help='Description of the project')
    parser.add_argument('--s3-bucket', type=str, default='lucaskle-ab3-project-pv', help='S3 bucket for project artifacts')
    return parser.parse_args()

def create_seed_code_zip(source_dir, output_zip):
    """Create a zip file from a directory."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

def setup_sagemaker_project(args):
    """Set up SageMaker Project for CI/CD."""
    # Set up AWS session
    session = boto3.Session(profile_name=args.profile)
    sagemaker_client = session.client('sagemaker')
    s3_client = session.client('s3')
    
    # Create temporary directory for zip files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create seed code zip files
        model_build_zip = os.path.join(temp_dir, 'model-build-seed.zip')
        model_deploy_zip = os.path.join(temp_dir, 'model-deploy-seed.zip')
        
        logger.info(f"Creating model build seed code zip: {model_build_zip}")
        create_seed_code_zip('configs/sagemaker_projects/seed_code/model_build', model_build_zip)
        
        logger.info(f"Creating model deploy seed code zip: {model_deploy_zip}")
        create_seed_code_zip('configs/sagemaker_projects/seed_code/model_deploy', model_deploy_zip)
        
        # Upload seed code zip files to S3
        logger.info(f"Uploading seed code zip files to S3 bucket: {args.s3_bucket}")
        s3_client.upload_file(model_build_zip, args.s3_bucket, 'sagemaker-projects/seed-code/model-build-seed.zip')
        s3_client.upload_file(model_deploy_zip, args.s3_bucket, 'sagemaker-projects/seed-code/model-deploy-seed.zip')
    
    # Upload template files to S3
    logger.info(f"Uploading template files to S3 bucket: {args.s3_bucket}")
    s3_client.upload_file(
        'configs/sagemaker_projects/templates/model_building/template.yaml',
        args.s3_bucket,
        'sagemaker-projects/templates/model-building-template.yaml'
    )
    s3_client.upload_file(
        'configs/sagemaker_projects/templates/model_deployment/template.yaml',
        args.s3_bucket,
        'sagemaker-projects/templates/model-deployment-template.yaml'
    )
    
    # Create service catalog product for model building
    logger.info("Creating service catalog product for model building")
    service_catalog_client = session.client('servicecatalog')
    
    # Check if portfolio exists
    portfolio_name = 'SageMaker Organization Templates'
    portfolios = service_catalog_client.list_portfolios()
    portfolio_id = None
    
    for portfolio in portfolios.get('PortfolioDetails', []):
        if portfolio['DisplayName'] == portfolio_name:
            portfolio_id = portfolio['Id']
            break
    
    if not portfolio_id:
        # Create portfolio
        logger.info(f"Creating portfolio: {portfolio_name}")
        portfolio_response = service_catalog_client.create_portfolio(
            DisplayName=portfolio_name,
            Description='SageMaker Organization Templates Portfolio',
            ProviderName='MLOps Team'
        )
        portfolio_id = portfolio_response['PortfolioDetail']['Id']
    
    # Create product for model building
    model_building_product_name = f"{args.project_name}-model-building"
    logger.info(f"Creating product: {model_building_product_name}")
    
    model_building_product_response = service_catalog_client.create_product(
        Name=model_building_product_name,
        Owner='MLOps Team',
        Description='SageMaker Model Building Pipeline',
        Distributor='MLOps Team',
        SupportDescription='Contact MLOps Team for support',
        ProductType='CLOUD_FORMATION_TEMPLATE',
        ProvisioningArtifactParameters={
            'Name': 'v1.0',
            'Description': 'Initial version',
            'Info': {
                'LoadTemplateFromURL': f"https://{args.s3_bucket}.s3.amazonaws.com/sagemaker-projects/templates/model-building-template.yaml"
            },
            'Type': 'CLOUD_FORMATION_TEMPLATE'
        }
    )
    
    model_building_product_id = model_building_product_response['ProductViewDetail']['ProductViewSummary']['ProductId']
    
    # Create product for model deployment
    model_deployment_product_name = f"{args.project_name}-model-deployment"
    logger.info(f"Creating product: {model_deployment_product_name}")
    
    model_deployment_product_response = service_catalog_client.create_product(
        Name=model_deployment_product_name,
        Owner='MLOps Team',
        Description='SageMaker Model Deployment Pipeline',
        Distributor='MLOps Team',
        SupportDescription='Contact MLOps Team for support',
        ProductType='CLOUD_FORMATION_TEMPLATE',
        ProvisioningArtifactParameters={
            'Name': 'v1.0',
            'Description': 'Initial version',
            'Info': {
                'LoadTemplateFromURL': f"https://{args.s3_bucket}.s3.amazonaws.com/sagemaker-projects/templates/model-deployment-template.yaml"
            },
            'Type': 'CLOUD_FORMATION_TEMPLATE'
        }
    )
    
    model_deployment_product_id = model_deployment_product_response['ProductViewDetail']['ProductViewSummary']['ProductId']
    
    # Associate products with portfolio
    logger.info(f"Associating products with portfolio: {portfolio_id}")
    service_catalog_client.associate_product_with_portfolio(
        ProductId=model_building_product_id,
        PortfolioId=portfolio_id
    )
    
    service_catalog_client.associate_product_with_portfolio(
        ProductId=model_deployment_product_id,
        PortfolioId=portfolio_id
    )
    
    # Create SageMaker project template
    logger.info(f"Creating SageMaker project template: {args.project_name}")
    
    try:
        sagemaker_client.create_project(
            ProjectName=args.project_name,
            ProjectDescription=args.project_description,
            ServiceCatalogProvisioningDetails={
                'ProductId': model_building_product_id,
                'ProvisioningArtifactId': model_building_product_response['ProvisioningArtifactDetail']['Id']
            },
            Tags=[
                {
                    'Key': 'Project',
                    'Value': args.project_name
                }
            ]
        )
        logger.info(f"SageMaker project created: {args.project_name}")
    except sagemaker_client.exceptions.ResourceInUse:
        logger.warning(f"SageMaker project already exists: {args.project_name}")
    
    logger.info("SageMaker Project setup completed")

def main():
    """Main function."""
    args = parse_args()
    setup_sagemaker_project(args)

if __name__ == '__main__':
    main()