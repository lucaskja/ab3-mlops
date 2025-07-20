#!/usr/bin/env python3
"""
Deploy Core Notebooks to SageMaker Studio

This script deploys the core notebooks to SageMaker Studio for Data Scientists and ML Engineers.
It creates the appropriate directory structure and sets up the notebooks for each user role.
"""

import boto3
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
AWS_PROFILE = 'ab'
REGION = 'us-east-1'
DOMAIN_NAME = 'mlops-sagemaker-demo'  # Update with your actual domain name
BUCKET_NAME = 'lucaskle-ab3-project-pv'

# Notebook configurations for each role
NOTEBOOK_CONFIGS = {
    'data-scientist': {
        'notebooks': [
            'notebooks/data-scientist-core-enhanced.ipynb',
            'notebooks/create_labeling_job.ipynb'  # From data-labeling directory
        ],
        'target_directory': '/home/sagemaker-user/data-scientist-notebooks/',
        'description': 'Core notebooks for Data Scientists'
    },
    'ml-engineer': {
        'notebooks': [
            'notebooks/ml-engineer-core-enhanced.ipynb'
        ],
        'target_directory': '/home/sagemaker-user/ml-engineer-notebooks/',
        'description': 'Core notebooks for ML Engineers'
    }
}

def setup_aws_session():
    """Set up AWS session with the specified profile"""
    try:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        sagemaker_client = session.client('sagemaker')
        s3_client = session.client('s3')
        
        # Test the connection
        sagemaker_client.list_domains()
        
        print(f"✅ Successfully connected to AWS using profile '{AWS_PROFILE}' in region '{REGION}'")
        return session, sagemaker_client, s3_client
        
    except Exception as e:
        print(f"❌ Error connecting to AWS: {str(e)}")
        print(f"Please ensure AWS CLI is configured with profile '{AWS_PROFILE}'")
        sys.exit(1)

def get_sagemaker_domain_info(sagemaker_client):
    """Get SageMaker domain information"""
    try:
        # List domains
        response = sagemaker_client.list_domains()
        domains = response.get('Domains', [])
        
        if not domains:
            print("❌ No SageMaker domains found")
            print("Please create a SageMaker domain first")
            sys.exit(1)
        
        # Find the target domain or use the first one
        target_domain = None
        for domain in domains:
            if domain['DomainName'] == DOMAIN_NAME:
                target_domain = domain
                break
        
        if not target_domain:
            target_domain = domains[0]
            print(f"⚠️  Domain '{DOMAIN_NAME}' not found, using '{target_domain['DomainName']}'")
        
        domain_id = target_domain['DomainId']
        
        # Get user profiles
        response = sagemaker_client.list_user_profiles(DomainIdEquals=domain_id)
        user_profiles = response.get('UserProfiles', [])
        
        print(f"📋 Found SageMaker domain: {target_domain['DomainName']} (ID: {domain_id})")
        print(f"👥 Found {len(user_profiles)} user profiles:")
        for profile in user_profiles:
            print(f"   - {profile['UserProfileName']}")
        
        return domain_id, user_profiles
        
    except Exception as e:
        print(f"❌ Error getting SageMaker domain info: {str(e)}")
        sys.exit(1)

def upload_notebooks_to_s3(s3_client, notebook_configs):
    """Upload notebooks to S3 for distribution"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"sagemaker-studio-notebooks/{timestamp}/"
        
        uploaded_notebooks = {}
        
        for role, config in notebook_configs.items():
            print(f"📤 Uploading notebooks for {role}...")
            
            role_notebooks = []
            for notebook_path in config['notebooks']:
                if os.path.exists(notebook_path):
                    # Upload notebook to S3
                    notebook_name = os.path.basename(notebook_path)
                    s3_key = f"{s3_prefix}{role}/{notebook_name}"
                    
                    s3_client.upload_file(notebook_path, BUCKET_NAME, s3_key)
                    
                    s3_url = f"s3://{BUCKET_NAME}/{s3_key}"
                    role_notebooks.append({
                        'local_path': notebook_path,
                        'notebook_name': notebook_name,
                        's3_url': s3_url,
                        's3_key': s3_key
                    })
                    
                    print(f"   ✅ Uploaded {notebook_name} to {s3_url}")
                else:
                    print(f"   ⚠️  Notebook not found: {notebook_path}")
            
            uploaded_notebooks[role] = {
                'notebooks': role_notebooks,
                'target_directory': config['target_directory'],
                'description': config['description']
            }
        
        return uploaded_notebooks
        
    except Exception as e:
        print(f"❌ Error uploading notebooks to S3: {str(e)}")
        sys.exit(1)

def create_deployment_script(uploaded_notebooks, domain_id, user_profiles):
    """Create a deployment script for SageMaker Studio"""
    
    script_content = f'''#!/bin/bash
# SageMaker Studio Notebook Deployment Script
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

set -e

echo "🚀 Deploying core notebooks to SageMaker Studio..."
echo "Domain ID: {domain_id}"
echo "Bucket: {BUCKET_NAME}"

# Function to download and setup notebooks for a role
setup_notebooks_for_role() {{
    local role=$1
    local target_dir=$2
    local description="$3"
    
    echo "📁 Setting up notebooks for $role..."
    echo "   Target directory: $target_dir"
    echo "   Description: $description"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Set permissions
    chmod 755 "$target_dir"
    
    echo "   ✅ Directory created: $target_dir"
}}

# Function to download notebook from S3
download_notebook() {{
    local s3_url=$1
    local target_path=$2
    local notebook_name=$3
    
    echo "   📥 Downloading $notebook_name..."
    aws s3 cp "$s3_url" "$target_path" --profile {AWS_PROFILE}
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Downloaded: $notebook_name"
        # Set appropriate permissions
        chmod 644 "$target_path"
    else
        echo "   ❌ Failed to download: $notebook_name"
        return 1
    fi
}}

# Main deployment logic
main() {{
    echo "Starting notebook deployment..."
    
'''
    
    # Add deployment logic for each role
    for role, config in uploaded_notebooks.items():
        script_content += f'''
    # Deploy notebooks for {role}
    echo "🔧 Deploying notebooks for {role}..."
    setup_notebooks_for_role "{role}" "{config['target_directory']}" "{config['description']}"
    
'''
        
        for notebook in config['notebooks']:
            target_path = f"{config['target_directory']}{notebook['notebook_name']}"
            script_content += f'''    download_notebook "{notebook['s3_url']}" "{target_path}" "{notebook['notebook_name']}"
'''
    
    script_content += '''
    echo "✅ Notebook deployment completed successfully!"
    echo ""
    echo "📋 Deployed notebooks:"
'''
    
    # Add summary of deployed notebooks
    for role, config in uploaded_notebooks.items():
        script_content += f'''    echo "   {role.upper()}:"
'''
        for notebook in config['notebooks']:
            script_content += f'''    echo "     - {notebook['notebook_name']}"
'''
    
    script_content += '''
    echo ""
    echo "🎯 Next steps:"
    echo "1. Open SageMaker Studio"
    echo "2. Navigate to the appropriate notebook directory for your role"
    echo "3. Start exploring and running the notebooks"
    echo "4. Refer to the README files for detailed instructions"
}

# Run main function
main "$@"
'''
    
    # Write the script
    script_path = 'scripts/setup/deploy_studio_notebooks.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"📝 Created deployment script: {script_path}")
    return script_path

def create_readme_files(uploaded_notebooks):
    """Create README files for each role"""
    
    for role, config in uploaded_notebooks.items():
        readme_content = f'''# {role.replace('-', ' ').title()} Notebooks

{config['description']}

## Notebooks Included

'''
        
        for notebook in config['notebooks']:
            readme_content += f'''### {notebook['notebook_name']}

- **Purpose**: Core functionality for {role.replace('-', ' ')}
- **Location**: `{config['target_directory']}{notebook['notebook_name']}`

'''
        
        readme_content += f'''## Getting Started

1. **Open SageMaker Studio**
   - Navigate to your SageMaker Studio environment
   - Go to the file browser

2. **Navigate to Notebooks**
   - Go to: `{config['target_directory']}`
   - You'll find all the notebooks for your role

3. **Prerequisites**
   - AWS CLI configured with "ab" profile
   - Access to S3 bucket: `{BUCKET_NAME}`
   - Appropriate IAM permissions for your role

4. **Running Notebooks**
   - Open any notebook by double-clicking
   - Select the appropriate kernel (Python 3)
   - Run cells sequentially

## Notebook Descriptions

'''
        
        if role == 'data-scientist':
            readme_content += '''### Data Scientist Core Enhanced
- **Purpose**: Data exploration and preparation with MLFlow tracking
- **Features**:
  - S3 data access and exploration
  - Image analysis and visualization
  - Data preparation for YOLOv11
  - MLFlow experiment tracking
  - Dataset structure creation

### Create Labeling Job
- **Purpose**: Create SageMaker Ground Truth labeling jobs
- **Features**:
  - Interactive labeling job creation
  - Cost estimation
  - Job monitoring
  - Label format conversion

'''
        elif role == 'ml-engineer':
            readme_content += '''### ML Engineer Core Enhanced
- **Purpose**: Training pipeline execution with MLFlow and Model Registry
- **Features**:
  - Pipeline configuration and execution
  - SageMaker training job management
  - MLFlow experiment tracking
  - Model Registry integration
  - Training monitoring and metrics

'''
        
        readme_content += f'''## MLFlow Integration

All notebooks include MLFlow integration for experiment tracking:

- **Experiment Tracking**: All runs are automatically tracked
- **Parameter Logging**: Training parameters and configurations
- **Metric Logging**: Performance metrics and results
- **Artifact Storage**: Visualizations and model artifacts
- **Model Registry**: Automatic model registration (ML Engineer)

### Accessing MLFlow UI

1. In SageMaker Studio, go to "Experiments and trials"
2. Click on "MLflow" to access the tracking UI
3. View experiments, runs, and artifacts

## Support

For issues or questions:
1. Check the notebook documentation
2. Review the MLOps SageMaker Demo project documentation
3. Contact your team lead or DevOps team

## Last Updated

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        
        # Write README file
        readme_path = f'notebooks/README_{role}.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"📄 Created README: {readme_path}")

def create_user_instructions():
    """Create instructions for users on how to access notebooks"""
    
    instructions = f'''# SageMaker Studio Notebook Deployment Instructions

## Overview

This guide explains how to access and use the core notebooks deployed to your SageMaker Studio environment.

## For Data Scientists

### Accessing Your Notebooks

1. **Open SageMaker Studio**
   - Go to the AWS Console
   - Navigate to SageMaker > Studio
   - Click "Open Studio" for your user profile

2. **Navigate to Your Notebooks**
   - In the file browser, go to: `/home/sagemaker-user/data-scientist-notebooks/`
   - You'll find these notebooks:
     - `data-scientist-core-enhanced.ipynb` - Core data exploration with MLFlow
     - `create_labeling_job.ipynb` - Ground Truth labeling job creation

3. **Getting Started**
   - Open `data-scientist-core-enhanced.ipynb` first
   - Follow the instructions in the notebook
   - Ensure your AWS CLI is configured with the "ab" profile

### Key Features for Data Scientists

- **Data Exploration**: Analyze drone imagery datasets
- **MLFlow Tracking**: All exploration activities are tracked
- **Ground Truth Integration**: Create labeling jobs for annotation
- **Data Preparation**: Prepare data for YOLOv11 training

## For ML Engineers

### Accessing Your Notebooks

1. **Open SageMaker Studio**
   - Go to the AWS Console
   - Navigate to SageMaker > Studio
   - Click "Open Studio" for your user profile

2. **Navigate to Your Notebooks**
   - In the file browser, go to: `/home/sagemaker-user/ml-engineer-notebooks/`
   - You'll find this notebook:
     - `ml-engineer-core-enhanced.ipynb` - Training pipeline with MLFlow and Model Registry

3. **Getting Started**
   - Open `ml-engineer-core-enhanced.ipynb`
   - Follow the instructions in the notebook
   - Ensure your AWS CLI is configured with the "ab" profile

### Key Features for ML Engineers

- **Training Pipeline**: Execute YOLOv11 training jobs
- **MLFlow Integration**: Complete experiment tracking
- **Model Registry**: Automatic model registration
- **Pipeline Monitoring**: Real-time training monitoring

## MLFlow Integration

### Accessing MLFlow UI

1. In SageMaker Studio, look for "Experiments and trials" in the left sidebar
2. Click on "MLflow" to access the MLFlow tracking UI
3. Here you can:
   - View all experiments and runs
   - Compare different training runs
   - Access model artifacts and visualizations
   - Track model lineage

### Benefits of MLFlow Integration

- **Experiment Tracking**: Every data exploration and training run is tracked
- **Reproducibility**: All parameters and configurations are logged
- **Collaboration**: Team members can view and compare experiments
- **Model Management**: Complete model lifecycle tracking

## Prerequisites

### AWS Configuration

Ensure your environment has:
- AWS CLI configured with "ab" profile
- Access to S3 bucket: `{BUCKET_NAME}`
- Appropriate IAM permissions for your role

### Checking AWS Configuration

Run this in a notebook cell to verify:

```python
import boto3
session = boto3.Session(profile_name='ab')
print(f"Account ID: {{session.client('sts').get_caller_identity()['Account']}}")
print(f"Region: {{session.region_name}}")
```

## Troubleshooting

### Common Issues

1. **AWS Profile Not Found**
   - Ensure AWS CLI is configured with "ab" profile
   - Run `aws configure list --profile ab` in terminal

2. **S3 Access Denied**
   - Check IAM permissions for your role
   - Ensure you have access to the data bucket

3. **MLFlow Connection Issues**
   - Verify SageMaker Studio has MLFlow enabled
   - Check network connectivity

4. **Notebook Kernel Issues**
   - Select "Python 3" kernel when opening notebooks
   - Restart kernel if needed

### Getting Help

1. Check notebook documentation and comments
2. Review error messages carefully
3. Contact your team lead or DevOps team
4. Refer to AWS SageMaker documentation

## Best Practices

### For Data Scientists

1. **Start Small**: Begin with small datasets for exploration
2. **Track Everything**: Use MLFlow to track all experiments
3. **Document Findings**: Add markdown cells with insights
4. **Collaborate**: Share interesting findings with the team

### For ML Engineers

1. **Monitor Resources**: Keep an eye on training costs
2. **Use Spot Instances**: Enable spot instances for cost savings
3. **Track Experiments**: Log all training parameters and metrics
4. **Model Governance**: Use Model Registry approval workflow

## Next Steps

1. **Complete Setup**: Ensure all prerequisites are met
2. **Run Notebooks**: Start with the core notebooks for your role
3. **Explore MLFlow**: Familiarize yourself with the MLFlow UI
4. **Collaborate**: Share experiments and findings with your team

---

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
    
    # Write instructions file
    instructions_path = 'SAGEMAKER_STUDIO_INSTRUCTIONS.md'
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📋 Created user instructions: {instructions_path}")

def main():
    """Main function to deploy notebooks to SageMaker Studio"""
    
    print("🚀 SageMaker Studio Notebook Deployment")
    print("=" * 50)
    
    # Setup AWS session
    session, sagemaker_client, s3_client = setup_aws_session()
    
    # Get SageMaker domain information
    domain_id, user_profiles = get_sagemaker_domain_info(sagemaker_client)
    
    # Upload notebooks to S3
    print("\\n📤 Uploading notebooks to S3...")
    uploaded_notebooks = upload_notebooks_to_s3(s3_client, NOTEBOOK_CONFIGS)
    
    # Create deployment script
    print("\\n📝 Creating deployment script...")
    script_path = create_deployment_script(uploaded_notebooks, domain_id, user_profiles)
    
    # Create README files
    print("\\n📄 Creating README files...")
    create_readme_files(uploaded_notebooks)
    
    # Create user instructions
    print("\\n📋 Creating user instructions...")
    create_user_instructions()
    
    print("\\n" + "=" * 50)
    print("✅ Deployment preparation completed successfully!")
    print("\\n📋 Summary:")
    print(f"   - Domain ID: {domain_id}")
    print(f"   - User Profiles: {len(user_profiles)}")
    print(f"   - Notebooks uploaded to S3: {BUCKET_NAME}")
    print(f"   - Deployment script: {script_path}")
    
    print("\\n🎯 Next Steps:")
    print("1. Review the generated files:")
    print(f"   - {script_path}")
    print("   - SAGEMAKER_STUDIO_INSTRUCTIONS.md")
    print("   - notebooks/README_*.md files")
    print("\\n2. To deploy to SageMaker Studio:")
    print(f"   - Run: ./{script_path}")
    print("   - Or manually copy notebooks using the S3 URLs")
    print("\\n3. Share instructions with users:")
    print("   - Distribute SAGEMAKER_STUDIO_INSTRUCTIONS.md")
    print("   - Ensure users have proper AWS configuration")

if __name__ == "__main__":
    main()