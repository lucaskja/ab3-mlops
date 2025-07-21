#!/usr/bin/env python3
"""
Test SageMaker Managed MLflow Integration

This script tests the MLflow integration with the SageMaker managed MLflow server.
"""

import os
import sys
import boto3
from datetime import datetime

# Set AWS profile
os.environ['AWS_PROFILE'] = 'ab'

def test_mlflow_integration():
    """Test the SageMaker managed MLflow integration."""
    
    print("🧪 Testing SageMaker Managed MLflow Integration")
    print("=" * 50)
    
    try:
        # Download the helper script
        print("📥 Downloading MLflow helper...")
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.download_file(
            'lucaskle-ab3-project-pv', 
            'mlflow-sagemaker/utils/sagemaker_mlflow_helper.py', 
            'sagemaker_mlflow_helper.py'
        )
        print("✅ Helper script downloaded")
        
        # Import the helper
        from sagemaker_mlflow_helper import get_sagemaker_mlflow_helper
        
        # Initialize MLflow helper
        print("🔗 Connecting to SageMaker managed MLflow server...")
        mlflow_helper = get_sagemaker_mlflow_helper(aws_profile='ab')
        
        # Get server info
        server_info = mlflow_helper.get_tracking_server_info()
        print(f"✅ Connected to MLflow server!")
        print(f"   Server URL: {server_info.get('url', 'Unknown')}")
        print(f"   Status: {server_info.get('status', 'Unknown')}")
        print(f"   MLflow Version: {server_info.get('mlflow_version', 'Unknown')}")
        print(f"   Server Size: {server_info.get('size', 'Unknown')}")
        
        # Create a test experiment
        print("\n🧪 Creating test experiment...")
        experiment_name = "mlflow-integration-test"
        experiment_id = mlflow_helper.create_experiment(experiment_name)
        print(f"✅ Test experiment created: {experiment_name}")
        
        # Start a test run
        print("\n🏃 Starting test run...")
        with mlflow_helper.start_run(run_name=f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}", experiment_name=experiment_name) as run:
            # Log test parameters
            test_params = {
                "test_parameter_1": "test_value_1",
                "test_parameter_2": 42,
                "test_parameter_3": 3.14159
            }
            mlflow_helper.log_params(test_params)
            print("✅ Test parameters logged")
            
            # Log test metrics
            test_metrics = {
                "test_metric_1": 0.95,
                "test_metric_2": 0.87,
                "test_metric_3": 1.23
            }
            mlflow_helper.log_metrics(test_metrics)
            print("✅ Test metrics logged")
            
            # Create and log a test artifact
            with open('test_artifact.txt', 'w') as f:
                f.write(f"Test artifact created at {datetime.now()}\n")
                f.write("This is a test of the SageMaker managed MLflow integration.\n")
            
            mlflow_helper.log_artifact('test_artifact.txt')
            print("✅ Test artifact logged")
            
            print(f"\n📊 Test run completed!")
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Experiment ID: {run.info.experiment_id}")
        
        # List experiments
        print("\n📋 Listing experiments...")
        experiments = mlflow_helper.list_experiments()
        print(f"✅ Found {len(experiments)} experiments")
        
        # List runs in the test experiment
        print(f"\n📋 Listing runs in '{experiment_name}'...")
        runs = mlflow_helper.list_runs(experiment_name, max_results=5)
        if runs is not None and not runs.empty:
            print(f"✅ Found {len(runs)} runs in the experiment")
            print("\nRecent runs:")
            for _, run in runs.head(3).iterrows():
                run_name = run.get('tags.mlflow.runName', 'Unnamed')
                status = run.get('status', 'Unknown')
                print(f"   - {run_name} ({status})")
        else:
            print("⚠️  No runs found in the experiment")
        
        # Get MLflow UI URL
        ui_url = mlflow_helper.get_mlflow_ui_url()
        print(f"\n🌐 MLflow UI URL: {ui_url}")
        
        print("\n🎉 MLflow integration test completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Open the MLflow UI to view your experiments:")
        print(f"   {ui_url}")
        print("2. Use the enhanced notebooks with MLflow integration:")
        print("   - notebooks/data-scientist-core-enhanced.ipynb")
        print("   - notebooks/ml-engineer-core-enhanced.ipynb")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure the SageMaker managed MLflow server is running")
        print("2. Check your AWS profile 'ab' is configured correctly")
        print("3. Verify you have access to the S3 bucket")
        return False
    
    finally:
        # Clean up test files
        try:
            os.remove('test_artifact.txt')
            os.remove('sagemaker_mlflow_helper.py')
        except:
            pass


if __name__ == "__main__":
    success = test_mlflow_integration()
    sys.exit(0 if success else 1)
