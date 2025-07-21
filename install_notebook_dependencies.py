#!/usr/bin/env python3
"""
Install Notebook Dependencies

This script installs all required dependencies for the enhanced notebooks
including SageMaker managed MLflow integration.
"""

import subprocess
import sys

def install_dependencies():
    """Install all required dependencies for the enhanced notebooks."""
    
    print("📦 Installing Enhanced Notebook Dependencies")
    print("=" * 50)
    
    # List of required packages
    packages = [
        "mlflow>=3.0.0",
        "requests-auth-aws-sigv4>=0.7",
        "boto3>=1.28.0", 
        "sagemaker>=2.190.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "xmltodict>=0.13.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.8.0"
    ]
    
    print("Installing packages:")
    for package in packages:
        print(f"  - {package}")
    
    try:
        # Install packages
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("\n✅ All packages installed successfully!")
        
        # Test MLflow import
        try:
            import mlflow
            print(f"✅ MLflow {mlflow.__version__} imported successfully")
        except ImportError as e:
            print(f"⚠️  MLflow import failed: {e}")
        
        # Test SageMaker import
        try:
            import sagemaker
            print(f"✅ SageMaker {sagemaker.__version__} imported successfully")
        except ImportError as e:
            print(f"⚠️  SageMaker import failed: {e}")
        
        # Test auth package import
        try:
            import requests_auth_aws_sigv4
            print("✅ AWS SigV4 authentication package imported successfully")
        except ImportError as e:
            print(f"⚠️  AWS SigV4 auth package import failed: {e}")
        
        print("\n🎉 Installation completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Open your enhanced notebooks:")
        print("   - notebooks/data-scientist-core-enhanced.ipynb")
        print("   - notebooks/ml-engineer-core-enhanced.ipynb")
        print("2. The notebooks will automatically connect to the SageMaker managed MLflow server")
        print("3. Start tracking your experiments!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)
