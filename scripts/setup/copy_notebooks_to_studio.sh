#!/bin/bash
# Simple script to copy notebooks from S3 to SageMaker Studio

set -e

echo "📋 Copying notebooks from S3 to SageMaker Studio..."

# S3 locations (from the deployment script output)
DATA_SCIENTIST_NOTEBOOK="s3://lucaskle-ab3-project-pv/sagemaker-studio-notebooks/20250720_131422/data-scientist/data-scientist-core-enhanced.ipynb"
ML_ENGINEER_NOTEBOOK="s3://lucaskle-ab3-project-pv/sagemaker-studio-notebooks/20250720_131422/ml-engineer/ml-engineer-core-enhanced.ipynb"

echo "📥 Available notebooks in S3:"
echo "   Data Scientist: $DATA_SCIENTIST_NOTEBOOK"
echo "   ML Engineer: $ML_ENGINEER_NOTEBOOK"

echo ""
echo "🎯 To access these notebooks in SageMaker Studio:"
echo ""
echo "1. Open SageMaker Studio in your browser"
echo "2. Open a terminal in SageMaker Studio"
echo "3. Run these commands to download the notebooks:"
echo ""
echo "   # For Data Scientists:"
echo "   mkdir -p ~/data-scientist-notebooks"
echo "   aws s3 cp $DATA_SCIENTIST_NOTEBOOK ~/data-scientist-notebooks/ --profile ab"
echo ""
echo "   # For ML Engineers:"
echo "   mkdir -p ~/ml-engineer-notebooks"
echo "   aws s3 cp $ML_ENGINEER_NOTEBOOK ~/ml-engineer-notebooks/ --profile ab"
echo ""
echo "4. Navigate to the appropriate directory in the SageMaker Studio file browser"
echo "5. Open the notebook for your role"
echo ""
echo "✅ Notebooks are ready for deployment!"