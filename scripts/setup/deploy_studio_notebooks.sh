#!/bin/bash
# SageMaker Studio Notebook Deployment Script
# Generated on 2025-07-20 13:29:32

set -e

echo "🚀 Deploying core notebooks to SageMaker Studio..."
echo "Domain ID: d-kzi9b2bvwfvb"
echo "Bucket: lucaskle-ab3-project-pv"

# Function to download and setup notebooks for a role
setup_notebooks_for_role() {
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
}

# Function to download notebook from S3
download_notebook() {
    local s3_url=$1
    local target_path=$2
    local notebook_name=$3
    
    echo "   📥 Downloading $notebook_name..."
    aws s3 cp "$s3_url" "$target_path" --profile ab
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Downloaded: $notebook_name"
        # Set appropriate permissions
        chmod 644 "$target_path"
    else
        echo "   ❌ Failed to download: $notebook_name"
        return 1
    fi
}

# Main deployment logic
main() {
    echo "Starting notebook deployment..."
    

    # Deploy notebooks for data-scientist
    echo "🔧 Deploying notebooks for data-scientist..."
    setup_notebooks_for_role "data-scientist" "/home/sagemaker-user/data-scientist-notebooks/" "Core notebooks for Data Scientists"
    
    download_notebook "s3://lucaskle-ab3-project-pv/sagemaker-studio-notebooks/20250720_132928/data-scientist/data-scientist-core-enhanced.ipynb" "/home/sagemaker-user/data-scientist-notebooks/data-scientist-core-enhanced.ipynb" "data-scientist-core-enhanced.ipynb"

    # Deploy notebooks for ml-engineer
    echo "🔧 Deploying notebooks for ml-engineer..."
    setup_notebooks_for_role "ml-engineer" "/home/sagemaker-user/ml-engineer-notebooks/" "Core notebooks for ML Engineers"
    
    download_notebook "s3://lucaskle-ab3-project-pv/sagemaker-studio-notebooks/20250720_132928/ml-engineer/ml-engineer-core-enhanced.ipynb" "/home/sagemaker-user/ml-engineer-notebooks/ml-engineer-core-enhanced.ipynb" "ml-engineer-core-enhanced.ipynb"

    echo "✅ Notebook deployment completed successfully!"
    echo ""
    echo "📋 Deployed notebooks:"
    echo "   DATA-SCIENTIST:"
    echo "     - data-scientist-core-enhanced.ipynb"
    echo "   ML-ENGINEER:"
    echo "     - ml-engineer-core-enhanced.ipynb"

    echo ""
    echo "🎯 Next steps:"
    echo "1. Open SageMaker Studio"
    echo "2. Navigate to the appropriate notebook directory for your role"
    echo "3. Start exploring and running the notebooks"
    echo "4. Refer to the README files for detailed instructions"
}

# Run main function
main "$@"
