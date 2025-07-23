#!/bin/bash

# Build and push YOLOv11 training container to ECR for AMD64 architecture

set -e

# Configuration
AWS_PROFILE="ab"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="192771711075"
REPOSITORY_NAME="yolov11-training"
IMAGE_TAG="latest"

# Full image URI
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Building and pushing training container for AMD64 architecture..."
echo "Image URI: ${IMAGE_URI}"

# Get the login token and login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build the Docker image for AMD64 platform and load it automatically
echo "Building Docker image for AMD64 platform..."
docker buildx build --platform linux/amd64 --load -t ${REPOSITORY_NAME}:${IMAGE_TAG} .

# Tag the image for ECR
echo "Tagging image for ECR..."
docker tag ${REPOSITORY_NAME}:${IMAGE_TAG} ${IMAGE_URI}

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${IMAGE_URI}

echo "Successfully built and pushed training container for AMD64!"
echo "Image URI: ${IMAGE_URI}"
