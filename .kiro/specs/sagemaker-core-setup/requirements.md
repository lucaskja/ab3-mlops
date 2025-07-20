# Requirements Document

## Introduction

This document outlines the requirements for setting up the core AWS SageMaker infrastructure for a machine learning operations (MLOps) environment. The focus is on creating a minimal viable setup with SageMaker Studio notebooks for both Data Scientists and ML Engineers, along with a pipeline that ML Engineers can activate to train YOLOv11 models using data stored in an S3 bucket. This implementation prioritizes simplicity and core functionality over comprehensive features.

## Requirements

### Requirement 1

**User Story:** As a Data Scientist, I want to access SageMaker Studio with appropriate notebooks, so that I can explore and prepare data for YOLOv11 model training.

#### Acceptance Criteria
1. WHEN a Data Scientist logs into AWS THEN they SHALL have access to a SageMaker Studio environment with their role permissions
2. WHEN a Data Scientist opens SageMaker Studio THEN they SHALL find pre-configured notebooks for data exploration and preparation
3. WHEN a Data Scientist runs the notebooks THEN they SHALL be able to access the drone imagery data in the S3 bucket
4. IF a Data Scientist attempts to access production resources THEN the system SHALL deny access based on IAM policies

### Requirement 2

**User Story:** As an ML Engineer, I want to access SageMaker Studio with appropriate notebooks, so that I can manage and execute training pipelines for YOLOv11 models.

#### Acceptance Criteria
1. WHEN an ML Engineer logs into AWS THEN they SHALL have access to a SageMaker Studio environment with their role permissions
2. WHEN an ML Engineer opens SageMaker Studio THEN they SHALL find pre-configured notebooks for pipeline management
3. WHEN an ML Engineer runs the notebooks THEN they SHALL be able to start and monitor YOLOv11 training pipelines
4. IF an ML Engineer executes a pipeline THEN the system SHALL use the data from the S3 bucket for training

### Requirement 3

**User Story:** As a user, I want a SageMaker Pipeline for YOLOv11 model training, so that models can be trained in a consistent and reproducible manner.

#### Acceptance Criteria
1. WHEN the pipeline is executed THEN it SHALL use YOLOv11 architecture for object detection
2. WHEN the pipeline runs THEN it SHALL access training data from the specified S3 bucket
3. WHEN training completes THEN the model artifacts SHALL be stored in the S3 bucket
4. WHEN the pipeline is configured THEN it SHALL allow customization of basic hyperparameters

### Requirement 4

**User Story:** As a system administrator, I want proper IAM roles and permissions, so that users have appropriate access based on their roles.

#### Acceptance Criteria
1. WHEN users are created THEN they SHALL be assigned either Data Scientist or ML Engineer roles
2. WHEN a Data Scientist accesses resources THEN they SHALL have read-only access to the dataset
3. WHEN an ML Engineer accesses resources THEN they SHALL have permissions to execute training pipelines
4. WHEN any AWS resources are created THEN they SHALL exclusively use the "ab" AWS CLI profile
5. WHEN any AWS CLI command is executed THEN it SHALL include the "--profile ab" parameter or use the AWS_PROFILE=ab environment variable

### Requirement 5

**User Story:** As a user, I want simple deployment and cleanup procedures, so that I can easily set up and tear down the environment.

#### Acceptance Criteria
1. WHEN the deployment script is executed THEN it SHALL create all necessary AWS resources using the "ab" AWS CLI profile
2. WHEN the deployment completes THEN users SHALL receive instructions on how to access SageMaker Studio
3. WHEN the cleanup script is executed THEN it SHALL remove all created AWS resources using the "ab" AWS CLI profile
4. IF deployment fails THEN the script SHALL provide clear error messages and recovery steps
5. WHEN any script is executed THEN it SHALL verify the "ab" profile is configured before proceeding