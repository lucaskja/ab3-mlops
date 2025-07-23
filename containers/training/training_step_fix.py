# Suggested fix for your training step:

def create_training_step_corrected(parameters, validation_step):
    """Create YOLOv11 training step - CORRECTED"""
    
    # Training estimator - Use parameter objects in hyperparameters
    estimator = Estimator(
        image_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/yolov11-training:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type=parameters['instance_type'],  # ParameterString object
        output_path=parameters['model_output_path'],  # ParameterString object
        sagemaker_session=pipeline_session,
        use_spot_instances=True,  # Use fixed value for now
        max_wait=3600,
        max_run=3600,
        hyperparameters={
            "model_variant": parameters['model_variant'],    # ParameterString object
            "image_size": parameters['image_size'],          # ParameterInteger object
            "batch_size": parameters['batch_size'],          # ParameterInteger object
            "epochs": parameters['epochs'],                  # ParameterInteger object
            "learning_rate": parameters['learning_rate'],    # ParameterFloat object
            "dataset_name": parameters['dataset_name']       # ParameterString object
        }
    )
    
    # Training step
    training_step = TrainingStep(
        name="YOLOv11Training",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=parameters['dataset_path'],  # ParameterString object
                content_type="application/x-image"   # ✅ FIXED: Better for image datasets
                # OR you can omit content_type entirely for auto-detection
            )
        },
        depends_on=[validation_step.name] if validation_step else None
    )
    
    return training_step
