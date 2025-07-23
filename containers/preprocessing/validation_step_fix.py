# SOLUTION 1: Use ScriptProcessor (Best Practice)
from sagemaker.processing import ScriptProcessor

def create_data_validation_step_fixed(parameters):
    """Create data validation processing step - FIXED with ScriptProcessor"""
    
    # Use ScriptProcessor instead of generic Processor
    validation_processor = ScriptProcessor(
        command=["python3"],  # This is the key difference
        image_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/yolov11-preprocessing:latest",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=pipeline_session
    )
    
    # Define processing step
    validation_step = ProcessingStep(
        name="DataValidation",
        processor=validation_processor,
        inputs=[
            ProcessingInput(
                source=parameters['dataset_path'],
                destination="/opt/ml/processing/input",
                input_name="dataset"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="validation_report",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET_NAME}/pipeline-artifacts/validation"
            )
        ],
        code="scripts/validate_dataset.py",  # ScriptProcessor handles this correctly
        job_arguments=[
            "--dataset-name", parameters['dataset_name'],
            "--model-variant", parameters['model_variant']
        ]
    )
    
    return validation_step
