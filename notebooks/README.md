# Jupyter Notebooks

This directory contains Jupyter notebooks for interactive development, exploration, and demonstration of the MLOps SageMaker Demo project.

## Directory Structure

```
notebooks/
├── data-exploration/       # Data analysis and profiling notebooks
│   ├── data-profiling.ipynb # Data profiling and visualization
│   └── dataset-analysis.ipynb # Dataset analysis and statistics
├── data-labeling/          # Ground Truth labeling notebooks
│   └── create_labeling_job.ipynb # Create and manage labeling jobs
├── model-development/      # Model training and experimentation
│   ├── yolov11-training.ipynb # YOLOv11 model training
│   ├── hyperparameter-tuning.ipynb # Hyperparameter optimization
│   └── model-evaluation.ipynb # Model evaluation and comparison
└── pipeline-development/   # Pipeline development notebooks
    ├── preprocessing-pipeline.ipynb # Data preprocessing pipeline
    └── training-pipeline.ipynb # End-to-end training pipeline
```

## Notebook Descriptions

### Data Exploration Notebooks

- **data-profiling.ipynb**: Analyze and visualize drone imagery dataset characteristics
- **dataset-analysis.ipynb**: Statistical analysis of the dataset with distribution plots

### Data Labeling Notebooks

- **create_labeling_job.ipynb**: Create and manage Ground Truth labeling jobs for object detection

### Model Development Notebooks

- **yolov11-training.ipynb**: Train YOLOv11 models on drone imagery
- **hyperparameter-tuning.ipynb**: Optimize YOLOv11 hyperparameters
- **model-evaluation.ipynb**: Evaluate and compare model performance

### Pipeline Development Notebooks

- **preprocessing-pipeline.ipynb**: Develop data preprocessing pipelines
- **training-pipeline.ipynb**: Create end-to-end training pipelines

## Usage Guidelines

### Environment Setup

All notebooks should be run in SageMaker Studio with the appropriate IAM role:

- **Data Scientist Role**: For data exploration, labeling, and model development
- **ML Engineer Role**: For pipeline development and deployment

### AWS Profile

All notebooks use the "ab" AWS CLI profile for resource access:

```python
import boto3
session = boto3.Session(profile_name='ab')
```

### Code Organization

Notebooks should follow these best practices:

1. **Import from Source Modules**: Use functions from `src/` modules rather than duplicating code
2. **Markdown Documentation**: Include clear markdown cells explaining each step
3. **Parameter Configuration**: Keep configurable parameters at the top of the notebook
4. **Error Handling**: Include proper error handling for AWS API calls
5. **Resource Cleanup**: Include cells for cleaning up resources at the end

### Interactive Widgets

Notebooks use interactive widgets for parameter tuning and visualization:

```python
import ipywidgets as widgets
from IPython.display import display

# Create interactive widgets
learning_rate = widgets.FloatSlider(
    value=0.001,
    min=0.0001,
    max=0.01,
    step=0.0001,
    description='Learning Rate:',
    continuous_update=False
)

batch_size = widgets.IntSlider(
    value=16,
    min=1,
    max=64,
    step=1,
    description='Batch Size:',
    continuous_update=False
)

# Display widgets
display(learning_rate, batch_size)

# Use widget values in code
def train_model(lr, bs):
    print(f"Training with learning rate: {lr}, batch size: {bs}")
    
# Connect widgets to function
widgets.interactive(train_model, lr=learning_rate, bs=batch_size)
```

### Progress Tracking

Long-running operations use progress bars:

```python
from tqdm.notebook import tqdm
import time

# Create a progress bar
with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.1)  # Simulate work
        pbar.update(1)
```

### Visualization

Notebooks use standardized visualization components:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Create visualization
def plot_metrics(metrics_dict):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax[0].plot(metrics_dict['train_loss'], label='Training Loss')
    ax[0].plot(metrics_dict['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss Curves')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Plot metrics
    ax[1].plot(metrics_dict['precision'], label='Precision')
    ax[1].plot(metrics_dict['recall'], label='Recall')
    ax[1].plot(metrics_dict['mAP'], label='mAP')
    ax[1].set_title('Performance Metrics')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].legend()
    
    plt.tight_layout()
    return fig
```

## Running Notebooks

To run these notebooks:

1. Launch SageMaker Studio with the appropriate IAM role
2. Clone this repository to your SageMaker Studio environment
3. Open the desired notebook
4. Run all cells or step through them individually

## Contributing New Notebooks

When creating new notebooks:

1. Place them in the appropriate subdirectory
2. Follow the code organization guidelines
3. Include comprehensive markdown documentation
4. Use interactive widgets for parameter tuning
5. Implement proper error handling and resource cleanup