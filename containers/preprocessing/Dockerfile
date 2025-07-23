FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY preprocessing.py .
COPY validate_dataset.py .
COPY scripts/ ./scripts/

# Make all scripts executable
RUN chmod +x *.py scripts/*.py

# Set entrypoint for SageMaker processing
ENTRYPOINT ["python3"]
