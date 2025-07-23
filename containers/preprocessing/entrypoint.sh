#!/bin/bash
# SOLUTION 2: Custom entrypoint script

# This script handles SageMaker's code parameter properly
# Usage: entrypoint.sh script_name.py [arguments...]

set -e

SCRIPT_NAME="$1"
shift  # Remove script name from arguments

# Check if script exists
if [ ! -f "/opt/ml/code/$SCRIPT_NAME" ]; then
    echo "Error: Script $SCRIPT_NAME not found in /opt/ml/code/"
    exit 1
fi

# Execute the script with remaining arguments
echo "Executing: python3 /opt/ml/code/$SCRIPT_NAME $@"
exec python3 "/opt/ml/code/$SCRIPT_NAME" "$@"
