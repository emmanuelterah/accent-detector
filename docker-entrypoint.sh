#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p ${MODEL_CACHE_DIR} ${TEMP_DIR}

# Set permissions
chmod -R 777 ${MODEL_CACHE_DIR} ${TEMP_DIR}

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi
else
    echo "No GPU detected, running in CPU mode"
fi

# Start the application
echo "Starting Accent Analysis Tool..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 