#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the Docker image name
IMAGE_NAME="nlp-webapp"

# Step 1: Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 2: Run unit tests
echo "Running unit tests..."
#docker run --rm $IMAGE_NAME pytest test_unittest.py -vv

# Example of integration tests
echo "Running integration tests..."
#docker run --rm -v $(pwd):/root/web_app $IMAGE_NAME pytest test_api.py -vv
#MODEL=quantized docker-compose up --build --abort-on-container-exit

echo "Deploying the model..."
docker run -e MODEL=quantized -p 5000:5000 nlp-webapp