#!/usr/bin/env bash
# Script to build and push Loan Default Prediction Image to Docker Hub

set -e

DOCKER_USER="${1}"
IMAGE_NAME="loan-default-prediction"
TAG="${2:-latest}"

if [ -z "$DOCKER_USER" ]; then
    echo "=================================================================="
    echo "Usage: ./docker_push.sh <your_dockerhub_username> [tag]"
    echo "Example: ./docker_push.sh raghav latest"
    echo "=================================================================="
    exit 1
fi

FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "1. Checking Docker login..."
docker login

echo "2. Building Docker image for cloud (linux/amd64): ${FULL_IMAGE}..."
docker build --platform linux/amd64 -t "${FULL_IMAGE}" .

echo "3. Pushing image to Docker Hub: ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo ""
echo "=================================================================="
echo "Successfully uploaded to Docker Hub!"
echo "Anyone can now run your project anywhere using:"
echo "docker run -p 8501:8501 -p 8000:8000 ${FULL_IMAGE}"
echo "=================================================================="
