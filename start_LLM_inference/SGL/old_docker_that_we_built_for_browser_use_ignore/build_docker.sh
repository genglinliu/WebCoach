#!/bin/bash

IMAGE_NAME="browser-use-ready"
SCRIPT_DIR="$(dirname "$0")"

echo "Building $IMAGE_NAME Docker image..."
docker build -t $IMAGE_NAME "$SCRIPT_DIR"

echo "Setting up browser-use environment..."

# Install dependencies and playwright in the Docker image itself
docker run --rm \
    -v "/home/genglin/browser-use":/app/browser-use \
    -v "browser-use-uv-cache":/root/.cache \
    $IMAGE_NAME bash -c "cd /app/browser-use && uv sync --all-extras --dev"

echo "Docker image $IMAGE_NAME is ready!"