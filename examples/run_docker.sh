#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BROWSER_USE_DIR="/home/genglin/browser-use"

# Load .env file
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"
[ -f "$BROWSER_USE_DIR/.env" ] && source "$BROWSER_USE_DIR/.env"

# Get script to run (default: example.py)
SCRIPT_TO_RUN="${1:-example.py}"

# Run script in Docker
docker run --rm -it \
    --entrypoint="" \
    --privileged \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$SCRIPT_DIR":/app/scripts_gl \
    -w /app/scripts_gl \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    browseruse \
    python "$SCRIPT_TO_RUN"