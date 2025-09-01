#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_GL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BROWSER_USE_DIR="/home/genglin/browser-use"

# Load .env file
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"
[ -f "$BROWSER_USE_DIR/.env" ] && source "$BROWSER_USE_DIR/.env"

echo "🚀 Starting WebVoyager benchmark in Docker..."
echo "📁 Script directory: $SCRIPT_DIR"
echo "📁 Scripts GL directory: $SCRIPTS_GL_DIR"
echo "🌐 Browser-use directory: $BROWSER_USE_DIR"

# Run WebVoyager benchmark in Docker
docker run --rm -it \
    --entrypoint="" \
    --privileged \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$SCRIPTS_GL_DIR":/app/scripts_gl \
    -w /app/scripts_gl/run_benchmark \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    browseruse \
    python run_webvoyager.py
