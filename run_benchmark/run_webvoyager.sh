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
    --user root \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$SCRIPTS_GL_DIR":/app/scripts_gl \
    -w /app/scripts_gl/run_benchmark \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    browseruse \
    bash -c "
    echo '🔍 Debugging Python environment...'
    which python
    which pip
    python --version
    echo '📦 Installing WebCoach dependencies using system pip with target...'
    pip install --target /app/.venv/lib/python3.12/site-packages -r /app/scripts_gl/WebCoach/requirements.txt
    echo '✅ Testing numpy import with explicit venv python...'
    /app/.venv/bin/python -c 'import numpy; print(\"NumPy version:\", numpy.__version__)'
    echo '✅ Testing faiss import...'
    /app/.venv/bin/python -c 'import faiss; print(\"FAISS imported successfully\")'
    echo '🚀 Starting WebVoyager...'
    /app/.venv/bin/python run_webvoyager.py
    "
