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

# Read output directory from config.yaml
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract output base directory from config.yaml
# Try Python YAML parsing first, fallback to basic grep if PyYAML not available
OUTPUT_BASE_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print(config['output']['base_dir'])
except ImportError:
    # PyYAML not available, fallback to basic parsing
    import re
    with open('$CONFIG_FILE', 'r') as f:
        content = f.read()
    # Look for base_dir: "path" pattern
    match = re.search(r'base_dir:\s*[\"\\']([^\"\\']+)[\"\\']', content)
    if match:
        print(match.group(1))
    else:
        print('Error: Could not find base_dir in config', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print('Error reading config:', e, file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

# If Python parsing failed, try basic grep as last resort
if [ -z "$OUTPUT_BASE_DIR" ]; then
    echo "⚠️  Python YAML parsing failed, trying basic grep..."
    OUTPUT_BASE_DIR=$(grep -E '^\s*base_dir:\s*["'"'"']' "$CONFIG_FILE" | sed -E 's/^\s*base_dir:\s*["'"'"']([^"'"'"']+)["'"'"'].*/\1/')
fi

if [ -z "$OUTPUT_BASE_DIR" ]; then
    echo "❌ Failed to read output base directory from config.yaml"
    exit 1
fi

echo "📁 Output base directory from config: $OUTPUT_BASE_DIR"

# Create output directory on host
mkdir -p "$OUTPUT_BASE_DIR"

# Run WebVoyager benchmark in Docker
docker run --rm -it \
    --entrypoint="" \
    --privileged \
    --user root \
    --add-host host.docker.internal:host-gateway \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$SCRIPTS_GL_DIR":/app/scripts_gl \
    -v "$OUTPUT_BASE_DIR":"$OUTPUT_BASE_DIR" \
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
