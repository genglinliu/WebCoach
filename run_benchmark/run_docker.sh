#!/bin/bash

# WebVoyager Benchmark Docker Runner
# This script runs the WebVoyager benchmark in a Docker container

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
BROWSER_USE_DIR="/home/genglin/browser-use"

# Load .env file (try current directory first, then parent)
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"
[ -f "$PARENT_DIR/.env" ] && source "$PARENT_DIR/.env"
[ -f "$BROWSER_USE_DIR/.env" ] && source "$BROWSER_USE_DIR/.env"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please create a .env file with OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Parse command line arguments
CONFIG_FILE="config.yaml"
SUBTASKS=""
SUBTASK=""
LIST_SUBTASKS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --subtasks)
            shift
            SUBTASKS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUBTASKS="$SUBTASKS \"$1\""
                shift
            done
            ;;
        --subtask)
            SUBTASK="$2"
            shift 2
            ;;
        --list-subtasks)
            LIST_SUBTASKS="--list-subtasks"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE         Configuration YAML file (default: config.yaml)"
            echo "  --subtasks LIST       Space-separated list of subtasks to run"
            echo "  --subtask NAME        Run a single subtask by name"
            echo "  --list-subtasks      List available subtasks and exit"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build Python command
PYTHON_CMD="python run_webvoyager_without_coach.py --config $CONFIG_FILE"

if [ -n "$SUBTASKS" ]; then
    PYTHON_CMD="$PYTHON_CMD --subtasks$SUBTASKS"
fi

if [ -n "$SUBTASK" ]; then
    PYTHON_CMD="$PYTHON_CMD --subtask $SUBTASK"
fi

if [ -n "$LIST_SUBTASKS" ]; then
    PYTHON_CMD="$PYTHON_CMD $LIST_SUBTASKS"
fi

echo "Running WebVoyager benchmark..."
echo "Command: $PYTHON_CMD"
echo ""

# Create output directories
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/screenshots"

# Run the benchmark in Docker
docker run --rm -it \
    --entrypoint="" \
    --privileged \
    --user "$(id -u):$(id -g)" \
    --shm-size=2g \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$PARENT_DIR":/app/scripts_gl \
    -w /app/scripts_gl/run_benchmark \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e PYTHONPATH="/app/browser-use:/app/scripts_gl/run_benchmark" \
    -e BROWSER_USE_CONFIG_DIR="/app/scripts_gl/run_benchmark/.browser-use-config" \
    -e BROWSER_USE_PROFILES_DIR="/app/scripts_gl/run_benchmark/.browser-use-profiles" \
    -e BROWSER_USE_DOWNLOADS_DIR="/app/scripts_gl/run_benchmark/.browser-use-downloads" \
    browseruse \
    bash -c "$PYTHON_CMD"

# Check exit code and show results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "📁 Results: $SCRIPT_DIR/results"
    echo "📸 Screenshots: $SCRIPT_DIR/screenshots"
else
    echo ""
    echo "❌ Benchmark failed!"
    exit 1
fi
