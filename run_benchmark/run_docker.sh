#!/bin/bash

# WebVoyager Benchmark Docker Runner
# This script runs the WebVoyager benchmark in a Docker container with browser-use

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
BROWSER_USE_DIR="/home/genglin/browser-use"

# Load .env file from multiple possible locations
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"
[ -f "$PARENT_DIR/.env" ] && source "$PARENT_DIR/.env"
[ -f "$BROWSER_USE_DIR/.env" ] && source "$BROWSER_USE_DIR/.env"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please create a .env file with OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Default values
CONFIG_FILE="config.yaml"
SUBTASKS=""
SUBTASK=""
LIST_SUBTASKS=""

# Parse command line arguments
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
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default config"
            echo "  $0 --subtasks Amazon Google           # Run specific subtasks"
            echo "  $0 --subtask Amazon                   # Run single subtask"
            echo "  $0 --list-subtasks                   # List available subtasks"
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
echo "Working directory: /app/scripts_gl/run_benchmark"
echo "Browser-use directory: /app/browser-use"
echo ""

# Create output directories on host with proper permissions
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/screenshots"
chmod 755 "$SCRIPT_DIR/results"
chmod 755 "$SCRIPT_DIR/screenshots"

echo "Created output directories with proper permissions"

# Run the benchmark in Docker
docker run --rm -it \
    --entrypoint="" \
    --privileged \
    --shm-size=2g \
    -v "$BROWSER_USE_DIR":/app/browser-use \
    -v "$PARENT_DIR":/app/scripts_gl \
    -w /app/scripts_gl/run_benchmark \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e PYTHONPATH="/app/browser-use:/app/scripts_gl/run_benchmark" \
    browseruse \
    bash -c "$PYTHON_CMD"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "📁 Results saved to: $SCRIPT_DIR/results"
    echo "📸 Screenshots saved to: $SCRIPT_DIR/screenshots"
    
    # List generated files
    if [ -d "$SCRIPT_DIR/results" ] && [ "$(ls -A "$SCRIPT_DIR/results")" ]; then
        echo "📋 Generated result files:"
        ls -la "$SCRIPT_DIR/results"
    fi
    
    if [ -d "$SCRIPT_DIR/screenshots" ] && [ "$(ls -A "$SCRIPT_DIR/screenshots")" ]; then
        echo "📸 Generated screenshot files:"
        ls -la "$SCRIPT_DIR/screenshots"
    fi
else
    echo ""
    echo "❌ Benchmark failed!"
    exit 1
fi
