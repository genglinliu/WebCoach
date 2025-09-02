# WebCoach Framework

A real-time coaching system for browser-use agents that learns from past experiences and provides guidance during task execution.

## Overview

The WebCoach Framework implements a coaching system that:

1. **Learns from Experience**: Processes and stores past agent trajectories in a vector database
2. **Provides Real-time Guidance**: Analyzes current agent state and retrieves relevant past experiences  
3. **Offers Strategic Advice**: Uses LLM to generate actionable coaching advice when similar failures are detected
4. **Integrates Seamlessly**: Works with browser-use agents through callback hooks without modifying core code

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Condenser     │────│     EMS      │────│   WebCoach      │
│ Raw → Summary   │    │ Vector Store │    │ Advice Generation│
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                    │
         └───────────────────────┼────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │    coach_callback       │
                    │   Integration Point     │
                    └─────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │     browser-use         │
                    │       Agent             │
                    └─────────────────────────┘
```

## Components

### 1. Condenser (`condenser.py`)
- Processes raw agent trajectories into structured summaries
- Generates embeddings for similarity search
- Determines trajectory type (complete vs partial)
- Extracts failure patterns and success workflows

### 2. External Memory Store (`ems.py`) 
- Vector database for storing condensed experiences
- Uses cosine similarity for experience retrieval
- Persistent JSON storage with numpy-based similarity calculations
- Supports filtering by domain, recency, and relevance

### 3. WebCoach (`web_coach.py`)
- Retrieves relevant past experiences from EMS
- Uses LLM to analyze current state vs past patterns
- Generates structured advice with intervention decisions
- Focuses on preventing repeated failure patterns

### 4. Coach Callback (`coach_callback.py`)
- Main integration point with browser-use agents
- Handles lazy loading of coach components
- Manages configuration and error handling
- Injects advice as system messages

## Usage

### 1. Configuration

Add to your `config.yaml`:

```yaml
coach:
  enabled: true          # Enable/disable coaching
  model: "gpt-4o"        # LLM model for advice generation
  frequency: 5           # Provide coaching every N steps
  storage_dir: "./coach_storage"  # Directory for storing experiences
  debug: false           # Enable debug logging
```

### 2. Integration

The framework integrates automatically with `run_webvoyager.py`:

```python
# Coach is set up automatically based on config
coach_callback = setup_coach(config)

# Agent runs with coaching if enabled
if coach_callback:
    history = await agent.run(
        max_steps=max_steps,
        on_step_end=coach_callback
    )
else:
    history = await agent.run(max_steps=max_steps)
```

### 3. Running with Coaching

1. Enable coaching in `config.yaml`: `coach.enabled: true`
2. Run the benchmark: `python run_webvoyager.py`
3. Watch for coaching messages in the logs:

```
🤖 WebCoach enabled - Model: gpt-4o, Frequency: 5
🎯 Running with WebCoach guidance...
🤖 Coach intervention: Based on similar past failures, try alternative search terms...
```

## How It Works

1. **During Execution**: After every N steps (configurable), the coach callback is triggered
2. **Trajectory Analysis**: Current agent history is processed by the condenser
3. **Experience Retrieval**: Similar past experiences are retrieved from the EMS
4. **Advice Generation**: WebCoach analyzes patterns and decides whether to intervene
5. **Guidance Injection**: If intervention is needed, advice is injected as a system message

## Key Features

- **Non-invasive**: Uses browser-use's official hook system, no core modifications
- **Configurable**: Enable/disable, adjust frequency, choose models
- **Persistent Learning**: Experiences accumulate across runs and sessions
- **Smart Intervention**: Only provides advice when relevant patterns are detected
- **Failure Prevention**: Focuses on avoiding repeated mistakes and dead ends

## Example Coaching Scenarios

- **Repeated Errors**: "Previous similar attempts failed with timeouts. Try refreshing the page first."
- **Navigation Loops**: "Similar cases got stuck in navigation loops. Consider using direct URL navigation."
- **Search Failures**: "Past experiences show this search term doesn't work. Try 'X' instead."
- **Authentication Issues**: "Similar tasks failed due to login requirements. Check if authentication is needed."

## Dependencies

- `openai`: For LLM calls and embeddings
- `numpy`: For vector similarity calculations
- `pydantic`: For data validation
- `pathlib`: For file management
- `browser-use`: For agent integration

## File Structure

```
WebCoach/
├── __init__.py              # Package initialization
├── coach_callback.py        # Main integration callback
├── condenser.py            # Trajectory processing
├── ems.py                  # External Memory Store
├── web_coach.py            # Advice generation
├── config.py               # Configuration management
└── README.md               # This file
```

## Development

The framework is designed to be modular and extensible:

- **New Models**: Add support for different LLMs by modifying the config system
- **Better Retrieval**: Enhance EMS with more sophisticated ranking algorithms
- **Richer Summaries**: Improve condenser to extract more detailed patterns
- **Advanced Coaching**: Enhance WebCoach with more sophisticated reasoning

## Troubleshooting

- **Coach Not Starting**: Check OPENAI_API_KEY environment variable
- **No Advice Given**: Verify EMS has stored experiences from previous runs
- **High Frequency Issues**: Reduce `frequency` setting to avoid overwhelming the agent
- **Storage Issues**: Check write permissions for `storage_dir`
