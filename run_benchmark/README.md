# WebVoyager Benchmark Runner

A simple and efficient system for running WebVoyager tasks using the browser-use library. This system focuses on task execution and result collection, organizing results by subtask groups for easy analysis.

## Features

- **Task Execution**: Run WebVoyager tasks using browser-use agents
- **Concurrent Processing**: Execute multiple tasks concurrently with configurable limits
- **Result Organization**: Results are automatically organized by subtask (web_name)
- **Agent History**: Save complete agent execution history for each task
- **Screenshot Capture**: Save visual states during task execution
- **Flexible Configuration**: YAML-based configuration for easy customization

## System Overview

The system consists of several key components:

1. **Data Loader** (`data_loader.py`): Parses WebVoyager JSONL data and groups tasks by web_name
2. **Task Runner** (`task_runner.py`): Executes individual tasks using browser-use agents
3. **Concurrent Runner** (`concurrent_runner.py`): Manages concurrent execution of multiple tasks
4. **Main Script** (`run_webvoyager_without_coach.py`): CLI interface for running benchmarks

## Available Subtasks

The system supports all 15 WebVoyager subtasks:

- Allrecipes
- Amazon
- Apple
- ArXiv
- BBC News
- Booking
- Cambridge Dictionary
- Coursera
- ESPN
- GitHub
- Google Flights
- Google Map
- Google Search
- Huggingface
- Wolfram Alpha

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright**:
   ```bash
   playwright install chromium
   ```

3. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Configuration

Edit `config.yaml` to customize:

- **Concurrency**: Set `max_concurrent` for parallel task execution
- **LLM Settings**: Configure model, temperature, and token limits
- **Browser Settings**: Control headless mode, window size, and security options
- **Output Options**: Configure result saving, screenshots, and agent history
- **Data Source**: Specify the WebVoyager data file path

## Usage

### List Available Subtasks
```bash
python run_webvoyager_without_coach.py --list-subtasks
```

### Run All Subtasks
```bash
python run_webvoyager_without_coach.py
```

### Run Specific Subtasks
```bash
python run_webvoyager_without_coach.py --subtasks Amazon Google
```

### Run Single Subtask
```bash
python run_webvoyager_without_coach.py --subtask Amazon
```

### Use Custom Configuration
```bash
python run_webvoyager_without_coach.py --config custom_config.yaml
```

## Output Structure

Results are organized by subtask in the following structure:

```
results/
├── summary.json                    # Overall benchmark summary
├── Allrecipes/                     # Subtask-specific directory
│   ├── Allrecipes--0_result.json  # Individual task result
│   ├── Allrecipes--0_history.json # Agent execution history
│   └── Allrecipes--1_result.json
├── Amazon/
│   ├── Amazon--0_result.json
│   └── Amazon--0_history.json
└── ...
```

### Result Files

- **`*_result.json`**: Task execution results including success status, output, and error messages
- **`*_history.json`**: Complete agent execution history with URLs visited, actions taken, and extracted content
- **`summary.json`**: Overall benchmark summary with success rates and timing information

### Screenshots

If enabled, screenshots are saved in:
```
screenshots/
├── Allrecipes/
│   ├── Allrecipes--0_screenshot.png
│   └── Allrecipes--1_screenshot.png
├── Amazon/
│   └── Amazon--0_screenshot.png
└── ...
```

## Docker Support

Run the benchmark in a Docker container:

```bash
./run_docker.sh
```

This script mounts the necessary directories and sets up the environment for containerized execution.

## Key Benefits

1. **No Evaluation Complexity**: Focuses purely on task execution and result collection
2. **Efficient Organization**: Results are automatically organized by subtask for easy analysis
3. **Complete History**: Saves full agent execution history for debugging and analysis
4. **Flexible Concurrency**: Configurable parallel execution for optimal performance
5. **Browser-use Integration**: Leverages the powerful browser-use library for web automation

## Example Result Structure

```json
{
  "task_id": "Amazon--0",
  "web_name": "Amazon",
  "question": "Find a wireless mouse under $50",
  "url": "https://www.amazon.com/",
  "success": true,
  "output": "Found Logitech M705 wireless mouse for $39.99",
  "screenshot_path": "./screenshots/Amazon/Amazon--0_screenshot.png",
  "agent_history_path": "./results/Amazon/Amazon--0_history.json",
  "execution_time": 45.2
}
```

## Troubleshooting

1. **API Key Issues**: Ensure `OPENAI_API_KEY` is set correctly
2. **Browser Launch Problems**: Install Playwright browsers with `playwright install chromium`
3. **Memory Issues**: Reduce `max_concurrent` in configuration
4. **Timeout Issues**: Increase `max_steps` in the task runner for complex tasks

## Performance Tips

1. **Concurrency**: Adjust `max_concurrent` based on your system's capabilities
2. **Headless Mode**: Use `headless: true` for faster execution
3. **Model Selection**: Choose appropriate LLM models based on task complexity
4. **Task Filtering**: Run specific subtasks instead of all for focused testing

## Contributing

This system is designed to be simple and extensible. Key areas for enhancement:

- Additional result analysis tools
- Custom task filtering and sampling
- Integration with other evaluation frameworks
- Enhanced error handling and recovery
