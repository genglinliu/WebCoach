#!/usr/bin/env python3
"""
WebVoyager Benchmark Runner

This script runs WebVoyager tasks using the browser-use library without evaluation.
It focuses on task execution and result collection, organizing results by subtask groups.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv
import os

from simple_runner import WebVoyagerRunner
from data_loader import WebVoyagerDataLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def list_subtasks(config: Dict):
    """List all available subtasks."""
    try:
        data_loader = WebVoyagerDataLoader(config.get('data', {}).get('input_file'))
        tasks_by_subtask = data_loader.load_tasks_by_web_name()
        
        print(f"\nAvailable WebVoyager Subtasks ({len(tasks_by_subtask)}):")
        print("=" * 60)
        
        for web_name, tasks in tasks_by_subtask.items():
            print(f"\n{web_name}:")
            print(f"  Tasks: {len(tasks)}")
            print(f"  Sample task: {tasks[0].ques[:80]}...")
        
        print(f"\nTotal tasks across all subtasks: {sum(len(tasks) for tasks in tasks_by_subtask.values())}")
        
    except Exception as e:
        logger.error(f"Error listing subtasks: {e}")
        sys.exit(1)


async def run_benchmark(config: Dict, subtask_names: List[str] = None):
    """Run the WebVoyager benchmark."""
    try:
        logger.info("Starting WebVoyager benchmark...")
        
        # Create runner
        runner = WebVoyagerRunner(config)
        
        # Run tasks
        if subtask_names:
            logger.info(f"Running subtasks: {subtask_names}")
            results = await runner.run_subtasks(subtask_names)
        else:
            logger.info("Running all subtasks")
            results = await runner.run_subtasks()
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"\nBenchmark completed!")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        
        # Show results by subtask
        results_by_subtask = {}
        for result in results:
            if result.web_name not in results_by_subtask:
                results_by_subtask[result.web_name] = []
            results_by_subtask[result.web_name].append(result)
        
        print(f"\nResults by subtask:")
        print("-" * 40)
        for web_name, subtask_results in results_by_subtask.items():
            subtask_successful = sum(1 for r in subtask_results if r.success)
            print(f"{web_name}: {subtask_successful}/{len(subtask_results)} successful")
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        sys.exit(1)


async def run_single_subtask(config: Dict, subtask_name: str):
    """Run a single subtask."""
    try:
        logger.info(f"Running single subtask: {subtask_name}")
        
        # Create runner
        runner = WebVoyagerRunner(config)
        
        # Verify subtask exists
        available_subtasks = runner.get_available_subtasks()
        if subtask_name not in available_subtasks:
            logger.error(f"Subtask '{subtask_name}' not found. Available: {available_subtasks}")
            sys.exit(1)
        
        # Run the subtask
        results = await runner.run_single_subtask(subtask_name)
        
        # Print results
        successful = sum(1 for r in results if r.success)
        print(f"\nSubtask '{subtask_name}' completed!")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        logger.info(f"Subtask '{subtask_name}' completed successfully")
        
    except Exception as e:
        logger.error(f"Error running subtask: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run WebVoyager benchmark using browser-use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available subtasks
  python run_webvoyager_without_coach.py --list-subtasks
  
  # Run all subtasks
  python run_webvoyager_without_coach.py
  
  # Run specific subtasks
  python run_webvoyager_without_coach.py --subtasks Amazon Google
  
  # Run single subtask
  python run_webvoyager_without_coach.py --subtask Amazon
  
  # Use custom config
  python run_webvoyager_without_coach.py --config custom_config.yaml

Note: Multiple subtasks run in parallel, but tasks within each subtask run sequentially.
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--list-subtasks',
        action='store_true',
        help='List all available subtasks and exit'
    )
    
    parser.add_argument(
        '--subtasks',
        nargs='+',
        help='Run specific subtasks by name'
    )
    
    parser.add_argument(
        '--subtask',
        help='Run a single subtask by name'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Handle different modes
    if args.list_subtasks:
        list_subtasks(config)
        return
    
    if args.subtask:
        asyncio.run(run_single_subtask(config, args.subtask))
        return
    
    # Run benchmark
    asyncio.run(run_benchmark(config, args.subtasks))


if __name__ == "__main__":
    main()
