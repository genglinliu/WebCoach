"""
WebVoyager Task Runner using browser-use

This module provides a simple task runner that executes WebVoyager tasks using the browser-use library.
It follows the correct browser-use patterns and focuses on task execution and result collection.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from browser_use import Agent, BrowserProfile, ChatOpenAI
from dotenv import load_dotenv

from data_loader import WebVoyagerDataLoader, WebVoyagerTask

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a WebVoyager task execution."""
    task_id: str
    web_name: str
    question: str
    url: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    screenshot_path: Optional[str] = None
    agent_history_path: Optional[str] = None
    execution_time: Optional[float] = None


class WebVoyagerTaskRunner:
    """Simple task runner for WebVoyager tasks using browser-use."""
    
    def __init__(self, config: Dict):
        """Initialize the task runner with configuration."""
        self.config = config
        self.results_dir = Path(config.get('output', {}).get('results_dir', './results'))
        self.screenshots_dir = Path(config.get('screenshots', {}).get('save_path', './screenshots'))
        self.save_agent_history = config.get('output', {}).get('save_agent_history', True)
        
        # Create output directories
        self.results_dir.mkdir(exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Initialize LLM - using correct parameters from example.py
        llm_config = config.get('llm', {})
        self.llm = ChatOpenAI(
            model=llm_config.get('model', 'gpt-4o')
            # Note: temperature and max_tokens are not supported by browser-use ChatOpenAI
        )
        
        # Initialize browser profile - using correct parameters from example.py
        browser_config = config.get('browser', {})
        self.browser_profile = BrowserProfile(
            headless=browser_config.get('headless', True),
            disable_dev_shm_usage=browser_config.get('disable_dev_shm_usage', True),
            no_sandbox=browser_config.get('no_sandbox', True)
            # Note: window_size is not supported by browser-use BrowserProfile
        )
    
    async def execute_task(self, task: WebVoyagerTask) -> TaskResult:
        """Execute a single WebVoyager task."""
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Executing task: {task.id} - {task.web_name}")
        
        try:
            # Create agent with the task - using correct pattern from example.py
            agent = Agent(
                task=task.question,
                llm=self.llm,
                browser_profile=self.browser_profile
            )
            
            # Execute the task
            history = await agent.run()
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Save agent history if enabled
            agent_history_path = None
            if self.save_agent_history:
                agent_history_path = self._save_agent_history(task, history)
            
            # Save screenshot if enabled
            screenshot_path = None
            if self.config.get('screenshots', {}).get('enabled', True):
                screenshot_path = self._save_screenshot(task, history)
            
            # Extract output
            output = history.final_result() if history.is_done() else "Task did not complete"
            
            return TaskResult(
                task_id=task.id,
                web_name=task.web_name,
                question=task.question,
                url=task.web_url,
                success=history.is_successful() is not None,
                output=output,
                screenshot_path=screenshot_path,
                agent_history_path=agent_history_path,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error executing task {task.id}: {e}")
            
            return TaskResult(
                task_id=task.id,
                web_name=task.web_name,
                question=task.question,
                url=task.web_url,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _save_agent_history(self, task: WebVoyagerTask, history) -> str:
        """Save agent execution history to file."""
        try:
            # Create web_name subdirectory
            web_dir = self.results_dir / task.web_name
            web_dir.mkdir(exist_ok=True)
            
            # Save history to file using the correct method from example.py
            history_file = web_dir / f"{task.id}_history.json"
            history.save_to_file(str(history_file))
            
            logger.info(f"Saved agent history to: {history_file}")
            return str(history_file)
            
        except Exception as e:
            logger.error(f"Error saving agent history: {e}")
            return None
    
    def _save_screenshot(self, task: WebVoyagerTask, history) -> str:
        """Save screenshot from task execution."""
        try:
            # Create web_name subdirectory
            web_dir = self.screenshots_dir / task.web_name
            web_dir.mkdir(exist_ok=True)
            
            # Get screenshots from history
            screenshots = history.screenshots()
            if not screenshots:
                return None
            
            # Save the last screenshot
            screenshot_file = web_dir / f"{task.id}_screenshot.png"
            
            # Convert base64 to image file
            import base64
            screenshot_data = base64.b64decode(screenshots[-1])
            
            with open(screenshot_file, 'wb') as f:
                f.write(screenshot_data)
            
            logger.info(f"Saved screenshot to: {screenshot_file}")
            return str(screenshot_file)
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
    
    async def run_tasks(self, tasks: List[WebVoyagerTask]) -> List[TaskResult]:
        """Run multiple tasks sequentially."""
        results = []
        
        for task in tasks:
            result = await self.execute_task(task)
            results.append(result)
            
            # Save individual result if enabled
            if self.config.get('output', {}).get('save_individual_results', True):
                self._save_individual_result(result)
            
            # Small delay between tasks
            await asyncio.sleep(1)
        
        return results
    
    def _save_individual_result(self, result: TaskResult):
        """Save individual task result to file."""
        try:
            # Create web_name subdirectory
            web_dir = self.results_dir / result.web_name
            web_dir.mkdir(exist_ok=True)
            
            # Save result to file
            result_file = web_dir / f"{result.task_id}_result.json"
            
            # Convert result to dict for JSON serialization
            result_dict = {
                'task_id': result.task_id,
                'web_name': result.web_name,
                'question': result.question,
                'url': result.url,
                'success': result.success,
                'output': result.output,
                'error': result.error,
                'screenshot_path': result.screenshot_path,
                'agent_history_path': result.agent_history_path,
                'execution_time': result.execution_time
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Saved individual result to: {result_file}")
            
        except Exception as e:
            logger.error(f"Error saving individual result: {e}")
    
    def save_summary(self, results: List[TaskResult]):
        """Save summary of all results."""
        try:
            summary_file = self.results_dir / "summary.json"
            
            # Create summary data
            summary = {
                'total_tasks': len(results),
                'successful_tasks': len([r for r in results if r.success]),
                'failed_tasks': len([r for r in results if not r.success]),
                'total_execution_time': sum(r.execution_time or 0 for r in results),
                'results_by_web_name': {}
            }
            
            # Group results by web_name
            for result in results:
                if result.web_name not in summary['results_by_web_name']:
                    summary['results_by_web_name'][result.web_name] = []
                
                summary['results_by_web_name'][result.web_name].append({
                    'task_id': result.task_id,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'error': result.error
                })
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
