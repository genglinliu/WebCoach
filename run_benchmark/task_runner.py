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

from dotenv import load_dotenv

from data_loader import WebVoyagerDataLoader, WebVoyagerTask

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    web_name: str  # Add web_name for proper directory organization
    success: bool
    result: Optional[str]
    execution_time: float
    urls_visited: List[str]
    action_names: List[str]
    error: Optional[str] = None


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
    
    async def execute_task(self, task: WebVoyagerTask) -> TaskResult:
        """Execute a single WebVoyager task."""
        # Import browser_use here to avoid import-time config directory creation
        from browser_use import Agent, BrowserProfile, ChatOpenAI
        
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Executing task: {task.id} - {task.web_name}")
        print(f"🚀 Starting task: {task.id} - {task.question[:60]}...")
        
        try:
            # Initialize LLM - using correct parameters from example.py
            llm_config = self.config.get('llm', {})
            llm = ChatOpenAI(
                model=llm_config.get('model', 'gpt-4o')
                # Note: temperature and max_tokens are not supported by browser-use ChatOpenAI
            )
            
            # Initialize browser profile - using correct parameters from example.py
            browser_config = self.config.get('browser', {})
            browser_profile = BrowserProfile(
                headless=browser_config.get('headless', True),
                disable_dev_shm_usage=browser_config.get('disable_dev_shm_usage', True),
                no_sandbox=browser_config.get('no_sandbox', True),
                # Disable extensions to avoid deadlocks
                extensions=[],
                # Add timeout to prevent hanging
                browser_start_timeout=30
            )
            
            # Create agent with the task - using correct pattern from example.py
            # For evaluation purposes, we directly navigate to the target URL as the first step
            # This ensures consistent behavior and reliable navigation for all tasks
            # The agent can then focus on the actual task rather than navigation
            # Benefits for evaluation:
            # - Consistent starting point for all tasks
            # - Reliable navigation (no LLM interpretation errors)
            # - Focus on task completion performance, not navigation
            # - Faster execution (skip navigation reasoning)
            enhanced_task = f"Task: {task.question}\n\nPlease complete this task on the website."
            
            # Create agent with initial action to navigate directly to the target URL
            # initial_actions format: list of dictionaries with action name and parameters
            # This runs before the LLM gets involved, ensuring reliable navigation
            agent = Agent(
                task=enhanced_task,
                llm=llm,
                browser_profile=browser_profile,
                initial_actions=[{'go_to_url': {'url': task.web_url, 'new_tab': False}}]
            )
            
            # Run the agent with the task - the agent will use go_to_url tool automatically
            try:
                # Add timeout to prevent deadlocks
                history = await asyncio.wait_for(agent.run(max_steps=50), timeout=300)  # 5 minute timeout
            except asyncio.TimeoutError:
                logger.error(f"Task {task.id} timed out after 5 minutes")
                print(f"⏰ Task {task.id} timed out after 5 minutes")
                raise Exception("Task execution timed out - possible deadlock detected")
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Save agent history if enabled
            agent_history_path = None
            if self.save_agent_history:
                agent_history_path = self._save_agent_history(task, history)
                print(f"💾 Agent history saved: {agent_history_path}")
                
                # Also save history with embedded screenshots like example.py does
                history_with_screenshots_path = self._save_history_with_screenshots(task, history)
                if history_with_screenshots_path:
                    print(f"📸 History with screenshots saved: {history_with_screenshots_path}")
            
            # Save final screenshot if enabled
            screenshot_path = None
            if self.config.get('screenshots', {}).get('enabled', True):
                screenshot_path = self._save_screenshot(task, history)
                if screenshot_path:
                    print(f"📸 Final screenshot saved: {screenshot_path}")
            
            # Extract output using proper history methods
            output = history.final_result() if history.final_result() else "Task did not complete"
            
            # Create result
            result = TaskResult(
                task_id=task.id,
                web_name=task.web_name,
                success=history.is_successful() is not None,
                result=output,
                execution_time=execution_time,
                urls_visited=history.urls(),
                action_names=history.action_names()
            )
            
            # Print real-time status
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"{status} Task {task.id} completed in {execution_time:.2f}s")
            if result.error:
                print(f"   Error: {result.error}")
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error executing task {task.id}: {e}")
            print(f"💥 Task {task.id} failed with error: {e}")
            
            # Create error result
            error_result = TaskResult(
                task_id=task.id,
                web_name=task.web_name,
                success=False,
                result="Task failed with error",
                execution_time=execution_time,
                urls_visited=[],
                action_names=[]
            )
            

            
            return error_result
    
    def _save_agent_history(self, task: WebVoyagerTask, history) -> str:
        """Save agent execution history to file."""
        try:
            # Create web_name subdirectory under history_without_screenshots
            history_dir = self.results_dir / "history_without_screenshots"
            web_dir = history_dir / task.web_name
            task_dir = web_dir / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Save history to file using the correct method from example.py
            history_file = task_dir / "agent_history.json"
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
    
    def _save_history_with_screenshots(self, task: WebVoyagerTask, history) -> str:
        """Save agent history with embedded screenshots like example.py does."""
        try:
            # Create history directory structure
            history_dir = self.results_dir / "history_with_screenshots"
            web_dir = history_dir / task.web_name
            task_dir = web_dir / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Get the raw data and replace screenshot_paths with base64 data
            history_data = history.model_dump()
            screenshots = history.screenshots()
            
            # Extract screenshots to PNG files if enabled
            if self.config.get('screenshots', {}).get('enabled', True) and screenshots:
                screenshots_dir = task_dir / "screenshots"
                screenshots_dir.mkdir(exist_ok=True)
                
                extracted_count = 0
                for i, screenshot in enumerate(screenshots):
                    if screenshot:
                        try:
                            # Decode base64 and save as PNG
                            import base64
                            screenshot_data = base64.b64decode(screenshot)
                            screenshot_file = screenshots_dir / f"step_{i+1}.png"
                            
                            with open(screenshot_file, 'wb') as f:
                                f.write(screenshot_data)
                            
                            extracted_count += 1
                            print(f"   📸 Saved step_{i+1}.png")
                        except Exception as e:
                            print(f"   ❌ Failed to save step_{i+1}.png: {e}")
                
                if extracted_count > 0:
                    print(f"📸 Extracted {extracted_count} screenshots to {screenshots_dir}")
            
            # Replace screenshot_paths with base64 data in history
            for i, item in enumerate(history_data['history']):
                if 'state' in item and item['state'].get('screenshot_path'):
                    # Replace file path with actual base64 data
                    if i < len(screenshots) and screenshots[i]:
                        item['state']['screenshot_base64'] = screenshots[i]
                        item['state']['screenshot_path'] = None  # Clear the file path
            
            # Save to file
            history_file = task_dir / "agent_history.json"
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Saved history with screenshots to: {history_file}")
            return str(history_file)
            
        except Exception as e:
            logger.error(f"Error saving history with screenshots: {e}")
            return None
    
    def _save_intermediate_screenshots(self, task: WebVoyagerTask, history):
        """Save intermediate screenshots from task execution."""
        try:
            # Create web_name subdirectory
            web_dir = self.screenshots_dir / task.web_name
            web_dir.mkdir(exist_ok=True)
            
            # Get all screenshots from history
            screenshots = history.screenshots()
            if not screenshots or len(screenshots) <= 1:
                return
            
            # Save intermediate screenshots (all except the last one)
            for i, screenshot in enumerate(screenshots[:-1]):
                screenshot_file = web_dir / f"{task.id}_step_{i+1}_screenshot.png"
                
                # Convert base64 to image file
                import base64
                screenshot_data = base64.b64decode(screenshot)
                
                with open(screenshot_file, 'wb') as f:
                    f.write(screenshot_data)
                
                print(f"📸 Intermediate screenshot saved: {screenshot_file}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate screenshots: {e}")
    
    def _display_screenshot_info(self, task: WebVoyagerTask, history):
        """Display screenshot information in terminal."""
        try:
            screenshots = history.screenshots()
            if not screenshots:
                return
            
            print(f"📊 Screenshot Summary for Task {task.id}:")
            print(f"   📸 Total screenshots captured: {len(screenshots)}")
            
            # Show timing info if available
            if hasattr(history, 'timestamps') and history.timestamps():
                timestamps = history.timestamps()
                print(f"   ⏱️  Screenshots taken at:")
                for i, timestamp in enumerate(timestamps):
                    print(f"      Step {i+1}: {timestamp}")
            
            # Show a simple ASCII representation
            print(f"   🖼️  Screenshot sequence: {'📱' * len(screenshots)}")
            
        except Exception as e:
            logger.error(f"Error displaying screenshot info: {e}")
    
    async def _execute_with_realtime_monitoring(self, agent, task: WebVoyagerTask):
        """Execute agent with real-time screenshot monitoring."""
        try:
            print(f"🔍 Starting task execution for {task.id}...")
            
            # Execute the agent
            history = await agent.run()
            
            # Get all screenshots captured during execution
            screenshots = history.screenshots()
            if screenshots:
                print(f"📸 Captured {len(screenshots)} screenshots during execution")
                
                # Save screenshots immediately as they become available
                for i, screenshot in enumerate(screenshots):
                    screenshot_path = self._save_realtime_screenshot(task, screenshot, i + 1)
                    if screenshot_path:
                        print(f"   📱 Step {i+1} screenshot: {screenshot_path}")
                        
                        # Try to display a simple ASCII preview if possible
                        self._display_screenshot_preview(screenshot, i + 1)
                        
                        # Also save intermediate screenshots for analysis
                        self._save_intermediate_screenshots(task, history)
            else:
                print(f"⚠️  No screenshots captured for task {task.id}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error in task execution: {e}")
            # Fall back to normal execution
            return await agent.run()
    
    def _save_realtime_screenshot(self, task: WebVoyagerTask, screenshot_base64: str, step: int) -> str:
        """Save a screenshot captured during real-time execution."""
        try:
            # Create web_name subdirectory
            web_dir = self.screenshots_dir / task.web_name
            web_dir.mkdir(exist_ok=True)
            
            # Save with step number and timestamp
            import time
            timestamp = int(time.time())
            screenshot_file = web_dir / f"{task.id}_step_{step}_{timestamp}.png"
            
            # Convert base64 to image file
            import base64
            screenshot_data = base64.b64decode(screenshot_base64)
            
            with open(screenshot_file, 'wb') as f:
                f.write(screenshot_data)
            
            return str(screenshot_file)
            
        except Exception as e:
            logger.error(f"Error saving real-time screenshot: {e}")
            return None
    
    def _display_screenshot_preview(self, screenshot_base64: str, step: int):
        """Display a simple screenshot preview."""
        try:
            import base64
            image_data = base64.b64decode(screenshot_base64)
            print(f"      🖼️  Step {step}: Screenshot saved ({len(image_data)} bytes)")
        except Exception as e:
            logger.error(f"Error displaying screenshot preview: {e}")
    

    
    async def run_tasks(self, tasks: List[WebVoyagerTask]) -> List[TaskResult]:
        """Run multiple tasks sequentially."""
        results = []
        
        for task in tasks:
            result = await self.execute_task(task)
            results.append(result)
            
            # Small delay between tasks
            await asyncio.sleep(1)
        
        return results

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
                if result.task_id not in summary['results_by_web_name']:
                    summary['results_by_web_name'][result.task_id] = []
                
                summary['results_by_web_name'][result.task_id].append({
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'error': result.error
                })
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
