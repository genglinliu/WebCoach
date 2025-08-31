"""
Simple WebVoyager Task Runner

Runs multiple subsets in parallel, but tasks within each subset sequentially.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from data_loader import WebVoyagerDataLoader, WebVoyagerTask
from task_runner import WebVoyagerTaskRunner, TaskResult

logger = logging.getLogger(__name__)


class WebVoyagerRunner:
    """Simple runner: subsets in parallel, tasks within each subset sequentially."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.task_runner = WebVoyagerTaskRunner(config)
    
    async def run_subtasks(self, subtask_names: Optional[List[str]] = None) -> List[TaskResult]:
        """Run specified subtasks. Each subtask runs its tasks sequentially."""
        # Load data
        data_loader = WebVoyagerDataLoader(self.config.get('data', {}).get('input_file'))
        tasks_by_subtask = data_loader.get_tasks_by_web_name()
        
        # Determine which subtasks to run
        if subtask_names:
            target_subtasks = subtask_names
        else:
            config_subtasks = self.config.get('subtasks')
            target_subtasks = config_subtasks if config_subtasks else list(tasks_by_subtask.keys())
        
        # Filter tasks by target subtasks
        filtered_tasks = {name: tasks for name, tasks in tasks_by_subtask.items() 
                         if name in target_subtasks}
        
        logger.info(f"Running {len(filtered_tasks)} subtasks in parallel")
        logger.info(f"Target subtasks: {target_subtasks}")
        
        # Run each subtask (tasks within each run sequentially)
        subtask_coroutines = [
            self._run_subtask_sequentially(web_name, tasks) 
            for web_name, tasks in filtered_tasks.items()
        ]
        
        # Execute all subtasks concurrently
        subtask_results = await asyncio.gather(*subtask_coroutines, return_exceptions=True)
        
        # Collect all results
        all_results = []
        for i, result in enumerate(subtask_results):
            if isinstance(result, Exception):
                web_name = list(filtered_tasks.keys())[i]
                logger.error(f"Subtask '{web_name}' failed: {result}")
            else:
                all_results.extend(result)
        
        # Save summary
        self.task_runner.save_summary(all_results)
        return all_results
    
    async def _run_subtask_sequentially(self, web_name: str, tasks: List[WebVoyagerTask]) -> List[TaskResult]:
        """Run all tasks for a single subtask sequentially."""
        logger.info(f"Starting subtask '{web_name}' with {len(tasks)} tasks")
        results = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Executing task {i+1}/{len(tasks)} in '{web_name}': {task.question[:60]}...")
            try:
                result = await self.task_runner.execute_task(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                error_result = TaskResult(
                    task_id=task.id,
                    web_name=task.web_name,
                    question=task.question,
                    url=task.web_url,
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
        
        logger.info(f"Completed subtask '{web_name}' - {len(results)} tasks")
        return results
    
    async def run_single_subtask(self, subtask_name: str) -> List[TaskResult]:
        """Run a single subtask."""
        return await self.run_subtasks([subtask_name])
    
    def get_available_subtasks(self) -> List[str]:
        """Get list of available subtask names."""
        data_loader = WebVoyagerDataLoader(self.config.get('data', {}).get('input_file'))
        tasks_by_subtask = data_loader.get_tasks_by_web_name()
        return list(tasks_by_subtask.keys())
