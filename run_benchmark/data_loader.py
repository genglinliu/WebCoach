"""
Data loader utility for WebVoyager evaluation dataset.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class WebVoyagerTask:
    """Represents a single WebVoyager evaluation task."""
    web_name: str
    id: str
    question: str
    web_url: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WebVoyagerTask':
        """Create WebVoyagerTask from dictionary."""
        return cls(
            web_name=data['web_name'],
            id=data['id'],
            question=data['ques'],
            web_url=data['web']
        )


class WebVoyagerDataLoader:
    """Data loader for WebVoyager evaluation dataset."""
    
    def __init__(self, data_file: Union[str, Path]):
        """Initialize data loader.
        
        Args:
            data_file: Path to the WebVoyager JSONL data file
        """
        self.data_file = Path(data_file)
        self._tasks: Optional[List[WebVoyagerTask]] = None
        self._tasks_by_web_name: Optional[Dict[str, List[WebVoyagerTask]]] = None
    
    def load_tasks(self) -> List[WebVoyagerTask]:
        """Load all tasks from the data file.
        
        Returns:
            List of WebVoyagerTask objects
        """
        if self._tasks is not None:
            return self._tasks
            
        tasks = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    task = WebVoyagerTask.from_dict(data)
                    tasks.append(task)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                except KeyError as e:
                    print(f"Warning: Missing required field on line {line_num}: {e}")
        
        self._tasks = tasks
        return tasks
    
    def get_tasks_by_web_name(self) -> Dict[str, List[WebVoyagerTask]]:
        """Group tasks by web_name.
        
        Returns:
            Dictionary mapping web_name to list of tasks
        """
        if self._tasks_by_web_name is not None:
            return self._tasks_by_web_name
            
        tasks = self.load_tasks()
        tasks_by_web_name = {}
        
        for task in tasks:
            if task.web_name not in tasks_by_web_name:
                tasks_by_web_name[task.web_name] = []
            tasks_by_web_name[task.web_name].append(task)
        
        self._tasks_by_web_name = tasks_by_web_name
        return tasks_by_web_name
    
    def get_available_web_names(self) -> List[str]:
        """Get list of available web_name values.
        
        Returns:
            Sorted list of web_name values
        """
        tasks_by_web_name = self.get_tasks_by_web_name()
        return sorted(tasks_by_web_name.keys())
    
    def get_tasks_for_web_names(self, web_names: List[str], 
                               sample_size: Optional[int] = None) -> Dict[str, List[WebVoyagerTask]]:
        """Get tasks for specific web_names with optional sampling.
        
        Args:
            web_names: List of web_name values to include
            sample_size: Maximum number of tasks per web_name (None for all)
            
        Returns:
            Dictionary mapping web_name to list of tasks
        """
        all_tasks = self.get_tasks_by_web_name()
        filtered_tasks = {}
        
        for web_name in web_names:
            if web_name not in all_tasks:
                print(f"Warning: web_name '{web_name}' not found in dataset")
                continue
                
            tasks = all_tasks[web_name]
            
            if sample_size is not None and len(tasks) > sample_size:
                # Randomly sample tasks
                tasks = random.sample(tasks, sample_size)
            
            filtered_tasks[web_name] = tasks
        
        return filtered_tasks
    
    def get_task_counts(self) -> Dict[str, int]:
        """Get count of tasks per web_name.
        
        Returns:
            Dictionary mapping web_name to task count
        """
        tasks_by_web_name = self.get_tasks_by_web_name()
        return {web_name: len(tasks) for web_name, tasks in tasks_by_web_name.items()}
    
    def print_summary(self):
        """Print a summary of the dataset."""
        tasks = self.load_tasks()
        tasks_by_web_name = self.get_tasks_by_web_name()
        task_counts = self.get_task_counts()
        
        print(f"WebVoyager Dataset Summary")
        print(f"==========================")
        print(f"Total tasks: {len(tasks)}")
        print(f"Total web_names: {len(tasks_by_web_name)}")
        print(f"")
        print(f"Tasks per web_name:")
        for web_name in sorted(task_counts.keys()):
            count = task_counts[web_name]
            print(f"  {web_name}: {count} tasks")


if __name__ == "__main__":
    # Example usage
    data_file = "../benchmark_data/WebVoyager/WebVoyager_data.jsonl"
    loader = WebVoyagerDataLoader(data_file)
    loader.print_summary()
    
    # Example: Get tasks for specific web_names
    web_names = ["Amazon", "Google Search"]
    tasks = loader.get_tasks_for_web_names(web_names, sample_size=5)
    
    print(f"\nSample tasks for {web_names}:")
    for web_name, task_list in tasks.items():
        print(f"{web_name}: {len(task_list)} tasks")
        for task in task_list[:2]:  # Show first 2 tasks
            print(f"  - {task.id}: {task.question[:50]}...")
