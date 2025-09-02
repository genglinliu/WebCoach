#!/usr/bin/env python3
import json
import os
import yaml
from pathlib import Path
from collections import defaultdict

def analyze_task(agent_history_path):
    """Analyze task for success, timing, and step count"""
    try:
        with open(agent_history_path, 'r') as f:
            data = json.load(f)
        
        # Check for success by looking for both required strings in the JSON
        data_str = json.dumps(data)
        is_done = '"is_done": true' in data_str
        success = '"success": true' in data_str
        is_successful = is_done and success
        
        history = data.get('history', [])
        if not history:
            return is_successful, 0, 0, data.get('level')
        
        first_step_start = None
        last_step_end = None
        total_steps = len(history)
        
        for step in history:
            if step is None:
                continue
            metadata = step.get('metadata', {})
            if metadata is None:
                continue
            step_start = metadata.get('step_start_time')
            step_end = metadata.get('step_end_time')
            
            if step_start is not None:
                if first_step_start is None or step_start < first_step_start:
                    first_step_start = step_start
            
            if step_end is not None:
                if last_step_end is None or step_end > last_step_end:
                    last_step_end = step_end
        
        total_time = 0
        if first_step_start is not None and last_step_end is not None:
            total_time = last_step_end - first_step_start
        
        return is_successful, total_time, total_steps, data.get('level')
        
    except Exception as e:
        print(f"Error reading {agent_history_path}: {e}")
        return False, 0, 0, None

def load_config():
    """Load configuration from YAML file"""
    # Look for config.yaml in the run_benchmark directory
    config_path = Path(__file__).parent.parent / "run_benchmark" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def calculate_results(base_path=None):
    """Calculate results for the new directory structure: {base_dir}/{exp_config}/webvoyager/{model}/"""
    results = {}
    
    # Load config to get base directory if not provided
    if base_path is None:
        try:
            config = load_config()
            base_path = config['output']['base_dir']
            print(f"📁 Using output base directory from config: {base_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("Using default path: /home/genglin/benchmark_outputs")
            base_path = "/home/genglin/benchmark_outputs"
    
    # Look for experiment directories in the base path
    if not os.path.exists(base_path):
        print(f"Base output directory not found: {base_path}")
        return results
    
    # Find all experiment directories (e.g., baseline_no_coach, with_coach_*)
    exp_dirs = [d for d in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, d))]
    
    if not exp_dirs:
        print(f"No experiment directories found in: {base_path}")
        return results
    
    print(f"Found experiment directories: {exp_dirs}")
    
    # Process each experiment directory
    for exp_dir in exp_dirs:
        exp_path = os.path.join(base_path, exp_dir)
        webvoyager_path = os.path.join(exp_path, 'webvoyager')
        
        if not os.path.exists(webvoyager_path):
            print(f"WebVoyager output directory not found: {webvoyager_path}")
            continue
    
        # Get all model directories for this experiment
        model_dirs = [d for d in os.listdir(webvoyager_path) 
                      if os.path.isdir(os.path.join(webvoyager_path, d))]
        
        if not model_dirs:
            print(f"No model directories found in: {webvoyager_path}")
            continue
        
        print(f"Found models in {exp_dir}: {model_dirs}")
        
        for model_dir in model_dirs:
            model_path = os.path.join(webvoyager_path, model_dir)
            # Create a unique key that includes experiment directory
            model_key = f"{exp_dir}/{model_dir}"
            subtask_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'total_time': 0, 'total_steps': 0})
            
            # Process each subset (subtask) directory
            for subset_folder in os.listdir(model_path):
                subset_path = os.path.join(model_path, subset_folder)
                if not os.path.isdir(subset_path):
                    continue
                    
                # Process each task directory within the subset
                for task_folder in os.listdir(subset_path):
                    task_path = os.path.join(subset_path, task_folder)
                    if not os.path.isdir(task_path):
                        continue
                        
                    agent_history_path = os.path.join(task_path, 'agent_history.json')
                    
                    if os.path.exists(agent_history_path):
                        is_successful, total_time, total_steps, _ = analyze_task(agent_history_path)
                        subtask_stats[subset_folder]['total'] += 1
                        subtask_stats[subset_folder]['total_time'] += total_time
                        subtask_stats[subset_folder]['total_steps'] += total_steps
                        if is_successful:
                            subtask_stats[subset_folder]['successful'] += 1
            
            # Calculate statistics for this model
            model_results = {}
            for subtask, stats in subtask_stats.items():
                sr = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
                avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
                avg_steps = stats['total_steps'] / stats['total'] if stats['total'] > 0 else 0
                model_results[subtask] = {
                    'success_rate': sr,
                    'successful': stats['successful'],
                    'total': stats['total'],
                    'avg_time': avg_time,
                    'avg_steps': avg_steps
                }
            
            results[model_key] = model_results
    
    return results

def print_results(results):
    """Print results in formatted tables"""
    if not results:
        print("No results to display")
        return
    
    # Group results by experiment
    experiments = {}
    for model_key, model_results in results.items():
        if '/' in model_key:
            exp_name, model_name = model_key.split('/', 1)
        else:
            exp_name = "default"
            model_name = model_key
        
        if exp_name not in experiments:
            experiments[exp_name] = {}
        experiments[exp_name][model_name] = model_results
    
    # Print results for each experiment separately
    for exp_name, exp_results in experiments.items():
        print(f"\nWebVoyager Results - {exp_name}")
        
        # Get all subtasks across all models in this experiment
        all_subtasks = set()
        for model_results in exp_results.values():
            all_subtasks.update(model_results.keys())
        
        all_subtasks = sorted(all_subtasks)
        models = sorted(exp_results.keys())
        
        # Print a separate table for each model
        for model in models:
            # Shorten model name for display
            model_short = model.replace('gpt-4o', 'GPT-4o').replace('qwen_', 'Qwen-').replace('_', '')
            print(f"\n{model_short} Per-Subtask Results:")
            
            # Calculate column widths dynamically  
            subtask_width = max(20, max(len(subtask) for subtask in all_subtasks) + 2)
            
            # Simple header for single model
            header_parts = [f"{'Subtask':<{subtask_width}}", "Success Rate", "Success/Total", "Avg Time", "Avg Steps"]
            column_widths = [subtask_width, 12, 12, 10, 10]
            
            # Calculate total width for separator lines
            total_width = len(" ".join(header_parts))
            print("=" * total_width)
            print(" ".join(header_parts))
            print("-" * total_width)
            
            # Print data rows
            for subtask in all_subtasks:
                model_data = exp_results.get(model, {}).get(subtask, {
                    'success_rate': 0, 'successful': 0, 'total': 0, 'avg_time': 0, 'avg_steps': 0
                })
                
                sr = f"{model_data['success_rate']:.3f}"
                st = f"{model_data['successful']}/{model_data['total']}"
                time = f"{model_data['avg_time']:.0f}s"
                steps = f"{model_data['avg_steps']:.1f}"
                
                # Align data to match header column widths
                row_parts = [
                    f"{subtask:<{column_widths[0]}}",
                    f"{sr:<{column_widths[1]}}",
                    f"{st:<{column_widths[2]}}",
                    f"{time:<{column_widths[3]}}",
                    f"{steps:<{column_widths[4]}}"
                ]
                
                print(" ".join(row_parts))
            
            # Print overall statistics for this model
            total_successful = sum(stats['successful'] for stats in exp_results[model].values())
            total_tasks = sum(stats['total'] for stats in exp_results[model].values())
            overall_sr = total_successful / total_tasks if total_tasks > 0 else 0
            
            print("\n" + "=" * total_width)
            print(f"{model_short} Overall: {overall_sr:.3f} ({total_successful}/{total_tasks})")

if __name__ == "__main__":
    import sys
    
    # Allow command line override of base path
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        print(f"📁 Using command line base path: {base_path}")
        results = calculate_results(base_path)
    else:
        # Use config file
        results = calculate_results()
    
    print_results(results)