#!/usr/bin/env python3
import json
import os
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

def calculate_results(base_path):
    """Calculate results for the new directory structure: outputs/webvoyager/{model}/"""
    results = {}
    
    # Find all model directories in the webvoyager output folder
    webvoyager_path = os.path.join(base_path, 'webvoyager')
    if not os.path.exists(webvoyager_path):
        print(f"WebVoyager output directory not found: {webvoyager_path}")
        return results
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(webvoyager_path) 
                  if os.path.isdir(os.path.join(webvoyager_path, d))]
    
    if not model_dirs:
        print(f"No model directories found in: {webvoyager_path}")
        return results
    
    print(f"Found models: {model_dirs}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(webvoyager_path, model_dir)
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
        
        results[model_dir] = model_results
    
    return results

def print_results(results):
    """Print results in formatted tables"""
    if not results:
        print("No results to display")
        return
        
    print("\nWebVoyager Per-Subtask Success Rates")
    
    # Get all subtasks across all models
    all_subtasks = set()
    for model_results in results.values():
        all_subtasks.update(model_results.keys())
    
    all_subtasks = sorted(all_subtasks)
    models = sorted(results.keys())
    
    # Calculate column widths dynamically  
    subtask_width = max(20, max(len(subtask) for subtask in all_subtasks) + 2)
    
    # Print header and calculate column widths for proper alignment
    header_parts = [f"{'Subtask':<{subtask_width}}"]
    column_widths = [subtask_width]  # Track actual column widths
    
    for model in models:
        # Shorten model name for display
        model_short = model.replace('gpt-4o', 'GPT4o').replace('qwen_', 'Q').replace('_', '')
        
        # Calculate actual width needed for each header
        sr_header = f"{model_short} SR"
        st_header = f"{model_short} S/T"
        time_header = f"{model_short} Time"
        steps_header = f"{model_short} Steps"
        
        # Use the actual header width for each column
        header_parts.extend([sr_header, st_header, time_header, steps_header])
        column_widths.extend([len(sr_header), len(st_header), len(time_header), len(steps_header)])
    
    # Calculate total width for separator lines
    total_width = len(" ".join(header_parts))
    print("=" * total_width)
    print(" ".join(header_parts))
    print("-" * total_width)
    
    # Print data rows - align data to match header column widths
    for subtask in all_subtasks:
        row_parts = []
        col_idx = 0
        
        # Add subtask name
        row_parts.append(f"{subtask:<{column_widths[col_idx]}}")
        col_idx += 1
        
        for model in models:
            model_data = results.get(model, {}).get(subtask, {
                'success_rate': 0, 'successful': 0, 'total': 0, 'avg_time': 0, 'avg_steps': 0
            })
            
            sr = f"{model_data['success_rate']:.3f}"
            st = f"{model_data['successful']}/{model_data['total']}"
            time = f"{model_data['avg_time']:.0f}s"
            steps = f"{model_data['avg_steps']:.1f}"
            
            # Align each data value to its corresponding header column width
            row_parts.extend([
                f"{sr:<{column_widths[col_idx]}}",     # SR column
                f"{st:<{column_widths[col_idx+1]}}",   # S/T column  
                f"{time:<{column_widths[col_idx+2]}}", # Time column
                f"{steps:<{column_widths[col_idx+3]}}" # Steps column
            ])
            col_idx += 4
        
        print(" ".join(row_parts))
    
    print("\n" + "=" * total_width)
    print("WebVoyager Overall Statistics:")
    
    for model in models:
        total_successful = sum(stats['successful'] for stats in results[model].values())
        total_tasks = sum(stats['total'] for stats in results[model].values())
        overall_sr = total_successful / total_tasks if total_tasks > 0 else 0
        
        # Shorten model name for display
        model_short = model.replace('gpt-4o', 'GPT-4o').replace('qwen_', 'Qwen-').replace('_', '')
        print(f"{model_short}: {overall_sr:.3f} ({total_successful}/{total_tasks})")

if __name__ == "__main__":
    base_path = "/home/genglin/scripts_gl/run_benchmark/outputs"
    results = calculate_results(base_path)
    print_results(results)