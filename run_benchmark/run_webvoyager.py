from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import json
import sys
import yaml
from utils import extract_screenshots_from_history

load_dotenv()

def create_llm_instance(model_name, qwen_ports=None):
    """Create LLM instance for either OpenAI or specific Qwen models"""
    # Define the specific models we support
    supported_qwen_models = {
        "qwen_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen_vl_32b": "Qwen/Qwen2.5-VL-32B-Instruct", 
        "qwen3_8b": "Qwen/Qwen3-8B"  # Qwen3 8B for coach
    }
    
    # List of supported OpenAI models
    supported_openai_models = {"gpt-4o"}
    
    if model_name in supported_qwen_models:
        # Qwen model - use sglang server
        if not qwen_ports:
            raise ValueError(f"Qwen ports configuration missing for model: {model_name}")
        
        if model_name not in qwen_ports:
            raise ValueError(f"Port not configured for Qwen model: {model_name}")
        
        port = qwen_ports[model_name]
        actual_model = supported_qwen_models[model_name]
        
        # Determine model type for logging
        if model_name.startswith("qwen_vl_"):
            model_type = "Qwen VL"
        elif model_name.startswith("qwen3_"):
            model_type = "Qwen3"
        else:
            model_type = "Qwen"
        
        print(f"🤖 Using {model_type} model: {actual_model} on port {port}")
        return ChatOpenAI(
            model=actual_model,
            base_url=f"http://host.docker.internal:{port}/v1",
            api_key="dummy",
            temperature=0.5,
        )
    elif model_name in supported_openai_models:
        # OpenAI model - use OpenAI API
        print(f"🤖 Using OpenAI model: {model_name}")
        return ChatOpenAI(model=model_name)
    else:
        # Unsupported model - throw error
        all_supported = list(supported_qwen_models.keys()) + list(supported_openai_models)
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models are: {', '.join(all_supported)}"
        )

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required config sections
    required_sections = ['browser', 'llm', 'agent', 'task', 'output', 'error_handling', 'subsets']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate subsets configuration
    if not isinstance(config['subsets'], list) or len(config['subsets']) == 0:
        raise ValueError("subsets must be a non-empty list")
    
    print(f"✅ Loaded configuration from: {config_path}")
    return config

def setup_coach(config):
    """Setup WebCoach if enabled in configuration"""
    coach_config = config.get('coach', {})
    
    if not coach_config.get('enabled', False):
        print("🤖 WebCoach disabled")
        return None
        
    try:
        # Import coach components
        sys.path.append(str(Path(__file__).parent.parent))
        from WebCoach.coach_callback import configure_coach, coach_step_callback
        from WebCoach.config import get_coach_config_from_main_config
        
        # Configure coach with validated config
        validated_coach_config = get_coach_config_from_main_config(config)
        
        # Create coach LLM instance if it's a supported Qwen model
        coach_model = validated_coach_config.get('model', 'gpt-4o')
        supported_qwen_models = ["qwen_vl_7b", "qwen_vl_32b", "qwen3_8b"]
        if coach_model in supported_qwen_models:
            qwen_ports = config['llm'].get('qwen_ports', {})
            coach_llm = create_llm_instance(coach_model, qwen_ports)
            # Update the config with the actual LLM instance
            validated_coach_config['llm_instance'] = coach_llm
        
        configure_coach(validated_coach_config)
        
        print(f"🤖 WebCoach enabled - Model: {coach_model}, Frequency: {validated_coach_config['frequency']}")
        print(f"📁 Coach storage: {validated_coach_config['storage_dir']}")
        
        return coach_step_callback
        
    except Exception as e:
        print(f"❌ Failed to setup WebCoach: {e}")
        print("🤖 Continuing without coaching...")
        return None

def filter_tasks_by_subsets(tasks, selected_subsets):
    """Filter tasks based on selected subsets"""
    if "all" in selected_subsets:
        print("📋 Running all available subsets")
        return tasks
    
    # Get all available web names from tasks
    available_subsets = set(task['web_name'] for task in tasks)
    
    # Validate selected subsets
    invalid_subsets = set(selected_subsets) - available_subsets
    if invalid_subsets:
        print(f"⚠️  Warning: Invalid subsets specified: {invalid_subsets}")
        print(f"📋 Available subsets: {sorted(available_subsets)}")
    
    # Filter tasks by valid selected subsets
    valid_subsets = set(selected_subsets) & available_subsets
    if not valid_subsets:
        raise ValueError(f"No valid subsets found. Available: {sorted(available_subsets)}")
    
    filtered_tasks = [task for task in tasks if task['web_name'] in valid_subsets]
    
    print(f"📋 Selected subsets: {sorted(valid_subsets)}")
    print(f"📊 Filtered to {len(filtered_tasks)} tasks from {len(tasks)} total tasks")
    
    return filtered_tasks

def task_already_exists(output_dir, web_name, task_id):
    """Check if a task output directory already exists"""
    subset_output_dir = output_dir / web_name
    task_output_dir = subset_output_dir / task_id
    
    # Check if the task directory exists and has content
    if task_output_dir.exists():
        # Check if it has the expected files
        history_file = task_output_dir / "agent_history.json"
        if history_file.exists():
            return True
    
    return False

async def run_webvoyager_benchmark():
    """Run WebVoyager benchmark tasks one by one"""
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please create a config.yaml file in the same directory as this script.")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Load the WebVoyager dataset
    dataset_path = Path("/app/scripts_gl/benchmark_data/WebVoyager/WebVoyager_data.jsonl")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Read all tasks
    tasks = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"📊 Loaded {len(tasks)} WebVoyager tasks")
    
    # Filter tasks based on selected subsets
    try:
        tasks = filter_tasks_by_subsets(tasks, config['subsets'])
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Setup LLM from config (supports both OpenAI and Qwen models)
    llm = create_llm_instance(
        config['llm']['model'],
        config['llm'].get('qwen_ports')
    )
    
    # Setup WebCoach if enabled
    coach_callback = setup_coach(config)
    
    browser_profile = BrowserProfile(
        headless=config['browser']['headless'],
        disable_dev_shm_usage=config['browser']['disable_dev_shm_usage'],
        no_sandbox=config['browser']['no_sandbox'],
        window_size=config['browser']['window_size'],
        viewport=config['browser']['viewport'],
        minimum_wait_page_load_time=config['browser']['minimum_wait_page_load_time'],
        wait_between_actions=config['browser']['wait_between_actions'],
        wait_for_network_idle_page_load_time=config['browser']['wait_for_network_idle_page_load_time'],
        enable_default_extensions=config['browser']['enable_default_extensions'],
        highlight_elements=config['browser']['highlight_elements'],
        accept_downloads=config['browser']['accept_downloads'],
        auto_download_pdfs=config['browser']['auto_download_pdfs']
    )
    
    # Create output directory from config: {base_dir}/{exp_config}/{dataset}/{model}/
    dataset_name = "webvoyager"  # This could be made configurable in the future
    model_name = config['llm']['model'].replace("/", "_")  # Replace / with _ for valid directory names
    
    # Create experiment configuration identifier
    if config.get('coach', {}).get('enabled', False):
        coach_config = config['coach']
        frequency = coach_config.get('frequency', 5)
        coach_model = coach_config.get('model', 'gpt-4o').replace("/", "_").replace("-", "_")
        exp_config = f"with_coach_freq_{frequency}_model_{coach_model}"
    else:
        exp_config = "baseline_no_coach"
    
    # Get base output directory from config
    base_output_dir = config['output']['base_dir']
    
    # Structure: {base_dir}/{exp_config}/{dataset}/{model}/
    output_dir = Path(base_output_dir) / exp_config / dataset_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Process each task
    skipped_count = 0
    for i, task_data in enumerate(tasks):
        task_id = task_data['id']
        web_name = task_data['web_name']
        question = task_data['ques']
        url = task_data['web']
        
        # Check if task already exists
        if task_already_exists(output_dir, web_name, task_id):
            print(f"\n{'='*80}")
            print(f"⏭️  Task {i+1}/{len(tasks)}: {task_id} - SKIPPED (already exists)")
            print(f"🌐 Website: {web_name}")
            print(f"🔗 URL: {url}")
            print(f"❓ Question: {question}")
            print(f"{'='*80}")
            skipped_count += 1
            continue
        
        print(f"\n{'='*80}")
        print(f"🔄 Task {i+1}/{len(tasks)}: {task_id}")
        print(f"🌐 Website: {web_name}")
        print(f"🔗 URL: {url}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}")
        
        # Create task description that includes the URL and question
        task_description = f"Go to {url} and answer this question: {question}"
        
        try:
            # Create and run agent with config
            agent = Agent(
                task=task_description,
                llm=llm,
                browser_profile=browser_profile,
                directly_open_url=True,  # This will automatically navigate to the URL
                max_actions_per_step=config['agent']['max_actions_per_step'],
                max_failures=config['agent']['max_failures'],
                use_thinking=config['agent']['use_thinking'],
                use_vision=config['agent']['use_vision'],
                vision_detail_level=config['agent']['vision_detail_level'],
                flash_mode=config['agent']['flash_mode']
            )
            
            print(f"🤖 Starting agent for task {task_id}...")
            # Run agent with or without coaching
            if coach_callback:
                print(f"🎯 Running with WebCoach guidance...")
                history = await agent.run(
                    max_steps=config['agent']['max_steps'],
                    on_step_end=coach_callback
                )
            else:
                history = await agent.run(max_steps=config['agent']['max_steps'])
            
            # Save task-specific history with subset grouping
            # Structure: outputs/webvoyager/gpt-4o/{subset_name}/{task_id}/
            subset_output_dir = output_dir / web_name
            task_output_dir = subset_output_dir / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save basic history if configured
            if config['output']['save_history']:
                history_file = task_output_dir / "agent_history.json"
                history.save_to_file(str(history_file))
            
            # Extract screenshots as PNG files if configured
            if config['output']['save_screenshots_as_png']:
                print("🖼️  Extracting screenshots as PNG files...")
                # Save history with screenshots temporarily for PNG extraction
                history_with_screenshots = task_output_dir / "agent_history_with_screenshots.json"
                
                # Get the raw data and replace screenshot_paths with base64 data
                screenshots = history.screenshots()
                history_data = history.model_dump()
                for j, item in enumerate(history_data['history']):
                    if 'state' in item and item['state'].get('screenshot_path'):
                        # Replace file path with actual base64 data
                        if j < len(screenshots) and screenshots[j]:
                            item['state']['screenshot_base64'] = screenshots[j]
                            item['state']['screenshot_path'] = None  # Clear the file path
                
                # Save temporarily for PNG extraction
                with open(history_with_screenshots, 'w') as f:
                    json.dump(history_data, f, indent=2)
                
                # Create screenshots folder
                screenshots_dir = task_output_dir / "screenshots"
                screenshots_dir.mkdir(exist_ok=True)
                
                # Extract PNG files to screenshots folder
                extract_screenshots_from_history(str(history_with_screenshots), screenshots_dir)
                
                # Clean up the temporary JSON file if not needed
                if not config['output']['save_screenshots_as_base64']:
                    history_with_screenshots.unlink()
                    print("🧹 Cleaned up temporary screenshot JSON file")
                
                print(f"📸 Screenshots saved to: {screenshots_dir}")
            
            print(f"✅ Task {task_id} completed successfully")
            print(f"📁 Results saved to: {task_output_dir}")
            
        except Exception as e:
            print(f"❌ Error running task {task_id}: {e}")
            
            # Save error information if configured
            if config['error_handling']['save_error_logs']:
                # Use the same subset-grouped structure for error logs
                subset_output_dir = output_dir / web_name
                subset_output_dir.mkdir(parents=True, exist_ok=True)
                error_file = subset_output_dir / f"{task_id}_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Task: {task_description}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Question: {question}\n")
            
            # Continue or stop based on configuration
            if not config['error_handling']['continue_on_error']:
                print("❌ Stopping due to error (continue_on_error: false)")
                break
            continue
        
        # Delay between tasks from config
        await asyncio.sleep(config['task']['delay_between_tasks'])
    
    print(f"\n🎉 WebVoyager benchmark completed!")
    print(f"📁 All results saved to: {output_dir}")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} tasks that already existed")

if __name__ == "__main__":
    asyncio.run(run_webvoyager_benchmark())
