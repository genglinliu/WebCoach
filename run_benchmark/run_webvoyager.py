from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import json
import sys
import yaml
from utils import extract_screenshots_from_history

load_dotenv()

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required config sections
    required_sections = ['browser', 'llm', 'agent', 'task', 'output', 'error_handling']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    print(f"✅ Loaded configuration from: {config_path}")
    return config

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
    
    # Setup LLM and browser profile from config
    llm = ChatOpenAI(
        model=config['llm']['model']
    )
    
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
    
    # Create output directory from config
    output_dir = Path(config['output']['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task
    for i, task_data in enumerate(tasks):
        task_id = task_data['id']
        web_name = task_data['web_name']
        question = task_data['ques']
        url = task_data['web']
        
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
            history = await agent.run(max_steps=config['agent']['max_steps'])
            
            # Save task-specific history
            task_output_dir = output_dir / task_id
            task_output_dir.mkdir(exist_ok=True)
            
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
                error_file = output_dir / f"{task_id}_error.txt"
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

if __name__ == "__main__":
    asyncio.run(run_webvoyager_benchmark())
