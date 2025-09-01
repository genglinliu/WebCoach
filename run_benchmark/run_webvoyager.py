from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import json
import sys
from utils import extract_screenshots_from_history

load_dotenv()

async def run_webvoyager_benchmark():
    """Run WebVoyager benchmark tasks one by one"""
    
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
    
    # Setup LLM and browser profile
    llm = ChatOpenAI(model="gpt-4o")
    browser_profile = BrowserProfile(
        headless=True,
        disable_dev_shm_usage=True,
        no_sandbox=True
    )
    
    # Create output directory (relative to current working directory)
    output_dir = Path("outputs/webvoyager")
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
            # Create and run agent
            agent = Agent(
                task=task_description,
                llm=llm,
                browser_profile=browser_profile,
                directly_open_url=True  # This will automatically navigate to the URL
            )
            
            print(f"🤖 Starting agent for task {task_id}...")
            history = await agent.run(max_steps=50)  # Limit steps to avoid infinite loops
            
            # Extract screenshots as base64
            print("📸 Extracting screenshots as base64...")
            screenshots = history.screenshots()
            print(f"   Found {len(screenshots)} screenshots")
            
            # Save task-specific history
            task_output_dir = output_dir / task_id
            task_output_dir.mkdir(exist_ok=True)
            
            # Save basic history
            history_file = task_output_dir / "agent_history.json"
            history.save_to_file(str(history_file))
            
            # Save history with embedded screenshots
            history_with_screenshots = task_output_dir / "agent_history_with_screenshots.json"
            
            # Get the raw data and replace screenshot_paths with base64 data
            history_data = history.model_dump()
            for j, item in enumerate(history_data['history']):
                if 'state' in item and item['state'].get('screenshot_path'):
                    # Replace file path with actual base64 data
                    if j < len(screenshots) and screenshots[j]:
                        item['state']['screenshot_base64'] = screenshots[j]
                        item['state']['screenshot_path'] = None  # Clear the file path
            
            with open(history_with_screenshots, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Extract screenshots as PNG files
            print("🖼️  Extracting screenshots as PNG files...")
            extract_screenshots_from_history(str(history_with_screenshots), task_output_dir)
            
            print(f"✅ Task {task_id} completed successfully")
            print(f"📁 Results saved to: {task_output_dir}")
            
        except Exception as e:
            print(f"❌ Error running task {task_id}: {e}")
            # Save error information
            error_file = output_dir / f"{task_id}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"Task: {task_description}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Question: {question}\n")
            continue
        
        # Small delay between tasks to avoid overwhelming the system
        await asyncio.sleep(2)
    
    print(f"\n🎉 WebVoyager benchmark completed!")
    print(f"📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(run_webvoyager_benchmark())
