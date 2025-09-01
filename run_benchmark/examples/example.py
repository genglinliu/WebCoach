from browser_use import Agent, ChatOpenAI, BrowserProfile
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import json
from utils import extract_screenshots_from_history

load_dotenv()

async def main():
    llm = ChatOpenAI(model="gpt-4o")
    task = "Find the number 1 post on Show HN"
    
    browser_profile = BrowserProfile(
        headless=True,
        disable_dev_shm_usage=True,
        no_sandbox=True
    )
    
    agent = Agent(task=task, llm=llm, browser_profile=browser_profile)
    history = await agent.run()
    
    # Extract screenshots as base64 before container exits
    print("📸 Extracting screenshots as base64...")
    screenshots = history.screenshots()
    print(f"   Found {len(screenshots)} screenshots")
    
    # Save history to output directory
    output_dir = Path("outputs")  # Relative to /app/scripts_gl in container
    output_dir.mkdir(exist_ok=True)
    
    history_file = output_dir / "agent_history.json"
    history.save_to_file(str(history_file))
    
    # Also save a version with embedded base64 screenshots
    history_with_screenshots = output_dir / "agent_history_with_screenshots.json"
    
    # Get the raw data and replace screenshot_paths with base64 data
    history_data = history.model_dump()
    for i, item in enumerate(history_data['history']):
        if 'state' in item and item['state'].get('screenshot_path'):
            # Replace file path with actual base64 data
            if i < len(screenshots) and screenshots[i]:
                item['state']['screenshot_base64'] = screenshots[i]
                item['state']['screenshot_path'] = None  # Clear the file path
    
    with open(history_with_screenshots, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"✅ Agent completed {len(history)} steps")
    print(f"📁 History saved to: {history_file}")
    print(f"📸 Screenshots embedded in: {history_with_screenshots}")
    
    # Extract screenshots as PNG files
    print("\n🖼️  Extracting screenshots as PNG files...")
    extract_screenshots_from_history(str(history_with_screenshots), output_dir)

if __name__ == "__main__":
    asyncio.run(main())