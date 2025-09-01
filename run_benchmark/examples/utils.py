import json
import base64
from pathlib import Path

def extract_screenshots_from_history(history_file_path: str, output_dir: Path):
    """Extract screenshots from the history JSON and save them as PNG files"""
    try:
        with open(history_file_path, 'r') as f:
            history_data = json.load(f)
        
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        
        extracted_count = 0
        for i, item in enumerate(history_data['history']):
            if 'state' in item and item['state'].get('screenshot_base64'):
                screenshot_b64 = item['state']['screenshot_base64']
                
                # Decode base64 and save as PNG
                try:
                    screenshot_data = base64.b64decode(screenshot_b64)
                    screenshot_file = screenshots_dir / f"step_{i+1}.png"
                    
                    with open(screenshot_file, 'wb') as f:
                        f.write(screenshot_data)
                    
                    extracted_count += 1
                    print(f"   📸 Saved step_{i+1}.png")
                except Exception as e:
                    print(f"   ❌ Failed to save step_{i+1}.png: {e}")
        
        print(f"✅ Extracted {extracted_count} screenshots to {screenshots_dir}")
        return extracted_count
        
    except Exception as e:
        print(f"❌ Failed to extract screenshots: {e}")
        return 0
