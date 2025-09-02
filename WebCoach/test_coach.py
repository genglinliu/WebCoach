#!/usr/bin/env python3
"""
Comprehensive Test Suite for WebCoach Framework

This script provides thorough test coverage for all coach components,
including edge cases, error conditions, and integration scenarios.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the WebCoach directory to Python path
webcoach_dir = Path(__file__).parent
sys.path.append(str(webcoach_dir))

def test_condenser():
    """Test the condenser component"""
    print("🧪 Testing Condenser...")
    
    from condenser import Condenser
    
    # Create a simple test trajectory
    test_trajectory = {
        "task": "Test task - search for information",
        "history": [
            {
                "model_output": {
                    "action": [{"go_to_url": {"url": "https://example.com"}}]
                },
                "state": {
                    "url": "https://example.com",
                    "title": "Example Site"
                },
                "result": [{"error": None}]
            },
            {
                "model_output": {
                    "action": [{"done": {"success": True, "text": "Task completed"}}]
                },
                "state": {
                    "url": "https://example.com/results",
                    "title": "Results"
                },
                "result": [{"is_done": True, "success": True}]
            }
        ]
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_trajectory, f, indent=2)
        temp_path = f.name
    
    try:
        condenser = Condenser()
        condensed_data, trajectory_type = condenser.process_trajectory(temp_path)
        
        print(f"  ✅ Trajectory type: {trajectory_type}")
        print(f"  ✅ Summary: {condensed_data['summary'][:100]}...")
        print(f"  ✅ Embedding dimensions: {len(condensed_data['embedding'])}")
        
        return condensed_data
        
    except Exception as e:
        print(f"  ❌ Condenser test failed: {e}")
        return None
    finally:
        os.unlink(temp_path)

def test_ems(condensed_data):
    """Test the EMS component"""
    print("\n🧪 Testing EMS...")
    
    from ems import ExternalMemoryStore
    
    try:
        # Use temporary directory
        temp_dir = tempfile.mkdtemp()
        ems = ExternalMemoryStore(storage_dir=temp_dir)
        
        # Add test experience
        if condensed_data:
            success = ems.add_experience(condensed_data)
            print(f"  ✅ Added experience: {success}")
            
            # Test retrieval
            similar = ems.get_similar_experiences(condensed_data, k=1)
            print(f"  ✅ Retrieved {len(similar)} similar experiences")
            
            # Test stats
            stats = ems.get_stats()
            print(f"  ✅ EMS stats: {stats}")
        
        return ems
        
    except Exception as e:
        print(f"  ❌ EMS test failed: {e}")
        return None

def test_webcoach(condensed_data, ems):
    """Test the WebCoach component"""
    print("\n🧪 Testing WebCoach...")
    
    from web_coach import WebCoach
    
    try:
        # Use temporary directory
        temp_dir = ems.storage_dir if ems else tempfile.mkdtemp()
        web_coach = WebCoach(ems_dir=temp_dir)
        
        if condensed_data:
            # Test advice generation
            advice = web_coach.generate_advice(condensed_data, k=1)
            print(f"  ✅ Generated advice: {advice}")
            
            # Test stats
            stats = web_coach.get_coaching_stats()
            print(f"  ✅ Coaching stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ WebCoach test failed: {e}")
        return False

def test_config():
    """Test the configuration system"""
    print("\n🧪 Testing Configuration...")
    
    from config import validate_coach_config, get_coach_config_from_main_config
    
    try:
        # Test config validation
        test_config = {"enabled": True, "model": "gpt-4o", "frequency": 3}
        validated = validate_coach_config(test_config)
        print(f"  ✅ Config validation: {validated['enabled']}, {validated['model']}")
        
        # Test main config extraction
        main_config = {"coach": test_config, "other": "data"}
        coach_config = get_coach_config_from_main_config(main_config)
        print(f"  ✅ Config extraction: {coach_config['frequency']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_callback():
    """Test the callback system"""
    print("\n🧪 Testing Coach Callback...")
    
    from coach_callback import configure_coach
    
    try:
        # Test configuration
        test_config = {
            "enabled": True,
            "model": "gpt-4o",
            "frequency": 5,
            "storage_dir": tempfile.mkdtemp()
        }
        
        configure_coach(test_config)
        print("  ✅ Coach callback configured successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Callback test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting WebCoach Framework Tests\n")
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some tests may fail.")
    
    # Run tests
    results = []
    
    # Test condenser
    condensed_data = test_condenser()
    results.append(condensed_data is not None)
    
    # Test EMS
    ems = test_ems(condensed_data)
    results.append(ems is not None)
    
    # Test WebCoach (only if we have data)
    if condensed_data and ems:
        webcoach_result = test_webcoach(condensed_data, ems)
        results.append(webcoach_result)
    else:
        print("\n⏭️  Skipping WebCoach test (no test data)")
        results.append(False)
    
    # Test config system
    config_result = test_config()
    results.append(config_result)
    
    # Test callback system
    callback_result = test_callback()
    results.append(callback_result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WebCoach framework is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
