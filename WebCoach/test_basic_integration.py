#!/usr/bin/env python3
"""
Basic Integration Test for WebCoach Framework

This script provides a lightweight test that can be run to verify basic functionality
without requiring heavy dependencies or API keys.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the WebCoach directory to Python path
webcoach_dir = Path(__file__).parent
sys.path.append(str(webcoach_dir))

def test_basic_config():
    """Test basic configuration functionality"""
    print("🧪 Testing Basic Configuration...")
    
    try:
        from config import validate_coach_config, get_coach_config_from_main_config
        
        # Test basic validation
        config = {"enabled": True, "model": "gpt-4o"}
        validated = validate_coach_config(config)
        
        assert validated["enabled"] == True
        assert validated["model"] == "gpt-4o"
        assert validated["frequency"] >= 1
        
        # Test main config extraction
        main_config = {"coach": config, "other": "data"}
        extracted = get_coach_config_from_main_config(main_config)
        
        assert extracted["enabled"] == True
        
        print("  ✅ Configuration validation works")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_basic_ems():
    """Test basic EMS functionality without OpenAI"""
    print("🧪 Testing Basic EMS...")
    
    try:
        from ems import ExternalMemoryStore
        import numpy as np
        
        # Use temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            ems = ExternalMemoryStore(storage_dir=temp_dir)
            
            # Create test experience with valid embedding
            experience = {
                'meta': {
                    'domain': 'example.com',
                    'final_success': True,
                    'total_steps': 3,
                    'goal': 'Test goal'
                },
                'summary': 'Test summary',
                'embedding': np.random.random(256).tolist()
            }
            
            # Test adding experience
            success = ems.add_experience(experience)
            assert success == True
            
            # Test retrieving experience
            similar = ems.get_similar_experiences(experience, k=1)
            assert len(similar) == 1
            
            # Test stats
            stats = ems.get_stats()
            assert stats['total_experiences'] == 1
            assert stats['successful_experiences'] == 1
            
            # Test persistence
            ems2 = ExternalMemoryStore(storage_dir=temp_dir)
            assert len(ems2.data) == 1
            
            print("  ✅ EMS basic functionality works")
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"  ❌ EMS test failed: {e}")
        return False

def test_basic_condenser_structure():
    """Test basic condenser structure without OpenAI calls"""
    print("🧪 Testing Basic Condenser Structure...")
    
    try:
        from condenser import Condenser
        
        condenser = Condenser()
        
        # Test domain extraction
        domain = condenser.extract_domain("https://example.com/path")
        assert domain == "example.com"
        
        # Test trajectory type determination
        complete_trajectory = {
            "history": [
                {
                    "model_output": {
                        "action": [{"done": {"success": True}}]
                    },
                    "result": [{"is_done": True}]
                }
            ]
        }
        
        trajectory_type = condenser._determine_trajectory_type(complete_trajectory)
        assert trajectory_type == "complete"
        
        partial_trajectory = {
            "history": [
                {
                    "model_output": {
                        "action": [{"click": {"index": 1}}]
                    },
                    "result": [{"error": None}]
                }
            ]
        }
        
        trajectory_type = condenser._determine_trajectory_type(partial_trajectory)
        assert trajectory_type == "partial"
        
        print("  ✅ Condenser basic structure works")
        return True
        
    except Exception as e:
        print(f"  ❌ Condenser test failed: {e}")
        return False

def test_callback_structure():
    """Test basic callback structure"""
    print("🧪 Testing Basic Callback Structure...")
    
    try:
        from coach_callback import configure_coach, CoachComponents
        
        # Test configuration without actually initializing components
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = {
                'model': 'gpt-4o',
                'frequency': 5,
                'storage_dir': temp_dir,
                'enabled': True
            }
            
            # This should work without errors
            configure_coach(config)
            
            # Test component structure
            components = CoachComponents(config)
            assert components.config == config
            assert components.storage_dir.exists()
            
            print("  ✅ Callback structure works")
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"  ❌ Callback test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("🧪 Testing File Structure...")
    
    try:
        required_files = [
            '__init__.py',
            'coach_callback.py',
            'condenser.py',
            'ems.py',
            'web_coach.py',
            'config.py',
            'README.md'
        ]
        
        for filename in required_files:
            filepath = webcoach_dir / filename
            assert filepath.exists(), f"Missing required file: {filename}"
        
        print("  ✅ All required files present")
        return True
        
    except Exception as e:
        print(f"  ❌ File structure test failed: {e}")
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing Module Imports...")
    
    try:
        # Test individual imports
        from config import validate_coach_config
        from ems import ExternalMemoryStore
        from condenser import Condenser
        from coach_callback import configure_coach
        
        # Test package import (this is optional for our setup)
        # import WebCoach  # Skip this for now
        
        print("  ✅ All modules import successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_integration_with_config():
    """Test integration with a realistic config"""
    print("🧪 Testing Integration with Config...")
    
    try:
        # Simulate the config that would come from run_webvoyager.py
        main_config = {
            "coach": {
                "enabled": True,
                "model": "gpt-4o",
                "frequency": 5,
                "storage_dir": "./test_coach_storage",
                "debug": False
            },
            "llm": {
                "model": "gpt-4o"
            },
            "agent": {
                "max_steps": 20
            }
        }
        
        # Test the setup function similar to run_webvoyager.py
        sys.path.append(str(Path(__file__).parent.parent / "run_benchmark"))
        
        # Simulate the setup_coach function
        from config import get_coach_config_from_main_config
        from coach_callback import configure_coach
        
        coach_config = main_config.get('coach', {})
        
        if coach_config.get('enabled', False):
            validated_coach_config = get_coach_config_from_main_config(main_config)
            
            # Use temporary directory for testing
            temp_dir = tempfile.mkdtemp()
            validated_coach_config['storage_dir'] = temp_dir
            
            try:
                configure_coach(validated_coach_config)
                print(f"  ✅ Coach configured successfully")
                print(f"  ✅ Model: {validated_coach_config['model']}")
                print(f"  ✅ Frequency: {validated_coach_config['frequency']}")
                return True
                
            finally:
                shutil.rmtree(temp_dir)
        else:
            print("  ⚠️  Coach disabled in config")
            return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run basic integration tests"""
    print("🚀 Starting Basic WebCoach Integration Tests\n")
    
    tests = [
        test_file_structure,
        test_imports,
        test_basic_config,
        test_basic_ems,
        test_basic_condenser_structure,
        test_callback_structure,
        test_integration_with_config,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
            results.append(False)
        print()  # Add spacing between tests
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"📊 Basic Integration Test Results:")
    print(f"  Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 All basic tests passed! WebCoach framework basic functionality is working.")
        print("💡 To test full functionality, ensure OPENAI_API_KEY is set and run:")
        print("   python test_coach_comprehensive.py")
    else:
        print("\n⚠️  Some basic tests failed. Please check the framework setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
