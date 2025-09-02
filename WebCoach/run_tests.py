#!/usr/bin/env python3
"""
Test Runner for WebCoach Framework

This script provides different levels of testing:
- basic: Quick tests without API dependencies
- comprehensive: Full test suite with mocked API calls
- integration: Real API tests (requires OPENAI_API_KEY)
"""

import os
import sys
import argparse
from pathlib import Path

def run_basic_tests():
    """Run basic integration tests"""
    print("🔧 Running Basic Integration Tests")
    print("=" * 50)
    
    try:
        from test_basic_integration import main as run_basic
        return run_basic()
    except ImportError as e:
        print(f"❌ Could not import basic tests: {e}")
        return False

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        from test_coach_comprehensive import run_comprehensive_tests
        return run_comprehensive_tests()
    except ImportError as e:
        print(f"❌ Could not import comprehensive tests: {e}")
        return False

def run_simple_functionality_test():
    """Run the original simple functionality test"""
    print("⚡ Running Simple Functionality Test")
    print("=" * 50)
    
    try:
        from test_coach import main as run_simple
        return run_simple()
    except ImportError as e:
        print(f"❌ Could not import simple tests: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("📦 Checking Dependencies...")
    
    required_packages = [
        'openai',
        'numpy',
        'pydantic',
        'pathlib',
        'json',
        'tempfile',
        'unittest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("  🎉 All dependencies available")
    return True

def check_environment():
    """Check environment setup"""
    print("\n🌍 Checking Environment...")
    
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        print("  ✅ OPENAI_API_KEY is set")
        api_key_available = True
    else:
        print("  ⚠️  OPENAI_API_KEY not set (some tests will be limited)")
        api_key_available = False
    
    # Check file structure
    webcoach_dir = Path(__file__).parent
    required_files = [
        '__init__.py',
        'coach_callback.py',
        'condenser.py',
        'ems.py',
        'web_coach.py',
        'config.py'
    ]
    
    all_files_present = True
    for filename in required_files:
        filepath = webcoach_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")
            all_files_present = False
    
    return api_key_available, all_files_present

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='WebCoach Framework Test Runner')
    parser.add_argument('test_type', nargs='?', default='basic',
                        choices=['basic', 'comprehensive', 'simple', 'all'],
                        help='Type of tests to run (default: basic)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check dependencies and environment, don\'t run tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 WebCoach Framework Test Runner")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Cannot run tests due to missing dependencies")
        return False
    
    # Check environment
    api_key_available, files_ok = check_environment()
    if not files_ok:
        print("\n❌ Cannot run tests due to missing files")
        return False
    
    if args.check_only:
        print("\n✅ Environment check complete")
        return True
    
    print("\n" + "=" * 50)
    
    # Run specified tests
    results = []
    
    if args.test_type == 'basic':
        results.append(run_basic_tests())
        
    elif args.test_type == 'comprehensive':
        results.append(run_comprehensive_tests())
        
    elif args.test_type == 'simple':
        results.append(run_simple_functionality_test())
        
    elif args.test_type == 'all':
        print("🔄 Running All Test Suites\n")
        
        # Run basic tests first
        print("1️⃣ Basic Tests:")
        results.append(run_basic_tests())
        print("\n" + "=" * 50 + "\n")
        
        # Run simple tests
        print("2️⃣ Simple Tests:")
        results.append(run_simple_functionality_test())
        print("\n" + "=" * 50 + "\n")
        
        # Run comprehensive tests
        print("3️⃣ Comprehensive Tests:")
        results.append(run_comprehensive_tests())
    
    # Final summary
    total_suites = len(results)
    passed_suites = sum(results)
    
    print("\n" + "=" * 50)
    print("📊 Final Test Summary:")
    print(f"  Test Suites Run: {total_suites}")
    print(f"  Test Suites Passed: {passed_suites}")
    print(f"  Success Rate: {(passed_suites/total_suites*100):.1f}%")
    
    if passed_suites == total_suites:
        print("\n🎉 All test suites passed! WebCoach framework is ready for use.")
        
        if not api_key_available:
            print("\n💡 Note: Set OPENAI_API_KEY environment variable for full functionality")
        
        print("\n📖 Next steps:")
        print("  1. Enable coaching in config.yaml: coach.enabled: true")
        print("  2. Run: python run_webvoyager.py")
        print("  3. Watch for coaching messages in the output")
        
        return True
    else:
        print(f"\n⚠️  {total_suites - passed_suites} test suite(s) failed")
        print("Please review the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
