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
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Add the WebCoach directory to Python path
webcoach_dir = Path(__file__).parent
sys.path.append(str(webcoach_dir))

class TestCondenser(unittest.TestCase):
    """Comprehensive tests for the Condenser component"""
    
    def setUp(self):
        """Set up test fixtures"""
        from condenser import Condenser
        self.condenser = Condenser()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_trajectory(self, task="Test task", completed=True, has_errors=False):
        """Create a test trajectory with various scenarios"""
        history = [
            {
                "model_output": {
                    "action": [{"go_to_url": {"url": "https://example.com"}}]
                },
                "state": {
                    "url": "https://example.com",
                    "title": "Example Site"
                },
                "result": [{"error": "Connection failed" if has_errors else None}]
            }
        ]
        
        if completed:
            history.append({
                "model_output": {
                    "action": [{"done": {"success": not has_errors, "text": "Task completed"}}]
                },
                "state": {
                    "url": "https://example.com/results",
                    "title": "Results"
                },
                "result": [{"is_done": True, "success": not has_errors}]
            })
        
        return {
            "task": task,
            "history": history
        }
    
    def save_trajectory(self, trajectory):
        """Save trajectory to temporary file"""
        temp_path = os.path.join(self.temp_dir, 'test_trajectory.json')
        with open(temp_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        return temp_path
    
    def test_process_complete_trajectory(self):
        """Test processing a complete trajectory"""
        trajectory = self.create_test_trajectory(completed=True)
        temp_path = self.save_trajectory(trajectory)
        
        condensed_data, trajectory_type = self.condenser.process_trajectory(temp_path)
        
        self.assertEqual(trajectory_type, 'complete')
        self.assertIn('meta', condensed_data)
        self.assertIn('summary', condensed_data)
        self.assertIn('embedding', condensed_data)
        self.assertTrue(condensed_data['meta']['final_success'])
        self.assertEqual(len(condensed_data['embedding']), 256)
    
    def test_process_partial_trajectory(self):
        """Test processing a partial trajectory"""
        trajectory = self.create_test_trajectory(completed=False)
        temp_path = self.save_trajectory(trajectory)
        
        condensed_data, trajectory_type = self.condenser.process_trajectory(temp_path)
        
        self.assertEqual(trajectory_type, 'partial')
        self.assertFalse(condensed_data['meta']['final_success'])
    
    def test_process_failed_trajectory(self):
        """Test processing a trajectory with errors"""
        trajectory = self.create_test_trajectory(completed=True, has_errors=True)
        temp_path = self.save_trajectory(trajectory)
        
        condensed_data, trajectory_type = self.condenser.process_trajectory(temp_path)
        
        self.assertEqual(trajectory_type, 'complete')
        self.assertFalse(condensed_data['meta']['final_success'])
        self.assertIn('fail_modes', condensed_data)
    
    def test_empty_trajectory(self):
        """Test processing an empty trajectory"""
        trajectory = {"task": "Empty task", "history": []}
        temp_path = self.save_trajectory(trajectory)
        
        condensed_data, trajectory_type = self.condenser.process_trajectory(temp_path)
        
        self.assertEqual(trajectory_type, 'partial')
        self.assertEqual(condensed_data['meta']['total_steps'], 0)
    
    def test_invalid_file(self):
        """Test processing non-existent file"""
        with self.assertRaises(Exception):
            self.condenser.process_trajectory('/nonexistent/file.json')
    
    def test_malformed_json(self):
        """Test processing malformed JSON"""
        temp_path = os.path.join(self.temp_dir, 'malformed.json')
        with open(temp_path, 'w') as f:
            f.write('{"invalid": json}')
        
        with self.assertRaises(Exception):
            self.condenser.process_trajectory(temp_path)
    
    @patch('openai.OpenAI')
    def test_embedding_generation_failure(self, mock_openai):
        """Test handling of embedding generation failure"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        embedding = self.condenser.generate_embedding("test text")
        
        # Should return zero embedding on failure
        self.assertEqual(len(embedding), 256)
        self.assertTrue(all(x == 0.0 for x in embedding))
    
    @patch('openai.OpenAI')
    def test_llm_summary_failure(self, mock_openai):
        """Test handling of LLM summary generation failure"""
        trajectory = self.create_test_trajectory()
        temp_path = self.save_trajectory(trajectory)
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Should fallback to simple summary
        condensed_data, _ = self.condenser.process_trajectory(temp_path)
        self.assertIn('summary', condensed_data)
        self.assertIsInstance(condensed_data['summary'], str)

class TestEMS(unittest.TestCase):
    """Comprehensive tests for the External Memory Store"""
    
    def setUp(self):
        """Set up test fixtures"""
        from ems import ExternalMemoryStore
        self.temp_dir = tempfile.mkdtemp()
        self.ems = ExternalMemoryStore(storage_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_experience(self, domain="example.com", success=True):
        """Create a test experience"""
        return {
            'meta': {
                'domain': domain,
                'final_success': success,
                'total_steps': 5,
                'goal': 'Test goal'
            },
            'summary': f'Test summary for {domain}',
            'embedding': np.random.random(256).tolist()
        }
    
    def test_add_experience_success(self):
        """Test successfully adding an experience"""
        experience = self.create_test_experience()
        result = self.ems.add_experience(experience)
        
        self.assertTrue(result)
        self.assertEqual(len(self.ems.data), 1)
    
    def test_add_experience_no_embedding(self):
        """Test adding experience without embedding"""
        experience = self.create_test_experience()
        del experience['embedding']
        
        result = self.ems.add_experience(experience)
        self.assertFalse(result)
    
    def test_add_experience_invalid_embedding(self):
        """Test adding experience with invalid embedding"""
        experience = self.create_test_experience()
        experience['embedding'] = [1, 2, 3]  # Wrong dimension
        
        result = self.ems.add_experience(experience)
        self.assertFalse(result)
    
    def test_get_similar_experiences(self):
        """Test retrieving similar experiences"""
        # Add multiple experiences
        exp1 = self.create_test_experience("example.com", True)
        exp2 = self.create_test_experience("test.com", False)
        exp3 = self.create_test_experience("example.com", True)
        
        self.ems.add_experience(exp1)
        self.ems.add_experience(exp2)
        self.ems.add_experience(exp3)
        
        # Query for similar experiences
        query = self.create_test_experience("example.com", True)
        results = self.ems.get_similar_experiences(query, k=2)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all('similarity_score' in r for r in results))
    
    def test_get_similar_experiences_empty_store(self):
        """Test retrieving from empty store"""
        query = self.create_test_experience()
        results = self.ems.get_similar_experiences(query, k=3)
        
        self.assertEqual(len(results), 0)
    
    def test_get_similar_experiences_no_embedding(self):
        """Test querying without embedding"""
        self.ems.add_experience(self.create_test_experience())
        
        query = self.create_test_experience()
        del query['embedding']
        
        results = self.ems.get_similar_experiences(query, k=1)
        self.assertEqual(len(results), 0)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        sim1 = self.ems._cosine_similarity(vec1, vec2)
        sim2 = self.ems._cosine_similarity(vec1, vec3)
        
        self.assertAlmostEqual(sim1, 0.0, places=5)
        self.assertAlmostEqual(sim2, 1.0, places=5)
    
    def test_get_stats(self):
        """Test statistics generation"""
        self.ems.add_experience(self.create_test_experience("example.com", True))
        self.ems.add_experience(self.create_test_experience("test.com", False))
        
        stats = self.ems.get_stats()
        
        self.assertEqual(stats['total_experiences'], 2)
        self.assertEqual(stats['successful_experiences'], 1)
        self.assertEqual(stats['failed_experiences'], 1)
        self.assertIn('example.com', stats['domains'])
    
    def test_persistence(self):
        """Test data persistence across instances"""
        # Add experience to first instance
        exp = self.create_test_experience()
        self.ems.add_experience(exp)
        
        # Create new instance with same storage dir
        from ems import ExternalMemoryStore
        ems2 = ExternalMemoryStore(storage_dir=self.temp_dir)
        
        self.assertEqual(len(ems2.data), 1)
    
    def test_clear(self):
        """Test clearing all data"""
        self.ems.add_experience(self.create_test_experience())
        self.assertEqual(len(self.ems.data), 1)
        
        self.ems.clear()
        self.assertEqual(len(self.ems.data), 0)
    
    def test_search_by_domain(self):
        """Test domain-based search"""
        self.ems.add_experience(self.create_test_experience("example.com"))
        self.ems.add_experience(self.create_test_experience("test.com"))
        self.ems.add_experience(self.create_test_experience("example.org"))
        
        results = self.ems.search_by_domain("example", k=5)
        self.assertEqual(len(results), 2)

class TestWebCoach(unittest.TestCase):
    """Comprehensive tests for the WebCoach component"""
    
    def setUp(self):
        """Set up test fixtures"""
        from web_coach import WebCoach
        from ems import ExternalMemoryStore
        
        self.temp_dir = tempfile.mkdtemp()
        self.ems = ExternalMemoryStore(storage_dir=self.temp_dir)
        self.web_coach = WebCoach(ems_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_state(self, domain="example.com", success=True):
        """Create a test current state"""
        return {
            'meta': {
                'domain': domain,
                'total_steps': 3,
                'goal': 'Test current goal'
            },
            'summary': 'Current agent is trying to complete a task',
            'embedding': np.random.random(256).tolist()
        }
    
    def create_test_experience(self, domain="example.com", success=True, fail_modes=None):
        """Create a test experience for EMS"""
        experience = {
            'meta': {
                'domain': domain,
                'final_success': success,
                'total_steps': 5,
                'goal': 'Past test goal'
            },
            'summary': f'Past agent {"succeeded" if success else "failed"} at similar task',
            'embedding': np.random.random(256).tolist()
        }
        
        if not success and fail_modes:
            experience['fail_modes'] = fail_modes
        elif success:
            experience['success_workflows'] = [
                {
                    'name': 'Successful completion',
                    'evidence_steps': [3, 4, 5],
                    'description': 'Successfully completed the task'
                }
            ]
        
        return experience
    
    def test_generate_advice_no_experiences(self):
        """Test advice generation with no past experiences"""
        current_state = self.create_test_state()
        advice = self.web_coach.generate_advice(current_state, k=3)
        
        self.assertIn('intervene', advice)
        self.assertIn('advice', advice)
        self.assertFalse(advice['intervene'])
    
    @patch('openai.OpenAI')
    def test_generate_advice_with_failures(self, mock_openai):
        """Test advice generation with relevant failure experiences"""
        # Add failure experience
        fail_modes = [
            {
                'name': 'Timeout error',
                'evidence_steps': [2, 3],
                'description': 'Connection timed out repeatedly'
            }
        ]
        experience = self.create_test_experience(success=False, fail_modes=fail_modes)
        self.ems.add_experience(experience)
        
        # Mock LLM response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intervene": true, "advice": "Try refreshing the page"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        current_state = self.create_test_state()
        advice = self.web_coach.generate_advice(current_state, k=3)
        
        self.assertTrue(advice['intervene'])
        self.assertIn('advice', advice)
    
    @patch('openai.OpenAI')
    def test_generate_advice_with_successes(self, mock_openai):
        """Test advice generation with successful experiences"""
        # Add success experience
        experience = self.create_test_experience(success=True)
        self.ems.add_experience(experience)
        
        # Mock LLM response indicating no intervention needed
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intervene": false, "advice": ""}'
        mock_client.chat.completions.create.return_value = mock_response
        
        current_state = self.create_test_state()
        advice = self.web_coach.generate_advice(current_state, k=3)
        
        self.assertFalse(advice['intervene'])
    
    @patch('openai.OpenAI')
    def test_llm_json_parsing_failure(self, mock_openai):
        """Test handling of invalid JSON from LLM"""
        self.ems.add_experience(self.create_test_experience())
        
        # Mock invalid JSON response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = 'Invalid JSON response'
        mock_client.chat.completions.create.return_value = mock_response
        
        current_state = self.create_test_state()
        advice = self.web_coach.generate_advice(current_state, k=3)
        
        # Should gracefully handle parsing failure
        self.assertIn('intervene', advice)
        self.assertIn('advice', advice)
    
    @patch('openai.OpenAI')
    def test_llm_api_failure(self, mock_openai):
        """Test handling of LLM API failure"""
        self.ems.add_experience(self.create_test_experience())
        
        # Mock API failure
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        current_state = self.create_test_state()
        advice = self.web_coach.generate_advice(current_state, k=3)
        
        # Should return safe default
        self.assertFalse(advice['intervene'])
        self.assertEqual(advice['advice'], "")
    
    def test_format_experiences_for_prompt(self):
        """Test experience formatting for LLM prompt"""
        experiences = [
            {
                'experience': {
                    'data': self.create_test_experience(success=False)
                },
                'similarity_score': 0.8
            }
        ]
        
        formatted = self.web_coach._format_experiences_for_prompt(experiences)
        
        self.assertIn('RELEVANT PAST EXPERIENCES', formatted)
        self.assertIn('FAILURE', formatted)
        self.assertIn('0.800', formatted)
    
    def test_extract_advice_from_text(self):
        """Test advice extraction from free text"""
        # Test with intervention keywords
        text_with_advice = "You should try a different approach to avoid this error"
        result = self.web_coach._extract_advice_from_text(text_with_advice)
        self.assertTrue(result['intervene'])
        
        # Test without intervention keywords
        text_without_advice = "Everything looks good, continue as normal"
        result = self.web_coach._extract_advice_from_text(text_without_advice)
        self.assertFalse(result['intervene'])
    
    def test_get_coaching_stats(self):
        """Test coaching statistics"""
        self.ems.add_experience(self.create_test_experience(success=True))
        self.ems.add_experience(self.create_test_experience(success=False))
        
        stats = self.web_coach.get_coaching_stats()
        
        self.assertIn('model', stats)
        self.assertIn('ems_experiences', stats)
        self.assertEqual(stats['ems_experiences'], 2)

class TestCoachCallback(unittest.TestCase):
    """Comprehensive tests for the coach callback system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_configure_coach(self):
        """Test coach configuration"""
        from coach_callback import configure_coach
        
        config = {
            'model': 'gpt-4o',
            'frequency': 3,
            'storage_dir': self.temp_dir,
            'enabled': True
        }
        
        # Should not raise any exceptions
        configure_coach(config)
    
    def test_configure_coach_validation(self):
        """Test coach configuration with validation"""
        from coach_callback import configure_coach
        
        # Test with incomplete config - should use defaults
        config = {'enabled': True}
        configure_coach(config)
        
        # Test with invalid frequency - should be corrected
        config = {'frequency': 0, 'storage_dir': self.temp_dir}
        configure_coach(config)
    
    @patch('tempfile.NamedTemporaryFile')
    def test_process_trajectory_async(self, mock_tempfile):
        """Test async trajectory processing"""
        from coach_callback import process_trajectory_async
        from condenser import Condenser
        
        # Mock temporary file
        mock_file = MagicMock()
        mock_file.name = '/tmp/test.json'
        mock_tempfile.return_value.__enter__.return_value = mock_file
        
        with patch('os.unlink'), patch.object(Condenser, 'process_trajectory') as mock_process:
            mock_process.return_value = ({'summary': 'test'}, 'complete')
            
            history_dict = {'task': 'test', 'history': []}
            condenser = Condenser()
            
            result = process_trajectory_async(condenser, history_dict)
            
            self.assertEqual(result, ({'summary': 'test'}, 'complete'))

class TestConfig(unittest.TestCase):
    """Comprehensive tests for the configuration system"""
    
    def test_validate_coach_config_defaults(self):
        """Test config validation with defaults"""
        from config import validate_coach_config
        
        result = validate_coach_config({})
        
        self.assertFalse(result['enabled'])
        self.assertEqual(result['model'], 'gpt-4o')
        self.assertEqual(result['frequency'], 5)
    
    def test_validate_coach_config_overrides(self):
        """Test config validation with overrides"""
        from config import validate_coach_config
        
        config = {
            'enabled': True,
            'model': 'gpt-3.5-turbo',
            'frequency': 10,
            'similarity_threshold': 0.5
        }
        
        result = validate_coach_config(config)
        
        self.assertTrue(result['enabled'])
        self.assertEqual(result['model'], 'gpt-3.5-turbo')
        self.assertEqual(result['frequency'], 10)
        self.assertEqual(result['similarity_threshold'], 0.5)
    
    def test_validate_coach_config_boundaries(self):
        """Test config validation with boundary values"""
        from config import validate_coach_config
        
        config = {
            'frequency': -1,
            'max_experiences': 5,
            'similarity_threshold': 1.5
        }
        
        result = validate_coach_config(config)
        
        # Should correct invalid values
        self.assertEqual(result['frequency'], 1)
        self.assertEqual(result['max_experiences'], 10)
        self.assertEqual(result['similarity_threshold'], 0.3)
    
    def test_get_coach_config_from_main_config(self):
        """Test extracting coach config from main config"""
        from config import get_coach_config_from_main_config
        
        main_config = {
            'coach': {
                'enabled': True,
                'model': 'custom-model'
            },
            'other_section': {
                'some_value': 'test'
            }
        }
        
        result = get_coach_config_from_main_config(main_config)
        
        self.assertTrue(result['enabled'])
        self.assertEqual(result['model'], 'custom-model')
        # Should still have defaults for missing values
        self.assertEqual(result['frequency'], 5)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete WebCoach system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('openai.OpenAI')
    def test_end_to_end_workflow(self, mock_openai):
        """Test complete end-to-end workflow"""
        from condenser import Condenser
        from ems import ExternalMemoryStore
        from web_coach import WebCoach
        
        # Mock OpenAI responses
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data[0].embedding = np.random.random(256).tolist()
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Mock chat response
        mock_chat_response = MagicMock()
        mock_chat_response.choices[0].message.content = 'Agent successfully completed the web browsing task on example.com in 2 steps.'
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        # Create test trajectory
        trajectory = {
            "task": "Test end-to-end workflow",
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
        
        # Save trajectory
        trajectory_path = os.path.join(self.temp_dir, 'test_trajectory.json')
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        # Step 1: Process with condenser
        condenser = Condenser()
        condensed_data, trajectory_type = condenser.process_trajectory(trajectory_path)
        
        self.assertEqual(trajectory_type, 'complete')
        self.assertIn('embedding', condensed_data)
        
        # Step 2: Store in EMS
        ems = ExternalMemoryStore(storage_dir=self.temp_dir)
        success = ems.add_experience(condensed_data)
        self.assertTrue(success)
        
        # Step 3: Query with WebCoach
        web_coach = WebCoach(ems_dir=self.temp_dir)
        
        # Mock advice response
        mock_chat_response.choices[0].message.content = '{"intervene": false, "advice": ""}'
        
        advice = web_coach.generate_advice(condensed_data, k=1)
        
        self.assertIn('intervene', advice)
        self.assertIn('advice', advice)

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive WebCoach Framework Tests\n")
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some tests may be skipped.")
    
    # Create test suite
    test_classes = [
        TestCondenser,
        TestEMS,
        TestWebCoach,
        TestCoachCallback,
        TestConfig,
        TestIntegration
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__} tests...")
        
        # Create test suite for this class
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        # Print summary for this class
        class_success = result.testsRun - len(result.failures) - len(result.errors)
        print(f"  ✅ {class_success}/{result.testsRun} tests passed")
        
        if result.failures:
            print(f"  ❌ {len(result.failures)} failures")
        if result.errors:
            print(f"  💥 {len(result.errors)} errors")
    
    # Final summary
    total_success = total_tests - total_failures - total_errors
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Final Test Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_success}")
    print(f"  Failed: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed! WebCoach framework is ready for production.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the output above for details.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
