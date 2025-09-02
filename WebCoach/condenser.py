#!/usr/bin/env python3
"""
Condenser for WebCoach Framework

This module processes raw agent trajectories from browser-use and condenses them into
structured summaries that can be stored in the External Memory Store (EMS).

Adapted from the original Coach Framework implementation for simplified use.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import openai
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Condenser:
    """
    Condenser processes raw agent trajectories and creates concise structured summaries.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the condenser.
        
        Args:
            model_name: The model to use for summarization (default: gpt-4o)
        """
        self.model_name = model_name
        
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found. Condenser will not work without it.")
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            return domain if domain else url
        except:
            return url
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using OpenAI's API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=256
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero embedding as fallback
            return [0.0] * 256
    
    def process_trajectory(self, trajectory_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Process a trajectory file and return condensed data and trajectory type.
        
        Args:
            trajectory_path: Path to the agent history JSON file
            
        Returns:
            Tuple of (condensed_data, trajectory_type)
            trajectory_type is either 'complete' or 'partial'
        """
        try:
            # Load trajectory data
            with open(trajectory_path, 'r') as f:
                trajectory_data = json.load(f)
            
            # Determine if trajectory is complete
            trajectory_type = self._determine_trajectory_type(trajectory_data)
            
            # Generate condensed summary
            condensed_data = self._create_condensed_summary(trajectory_data)
            
            logger.info(f"Processed trajectory as {trajectory_type}")
            return condensed_data, trajectory_type
            
        except Exception as e:
            logger.error(f"Error processing trajectory {trajectory_path}: {e}")
            raise
    
    def _determine_trajectory_type(self, trajectory_data: Dict[str, Any]) -> str:
        """
        Determine if a trajectory is complete or partial.
        
        Args:
            trajectory_data: The trajectory data dictionary
            
        Returns:
            'complete' if trajectory is finished, 'partial' otherwise
        """
        history = trajectory_data.get('history', [])
        if not history:
            return 'partial'
        
        # Check if the last step has a 'done' action
        last_step = history[-1]
        if 'model_output' in last_step and last_step['model_output']:
            actions = last_step['model_output'].get('action', [])
            for action in actions:
                if isinstance(action, dict) and 'done' in action:
                    return 'complete'
        
        # Check if there are any results indicating completion
        if 'result' in last_step:
            results = last_step['result']
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and result.get('is_done'):
                        return 'complete'
        
        return 'partial'
    
    def _create_condensed_summary(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a condensed summary of the trajectory.
        
        Args:
            trajectory_data: The trajectory data dictionary
            
        Returns:
            Condensed summary dictionary
        """
        history = trajectory_data.get('history', [])
        
        # Extract basic metadata
        meta = self._extract_metadata(trajectory_data, history)
        
        # Generate summary using LLM
        summary = self._generate_llm_summary(trajectory_data, history)
        
        # Extract failure modes or success patterns
        patterns = self._extract_patterns(history, meta['final_success'])
        
        # Create embedding for the summary
        embedding_text = f"{meta.get('goal', '')} {summary} {meta.get('domain', '')}"
        embedding = self.generate_embedding(embedding_text)
        
        # Build condensed data structure
        condensed_data = {
            'meta': meta,
            'summary': summary,
            'embedding': embedding,
            **patterns
        }
        
        return condensed_data
    
    def _extract_metadata(self, trajectory_data: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:
        """Extract metadata from trajectory."""
        meta = {
            'total_steps': len(history),
            'final_success': self._determine_trajectory_type(trajectory_data) == 'complete'
        }
        
        # Extract domain from URLs visited
        urls = []
        for step in history:
            if 'state' in step and 'url' in step['state']:
                urls.append(step['state']['url'])
        
        if urls:
            meta['domain'] = self.extract_domain(urls[0])  # Use first URL's domain
        
        # Try to extract goal/task from various places
        if 'task' in trajectory_data:
            meta['goal'] = trajectory_data['task']
        
        return meta
    
    def _generate_llm_summary(self, trajectory_data: Dict[str, Any], history: List[Dict]) -> str:
        """Generate a summary using LLM."""
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(trajectory_data, history)
            
            prompt = f"""Please provide a concise 2-3 sentence summary of this web browsing session:

{context}

Focus on:
1. What the agent was trying to accomplish
2. What actions were taken
3. Whether it succeeded or failed and why

Summary:"""

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            # Fallback to simple summary
            return self._generate_simple_summary(trajectory_data, history)
    
    def _prepare_llm_context(self, trajectory_data: Dict[str, Any], history: List[Dict]) -> str:
        """Prepare context for LLM summarization."""
        context_parts = []
        
        # Add task/goal if available
        if 'task' in trajectory_data:
            context_parts.append(f"Task: {trajectory_data['task']}")
        
        # Add key steps (limit to avoid token overflow)
        context_parts.append("Key steps:")
        for i, step in enumerate(history[-10:], 1):  # Last 10 steps
            step_summary = self._summarize_step(step)
            if step_summary:
                context_parts.append(f"{i}. {step_summary}")
        
        return "\n".join(context_parts)
    
    def _summarize_step(self, step: Dict[str, Any]) -> str:
        """Create a brief summary of a single step."""
        parts = []
        
        # Add URL if available
        if 'state' in step and 'url' in step['state']:
            url = step['state']['url']
            parts.append(f"on {self.extract_domain(url)}")
        
        # Add actions if available
        if 'model_output' in step and step['model_output']:
            actions = step['model_output'].get('action', [])
            action_names = []
            for action in actions:
                if isinstance(action, dict):
                    action_names.extend(action.keys())
            if action_names:
                parts.append(f"performed {', '.join(action_names)}")
        
        # Add results if available
        if 'result' in step and step['result']:
            results = step['result']
            if isinstance(results, list) and results:
                result = results[0]
                if isinstance(result, dict):
                    if result.get('error'):
                        parts.append("(failed)")
                    elif result.get('is_done'):
                        parts.append("(completed)")
        
        return " ".join(parts) if parts else ""
    
    def _generate_simple_summary(self, trajectory_data: Dict[str, Any], history: List[Dict]) -> str:
        """Generate a simple fallback summary without LLM."""
        total_steps = len(history)
        
        # Get domains visited
        domains = set()
        for step in history:
            if 'state' in step and 'url' in step['state']:
                domains.add(self.extract_domain(step['state']['url']))
        
        # Check if completed
        is_complete = self._determine_trajectory_type(trajectory_data) == 'complete'
        status = "completed" if is_complete else "attempted"
        
        domain_text = f" on {', '.join(list(domains)[:2])}" if domains else ""
        
        return f"Agent {status} a web browsing task{domain_text} in {total_steps} steps."
    
    def _extract_patterns(self, history: List[Dict], final_success: bool) -> Dict[str, Any]:
        """Extract failure modes or success patterns from history."""
        if final_success:
            return {'success_workflows': self._extract_success_patterns(history)}
        else:
            return {'fail_modes': self._extract_failure_patterns(history)}
    
    def _extract_failure_patterns(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """Extract failure patterns from unsuccessful trajectory."""
        fail_modes = []
        
        # Look for error patterns
        error_steps = []
        for i, step in enumerate(history):
            if 'result' in step and step['result']:
                results = step['result']
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and result.get('error'):
                            error_steps.append(i)
        
        if error_steps:
            fail_modes.append({
                'name': 'Action errors',
                'evidence_steps': error_steps[-3:],  # Last 3 errors
                'description': 'Multiple actions failed with errors'
            })
        
        # Look for repeated actions (potential loops)
        action_sequence = []
        for step in history:
            if 'model_output' in step and step['model_output']:
                actions = step['model_output'].get('action', [])
                for action in actions:
                    if isinstance(action, dict):
                        action_sequence.extend(action.keys())
        
        # Simple pattern detection for repeated actions
        if len(action_sequence) > 5:
            recent_actions = action_sequence[-5:]
            if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions in last 5
                fail_modes.append({
                    'name': 'Repetitive behavior',
                    'evidence_steps': list(range(len(history)-5, len(history))),
                    'description': f'Repeated similar actions: {", ".join(set(recent_actions))}'
                })
        
        return fail_modes
    
    def _extract_success_patterns(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """Extract success patterns from successful trajectory."""
        success_workflows = []
        
        # Identify key successful steps
        key_steps = []
        for i, step in enumerate(history):
            # Look for steps that led to completion
            if 'model_output' in step and step['model_output']:
                actions = step['model_output'].get('action', [])
                for action in actions:
                    if isinstance(action, dict) and 'done' in action:
                        key_steps.append(i)
                        break
        
        if key_steps:
            success_workflows.append({
                'name': 'Task completion',
                'evidence_steps': key_steps,
                'description': 'Successfully completed the assigned task'
            })
        
        return success_workflows
