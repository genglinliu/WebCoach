#!/usr/bin/env python3
"""
WebCoach for WebCoach Framework

This module implements the WebCoach component that evaluates the current state of the
actor agent, retrieves relevant past experiences from the EMS, and decides whether to
intervene with advice.

Adapted from the original Coach Framework implementation for simplified use.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv
from ems import ExternalMemoryStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebCoach:
    """
    WebCoach evaluates the current state of the actor agent and provides advice
    based on relevant past experiences.
    """
    
    def __init__(self, model_name: str = "gpt-4o", ems_dir: str = "./coach_storage"):
        """
        Initialize the WebCoach.
        
        Args:
            model_name: The model to use for generating advice (default: gpt-4o)
            ems_dir: Directory where the EMS data is stored
        """
        self.model_name = model_name
        self.ems = ExternalMemoryStore(storage_dir=ems_dir)
        
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found. WebCoach will not work without it.")
    
    def generate_advice(self, current_state: Dict[str, Any], k: int = 3) -> Dict[str, Any]:
        """
        Generate advice based on the current state and relevant past experiences.
        
        Args:
            current_state: Current condensed state from the condenser
            k: Number of relevant experiences to retrieve from EMS
            
        Returns:
            Dictionary with 'intervene' (bool) and 'advice' (str) fields
        """
        try:
            # Retrieve relevant past experiences
            similar_experiences = self.ems.get_similar_experiences(current_state, k=k)
            
            logger.info(f"Retrieved {len(similar_experiences)} similar experiences for coaching")
            
            if not similar_experiences:
                logger.info("No similar experiences found, no intervention needed")
                return {"intervene": False, "advice": ""}
            
            # Generate advice using LLM
            advice = self._generate_llm_advice(current_state, similar_experiences)
            
            return advice
            
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            return {"intervene": False, "advice": ""}
    
    def _generate_llm_advice(self, current_state: Dict[str, Any], experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate advice using LLM based on current state and past experiences.
        
        Args:
            current_state: Current condensed state
            experiences: List of relevant past experiences
            
        Returns:
            Dictionary with intervention decision and advice
        """
        try:
            # Format experiences for the prompt
            experiences_text = self._format_experiences_for_prompt(experiences)
            
            # Prepare current state information
            current_summary = current_state.get('summary', 'No summary available')
            current_meta = current_state.get('meta', {})
            
            # Create the coaching prompt
            prompt = f"""You are an AI coach analyzing a web browsing agent's current situation based on similar past experiences.

CURRENT SITUATION:
- Summary: {current_summary}
- Domain: {current_meta.get('domain', 'unknown')}
- Steps taken: {current_meta.get('total_steps', 0)}
- Goal: {current_meta.get('goal', 'not specified')}

{experiences_text}

Based on the above analysis, decide whether to intervene with advice:

1. If you see patterns of failure in similar past experiences that apply to the current situation, you should intervene
2. If the agent appears to be making progress or if past experiences were mostly successful, you may choose not to intervene
3. Focus on actionable advice that can help avoid known failure patterns

Respond with a JSON object containing:
- "intervene": true/false (whether to provide advice)
- "advice": "string" (specific actionable advice if intervening, empty string if not)

Example response:
{{"intervene": true, "advice": "Based on similar past failures, try using alternative search terms or check if you're on the correct website section before proceeding."}}

Response:"""

            # Call LLM
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Try to parse as JSON
                advice_json = json.loads(response_text)
                
                # Validate the response structure
                if 'intervene' in advice_json and 'advice' in advice_json:
                    logger.info(f"WebCoach decision: intervene={advice_json['intervene']}")
                    return advice_json
                else:
                    logger.warning("Invalid response structure from LLM")
                    return {"intervene": False, "advice": ""}
                    
            except json.JSONDecodeError:
                logger.warning("LLM response was not valid JSON, attempting to extract advice")
                # Fallback: try to extract advice from free text
                return self._extract_advice_from_text(response_text)
                
        except Exception as e:
            logger.error(f"Error generating LLM advice: {e}")
            return {"intervene": False, "advice": ""}
    
    def _format_experiences_for_prompt(self, experiences: List[Dict[str, Any]]) -> str:
        """
        Format retrieved experiences for inclusion in the prompt.
        
        Args:
            experiences: List of relevant experiences retrieved from the EMS
            
        Returns:
            Formatted string of experiences
        """
        if not experiences:
            return "No relevant past experiences found."
        
        formatted = "RELEVANT PAST EXPERIENCES:\n"
        
        for i, exp in enumerate(experiences):
            # Get the experience data
            if 'experience' in exp:
                data = exp['experience']['data']  # From EMS format
                similarity = exp.get('similarity_score', 0.0)
            else:
                data = exp  # Direct format
                similarity = exp.get('similarity_score', 0.0)
            
            # Determine success status
            success_status = "SUCCESS" if data['meta'].get('final_success', False) else "FAILURE"
            
            formatted += f"\n{i+1}. {success_status} (similarity: {similarity:.3f})\n"
            formatted += f"   Summary: {data.get('summary', 'No summary')}\n"
            formatted += f"   Domain: {data['meta'].get('domain', 'unknown')}\n"
            formatted += f"   Steps: {data['meta'].get('total_steps', 0)}\n"
            
            # Add failure modes or success patterns
            if not data['meta'].get('final_success', False):
                fail_modes = data.get('fail_modes', [])
                if fail_modes:
                    formatted += "   Failure patterns:\n"
                    for mode in fail_modes[:2]:  # Limit to top 2
                        formatted += f"     - {mode['name']}: {mode['description']}\n"
            else:
                success_workflows = data.get('success_workflows', [])
                if success_workflows:
                    formatted += "   Success patterns:\n"
                    for workflow in success_workflows[:2]:  # Limit to top 2
                        formatted += f"     - {workflow['name']}: {workflow['description']}\n"
        
        return formatted
    
    def _extract_advice_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract advice from free text when JSON parsing fails.
        
        Args:
            text: Free text response from LLM
            
        Returns:
            Dictionary with intervention decision and advice
        """
        text_lower = text.lower()
        
        # Simple heuristics to determine if intervention is suggested
        intervention_keywords = ['should', 'try', 'avoid', 'consider', 'recommend', 'suggest']
        intervene = any(keyword in text_lower for keyword in intervention_keywords)
        
        if intervene:
            # Extract the text as advice
            advice = text.strip()
            # Clean up common JSON artifacts
            advice = advice.replace('{', '').replace('}', '').replace('"', '')
            return {"intervene": True, "advice": advice}
        else:
            return {"intervene": False, "advice": ""}
    
    def get_coaching_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the coaching system.
        
        Returns:
            Dictionary with coaching statistics
        """
        ems_stats = self.ems.get_stats()
        
        return {
            'model': self.model_name,
            'ems_experiences': ems_stats['total_experiences'],
            'successful_experiences': ems_stats['successful_experiences'],
            'failed_experiences': ems_stats['failed_experiences'],
            'domains_covered': ems_stats['domains']
        }
