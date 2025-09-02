#!/usr/bin/env python3
"""
Coach Callback for Browser-Use Integration

This module provides the main callback function that integrates the Coach Framework
with browser-use agents during execution.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global coach components - initialized on first use
_coach_components = None
_coach_config = None

class CoachComponents:
    """Lazy-loaded coach components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.condenser = None
        self.ems = None
        self.web_coach = None
        self.storage_dir = Path(config.get('storage_dir', './coach_storage'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def get_condenser(self):
        """Lazy load condenser"""
        if self.condenser is None:
            from condenser import Condenser
            self.condenser = Condenser(
                model_name=self.config.get('model', 'gpt-4o')
            )
        return self.condenser
    
    def get_ems(self):
        """Lazy load EMS"""
        if self.ems is None:
            from ems import ExternalMemoryStore
            self.ems = ExternalMemoryStore(storage_dir=str(self.storage_dir))
        return self.ems
    
    def get_web_coach(self):
        """Lazy load WebCoach"""
        if self.web_coach is None:
            from web_coach import WebCoach
            self.web_coach = WebCoach(
                model_name=self.config.get('model', 'gpt-4o'),
                ems_dir=str(self.storage_dir)
            )
        return self.web_coach

def configure_coach(config: Dict[str, Any]):
    """Configure coach components with given config"""
    global _coach_components, _coach_config
    from config import validate_coach_config
    
    _coach_config = validate_coach_config(config)
    _coach_components = CoachComponents(_coach_config)
    logger.info(f"Coach configured with model: {_coach_config.get('model', 'gpt-4o')}, frequency: {_coach_config.get('frequency', 5)}")

async def coach_step_callback(agent) -> None:
    """
    Main callback function for coach integration with browser-use agents.
    
    This function is called after each agent step and:
    1. Extracts the current trajectory from agent history
    2. Processes it with the condenser to determine if complete or partial
    3. Routes to EMS (if complete) or WebCoach (if partial)
    4. Injects coaching advice back into the agent if intervention is needed
    
    Args:
        agent: The browser-use agent instance
    """
    global _coach_components, _coach_config
    
    try:
        # Skip if coach not configured
        if _coach_components is None:
            return
            
        current_step = agent.state.n_steps
        frequency = _coach_config.get('frequency', 1)
        
        # Check if we should run coach based on frequency
        if current_step % frequency != 0:
            logger.debug(f"Skipping coach - step {current_step} not divisible by frequency {frequency}")
            return
            
        logger.info(f"Coach callback triggered at step {current_step}")
        
        # Extract agent history
        history_dict = agent.history.model_dump()
        
        if not history_dict.get('history'):
            logger.debug("No history available yet, skipping coach")
            return
            
        logger.info(f"Processing trajectory with {len(history_dict['history'])} steps")
        
        # Process with condenser to determine trajectory type
        condenser = _coach_components.get_condenser()
        condensed_data, trajectory_type = await process_trajectory_async(condenser, history_dict)
        
        logger.info(f"Condenser determined trajectory type: {trajectory_type}")
        
        if trajectory_type == 'complete':
            # Store complete trajectory in EMS
            logger.info("Storing complete trajectory in EMS")
            ems = _coach_components.get_ems()
            success = ems.add_experience(condensed_data)
            if success:
                logger.info("Successfully stored trajectory in EMS")
            else:
                logger.warning("Failed to store trajectory in EMS")
                
        else:
            # Get advice for partial trajectory
            logger.info("Getting advice for partial trajectory")
            web_coach = _coach_components.get_web_coach()
            advice = web_coach.generate_advice(condensed_data, k=3)
            
            # Inject advice if intervention is needed
            if advice.get('intervene', False):
                advice_text = advice.get('advice', '')
                if advice_text.strip():
                    logger.info(f"Coach intervention: {advice_text[:100]}...")
                    await inject_coaching_advice(agent, advice_text)
                else:
                    logger.info("Coach decided to intervene but no advice text provided")
            else:
                logger.info("Coach determined no intervention needed")
                
    except Exception as e:
        logger.error(f"Error in coach callback: {e}", exc_info=True)

async def process_trajectory_async(condenser, history_dict: Dict[str, Any]):
    """
    Process trajectory with condenser in async context.
    
    Args:
        condenser: Condenser instance
        history_dict: Agent history dictionary
        
    Returns:
        Tuple of (condensed_data, trajectory_type)
    """
    import tempfile
    
    # Save history to temporary file for processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(history_dict, f, indent=2)
        temp_path = f.name
    
    try:
        # Process with condenser
        return condenser.process_trajectory(temp_path)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

async def inject_coaching_advice(agent, advice: str) -> None:
    """
    Inject coaching advice into the agent's message history.
    
    Args:
        agent: The browser-use agent instance
        advice: The coaching advice text
    """
    try:
        from browser_use.llm.messages import SystemMessage
        
        coaching_content = f"""🤖 COACHING ADVICE: Based on analysis of similar past experiences:

{advice}

Consider this guidance when planning your next actions. If you're encountering repeated issues, try alternative approaches or ask for human help."""
        
        logger.info("Injecting coaching advice into agent")
        
        coaching_message = SystemMessage(content=coaching_content)
        
        # Inject the advice as a system message
        if hasattr(agent, '_message_manager') and hasattr(agent._message_manager, '_add_message_with_type'):
            agent._message_manager._add_message_with_type(coaching_message, 'consistent')
            logger.info("Coaching advice successfully injected")
        else:
            logger.warning("Could not inject coaching advice - message manager not accessible")
            
    except Exception as e:
        logger.error(f"Error injecting coaching advice: {e}", exc_info=True)
