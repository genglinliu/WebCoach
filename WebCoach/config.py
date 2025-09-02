#!/usr/bin/env python3
"""
Configuration management for WebCoach Framework

This module provides default configurations and validation for the Coach components.
"""

import os
from typing import Dict, Any
from pathlib import Path

DEFAULT_COACH_CONFIG = {
    "enabled": False,
    "model": "gpt-4o",
    "frequency": 5,  # Intervene every N steps
    "storage_dir": "./coach_storage",
    "debug": False,
    "max_experiences": 1000,  # Maximum experiences to store in EMS
    "similarity_threshold": 0.3,  # Minimum similarity for experience retrieval
    "advice_max_length": 500,  # Maximum length of advice text
}

def validate_coach_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize coach configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration
    """
    # Start with default config
    validated_config = DEFAULT_COACH_CONFIG.copy()
    
    # Override with provided config
    if config:
        validated_config.update(config)
    
    # Validate specific fields
    if validated_config["frequency"] < 1:
        validated_config["frequency"] = 1
    
    if validated_config["max_experiences"] < 10:
        validated_config["max_experiences"] = 10
    
    if not 0 <= validated_config["similarity_threshold"] <= 1:
        validated_config["similarity_threshold"] = 0.3
    
    # Ensure storage directory is absolute path
    storage_dir = Path(validated_config["storage_dir"])
    if not storage_dir.is_absolute():
        # Make it relative to the current working directory
        validated_config["storage_dir"] = str(storage_dir.resolve())
    
    return validated_config

def get_coach_config_from_main_config(main_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate coach configuration from main config.
    
    Args:
        main_config: Main configuration dictionary (e.g., from config.yaml)
        
    Returns:
        Validated coach configuration
    """
    coach_config = main_config.get("coach", {})
    return validate_coach_config(coach_config)
