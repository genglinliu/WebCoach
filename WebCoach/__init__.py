"""
WebCoach Framework for Browser-Use Integration

A coaching system that provides real-time advice to browser-use agents
based on learned experiences from past executions.

Components:
- coach_callback: Main integration point with browser-use agents
- condenser: Processes raw trajectories into structured summaries
- ems: External Memory Store for experience storage and retrieval
- web_coach: Generates advice based on similar past experiences
"""

from coach_callback import coach_step_callback

__all__ = ['coach_step_callback']
