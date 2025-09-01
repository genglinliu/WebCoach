"""
Evaluation is very straightforward. Given a path, all you need to do is find matching string in the agent_history.json file.

If you find BOTH 
string 1: <"is_done": true >
string 2: <"success": true, >

then you have a successful run.
"""

import os
import json

def eval_webvoyager_results(path):
    """
    Evaluation of agent_history.json file.
    """
    agent_history_path = os.path.join(path, "agent_history.json")
    with open(agent_history_path, "r") as f:
        agent_history = json.load(f)

    # result field in the last history
    last_history = agent_history[-1]
    result = last_history["result"]
    if result["is_done"] and result["success"]:
        return True
    return False