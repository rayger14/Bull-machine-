import json
import os
import logging
from typing import Dict, Any

def load_state(path: str = ".bm_state.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {'prev_bias': 'neutral','last_signal_ts': None,'last_exit_ts': None}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load state from {path}: {e}")
        return {'prev_bias': 'neutral','last_signal_ts': None,'last_exit_ts': None}

def save_state(state: Dict[str, Any], path: str = ".bm_state.json") -> None:
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save state to {path}: {e}")
