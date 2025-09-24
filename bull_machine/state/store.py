"""Bull Machine v1.3 - State Persistence"""

import json
import logging
import os
from typing import Any, Dict


def load_state(path: str = ".bm_state.json") -> Dict[str, Any]:
    """
    Load state from JSON file.

    Args:
        path: Path to state file

    Returns:
        Dictionary with state data or empty dict if not found
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                state = json.load(f)
                logging.debug(f"Loaded state from {path}")
                return state
        except Exception as e:
            logging.warning(f"Could not load state from {path}: {e}")
            return {}
    else:
        logging.debug(f"No state file found at {path}, starting fresh")
        return {}


def save_state(state: Dict[str, Any], path: str = ".bm_state.json") -> None:
    """
    Save state to JSON file.

    Args:
        state: Dictionary to save
        path: Path to state file
    """
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            logging.debug(f"Saved state to {path}")
    except Exception as e:
        logging.error(f"Could not save state to {path}: {e}")


def update_state(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update state with new values.

    Common keys:
    - prev_bias: Previous LTF bias
    - last_signal_ts: Timestamp of last signal
    - last_htf_bias: Last HTF bias (for MTF)
    - last_mtf_bias: Last MTF bias (for MTF)
    - cooldown_until: Timestamp when cooldown ends
    - open_positions: List of open position info

    Args:
        state: Current state dict
        **kwargs: Key-value pairs to update

    Returns:
        Updated state dict
    """
    state.update(kwargs)
    return state


def check_cooldown(state: Dict[str, Any], current_ts: int) -> bool:
    """
    Check if we're in cooldown period.

    Args:
        state: Current state
        current_ts: Current timestamp

    Returns:
        True if in cooldown, False otherwise
    """
    cooldown_until = state.get("cooldown_until", 0)
    return current_ts < cooldown_until


def set_cooldown(state: Dict[str, Any], current_ts: int, bars: int = 5) -> Dict[str, Any]:
    """
    Set cooldown period.

    Args:
        state: Current state
        current_ts: Current timestamp
        bars: Number of bars for cooldown

    Returns:
        Updated state
    """
    # Assuming hourly bars, adjust as needed
    cooldown_duration = bars * 3600  # seconds
    state["cooldown_until"] = current_ts + cooldown_duration
    return state
