import json
import os
from typing import Any, Dict


def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Load Bull Machine v1.1 configuration from repo /config, else defaults."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {config_path}: {e}")
    return get_v1_1_defaults()


def get_v1_1_defaults() -> Dict[str, Any]:
    return {
        "version": "1.1",
        "features": {
            "wyckoff": True,
            "liquidity_basic": True,
            "enhanced_stops": True,
            "tp_ladder": True,
            "signal_ttl": True,
            "confidence_floor": True,
            "dynamic_ttl": True,
        },
        "signals": {
            "confidence_threshold": 0.72,
            "weights": {
                "wyckoff": 0.60,
                "liquidity": 0.40,
                "smt": 0.0,
                "macro": 0.0,
                "temporal": 0.0,
            },
        },
        "wyckoff": {"lookback_bars": 50, "bias_hysteresis_bars": 2},
        "range": {"time_in_range_bars_min": 20, "net_progress_threshold": 0.25},
        "risk": {
            "account_risk_percent": 1.0,
            "max_risk_per_trade": 200.0,
            "stop": {"method": "swing_with_atr_guardrail", "atr_mult": 2.0},
            "tp_ladder": {
                "tp1": {"r": 1.0, "pct": 33, "action": "move_stop_to_breakeven"},
                "tp2": {"r": 2.0, "pct": 33, "action": "trail_remainder"},
                "tp3": {"r": 3.0, "pct": 34, "action": "liquidate_or_hard_trail"},
                "range_mode_multiplier": 0.8,
            },
            "ttl_bars": 18,
            "ttl_dynamic": {
                "min": 8,
                "max": 30,
                "atr_period": 14,
                "atr_high_pct": 0.015,
                "atr_low_pct": 0.007,
                "delta_high_vol": 4,
                "delta_low_vol": -3,
                "trend_bonus": 3,
                "range_penalty": -3,
                "range_penetration_thresh": 0.25,
            },
            "trail_mode": "swing",
        },
    }
