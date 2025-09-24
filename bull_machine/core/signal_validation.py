"""
Signal Format Validation and Standardization
Ensures consistent signal schema between analyzers and backtest engine
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd


def validate_signal(signal: Dict[str, Any]) -> bool:
    """
    Validate signal format against expected schema.

    Expected schema:
    {
        "timestamp": pd.Timestamp,
        "symbol": str,
        "bias": "long" | "short",
        "score": float,
        "reasons": list[str],
        "stop": float,
        "tps": [float, float, float]
    }
    """
    if not isinstance(signal, dict):
        logging.error(f"Signal validation failed: not a dict - {type(signal)}")
        return False

    required_fields = {
        "timestamp": (pd.Timestamp, type(None)),
        "symbol": str,
        "bias": str,
        "score": (int, float),
        "reasons": list,
    }

    for field, expected_type in required_fields.items():
        if field not in signal:
            logging.error(f"Signal validation failed: missing field '{field}'")
            return False

        if not isinstance(signal[field], expected_type):
            logging.error(
                f"Signal validation failed: {field} type {type(signal[field])} != {expected_type}"
            )
            return False

    # Validate bias values
    if signal["bias"] not in ["long", "short"]:
        logging.error(f"Signal validation failed: invalid bias '{signal['bias']}'")
        return False

    # Validate score range
    if not (0.0 <= signal["score"] <= 1.0):
        logging.error(f"Signal validation failed: score {signal['score']} not in [0.0, 1.0]")
        return False

    # Validate reasons
    if len(signal["reasons"]) == 0:
        logging.warning("Signal validation warning: empty reasons list")

    return True


def standardize_signal(
    raw_signal: Any, symbol: str, timestamp: Optional[pd.Timestamp] = None
) -> Optional[Dict[str, Any]]:
    """
    Convert various signal formats to standardized schema.

    Args:
        raw_signal: Signal object from fusion engine
        symbol: Trading symbol
        timestamp: Bar timestamp

    Returns:
        Standardized signal dict or None if invalid
    """
    if raw_signal is None:
        return None

    try:
        # Handle Signal objects
        if hasattr(raw_signal, "side") and hasattr(raw_signal, "confidence"):
            standardized = {
                "timestamp": timestamp or pd.Timestamp.now(),
                "symbol": symbol,
                "bias": raw_signal.side,
                "score": float(raw_signal.confidence),
                "reasons": getattr(raw_signal, "reasons", ["fusion_signal"]),
                "stop": None,  # Will be filled by risk manager
                "tps": None,  # Will be filled by risk manager
            }

        # Handle dict signals
        elif isinstance(raw_signal, dict):
            standardized = {
                "timestamp": timestamp or pd.Timestamp.now(),
                "symbol": symbol,
                "bias": raw_signal.get("side", raw_signal.get("bias", "neutral")),
                "score": float(raw_signal.get("confidence", raw_signal.get("score", 0.0))),
                "reasons": raw_signal.get("reasons", ["dict_signal"]),
                "stop": raw_signal.get("stop"),
                "tps": raw_signal.get("tps"),
            }
        else:
            logging.error(f"Unknown signal format: {type(raw_signal)}")
            return None

        # Validate before returning
        if validate_signal(standardized):
            return standardized
        else:
            return None

    except Exception as e:
        logging.error(f"Signal standardization failed: {e}")
        return None


def add_risk_management(signal: Dict[str, Any], risk_plan: Any) -> Dict[str, Any]:
    """
    Add risk management fields to signal.

    Args:
        signal: Standardized signal dict
        risk_plan: Risk plan from risk manager

    Returns:
        Signal with risk management fields added
    """
    if risk_plan is None:
        # Default risk management
        entry_price = signal.get("entry_price", 0.0)
        signal["stop"] = entry_price * 0.98 if signal["bias"] == "long" else entry_price * 1.02
        signal["tps"] = [
            entry_price * 1.01 if signal["bias"] == "long" else entry_price * 0.99,  # TP1: 1R
            entry_price * 1.02 if signal["bias"] == "long" else entry_price * 0.98,  # TP2: 2R
            entry_price * 1.03 if signal["bias"] == "long" else entry_price * 0.97,  # TP3: 3R
        ]
    else:
        # Use risk plan
        signal["stop"] = getattr(risk_plan, "stop", signal.get("stop"))

        # Convert TP levels to list of floats
        tp_levels = getattr(risk_plan, "tp_levels", [])
        if tp_levels:
            signal["tps"] = [tp.get("price", 0.0) for tp in tp_levels]
        else:
            signal["tps"] = signal.get("tps", [])

        signal["size"] = getattr(risk_plan, "size", 1000.0)
        signal["risk_amount"] = getattr(risk_plan, "risk_amount", 100.0)

    return signal


def log_signal_stats(signals: List[Dict[str, Any]], stage: str = ""):
    """Log signal statistics for debugging."""
    if not signals:
        logging.info(f"[{stage}] No signals generated")
        return

    long_count = sum(1 for s in signals if s.get("bias") == "long")
    short_count = sum(1 for s in signals if s.get("bias") == "short")
    avg_score = sum(s.get("score", 0) for s in signals) / len(signals)

    logging.info(
        f"[{stage}] Generated {len(signals)} signals: {long_count}L/{short_count}S, avg_score={avg_score:.3f}"
    )

    # Log individual signals for detailed debugging
    for i, signal in enumerate(signals):
        logging.debug(
            f"[{stage}] Signal {i}: {signal['bias']} @ {signal['score']:.3f} - {signal['reasons'][:2]}"
        )
