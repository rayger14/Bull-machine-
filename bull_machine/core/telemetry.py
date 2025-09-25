"""
Bull Machine v1.5.0 - Enhanced Telemetry System
Comprehensive logging and metrics collection for analysis and debugging.
"""

import json
import datetime
import os
from typing import Dict, Any, List
from pathlib import Path


def log_telemetry(path: str, data: Dict[str, Any]) -> None:
    """
    Log telemetry data to specified file.

    Args:
        path: File path for telemetry log
        data: Dictionary of telemetry data

    Note:
        Each log entry includes ISO timestamp and the provided data.
        File is appended to, creating newline-delimited JSON.
    """
    # Ensure telemetry directory exists
    telemetry_dir = Path("telemetry")
    telemetry_dir.mkdir(exist_ok=True)

    # Full path
    full_path = telemetry_dir / path

    # Create record with timestamp
    record = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        **data
    }

    # Append to file
    with open(full_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_performance_metrics(metrics: Dict[str, Any]) -> None:
    """Log performance metrics for analysis."""
    log_telemetry("performance.json", {
        "type": "performance_metrics",
        **metrics
    })


def log_trade_decision(decision: Dict[str, Any]) -> None:
    """Log trade decision details."""
    log_telemetry("trade_decisions.json", {
        "type": "trade_decision",
        **decision
    })


def log_layer_analysis(layer: str, analysis: Dict[str, Any]) -> None:
    """Log analysis results from individual layers."""
    log_telemetry("layer_analysis.json", {
        "type": "layer_analysis",
        "layer": layer,
        **analysis
    })


def log_feature_flag_usage(feature: str, enabled: bool, context: Dict[str, Any] = None) -> None:
    """Log feature flag usage for analysis."""
    log_telemetry("feature_flags.json", {
        "type": "feature_flag",
        "feature": feature,
        "enabled": enabled,
        "context": context or {}
    })


def log_acceptance_gate_check(gate: str, result: Dict[str, Any]) -> None:
    """Log acceptance gate validation results."""
    log_telemetry("acceptance_gates.json", {
        "type": "acceptance_gate",
        "gate": gate,
        **result
    })


def clear_telemetry_logs() -> None:
    """Clear all telemetry log files."""
    telemetry_dir = Path("telemetry")
    if telemetry_dir.exists():
        for file in telemetry_dir.glob("*.json"):
            file.unlink()


def get_telemetry_summary() -> Dict[str, Any]:
    """
    Get summary of recent telemetry data.

    Returns:
        Dict containing telemetry statistics
    """
    telemetry_dir = Path("telemetry")
    if not telemetry_dir.exists():
        return {"error": "No telemetry directory found"}

    summary = {}

    for log_file in telemetry_dir.glob("*.json"):
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            summary[log_file.stem] = {
                "entries": len(lines),
                "latest": json.loads(lines[-1]) if lines else None,
                "file_size": log_file.stat().st_size
            }
        except (json.JSONDecodeError, IndexError):
            summary[log_file.stem] = {
                "entries": 0,
                "error": "Failed to parse log file"
            }

    return summary


class TelemetryContext:
    """Context manager for telemetry sessions."""

    def __init__(self, session_name: str, metadata: Dict[str, Any] = None):
        self.session_name = session_name
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.datetime.utcnow()
        log_telemetry("sessions.json", {
            "type": "session_start",
            "session": self.session_name,
            "metadata": self.metadata
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()

        log_telemetry("sessions.json", {
            "type": "session_end",
            "session": self.session_name,
            "duration_seconds": duration,
            "success": exc_type is None,
            "error": str(exc_val) if exc_val else None
        })

    def log(self, event: str, data: Dict[str, Any] = None):
        """Log event within the session context."""
        log_telemetry("sessions.json", {
            "type": "session_event",
            "session": self.session_name,
            "event": event,
            "data": data or {}
        })


def analyze_telemetry_trends() -> Dict[str, Any]:
    """
    Analyze telemetry data for trends and insights.

    Returns:
        Dict containing trend analysis
    """
    telemetry_dir = Path("telemetry")
    if not telemetry_dir.exists():
        return {"error": "No telemetry data available"}

    analysis = {
        "layer_performance": {},
        "feature_usage": {},
        "error_patterns": [],
        "recommendations": []
    }

    # Analyze layer analysis logs
    layer_file = telemetry_dir / "layer_analysis.json"
    if layer_file.exists():
        try:
            with open(layer_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    layer = entry.get("layer")
                    if layer:
                        if layer not in analysis["layer_performance"]:
                            analysis["layer_performance"][layer] = {
                                "calls": 0,
                                "avg_score": 0,
                                "scores": []
                            }

                        analysis["layer_performance"][layer]["calls"] += 1
                        if "score" in entry:
                            analysis["layer_performance"][layer]["scores"].append(entry["score"])

        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Calculate averages
    for layer, stats in analysis["layer_performance"].items():
        if stats["scores"]:
            stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])

    # Analyze feature flag usage
    feature_file = telemetry_dir / "feature_flags.json"
    if feature_file.exists():
        try:
            with open(feature_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    feature = entry.get("feature")
                    if feature:
                        if feature not in analysis["feature_usage"]:
                            analysis["feature_usage"][feature] = {
                                "enabled_count": 0,
                                "disabled_count": 0
                            }

                        if entry.get("enabled"):
                            analysis["feature_usage"][feature]["enabled_count"] += 1
                        else:
                            analysis["feature_usage"][feature]["disabled_count"] += 1

        except (json.JSONDecodeError, FileNotFoundError):
            pass

    return analysis