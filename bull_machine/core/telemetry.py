"""
Bull Machine v1.5.0 - Telemetry Module
Simple logging for analysis and debugging
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


def log_telemetry(filename: str, data: Dict[str, Any]) -> None:
    """
    Log telemetry data to JSON file for analysis.

    Args:
        filename: Name of the telemetry file (e.g., "layer_masks.json")
        data: Dictionary of data to log
    """
    try:
        # Create telemetry directory if it doesn't exist
        telemetry_dir = Path("telemetry")
        telemetry_dir.mkdir(exist_ok=True)

        # Add timestamp to data
        data_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **data
        }

        # Append to file (create if doesn't exist)
        telemetry_file = telemetry_dir / filename

        # Read existing data if file exists
        existing_data = []
        if telemetry_file.exists():
            try:
                with open(telemetry_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Try to parse as JSON array first, then as single object
                        try:
                            existing_data = json.loads(content)
                            if not isinstance(existing_data, list):
                                existing_data = [existing_data]
                        except json.JSONDecodeError:
                            # File might contain single objects, start fresh
                            existing_data = []
            except Exception:
                existing_data = []

        # Append new data
        existing_data.append(data_with_timestamp)

        # Keep only last 1000 entries to prevent file bloat
        if len(existing_data) > 1000:
            existing_data = existing_data[-1000:]

        # Write back to file
        with open(telemetry_file, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)

    except Exception as e:
        # Don't let telemetry errors break the main application
        logging.debug(f"Telemetry logging failed for {filename}: {e}")


def get_telemetry_summary(filename: str) -> Dict[str, Any]:
    """
    Get summary statistics from a telemetry file.

    Args:
        filename: Name of the telemetry file

    Returns:
        Dictionary with summary statistics
    """
    try:
        telemetry_dir = Path("telemetry")
        telemetry_file = telemetry_dir / filename

        if not telemetry_file.exists():
            return {"error": "File not found", "entries": 0}

        with open(telemetry_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"error": "Empty file", "entries": 0}

            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]

                return {
                    "entries": len(data),
                    "latest_timestamp": data[-1].get("timestamp") if data else None,
                    "first_timestamp": data[0].get("timestamp") if data else None
                }

            except json.JSONDecodeError:
                return {"error": "Invalid JSON", "entries": 0}

    except Exception as e:
        return {"error": str(e), "entries": 0}