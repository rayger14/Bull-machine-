#!/usr/bin/env python3
"""
Decision Gates Module

Provides adaptive entry and exit filtering based on runtime market intelligence.
"""

from engine.gates.decision import (
    check_gate5,
    check_assist_exit,
    compute_dynamic_sizing,
    apply_assist_exit_tighten,
    GateTelemetry
)

__all__ = [
    'check_gate5',
    'check_assist_exit',
    'compute_dynamic_sizing',
    'apply_assist_exit_tighten',
    'GateTelemetry'
]
