#!/usr/bin/env python3
"""
Archetype Detection and Exit Logic Module

Provides 16+1 market archetypes with YAML-driven hard gates, isolated fusion
scoring, whale conflict penalties, and Smart Exits V2 (composite invalidation,
distress half-exit, chop-aware trailing).
"""

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.archetypes.telemetry import ArchetypeTelemetry

__all__ = [
    'ArchetypeLogic',
    'ArchetypeTelemetry'
]
