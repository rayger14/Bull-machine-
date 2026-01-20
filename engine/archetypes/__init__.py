#!/usr/bin/env python3
"""
PR#6A: Rule-Based Archetype Expansion Module

Provides 11 distinct market archetypes (A-H + K, L, M) for clean labeled
data generation before PyTorch training.
"""

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.archetypes.telemetry import ArchetypeTelemetry

__all__ = [
    'ArchetypeLogic',
    'ArchetypeTelemetry'
]
