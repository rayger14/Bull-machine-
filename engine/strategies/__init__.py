"""
Strategy modules facade for Bull Machine.

This module provides a clean import interface for trading strategies
and archetype detection.
"""

# Re-export archetype logic for backward compatibility
from engine.archetypes.logic_v2_adapter import ArchetypeLogic

__all__ = ['ArchetypeLogic']
