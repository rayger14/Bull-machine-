"""
Archetype detection strategies.

This module will contain individual archetype detectors split by market regime:
- bull/: Bull market archetypes (A-M)
- bear/: Bear market archetypes (S1-S8)

Currently serves as a facade to the main logic_v2_adapter.py for backward compatibility.
Future refactoring will extract individual detectors into separate modules.
"""

# Re-export from main archetype logic
from engine.archetypes.logic_v2_adapter import ArchetypeLogic

__all__ = ['ArchetypeLogic']

# TODO: Future refactoring will add:
# from .bull.trap import TrapDetector
# from .bull.order_block import OrderBlockDetector
# from .bear.failed_rally import FailedRallyDetector
# from .bear.long_squeeze import LongSqueezeDetector
