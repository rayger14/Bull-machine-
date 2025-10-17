"""
Structure Analysis Module

Provides advanced market structure detection:
- Internal vs External structure tracking
- BOMS (Break of Market Structure) detection
- Pattern recognition (1-2-3 Squiggle)
- Range outcome classification
"""

from .internal_external import detect_structure_state, StructureState
from .boms_detector import detect_boms, BOMSSignal
from .squiggle_pattern import detect_squiggle_123, SquigglePattern
from .range_classifier import classify_range_outcome, RangeOutcome

__all__ = [
    'detect_structure_state',
    'StructureState',
    'detect_boms',
    'BOMSSignal',
    'detect_squiggle_123',
    'SquigglePattern',
    'classify_range_outcome',
    'RangeOutcome',
]
