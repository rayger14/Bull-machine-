"""
Temporal Engine - Minimal Gann/TPI Implementation

Provides bounded temporal analysis including Time Price Integration (TPI)
with conservative caps and simplified Gann square analysis.

v1.8.6 adds temporal_signal for comprehensive Gann/cycle/thermo-floor analysis.
v2.0   adds TemporalConfluenceEngine as unified interface for all temporal analysis.
"""

from .tpi import TemporalEngine, TPISignal, TPIType
from .gann import GannAnalyzer, GannLevel, GannTimeProject
from .cycles import CycleDetector, CycleType, CycleSignal
from .gann_cycles import temporal_signal
from .temporal_confluence import TemporalConfluenceEngine, TemporalBarResult

__all__ = [
    'TemporalEngine',
    'TPISignal',
    'TPIType',
    'GannAnalyzer',
    'GannLevel',
    'GannTimeProject',
    'CycleDetector',
    'CycleType',
    'CycleSignal',
    'temporal_signal',
    'TemporalConfluenceEngine',
    'TemporalBarResult',
]