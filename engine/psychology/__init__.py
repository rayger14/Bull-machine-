"""
Psychology Module

Quantifies herd behavior and trap detection without sentiment data.
Uses price action, volume, and structural patterns to detect psychological traps.

Components:
- PTI (Psychology Trap Index): Detects retail traps via divergences, exhaustion, wicks
- Fake-out Intensity: Quantifies failed breakout severity

Author: Bull Machine v2.0
"""

from .pti import calculate_pti, PTISignal
from .fakeout_intensity import detect_fakeout_intensity, FakeoutSignal

__all__ = [
    'calculate_pti',
    'PTISignal',
    'detect_fakeout_intensity',
    'FakeoutSignal',
]
