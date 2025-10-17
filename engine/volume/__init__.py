"""
Volume Module

Advanced volume analysis including:
- FRVP (Fixed Range Volume Profile)
- Volume DNA (absorption, climax, distribution patterns)

Author: Bull Machine v2.0
"""

from .frvp import calculate_frvp, FRVPProfile

__all__ = [
    'calculate_frvp',
    'FRVPProfile',
]
