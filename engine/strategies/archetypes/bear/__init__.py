"""
Bear market archetype runtime utilities

This package contains runtime feature enrichment and pattern detection
modules specifically designed for bear market (short-biased) trading patterns.
"""

from .failed_rally_runtime import S2RuntimeFeatures
from .funding_divergence_runtime import S4RuntimeFeatures
from .long_squeeze_runtime import S5RuntimeFeatures

__all__ = ['S2RuntimeFeatures', 'S4RuntimeFeatures', 'S5RuntimeFeatures']
