"""
Bear market archetype runtime utilities

This package contains runtime feature enrichment and pattern detection
modules specifically designed for bear market (short-biased) trading patterns.
"""

# Runtime feature enrichment (legacy)
from .failed_rally_runtime import S2RuntimeFeatures
from .funding_divergence_runtime import S4RuntimeFeatures
from .long_squeeze_runtime import S5RuntimeFeatures

# Production archetype detectors
from .liquidity_vacuum import LiquidityVacuumArchetype, detect_liquidity_vacuum_signal
from .funding_divergence import FundingDivergenceArchetype, detect_funding_divergence_signal
from .long_squeeze import LongSqueezeArchetype, detect_long_squeeze_signal

__all__ = [
    # Runtime features
    'S2RuntimeFeatures',
    'S4RuntimeFeatures',
    'S5RuntimeFeatures',
    # Archetype detectors
    'LiquidityVacuumArchetype',
    'FundingDivergenceArchetype',
    'LongSqueezeArchetype',
    # Detection helpers
    'detect_liquidity_vacuum_signal',
    'detect_funding_divergence_signal',
    'detect_long_squeeze_signal',
]
