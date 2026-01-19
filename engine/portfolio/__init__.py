"""
Portfolio management components.

Includes:
- RegimeWeightAllocator: Regime-conditioned allocation with soft gating
"""

from .regime_allocator import RegimeWeightAllocator

__all__ = ['RegimeWeightAllocator']
