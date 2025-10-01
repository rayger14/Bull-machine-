"""
Momentum Analysis Package

Provides momentum fallback signals with proper bounds and normalization.
"""

from .momentum_engine import MomentumEngine, MomentumSignal, momentum_delta, calculate_rsi, calculate_macd_norm

__all__ = ['MomentumEngine', 'MomentumSignal', 'momentum_delta', 'calculate_rsi', 'calculate_macd_norm']