"""
Model abstraction layer for Bull Machine trading strategies.

This module provides a clear interface for different model types:
- Archetype-based models (current pattern recognition system)
- Baseline models (simple benchmarks for comparison)
- Simple baselines (buy-hold-sell classifiers)
- ML models (future: LSTM, Random Forest, etc.)

All models implement the BaseModel interface for clean separation of concerns.
"""

from .base import BaseModel, Signal, Position
from .simple_classifier import BuyHoldSellClassifier
from .archetype_model import ArchetypeModel

# Import baseline models
from .baselines import (
    Baseline0_BuyAndHold,
    Baseline1_SMA200Trend,
    Baseline2_SMACrossover,
    Baseline3_RSIMeanReversion,
    Baseline4_VolTargetTrend,
    Baseline5_Cash,
    get_all_baselines,
)

__all__ = [
    # Core interfaces
    'BaseModel',
    'Signal',
    'Position',

    # Model types
    'BuyHoldSellClassifier',
    'ArchetypeModel',

    # Baseline models
    'Baseline0_BuyAndHold',
    'Baseline1_SMA200Trend',
    'Baseline2_SMACrossover',
    'Baseline3_RSIMeanReversion',
    'Baseline4_VolTargetTrend',
    'Baseline5_Cash',
    'get_all_baselines',
]
