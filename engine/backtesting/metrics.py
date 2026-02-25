"""
Performance metrics computation.

Placeholder for additional metrics beyond BacktestResults defaults.
"""

from typing import List
from dataclasses import dataclass
from .engine import Trade

@dataclass
class MetricsSummary:
    """Extended metrics summary."""
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    # Add more as needed


def compute_metrics(trades: List[Trade]) -> MetricsSummary:
    """
    Compute extended performance metrics.

    Args:
        trades: List of completed trades

    Returns:
        MetricsSummary with Sharpe, MDD, etc.
    """
    # Placeholder implementation
    return MetricsSummary(
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        avg_trade_duration=0.0
    )
