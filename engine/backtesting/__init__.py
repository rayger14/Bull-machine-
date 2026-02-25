"""
Model-agnostic backtesting framework.

This module provides clean separation between models and backtesting:
- BacktestEngine: Runs any BaseModel on historical data
- BacktestResults: Standardized performance metrics
- ModelComparison: Compare multiple models side-by-side
- Validator: Walk-forward, train/test splits

Usage:
    from engine.models import BuyHoldSellClassifier, ArchetypeModel
    from engine.backtesting import BacktestEngine, ModelComparison

    # Create models
    baseline = BuyHoldSellClassifier()
    archetype = ArchetypeModel(config)

    # Compare on same data
    comparison = ModelComparison(data)
    results = comparison.compare(
        models=[baseline, archetype],
        train_period=('2022-01-01', '2022-12-31'),
        test_period=('2023-01-01', '2023-12-31')
    )
    results.plot()
"""

from .engine import BacktestEngine, BacktestResults, Trade
from .metrics import compute_metrics, MetricsSummary
from .comparison import ModelComparison, ComparisonReport
from .validator import WalkForwardValidator, TrainTestSplit

__all__ = [
    'BacktestEngine',
    'BacktestResults',
    'Trade',
    'compute_metrics',
    'MetricsSummary',
    'ModelComparison',
    'ComparisonReport',
    'WalkForwardValidator',
    'TrainTestSplit',
]
