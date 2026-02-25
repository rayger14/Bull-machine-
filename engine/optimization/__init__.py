"""
Optimization utilities for Bull Machine trading system.

Provides institutional-grade multi-objective optimization with:
- True Pareto optimization (NSGA-II)
- Purging & embargo for lookahead bias prevention
- Portfolio correlation analysis
"""

from .multi_objective import (
    # Data structures
    Trade,
    OptimizationMetrics,

    # Purging & embargo
    purge_overlapping_trades,
    apply_embargo,
    purge_and_embargo_pipeline,

    # Multi-objective optimization
    create_pareto_study,
    select_best_from_pareto,
    create_multi_objective_function,

    # Portfolio management
    compute_strategy_correlation,
    build_uncorrelated_portfolio,

    # Utilities
    calculate_oos_consistency,
)

__all__ = [
    # Data structures
    'Trade',
    'OptimizationMetrics',

    # Purging & embargo
    'purge_overlapping_trades',
    'apply_embargo',
    'purge_and_embargo_pipeline',

    # Multi-objective optimization
    'create_pareto_study',
    'select_best_from_pareto',
    'create_multi_objective_function',

    # Portfolio management
    'compute_strategy_correlation',
    'build_uncorrelated_portfolio',

    # Utilities
    'calculate_oos_consistency',
]
