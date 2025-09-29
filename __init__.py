"""
Bull Machine Trading System v1.6.2

Institutional-grade multi-domain confluence trading framework with crash-resistant optimization.

Features:
- 5-Domain Confluence Strategy (Wyckoff, Liquidity, Momentum, Temporal, Fusion)
- Multi-stage optimization (Grid Search → Bayesian → Walk-Forward)
- Professional tearsheet generation with fund-style metrics
- Risk parameter scaling for institutional return targets (8-15% annual)
- Production monitoring and deployment validation
"""

__version__ = "1.6.2"
__author__ = "Bull Machine Capital"
__description__ = "Institutional-grade multi-domain confluence trading framework"

# Core system components
from .run_complete_confluence_system import (
    load_multi_timeframe_data,
    run_complete_confluence_backtest,
    ConfluenceSignalGenerator
)

from .safe_grid_runner import SafeGridRunner
from .generate_institutional_tearsheet import generate_tearsheet

__all__ = [
    "__version__",
    "load_multi_timeframe_data",
    "run_complete_confluence_backtest",
    "ConfluenceSignalGenerator",
    "SafeGridRunner",
    "generate_tearsheet"
]