"""
Bull Machine Trading System v1.8.6

Institutional-grade algorithmic trading framework with archetype-based pattern detection.

Features:
- Archetype-based trading strategies (13 production archetypes)
- Multi-domain engine layer (Liquidity, Wyckoff, Macro, Temporal, Funding)
- Regime-aware soft gating with Empirical Bayes shrinkage
- Multi-objective optimization (NSGA-II with purging & embargo)
- Production monitoring and deployment validation
"""

__version__ = "1.8.6"
__author__ = "Bull Machine Capital"
__description__ = "Institutional-grade multi-domain confluence trading framework"

# Core system components
try:
    from .run_complete_confluence_system import (
        load_multi_timeframe_data,
        run_complete_confluence_backtest,
        ConfluenceSignalGenerator
    )
    from .safe_grid_runner import SafeGridRunner
    from .generate_institutional_tearsheet import generate_tearsheet
except ImportError:
    # Fallback for direct execution contexts
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        from run_complete_confluence_system import (
            load_multi_timeframe_data,
            run_complete_confluence_backtest,
            ConfluenceSignalGenerator
        )
        from safe_grid_runner import SafeGridRunner
        from generate_institutional_tearsheet import generate_tearsheet
    except ImportError:
        # Graceful degradation if modules not available
        load_multi_timeframe_data = None
        run_complete_confluence_backtest = None
        ConfluenceSignalGenerator = None
        SafeGridRunner = None
        generate_tearsheet = None

__all__ = [
    "__version__",
    "load_multi_timeframe_data",
    "run_complete_confluence_backtest",
    "ConfluenceSignalGenerator",
    "SafeGridRunner",
    "generate_tearsheet"
]