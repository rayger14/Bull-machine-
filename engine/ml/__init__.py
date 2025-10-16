"""
Bull Machine ML Stack

Comprehensive ML enhancements for trading engine:
1. Kelly-Lite Dynamic Risk Sizing (GBM)
2. Fusion Scoring ML (XGBoost)
3. Enhanced Macro Signals (rule-based + ML-ready)
4. Smart Exit Optimization (coming soon)
5. Cooldown Optimization (coming soon)

All modules return bounded deltas compatible with existing engine architecture
"""

from .kelly_lite_sizer import KellyLiteSizer
from .fusion_scorer_ml import FusionScorerML
from .macro_signals_enhanced import MacroSignalsEnhanced

__all__ = [
    'KellyLiteSizer',
    'FusionScorerML',
    'MacroSignalsEnhanced'
]
