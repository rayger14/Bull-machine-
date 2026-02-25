"""
Bull Machine ML Stack

Comprehensive ML enhancements for trading engine:
1. Kelly-Lite Dynamic Risk Sizing (GBM) - requires sklearn
2. Fusion Scoring ML (XGBoost) - requires xgboost, sklearn
3. Enhanced Macro Signals (rule-based + ML-ready)
4. Smart Exit Optimization (coming soon)
5. Cooldown Optimization (coming soon)

All modules return bounded deltas compatible with existing engine architecture.

Integration:
    These modules are wired into IsolatedArchetypeEngine via opt-in config flags:
    - "use_ml_fusion": true  -> enables FusionScorerML blending
    - "use_kelly_sizing": true -> enables KellyLiteSizer position sizing

    See engine/integrations/isolated_archetype_engine.py for integration details.

Dependencies are imported lazily so the engine works without xgboost/sklearn installed.
"""

__all__ = [
    'KellyLiteSizer',
    'FusionScorerML',
    'MacroSignalsEnhanced'
]


def __getattr__(name):
    """Lazy import to avoid hard dependency on xgboost/sklearn at module load time."""
    if name == 'KellyLiteSizer':
        from .kelly_lite_sizer import KellyLiteSizer
        return KellyLiteSizer
    elif name == 'FusionScorerML':
        from .fusion_scorer_ml import FusionScorerML
        return FusionScorerML
    elif name == 'MacroSignalsEnhanced':
        from .macro_signals_enhanced import MacroSignalsEnhanced
        return MacroSignalsEnhanced
    raise AttributeError(f"module 'engine.ml' has no attribute {name!r}")
