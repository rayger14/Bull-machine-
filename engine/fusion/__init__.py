"""
Fusion Engine Package

Advanced signal fusion with delta routing architecture.
"""

from .advanced_fusion import AdvancedFusionEngine, AdvancedFusionSignal, FusionTelemetry

__all__ = ['AdvancedFusionEngine', 'AdvancedFusionSignal', 'FusionTelemetry']

# ---- Back-compat shim for legacy tests ----
try:
    from .advanced_fusion import AdvancedFusionEngine as _AdvancedFusionEngine
except Exception:  # pragma: no cover
    _AdvancedFusionEngine = None

# Legacy type aliases the old tests expect
try:
    from engine.fusion import FusionSignal as _FusionSignal, DomainSignal as _DomainSignal, FusionEngine as _FusionEngine
except Exception:  # pragma: no cover
    _FusionSignal = _DomainSignal = _FusionEngine = None

# Back-compat alias for legacy tests expecting FusionEngine from this package
if _AdvancedFusionEngine:
    class FusionEngine(_AdvancedFusionEngine):  # type: ignore[misc]
        """
        Back-compat alias for legacy tests expecting FusionEngine.
        Inherits AdvancedFusionEngine without behavior changes.
        """
        pass

# Re-export names old tests import directly
FusionSignal = _FusionSignal
DomainSignal = _DomainSignal