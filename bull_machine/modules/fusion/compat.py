"""Compatibility layer for legacy private APIs."""

import warnings
from typing import Any, Dict, List, Tuple

from .advanced import AdvancedFusionEngine as _AdvancedFusionEngine


class AdvancedFusionEngine(_AdvancedFusionEngine):
    """Back-compat shim for legacy private APIs. Prefer public evaluate(...) method."""

    def _check_vetoes(self, modules_data: Dict[str, Any]) -> List[str]:
        """Legacy private method - use evaluate() instead."""
        warnings.warn(
            "_check_vetoes is deprecated; use evaluate(...) and check 'vetoes' key.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            # Use the current public evaluation method
            result = self.fuse(modules_data)
            if result and hasattr(result, 'vetoes'):
                return list(result.vetoes)
            return []
        except Exception:
            # Safe fallback for any errors
            return []