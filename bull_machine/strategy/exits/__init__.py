"""
Exit Signal Framework
Intelligent exit signal detection and evaluation system.
"""

from .types import (
    ExitType, ExitAction, ExitSignal, ExitEvaluationResult,
    CHoCHContext, MomentumContext, TimeStopContext
)

from .rules import (
    CHoCHAgainstDetector, MomentumFadeDetector, TimeStopEvaluator
)

from .evaluators import (
    ExitSignalEvaluator, MTFDesyncEvaluator, create_default_exit_config
)

__all__ = [
    # Types
    'ExitType', 'ExitAction', 'ExitSignal', 'ExitEvaluationResult',
    'CHoCHContext', 'MomentumContext', 'TimeStopContext',

    # Rules
    'CHoCHAgainstDetector', 'MomentumFadeDetector', 'TimeStopEvaluator',

    # Evaluators
    'ExitSignalEvaluator', 'MTFDesyncEvaluator', 'create_default_exit_config'
]