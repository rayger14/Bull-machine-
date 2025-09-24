"""
Exit Signal Framework
Intelligent exit signal detection and evaluation system.
"""

from .evaluators import ExitSignalEvaluator, MTFDesyncEvaluator, create_default_exit_config
from .rules import CHoCHAgainstDetector, MomentumFadeDetector, TimeStopEvaluator
from .types import (
    CHoCHContext,
    ExitAction,
    ExitEvaluationResult,
    ExitSignal,
    ExitType,
    MomentumContext,
    TimeStopContext,
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
