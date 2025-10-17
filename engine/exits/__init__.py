"""
Exits Module

Enhanced exit system combining multiple strategies:
- Multi-modal exits (R-ladder + structural + liquidity + time)
- Macro echo rules (DXY/Oil/Yield correlations)

Author: Bull Machine v2.0
"""

from .multi_modal_exits import evaluate_multi_modal_exit, ExitSignal
from .macro_echo import analyze_macro_echo, MacroEchoSignal

__all__ = [
    'evaluate_multi_modal_exit',
    'ExitSignal',
    'analyze_macro_echo',
    'MacroEchoSignal',
]
