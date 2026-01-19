"""
Bull market archetype detectors.

This module contains specialized detectors for bull market patterns (long-biased):
- Spring/UTAD (A): Wyckoff spring accumulation reversals
- Order Block Retest (B): SMC demand zone retests
- BOS/CHOCH Reversal (C): Break of structure / change of character
- Liquidity Sweep (G): Stop hunt reversals
- Trap Within Trend (H): False breakdown continuations

MVP Status: 5 archetypes implemented (2025-12-12)
"""

from .spring_utad import SpringUTADArchetype, detect_spring_utad_signal
from .order_block_retest import OrderBlockRetestArchetype, detect_order_block_retest_signal
from .bos_choch_reversal import BOSCHOCHReversalArchetype, detect_bos_choch_reversal_signal
from .liquidity_sweep import LiquiditySweepArchetype, detect_liquidity_sweep_signal
from .trap_within_trend import TrapWithinTrendArchetype, detect_trap_within_trend_signal
from .wick_trap_moneytaur import WickTrapMoneytaurArchetype

__all__ = [
    # Archetype classes
    'SpringUTADArchetype',
    'OrderBlockRetestArchetype',
    'BOSCHOCHReversalArchetype',
    'LiquiditySweepArchetype',
    'TrapWithinTrendArchetype',
    'WickTrapMoneytaurArchetype',
    # Detection helpers
    'detect_spring_utad_signal',
    'detect_order_block_retest_signal',
    'detect_bos_choch_reversal_signal',
    'detect_liquidity_sweep_signal',
    'detect_trap_within_trend_signal',
]
