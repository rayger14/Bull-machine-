"""
Macro Context Engine - SMT Analysis Module

Provides Smart Money Theory (SMT) divergence detection and macro market context
analysis using USDT.D, BTC.D, and TOTAL3 for high-probability setup identification.
"""

from .signals import MacroContextEngine, SMTSignalType, HPS_Score
from .analysis import SMTAnalyzer, MacroRegime, ContextFilter

__all__ = [
    'MacroContextEngine',
    'SMTSignalType',
    'HPS_Score',
    'SMTAnalyzer',
    'MacroRegime',
    'ContextFilter'
]