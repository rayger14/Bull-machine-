"""
Advanced Exit Rules for Bull Machine v1.4.1
Implements Wyckoff phase-based exits, Moneytaur trailing, and Bojan protection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from .types import ExitSignal, ExitType, ExitAction


@dataclass
class ExitDecision:
    """Unified exit decision structure."""
    action: str  # 'partial', 'full', 'trail', 'flip', None
    size_pct: float  # 0.0-1.0 for partial exits
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    flip_bias: Optional[str] = None  # 'long'/'short' for flips
    cooldown_bars: int = 0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseExitRule:
    """Base class for all exit rules."""
    name = "base"
    required_keys = set()

    def __init__(self, cfg: Dict[str, Any], shared_cfg: Dict[str, Any]):
        self.cfg = cfg
        self.shared = shared_cfg
        self._validate_config()
        self._log_effective_params()

    def _validate_config(self):
        """Ensure all required keys are present."""
        missing = self.required_keys - set(self.cfg.keys())
        if missing:
            raise ValueError(f"{self.name} missing required keys: {missing}")

    def _log_effective_params(self):
        """Log parameters being used for telemetry."""
        logging.info(f"EXIT_RULE_INIT {self.name} params={self.cfg}")

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:
        """Override in subclasses."""
        raise NotImplementedError


# ============================================================================
# MARKUP LONG EXIT RULES
# ============================================================================

class MarkupSOWUTWarning(BaseExitRule):
    """
    SOW/UT Warning (Partial: 25-50%)
    Detects premium exhaustion, volume divergence, Bojan highs, MTF desync.
    """
    name = "markup_sow_ut_warning"
    required_keys = {"premium_floor", "vol_divergence_ratio", "wick_atr_mult",
                     "mtf_desync_floor", "veto_needed", "partial_pct", "trail_atr_buffer_R"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        if trade_plan['bias'] != 'long':
            return None

        # Get current bar and recent data
        current = df.iloc[-1]
        recent = df.tail(20)

        # 1. Premium exhaustion check (Layer 2 - Liquidity)
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        if range_size > 0:
            premium_position = (current['close'] - range_low) / range_size
        else:
            premium_position = 0.5

        premium_exhausted = premium_position > self.cfg['premium_floor']

        # 2. Volume divergence (Layer 5 - Volume)
        vol_sma = df['volume'].tail(10).mean()
        vol_divergence = current['volume'] < (self.cfg['vol_divergence_ratio'] * vol_sma)

        # 3. Bojan High check (wick analysis)
        atr = self._calculate_atr(df, period=self.shared['atr_period'])
        wick_up = current['high'] - max(current['open'], current['close'])
        bojan_wick = (wick_up / atr) > self.cfg['wick_atr_mult'] if atr > 0 else False
        close_below_open = current['close'] < current['open']
        bojan_high = bojan_wick and close_below_open

        # 4. MTF desync check (Layer 7)
        mtf_desync = False
        if mtf_context and 'mtf_score' in scores:
            mtf_desync = scores['mtf_score'] < self.cfg['mtf_desync_floor']

        # 5. Wyckoff weakness (Layer 1)
        wyckoff_weak = scores.get('wyckoff', 1.0) < 0.5

        # Count veto conditions
        veto_count = sum([
            premium_exhausted,
            vol_divergence,
            bojan_high,
            mtf_desync,
            wyckoff_weak
        ])

        # Trigger if enough vetos
        if veto_count >= self.cfg['veto_needed']:
            # Calculate new stop loss (BE + buffer)
            entry = trade_plan['entry_price']
            risk_r = abs(entry - trade_plan['sl'])
            new_sl = entry + (self.cfg['trail_atr_buffer_R'] * risk_r)

            return ExitDecision(
                action='partial',
                size_pct=self.cfg['partial_pct'],
                new_sl=new_sl,
                reason=f"SOW/UT Warning: {veto_count} vetos (premium:{premium_exhausted}, "
                      f"vol:{vol_divergence}, bojan:{bojan_high}, mtf:{mtf_desync})",
                metadata={
                    'premium_position': premium_position,
                    'volume_ratio': current['volume'] / vol_sma if vol_sma > 0 else 0,
                    'veto_count': veto_count,
                    'scores': scores
                }
            )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.tail(period).mean()


class MarkupUTADRejection(BaseExitRule):
    """
    UTAD Rejection (Partial: 50%)
    Detects false breakouts, liquidity sweep fails, structure breaks.
    """
    name = "markup_utad_rejection"
    required_keys = {"wick_close_frac", "fib_retrace", "partial_pct",
                     "trail_to_structure_minus_atr"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        if trade_plan['bias'] != 'long':
            return None

        current = df.iloc[-1]

        # Check for false breakout
        prior_high = trade_plan.get('initial_range_high', df.tail(20)['high'].max())
        false_breakout = (
            current['high'] > prior_high and
            current['close'] < (current['low'] + (current['high'] - current['low']) * self.cfg['wick_close_frac'])
        )

        # Check for liquidity sweep fail (Layer 2)
        liquidity_fail = scores.get('liquidity', 1.0) < 0.4

        # Check for structure break (Fib retracement)
        entry = trade_plan['entry_price']
        leg_high = df.tail(bars_since_entry)['high'].max() if bars_since_entry > 0 else current['high']
        fib_level = entry + (leg_high - entry) * self.cfg['fib_retrace']
        structure_break = current['close'] < fib_level

        if false_breakout and (liquidity_fail or structure_break):
            # Calculate new stop at structure low minus ATR
            atr = self._calculate_atr(df, self.shared['atr_period'])
            structure_low = df.tail(10)['low'].min()
            new_sl = structure_low - atr if self.cfg['trail_to_structure_minus_atr'] else None

            return ExitDecision(
                action='partial',
                size_pct=self.cfg['partial_pct'],
                new_sl=new_sl,
                reason=f"UTAD Rejection: false_breakout:{false_breakout}, "
                      f"liq_fail:{liquidity_fail}, struct_break:{structure_break}",
                metadata={
                    'prior_high': prior_high,
                    'fib_level': fib_level,
                    'current_close': current['close']
                }
            )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.tail(period).mean()


class MarkupExhaustion(BaseExitRule):
    """
    Markup Exhaustion (Full Exit)
    Detects 3rd leg completion, Phase E signals, cycle aggregate breakdown.
    """
    name = "markup_exhaustion"
    required_keys = {"min_bars_since_entry", "retest_frac", "wyckoff_drop_floor",
                     "aggregate_floor"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        if trade_plan['bias'] != 'long':
            return None

        current = df.iloc[-1]

        # Check 3rd leg completion
        third_leg = bars_since_entry > self.cfg['min_bars_since_entry']

        # Check range retest
        initial_high = trade_plan.get('initial_range_high', df.tail(50)['high'].max())
        range_retest = current['close'] < (initial_high * self.cfg['retest_frac'])

        # Check Wyckoff phase exhaustion
        wyckoff_exhausted = scores.get('wyckoff', 1.0) < self.cfg['wyckoff_drop_floor']

        # Check aggregate confluence breakdown
        aggregate_score = sum(scores.values()) / len(scores) if scores else 1.0
        aggregate_breakdown = aggregate_score < self.cfg['aggregate_floor']

        if third_leg and range_retest and (wyckoff_exhausted or aggregate_breakdown):
            return ExitDecision(
                action='full',
                size_pct=1.0,
                reason=f"Markup Exhaustion: 3rd_leg:{third_leg}, retest:{range_retest}, "
                      f"wyckoff:{wyckoff_exhausted}, aggregate:{aggregate_breakdown}",
                metadata={
                    'bars_held': bars_since_entry,
                    'wyckoff_score': scores.get('wyckoff', 0),
                    'aggregate_score': aggregate_score,
                    'phase': trade_plan.get('wyckoff_phase', 'unknown')
                }
            )

        return None


# ============================================================================
# MARKDOWN SHORT EXIT RULES
# ============================================================================

class MarkdownSOSSpringFlip(BaseExitRule):
    """
    SOS/Spring Flip (Partial: 50%)
    Detects discount mitigation, volume surge, Spring patterns, potential flips.
    """
    name = "markdown_sos_spring_flip"
    required_keys = {"discount_ceiling", "vol_surge_mult", "sos_green_in_6",
                     "wyckoff_flip_floor", "partial_pct"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        if trade_plan['bias'] != 'short':
            return None

        current = df.iloc[-1]
        recent = df.tail(20)

        # Check discount position (Layer 2)
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        if range_size > 0:
            discount_position = (current['close'] - range_low) / range_size
        else:
            discount_position = 0.5

        discount_mitigated = discount_position < self.cfg['discount_ceiling']

        # Check volume surge (Layer 5)
        vol_sma = df['volume'].tail(10).mean()
        vol_surge = current['volume'] > (self.cfg['vol_surge_mult'] * vol_sma)

        # Check Spring pattern (V-reversal)
        spring = False
        if len(df) >= 3:
            equal_lows = df.tail(20)['low'].min()
            dip_below = df.iloc[-2]['low'] < equal_lows
            v_reversal = current['close'] > df.iloc[-2]['close']
            spring = dip_below and v_reversal

        # Check SOS pattern (6-candle)
        sos_pattern = False
        if len(df) >= 6:
            last_6 = df.tail(6)
            green_count = sum(last_6['close'] > last_6['open'])
            sos_pattern = green_count >= self.cfg['sos_green_in_6']

        if discount_mitigated and vol_surge and (spring or sos_pattern):
            # Check for potential flip
            flip_bias = None
            if scores.get('wyckoff', 0) > self.cfg['wyckoff_flip_floor']:
                if mtf_context and mtf_context.get('htf_bias') != 'bearish':
                    flip_bias = 'long'

            # Calculate new stop at BE + ATR
            entry = trade_plan['entry_price']
            atr = self._calculate_atr(df, self.shared['atr_period'])
            new_sl = entry - atr  # For shorts, trail up

            return ExitDecision(
                action='partial',
                size_pct=self.cfg['partial_pct'],
                new_sl=new_sl,
                reason=f"SOS/Spring: discount:{discount_mitigated}, vol_surge:{vol_surge}, "
                      f"spring:{spring}, sos:{sos_pattern}",
                metadata={
                    'discount_position': discount_position,
                    'volume_ratio': current['volume'] / vol_sma if vol_sma > 0 else 0,
                    'potential_flip': flip_bias is not None
                },
                flip_bias=flip_bias
            )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.tail(period).mean()


# ============================================================================
# UNIVERSAL EXIT RULES
# ============================================================================

class MoneytaurTrailing(BaseExitRule):
    """
    Moneytaur Trailing System
    After 1R profit, trail to max(BE + 0.5R, structure_pivot - ATR).
    """
    name = "moneytaur_trailing"
    required_keys = {"activate_after_R", "trail_rule", "update_every_bars"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        # Check if we should update (every N bars)
        if bars_since_entry % self.cfg['update_every_bars'] != 0:
            return None

        current = df.iloc[-1]
        entry = trade_plan['entry_price']
        initial_sl = trade_plan['sl']
        risk_r = abs(entry - initial_sl)

        # Calculate current R multiple
        if trade_plan['bias'] == 'long':
            current_r = (current['close'] - entry) / risk_r if risk_r > 0 else 0
        else:
            current_r = (entry - current['close']) / risk_r if risk_r > 0 else 0

        # Only activate after minimum R achieved
        if current_r < self.cfg['activate_after_R']:
            return None

        # Calculate trailing stop
        atr = self._calculate_atr(df, self.shared['atr_period'])

        if trade_plan['bias'] == 'long':
            # For longs: trail below structure
            structure_pivot = df.tail(10)['low'].min()
            be_plus = entry + (0.5 * risk_r)
            structure_minus_atr = structure_pivot - atr
            new_sl = max(be_plus, structure_minus_atr)

            # Only update if new stop is higher than current
            current_sl = trade_plan.get('current_sl', initial_sl)
            if new_sl > current_sl:
                return ExitDecision(
                    action='trail',
                    size_pct=0,
                    new_sl=new_sl,
                    reason=f"Moneytaur Trail: R={current_r:.2f}, new_sl={new_sl:.2f}",
                    metadata={'current_r': current_r, 'structure': structure_pivot}
                )
        else:
            # For shorts: trail above structure
            structure_pivot = df.tail(10)['high'].max()
            be_minus = entry - (0.5 * risk_r)
            structure_plus_atr = structure_pivot + atr
            new_sl = min(be_minus, structure_plus_atr)

            # Only update if new stop is lower than current
            current_sl = trade_plan.get('current_sl', initial_sl)
            if new_sl < current_sl:
                return ExitDecision(
                    action='trail',
                    size_pct=0,
                    new_sl=new_sl,
                    reason=f"Moneytaur Trail: R={current_r:.2f}, new_sl={new_sl:.2f}",
                    metadata={'current_r': current_r, 'structure': structure_pivot}
                )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.tail(period).mean()


class GlobalVeto(BaseExitRule):
    """
    Global Veto System
    Immediate exit on aggregate breakdown, HTF invalidation, or macro stress.
    """
    name = "global_veto"
    required_keys = {"aggregate_floor", "context_floor", "cooldown_bars"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        # Check aggregate confluence
        aggregate_score = sum(scores.values()) / len(scores) if scores else 1.0
        aggregate_veto = aggregate_score < self.cfg['aggregate_floor']

        # Check context layer (macro stress)
        context_veto = scores.get('context', 1.0) < self.cfg['context_floor']

        # Check HTF bias invalidation
        htf_invalid = False
        if mtf_context:
            position_bias = trade_plan['bias']
            htf_bias = mtf_context.get('htf_bias')
            if position_bias == 'long' and htf_bias == 'bearish':
                htf_invalid = True
            elif position_bias == 'short' and htf_bias == 'bullish':
                htf_invalid = True

        if aggregate_veto or context_veto or htf_invalid:
            return ExitDecision(
                action='full',
                size_pct=1.0,
                reason=f"Global Veto: aggregate:{aggregate_veto}({aggregate_score:.2f}), "
                      f"context:{context_veto}, htf_invalid:{htf_invalid}",
                metadata={
                    'aggregate_score': aggregate_score,
                    'context_score': scores.get('context', 0),
                    'htf_bias': mtf_context.get('htf_bias') if mtf_context else None
                },
                cooldown_bars=self.cfg['cooldown_bars']
            )

        return None


class BojanExtremeProtection(BaseExitRule):
    """
    Bojan Extreme Protection (Phase-gated for v2.x)
    75% exit on extreme wicks without follow-through.
    """
    name = "bojan_extreme_protection"
    required_keys = {"wick_atr_mult", "vol_under_sma_mult", "exit_pct",
                     "require_htf_alignment"}

    def evaluate(self, df: pd.DataFrame, trade_plan: Dict, scores: Dict,
                bars_since_entry: int, mtf_context: Dict = None) -> Optional[ExitDecision]:

        # Phase-gated: only if explicitly enabled
        if not self.cfg.get('enabled', False):
            return None

        current = df.iloc[-1]
        atr = self._calculate_atr(df, self.shared['atr_period'])
        vol_sma = df['volume'].tail(10).mean()

        if trade_plan['bias'] == 'long':
            # Check for Bojan High
            wick_up = current['high'] - max(current['open'], current['close'])
            extreme_wick = (wick_up / atr) > self.cfg['wick_atr_mult'] if atr > 0 else False
            no_follow_through = current['close'] < current['open']
            low_volume = current['volume'] < (self.cfg['vol_under_sma_mult'] * vol_sma)

            bojan_high = extreme_wick and no_follow_through and low_volume

            # Check HTF alignment if required
            if bojan_high and self.cfg['require_htf_alignment']:
                if mtf_context and mtf_context.get('htf_resistance_near'):
                    return ExitDecision(
                        action='partial',
                        size_pct=self.cfg['exit_pct'],
                        reason=f"Bojan High Protection: wick={wick_up/atr:.1f}x ATR, "
                              f"vol={current['volume']/vol_sma:.1f}x",
                        metadata={'wick_ratio': wick_up/atr, 'volume_ratio': current['volume']/vol_sma}
                    )
        else:
            # Check for Bojan Low
            wick_down = min(current['open'], current['close']) - current['low']
            extreme_wick = (wick_down / atr) > self.cfg['wick_atr_mult'] if atr > 0 else False
            no_follow_through = current['close'] > current['open']
            volume_spike = current['volume'] > (1.5 * vol_sma)

            bojan_low = extreme_wick and no_follow_through and volume_spike

            # Check HTF alignment if required
            if bojan_low and self.cfg['require_htf_alignment']:
                if mtf_context and mtf_context.get('htf_support_near'):
                    return ExitDecision(
                        action='partial',
                        size_pct=self.cfg['exit_pct'],
                        reason=f"Bojan Low Protection: wick={wick_down/atr:.1f}x ATR, "
                              f"vol={current['volume']/vol_sma:.1f}x",
                        metadata={'wick_ratio': wick_down/atr, 'volume_ratio': current['volume']/vol_sma}
                    )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.tail(period).mean()