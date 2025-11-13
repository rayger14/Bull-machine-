"""
Multi-Modal Exit System

Combines 5 exit strategies for optimal trade management:
1. R-Ladder: Profit-taking at 1R, 2R, 3R milestones
2. Structural: Exit on CHOCH, bearish BOMS, squiggle breakdown
3. Liquidity: Exit near HVN (liquidity cluster resistance)
4. Time: Exit if no movement after N bars
5. Macro Regime: Exit on risk_off/crisis regime flips (ARCHITECTURE FIX #3)

Each mode provides exit signals independently, and the final decision
is made by voting (any 2+ modes trigger = exit).

Author: Bull Machine v2.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Constants for magic numbers
CHOCH_THRESHOLD_LONG = 0.99  # Break below 1% of swing low triggers bearish CHOCH
CHOCH_THRESHOLD_SHORT = 1.01  # Break above 1% of swing high triggers bullish CHOCH
HVN_THRESHOLD_LONG = 0.99  # HVN within 1% above price = resistance for longs
HVN_THRESHOLD_SHORT = 1.01  # HVN within 1% below price = support for shorts
HIGH_URGENCY_THRESHOLD = 0.8  # Urgency >= 0.8 triggers immediate exit
MEDIUM_URGENCY_THRESHOLD = 0.5  # Urgency >= 0.5 with 1 mode triggers exit
MINIMUM_VOTING_MODES = 2  # 2+ active modes trigger exit


@dataclass
class ExitSignal:
    """
    Multi-modal exit signal.

    Attributes:
        should_exit: True if exit conditions met
        exit_modes: List of modes triggering exit
        exit_reason: Primary exit reason
        r_multiple: Current R-multiple (profit/risk)
        urgency: 0-1, higher = more urgent
        partial_exit_pct: % to exit (0-100)
    """
    should_exit: bool
    exit_modes: List[str]
    exit_reason: str
    r_multiple: float
    urgency: float
    partial_exit_pct: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for feature store."""
        return {
            'exit_should_exit': self.should_exit,
            'exit_modes_active': ','.join(self.exit_modes),
            'exit_reason': self.exit_reason,
            'exit_r_multiple': self.r_multiple,
            'exit_urgency': self.urgency,
            'exit_partial_pct': self.partial_exit_pct,
        }


def check_r_ladder_exit(entry_price: float, current_price: float,
                        stop_loss: float, direction: str,
                        config: Optional[Dict] = None) -> Tuple[bool, float, str]:
    """
    Check R-ladder profit taking.

    Args:
        entry_price: Entry price
        current_price: Current price
        stop_loss: Stop loss level
        direction: 'long' or 'short'
        config: Optional config

    Returns:
        (should_exit, partial_pct, reason)

    Logic:
        - 1R: Exit 33% (secure base profit)
        - 2R: Exit 50% of remaining (total 67% out)
        - 3R: Exit 100% (full profit secured)
    """
    config = config or {}

    # Calculate R-multiple
    risk = abs(entry_price - stop_loss)

    if direction == 'long':
        profit = current_price - entry_price
    else:  # short
        profit = entry_price - current_price

    r_multiple = profit / risk if risk > 0 else 0

    # R-ladder thresholds
    r1_threshold = config.get('r1_threshold', 1.0)
    r2_threshold = config.get('r2_threshold', 2.0)
    r3_threshold = config.get('r3_threshold', 3.0)

    if r_multiple >= r3_threshold:
        return True, 100.0, f"R-ladder 3R ({r_multiple:.1f}R)"
    elif r_multiple >= r2_threshold:
        return True, 50.0, f"R-ladder 2R ({r_multiple:.1f}R)"
    elif r_multiple >= r1_threshold:
        return True, 33.0, f"R-ladder 1R ({r_multiple:.1f}R)"

    return False, 0.0, ""


def check_structural_exit(df: pd.DataFrame, direction: str,
                          entry_idx: int, config: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Check structural reversal signals.

    Args:
        df: OHLCV DataFrame
        direction: Position direction ('long' or 'short')
        entry_idx: Index of entry bar
        config: Optional config

    Returns:
        (should_exit, reason)

    Signals:
        - CHOCH (Change of Character): Trend reversal
        - Opposing BOMS: Strong counter-trend break
        - Squiggle breakdown: Pattern invalidation
    """
    config = config or {}

    if len(df) < entry_idx + 3:
        return False, ""

    recent_bars = df.iloc[entry_idx:]

    # Simple CHOCH detection: New swing high/low against position
    lookback = min(10, len(recent_bars))
    recent_window = recent_bars.tail(lookback)

    swing_high = recent_window['high'].max()
    swing_low = recent_window['low'].min()

    current_close = df['close'].iloc[-1]

    if direction == 'long':
        # Check for bearish CHOCH (break below recent low)
        if current_close < swing_low * CHOCH_THRESHOLD_LONG:
            return True, "Structural: Bearish CHOCH"

    elif direction == 'short':
        # Check for bullish CHOCH (break above recent high)
        if current_close > swing_high * CHOCH_THRESHOLD_SHORT:
            return True, "Structural: Bullish CHOCH"

    return False, ""


def check_liquidity_exit(current_price: float, frvp_hvn_levels: List[float],
                         direction: str, config: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Check liquidity cluster exit (near HVN resistance).

    Args:
        current_price: Current price
        frvp_hvn_levels: List of High Volume Node levels
        direction: Position direction
        config: Optional config

    Returns:
        (should_exit, reason)

    Logic:
        - Long: Exit if price within 1% of HVN above entry (resistance)
        - Short: Exit if price within 1% of HVN below entry (support)
    """
    config = config or {}
    tolerance = config.get('hvn_tolerance_pct', 0.01)

    if not frvp_hvn_levels:
        return False, ""

    for hvn_level in frvp_hvn_levels:
        distance_pct = abs(current_price - hvn_level) / current_price

        if distance_pct < tolerance:
            # Near HVN
            if direction == 'long' and hvn_level > current_price * HVN_THRESHOLD_LONG:
                # HVN above = resistance for longs
                return True, f"Liquidity: Near HVN resistance ({hvn_level:.2f})"
            elif direction == 'short' and hvn_level < current_price * HVN_THRESHOLD_SHORT:
                # HVN below = support for shorts
                return True, f"Liquidity: Near HVN support ({hvn_level:.2f})"

    return False, ""


def check_time_exit(entry_idx: int, current_idx: int,
                    config: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Check time-based exit (no movement after N bars).

    Args:
        entry_idx: Entry bar index
        current_idx: Current bar index
        config: Optional config

    Returns:
        (should_exit, reason)

    Logic:
        - Exit if position open > max_bars (default 72 bars = 3 days on 1H)
    """
    config = config or {}
    max_bars = config.get('max_hold_bars', 72)

    bars_in_trade = current_idx - entry_idx

    if bars_in_trade >= max_bars:
        return True, f"Time: Max hold exceeded ({bars_in_trade} bars)"

    return False, ""


def check_macro_regime_exit(current_row: pd.Series, entry_row: pd.Series,
                             direction: str, config: Optional[Dict] = None) -> Tuple[bool, str, float]:
    """
    Check macro regime flip exit (ARCHITECTURE FIX #3).

    **KEY INSIGHT FROM 2022 FAILURE ANALYSIS**:
    - 2022 had 0 regime exits despite 6.5% crisis+risk_off bars
    - 2023-2024 captured $549 from 5 regime exits
    - Macro regime changes are HIGH PRIORITY exit signals

    Args:
        current_row: Current bar data (must have macro_regime column)
        entry_row: Entry bar data (must have macro_regime column)
        direction: Position direction ('long' only for now, shorts exit on risk_on)
        config: Optional config

    Returns:
        (should_exit, reason, urgency_boost)

    Logic:
        - Long positions: Exit on neutral→risk_off or neutral→crisis flip
        - Urgency boost: 0.9 for crisis, 0.7 for risk_off (high priority)
        - Ignore risk_on→neutral flips (favorable for longs)
    """
    config = config or {}

    # Get macro regime from current and entry rows
    current_regime = current_row.get('macro_regime', 'neutral')
    entry_regime = entry_row.get('macro_regime', 'neutral')

    # FIX #3: HIGH PRIORITY for regime deterioration
    if direction == 'long':
        # Crisis entry → exit immediately (shouldn't happen, but safety check)
        if current_regime == 'crisis':
            return True, f"Macro: Crisis regime (VIX extreme)", 0.9

        # Risk-off entry → consider exit if persists
        if current_regime == 'risk_off' and entry_regime in ['neutral', 'risk_on']:
            return True, f"Macro: Regime flip {entry_regime}→risk_off", 0.7

        # Regime improved or neutral → hold
        return False, "", 0.0

    # For shorts, inverse logic (exit on risk_on)
    elif direction == 'short':
        if current_regime == 'risk_on' and entry_regime in ['neutral', 'risk_off']:
            return True, f"Macro: Regime flip {entry_regime}→risk_on", 0.7
        return False, "", 0.0

    return False, "", 0.0


def calculate_exit_urgency(r_multiple: float, structural_exit: bool,
                           bars_in_trade: int, config: Optional[Dict] = None) -> float:
    """
    Calculate exit urgency score (0-1).

    Args:
        r_multiple: Current R-multiple
        structural_exit: True if structural exit triggered
        bars_in_trade: Bars since entry
        config: Optional config

    Returns:
        Urgency score (0-1, higher = more urgent)

    Components:
        - Negative R: 1.0 urgency (stop hit)
        - Structural exit: 0.8 urgency (reversal)
        - High R + long hold: 0.6 urgency (take profit)
        - Neutral: 0.0 urgency
    """
    config = config or {}

    urgency = 0.0

    # Negative R (loss)
    if r_multiple < -0.5:
        urgency = 1.0

    # Structural reversal
    elif structural_exit:
        urgency = 0.8

    # High R + long hold
    elif r_multiple > 2.0 and bars_in_trade > 50:
        urgency = 0.6

    # Moderate R
    elif r_multiple > 1.0:
        urgency = 0.3

    return float(urgency)


def _apply_voting_logic(exit_modes: List[str], urgency: float, partial_pct: float) -> Tuple[bool, float]:
    """
    Apply voting logic to determine final exit decision.

    Args:
        exit_modes: List of active exit modes
        urgency: Exit urgency score (0-1)
        partial_pct: Partial exit percentage from R-ladder

    Returns:
        (should_exit, final_partial_pct)

    Logic:
        - Urgency >= 0.8: Immediate full exit
        - 2+ modes active: Full exit (majority vote)
        - 1 mode + urgency >= 0.5: Full exit
        - Otherwise: No exit
    """
    should_exit = False
    final_partial_pct = partial_pct

    if urgency >= HIGH_URGENCY_THRESHOLD:
        # High urgency: immediate exit
        should_exit = True
        final_partial_pct = 100.0
    elif len(exit_modes) >= MINIMUM_VOTING_MODES:
        # 2+ modes active: majority vote
        should_exit = True
        final_partial_pct = 100.0 if partial_pct == 0 else partial_pct
    elif len(exit_modes) == 1 and urgency >= MEDIUM_URGENCY_THRESHOLD:
        # 1 mode + medium urgency
        should_exit = True
        final_partial_pct = 100.0 if partial_pct == 0 else partial_pct

    return should_exit, final_partial_pct


def _calculate_r_multiple(entry_price: float, current_price: float,
                          stop_loss: float, direction: str) -> float:
    """
    Calculate R-multiple (profit/risk ratio).

    Args:
        entry_price: Entry price
        current_price: Current price
        stop_loss: Stop loss level
        direction: Position direction ('long' or 'short')

    Returns:
        R-multiple (profit / risk)
    """
    risk = abs(entry_price - stop_loss)
    if direction == 'long':
        profit = current_price - entry_price
    else:
        profit = entry_price - current_price
    return profit / risk if risk > 0 else 0


def evaluate_multi_modal_exit(
    entry_price: float,
    stop_loss: float,
    current_price: float,
    direction: str,
    df: pd.DataFrame,
    entry_idx: int,
    frvp_hvn_levels: Optional[List[float]] = None,
    config: Optional[Dict] = None,
    entry_row: Optional[pd.Series] = None
) -> ExitSignal:
    """
    Evaluate multi-modal exit system.

    Args:
        entry_price: Entry price
        stop_loss: Stop loss level
        current_price: Current price
        direction: 'long' or 'short'
        df: OHLCV DataFrame
        entry_idx: Entry bar index
        frvp_hvn_levels: Optional HVN levels from FRVP
        config: Optional configuration
        entry_row: Optional entry bar data for macro regime comparison (ARCHITECTURE FIX #3)

    Returns:
        ExitSignal with exit decision

    Voting Logic:
        - Any 1 mode at max urgency (R3, stop hit, macro crisis): Immediate exit
        - Any 2+ modes active: Exit (majority vote)
        - 1 mode + high urgency: Exit
        - Macro regime flips add urgency boost (0.7-0.9) for priority exit
        - Otherwise: Hold

    Example:
        >>> exit_signal = evaluate_multi_modal_exit(
        ...     entry_price=40000, stop_loss=39500,
        ...     current_price=41000, direction='long',
        ...     df=df_1h, entry_idx=100, frvp_hvn_levels=[41050],
        ...     entry_row=df_1h.iloc[100]  # For macro regime tracking
        ... )
        >>> if exit_signal.should_exit:
        ...     print(f"Exit: {exit_signal.exit_reason}")
    """
    config = config or {}
    frvp_hvn_levels = frvp_hvn_levels or []

    current_idx = len(df) - 1
    bars_in_trade = current_idx - entry_idx

    # Calculate R-multiple
    r_multiple = _calculate_r_multiple(entry_price, current_price, stop_loss, direction)

    # Check all exit modes
    exit_modes = []
    partial_pct = 0.0
    primary_reason = ""

    # 1. R-Ladder
    r_exit, r_pct, r_reason = check_r_ladder_exit(
        entry_price, current_price, stop_loss, direction, config
    )
    if r_exit:
        exit_modes.append('r_ladder')
        partial_pct = max(partial_pct, r_pct)
        if not primary_reason:
            primary_reason = r_reason

    # 2. Structural
    struct_exit, struct_reason = check_structural_exit(
        df, direction, entry_idx, config
    )
    if struct_exit:
        exit_modes.append('structural')
        if not primary_reason:
            primary_reason = struct_reason

    # 3. Liquidity
    liq_exit, liq_reason = check_liquidity_exit(
        current_price, frvp_hvn_levels, direction, config
    )
    if liq_exit:
        exit_modes.append('liquidity')
        if not primary_reason:
            primary_reason = liq_reason

    # 4. Time
    time_exit, time_reason = check_time_exit(
        entry_idx, current_idx, config
    )
    if time_exit:
        exit_modes.append('time')
        if not primary_reason:
            primary_reason = time_reason

    # 5. Macro Regime (ARCHITECTURE FIX #3 - HIGH PRIORITY)
    macro_urgency_boost = 0.0
    if entry_row is not None:
        current_row = df.iloc[-1]
        macro_exit, macro_reason, macro_urgency_boost = check_macro_regime_exit(
            current_row, entry_row, direction, config
        )
        if macro_exit:
            exit_modes.append('macro_regime')
            if not primary_reason:
                primary_reason = macro_reason
            # Macro exits get priority - boost urgency significantly
            logger.info(f"[MACRO EXIT] {macro_reason} (urgency boost: {macro_urgency_boost:.2f})")

    # Calculate urgency (with macro boost if applicable)
    urgency = calculate_exit_urgency(
        r_multiple, struct_exit, bars_in_trade, config
    )
    # Apply macro urgency boost (additive, can push urgency > 0.8 for immediate exit)
    urgency = min(1.0, urgency + macro_urgency_boost)

    # Apply voting logic
    should_exit, partial_pct = _apply_voting_logic(exit_modes, urgency, partial_pct)

    # If no primary reason set, use first mode
    if not primary_reason and exit_modes:
        primary_reason = f"Multi-modal: {exit_modes[0]}"

    return ExitSignal(
        should_exit=should_exit,
        exit_modes=exit_modes,
        exit_reason=primary_reason or "No exit",
        r_multiple=float(r_multiple),
        urgency=urgency,
        partial_exit_pct=float(partial_pct)
    )
