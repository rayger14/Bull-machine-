"""
Bull Machine ATR-Based Exit Logic - TRUE R-BASED IMPLEMENTATION
True R-based profit ladders with dynamic trailing stops
Fixed to use proper R = |entry - stop| calculations
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from bull_machine.core.numerics import safe_atr
from bull_machine.core.telemetry import log_telemetry

def _r_distance(position):
    """Calculate R distance from entry to stop in price units."""
    if position['side'] == 'LONG':
        return max(1e-9, position['entry_price'] - position['stop_loss'])
    else:
        return max(1e-9, position['stop_loss'] - position['entry_price'])

def _unrealized_r(price, position):
    """Calculate unrealized R multiple."""
    R = _r_distance(position)
    if position['side'] == 'LONG':
        return (price - position['entry_price']) / R
    else:
        return (position['entry_price'] - price) / R

def wick_magnet_distance(df):
    """Placeholder wick magnet logic - implement as needed."""
    return False  # Disabled for now

def check_exit(df, position, tf: str, config: dict) -> dict:
    """
    Enhanced exit check with TRUE R-BASED config-driven profit ladders and conditional trailing.

    Args:
        df: Price DataFrame
        position: Position details with side, entry_price, stop_loss, size
        tf: Timeframe string ('1H', '4H', '1D')
        config: Configuration dict with risk settings

    Returns:
        Exit signal dict with close_position, closed_percent, exit_price, reason
    """
    price = float(df['close'].iloc[-1])
    atr14 = float(safe_atr(df, 14).iloc[-1])
    side = position['side']  # 'LONG' or 'SHORT'

    # --- CONFIG-DRIVEN PROFIT LADDER ---
    ladder = config.get('risk', {}).get('profit_ladders', [
        {"ratio": 1.5, "percent": 0.25},
        {"ratio": 2.5, "percent": 0.50},
        {"ratio": 4.0, "percent": 0.25},
    ])
    trail_mult = config['risk'].get('trail_atr', 1.2)
    if tf in ('4H', '240'):
        trail_mult *= 0.8  # tighter on 4H

    # --- HARD STOP & WICK MAGNET ---
    if (side == 'LONG' and price <= position['stop_loss']) or (side == 'SHORT' and price >= position['stop_loss']):
        return {'close_position': True, 'closed_percent': 1.0, 'exit_price': position['stop_loss'], 'reason': 'stop_loss'}

    unreal_r = _unrealized_r(price, position)
    if config.get('features', {}).get('wick_magnet', False) and unreal_r >= 0.5:
        if wick_magnet_distance(df):
            return {'close_position': True, 'closed_percent': 1.0, 'exit_price': price, 'reason': 'wick_magnet'}

    # --- PROFIT LADDER (true R targets) ---
    closed = 0.0
    exit_price = price
    R = _r_distance(position)
    # track which tiers are already taken
    taken = position.setdefault('ladder_taken', [False] * len(ladder))

    for i, lvl in enumerate(ladder):
        if taken[i]:
            continue
        if side == 'LONG':
            target = position['entry_price'] + lvl['ratio'] * R
            hit = price >= target
        else:
            target = position['entry_price'] - lvl['ratio'] * R
            hit = price <= target
        if hit and position.get('size', 1.0) > 0:
            closed += float(lvl['percent'])
            exit_price = float(target)
            taken[i] = True

    if closed > 0.0:
        # caller should reduce position['size'] *= (1 - closed) after this returns
        log_telemetry('layer_masks.json', {
            'exit_signal': {
                'close_position': True,
                'closed_percent': closed,
                'exit_price': exit_price,
                'reason': 'profit_ladder'
            },
            'tf': tf,
            'R_distance': R,
            'unrealized_R': unreal_r,
            'ladder_tier': [i for i, t in enumerate(taken) if t]
        })
        return {'close_position': True, 'closed_percent': closed, 'exit_price': exit_price, 'reason': 'profit_ladder'}

    # --- DYNAMIC TRAILING (after ladder arm 1 fires or ≥1.0R) ---
    ladder_armed = any(taken) or (unreal_r >= 1.0)
    if ladder_armed:
        # update run-up for trailing logic
        if side == 'LONG':
            position['high_price'] = max(position.get('high_price', price), price)
            trail_trigger = position['high_price'] - trail_mult * atr14
            if price <= trail_trigger:
                return {'close_position': True, 'closed_percent': 1.0, 'exit_price': price, 'reason': 'trailing_stop'}
        else:
            position['low_price'] = min(position.get('low_price', price), price)
            trail_trigger = position['low_price'] + trail_mult * atr14
            if price >= trail_trigger:
                return {'close_position': True, 'closed_percent': 1.0, 'exit_price': price, 'reason': 'trailing_stop'}

    # no exit this bar
    log_telemetry('layer_masks.json', {
        'exit_signal': {
            'close_position': False,
            'closed_percent': 0.0,
            'exit_price': price,
            'reason': 'hold'
        },
        'tf': tf,
        'unrealized_R': unreal_r,
        'ladder_armed': ladder_armed
    })
    return {'close_position': False, 'closed_percent': 0.0, 'exit_price': price, 'reason': 'hold'}

# Legacy function compatibility
def enhanced_exit_check(df: pd.DataFrame, position: Dict, config: Dict, tf: str = None) -> Dict:
    """Legacy wrapper for compatibility with existing code."""
    return check_exit(df, position, tf or 'ensemble', config)

# Keep other legacy functions for compatibility
def compute_exit_levels(df: pd.DataFrame, side: str, risk_cfg: Dict) -> Tuple[float, float]:
    """Calculate stop loss and take profit levels based on ATR."""
    atr_window = risk_cfg.get("atr_window", 14)
    atr = float(safe_atr(df, atr_window).iloc[-1])

    sl_multiple = float(risk_cfg.get("sl_atr", 2.0))
    tp_multiple = float(risk_cfg.get("tp_atr", 3.0))
    current_price = float(df['close'].iloc[-1])

    if side == "long":
        stop_loss = current_price - sl_multiple * atr
        take_profit = current_price + tp_multiple * atr
    else:  # short
        stop_loss = current_price + sl_multiple * atr
        take_profit = current_price - tp_multiple * atr

    return stop_loss, take_profit

def maybe_trail_sl(df: pd.DataFrame, side: str, current_sl: float, risk_cfg: Dict) -> float:
    """Update trailing stop loss based on ATR."""
    trail_multiple = risk_cfg.get("trail_atr", 0.0)
    if not trail_multiple:
        return current_sl

    atr_window = risk_cfg.get("atr_window", 14)
    atr = float(safe_atr(df, atr_window).iloc[-1])
    current_price = float(df['close'].iloc[-1])
    trail_distance = trail_multiple * atr

    if side == "long":
        new_sl = current_price - trail_distance
        return max(current_sl, new_sl)  # Only move stop up
    else:  # short
        new_sl = current_price + trail_distance
        return min(current_sl, new_sl)  # Only move stop down