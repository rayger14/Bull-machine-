"""
Bull Machine ATR-Based Exit Logic
Dynamic stops, profit ladders, and trailing stops based on market volatility
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from bull_machine.core.numerics import safe_atr
from bull_machine.core.telemetry import log_telemetry

def compute_exit_levels(df: pd.DataFrame, side: str, risk_cfg: Dict) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit levels based on ATR.

    Args:
        df: Price DataFrame with OHLC data
        side: "long" or "short"
        risk_cfg: Risk configuration dict

    Returns:
        Tuple of (stop_loss, take_profit) prices
    """
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
    """
    Update trailing stop loss based on ATR.

    Args:
        df: Price DataFrame with OHLC data
        side: "long" or "short"
        current_sl: Current stop loss level
        risk_cfg: Risk configuration dict

    Returns:
        Updated stop loss level
    """
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

def check_profit_ladder_exit(df: pd.DataFrame, position: Dict, config: Dict) -> Dict:
    """
    Check for profit ladder exits with scaled position reduction.

    Args:
        df: Price DataFrame
        position: Position details
        config: Configuration dict

    Returns:
        Exit signal dict with close percentages and prices
    """
    current_price = df['close'].iloc[-1]
    entry_price = position['entry_price']
    side = position['side']
    atr = float(safe_atr(df, config.get('risk', {}).get('atr_window', 14)).iloc[-1])

    # Profit ladder configuration
    profit_levels = [
        {'ratio': 1.5, 'percent': 0.25},  # 25% at 1.5R
        {'ratio': 2.5, 'percent': 0.50},  # 50% at 2.5R
        {'ratio': 4.0, 'percent': 0.25}   # 25% at 4R+
    ]

    exit_signal = {
        'close_position': False,
        'closed_percent': 0.0,
        'exit_price': current_price,
        'exit_reason': 'none',
        'ladder_exits': []
    }

    # Calculate profit targets based on ATR
    already_closed = position.get('closed_percent', 0.0)
    remaining_position = 1.0 - already_closed

    if remaining_position <= 0:
        return exit_signal

    for level in profit_levels:
        if side == "long":
            target_price = entry_price + (level['ratio'] * atr)
            profit_achieved = current_price >= target_price
        else:  # short
            target_price = entry_price - (level['ratio'] * atr)
            profit_achieved = current_price <= target_price

        # Check if this level hasn't been triggered yet
        level_key = f"ladder_{level['ratio']}"
        if profit_achieved and not position.get(level_key, False):
            close_percent = level['percent'] * remaining_position

            exit_signal['close_position'] = True
            exit_signal['closed_percent'] += close_percent
            exit_signal['exit_price'] = target_price
            exit_signal['exit_reason'] = f"profit_ladder_{level['ratio']}R"
            exit_signal['ladder_exits'].append({
                'level': level['ratio'],
                'percent': close_percent,
                'price': target_price
            })

            # Mark this level as triggered
            position[level_key] = True

    log_telemetry("layer_masks.json", {
        "profit_ladder_check": True,
        "side": side,
        "current_price": current_price,
        "entry_price": entry_price,
        "atr": atr,
        "exit_signal": exit_signal,
        "remaining_position": remaining_position
    })

    return exit_signal

def check_dynamic_trailing_stop(df: pd.DataFrame, position: Dict, config: Dict) -> bool:
    """
    Check dynamic trailing stop based on momentum loss.

    Args:
        df: Price DataFrame
        position: Position details
        config: Configuration dict

    Returns:
        True if trailing stop triggered
    """
    current_price = df['close'].iloc[-1]
    side = position['side']
    high_price = position.get('high_price', position['entry_price'])
    low_price = position.get('low_price', position['entry_price'])

    atr = float(safe_atr(df, config.get('risk', {}).get('atr_window', 14)).iloc[-1])
    trail_mult = config.get('risk', {}).get('trail_atr', 1.2)

    # Update position high/low tracking
    if side == "long":
        position['high_price'] = max(high_price, current_price)
        trail_distance = trail_mult * atr
        trailing_stop = position['high_price'] - trail_distance

        stop_triggered = current_price <= trailing_stop

    else:  # short
        position['low_price'] = min(low_price, current_price)
        trail_distance = trail_mult * atr
        trailing_stop = position['low_price'] + trail_distance

        stop_triggered = current_price >= trailing_stop

    if stop_triggered:
        log_telemetry("layer_masks.json", {
            "dynamic_trailing_stop": True,
            "side": side,
            "current_price": current_price,
            "trailing_stop": trailing_stop,
            "trail_distance": trail_distance,
            "high_price": position.get('high_price'),
            "low_price": position.get('low_price')
        })

    return stop_triggered

def enhanced_exit_check(df: pd.DataFrame, position: Dict, config: Dict) -> Dict:
    """
    Enhanced exit check with profit ladders and dynamic trailing.

    Args:
        df: Price DataFrame
        position: Position details
        config: Configuration dict

    Returns:
        Comprehensive exit signal
    """
    current_price = df['close'].iloc[-1]
    side = position['side']
    stop_loss = position.get('stop_loss')

    # Initialize exit signal
    exit_signal = {
        'close_position': False,
        'closed_percent': 0.0,
        'exit_price': current_price,
        'exit_reason': 'none'
    }

    # 1. Check hard stop loss
    if stop_loss:
        if side == "long" and current_price <= stop_loss:
            exit_signal.update({
                'close_position': True,
                'closed_percent': 1.0,
                'exit_price': stop_loss,
                'exit_reason': 'stop_loss'
            })
            return exit_signal
        elif side == "short" and current_price >= stop_loss:
            exit_signal.update({
                'close_position': True,
                'closed_percent': 1.0,
                'exit_price': stop_loss,
                'exit_reason': 'stop_loss'
            })
            return exit_signal

    # 2. Check profit ladder exits
    ladder_signal = check_profit_ladder_exit(df, position, config)
    if ladder_signal['close_position']:
        return ladder_signal

    # 3. Check dynamic trailing stop
    if check_dynamic_trailing_stop(df, position, config):
        exit_signal.update({
            'close_position': True,
            'closed_percent': 1.0,
            'exit_price': current_price,
            'exit_reason': 'trailing_stop'
        })
        return exit_signal

    return exit_signal