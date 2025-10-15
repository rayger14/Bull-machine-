#!/usr/bin/env python3
"""
Smart Exits Simulator - Production-Faithful

Simulates exits EXACTLY like SmartExitPortfolio in production:
- TP1 partial exit at 1.0R with 50% scale-out
- Move SL to breakeven on TP1
- ATR trailing after TP1
- Regime-adaptive stops (ADX-based)
- Macro exit triggers (force close on crisis)
- Max bars in trade timeout
- Fees + slippage + leverage

This ensures optimizer results match live trading behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """Trade result"""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_r: float  # R-multiple
    exit_reason: str
    bars_held: int
    tp1_hit: bool


def dynamic_position_size(
    entry_idx: int,
    features: Dict[str, np.ndarray],
    fusion_score: float,
    config: Dict
) -> float:
    """
    Calculate position size using ADX and fusion score adaptation

    Matches production sizing logic:
    - Base risk: 0.75%
    - ADX boost: 0-50% additional
    - Fusion boost: 0-50% additional
    - Capped between 0.5% and 2.0%
    """
    base_risk_pct = config.get('base_risk_pct', 0.0075)  # 0.75%

    # ADX adaptation
    adx = features['adx_14'][entry_idx]
    adx_boost = np.clip((adx - 20) / 20, 0.0, 0.5)  # 0 to 0.5

    # Fusion adaptation
    threshold = config['fusion_threshold']
    fusion_boost = np.clip((fusion_score - threshold) / 0.2, 0.0, 0.5)  # 0 to 0.5

    # Combined risk
    risk_pct = base_risk_pct * (1.0 + 0.5 * adx_boost + 0.5 * fusion_boost)
    risk_pct = np.clip(risk_pct, 0.005, 0.02)  # 0.5% to 2.0%

    return risk_pct


def simulate_trade(
    entry_idx: int,
    features: Dict[str, np.ndarray],
    side: int,  # +1 for long, -1 for short
    fusion_score: float,
    config: Dict
) -> Optional[Trade]:
    """
    Simulate single trade with production-faithful Smart Exits logic

    Args:
        entry_idx: Entry bar index
        features: Dict of feature arrays (close, high, low, atr, adx, macro_exit_flag, etc.)
        side: +1 for long, -1 for short
        fusion_score: Fusion score at entry (for position sizing)
        config: Exit configuration dict

    Returns:
        Trade object or None if simulation fails
    """
    close = features['close']
    high = features['high']
    low = features['low']
    atr = features['atr_20']
    adx = features['adx_14']
    macro_exit = features['macro_exit_flag']

    max_bars = config.get('max_bars_in_trade', 96)
    entry_price = close[entry_idx]

    # Calculate position size
    risk_pct = dynamic_position_size(entry_idx, features, fusion_score, config)

    # Initial stop distance (regime-adaptive)
    base_stop_dist = config.get('stop_atr', 1.0) * atr[entry_idx]

    # Regime adaptation using ADX
    adx_trend_hi = config.get('adx_trend_hi', 25.0)
    adx_range_lo = config.get('adx_range_lo', 20.0)

    if adx[entry_idx] >= adx_trend_hi:
        # Strong trend - wider stop
        stop_dist = base_stop_dist * config.get('trend_stop_factor', 1.25)
    elif adx[entry_idx] <= adx_range_lo:
        # Range - tighter stop
        stop_dist = base_stop_dist * config.get('range_stop_factor', 0.75)
    else:
        stop_dist = base_stop_dist

    # Initial stop price
    if side == 1:  # Long
        stop_price = entry_price - stop_dist
    else:  # Short
        stop_price = entry_price + stop_dist

    # Target for TP1
    tp1_r = config.get('tp1_r', 1.0)
    r_distance = abs(entry_price - stop_price)
    tp1_price = entry_price + side * tp1_r * r_distance

    # Track state
    tp1_hit = False
    remaining_size = 1.0  # 100% of position
    realized_pnl_r = 0.0

    # Exit simulation loop
    max_idx = min(entry_idx + max_bars, len(close))

    for i in range(entry_idx + 1, max_idx):
        # Priority 1: Macro exit (overrides everything)
        if macro_exit[i]:
            exit_price = close[i]
            # Close remaining position
            remaining_pnl_r = side * (exit_price - entry_price) / r_distance * remaining_size
            total_pnl_r = realized_pnl_r + remaining_pnl_r

            # Apply fees/slippage
            fees_pct = apply_fees_slippage(entry_price, exit_price, config)
            leverage = config.get('leverage', 5.0)
            pnl_pct = (total_pnl_r * risk_pct * leverage) - fees_pct

            return Trade(
                entry_idx=entry_idx,
                exit_idx=i,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                pnl_r=total_pnl_r,
                exit_reason='macro',
                bars_held=i - entry_idx,
                tp1_hit=tp1_hit
            )

        # Priority 2: Check TP1 (partial exit)
        if not tp1_hit:
            tp1_triggered = (side == 1 and high[i] >= tp1_price) or \
                           (side == -1 and low[i] <= tp1_price)

            if tp1_triggered:
                # Partial exit
                scale_out_pct = config.get('scale_out_pct', 0.5)
                realized_pnl_r += tp1_r * scale_out_pct  # Lock in 1.0R on 50%
                remaining_size -= scale_out_pct
                tp1_hit = True

                # Move SL to breakeven
                if config.get('move_sl_to_be_on_tp1', True):
                    stop_price = entry_price

        # Priority 3: Update trailing stop (after TP1)
        if tp1_hit and config.get('trail_after_tp1', True):
            trail_mult = config.get('trail_atr_mult', 1.0)
            current_atr = atr[i]

            if side == 1:  # Long
                trail_candidate = close[i] - trail_mult * current_atr
                stop_price = max(stop_price, trail_candidate)  # Ratchet up
            else:  # Short
                trail_candidate = close[i] + trail_mult * current_atr
                stop_price = min(stop_price, trail_candidate)  # Ratchet down

        # Priority 4: Check stop
        stopped = (side == 1 and low[i] <= stop_price) or \
                 (side == -1 and high[i] >= stop_price)

        if stopped:
            exit_price = stop_price

            # Calculate remaining position PnL
            remaining_pnl_r = side * (exit_price - entry_price) / r_distance * remaining_size
            total_pnl_r = realized_pnl_r + remaining_pnl_r

            # Apply fees/slippage
            fees_pct = apply_fees_slippage(entry_price, exit_price, config)
            leverage = config.get('leverage', 5.0)
            pnl_pct = (total_pnl_r * risk_pct * leverage) - fees_pct

            exit_reason = 'stop_be' if tp1_hit and exit_price == entry_price else 'stop'

            return Trade(
                entry_idx=entry_idx,
                exit_idx=i,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                pnl_r=total_pnl_r,
                exit_reason=exit_reason,
                bars_held=i - entry_idx,
                tp1_hit=tp1_hit
            )

    # Timeout exit (max bars reached)
    exit_idx = max_idx - 1
    exit_price = close[exit_idx]

    remaining_pnl_r = side * (exit_price - entry_price) / r_distance * remaining_size
    total_pnl_r = realized_pnl_r + remaining_pnl_r

    fees_pct = apply_fees_slippage(entry_price, exit_price, config)
    leverage = config.get('leverage', 5.0)
    pnl_pct = (total_pnl_r * risk_pct * leverage) - fees_pct

    return Trade(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        pnl_r=total_pnl_r,
        exit_reason='timeout',
        bars_held=exit_idx - entry_idx,
        tp1_hit=tp1_hit
    )


def apply_fees_slippage(entry_price: float, exit_price: float, config: Dict) -> float:
    """Calculate fees + slippage as percentage"""
    fees_bps = config.get('fees_bps', 10.0)  # 10 bps = 0.1%
    slippage_bps = config.get('slippage_bps', 5.0)  # 5 bps = 0.05%

    total_bps = fees_bps + slippage_bps
    # Fees apply on both entry and exit
    return (total_bps / 10000) * 2


def simulate_trades_batch(
    entry_indices: np.ndarray,
    features: Dict[str, np.ndarray],
    fusion_scores: np.ndarray,
    side: int,
    config: Dict
) -> List[Trade]:
    """
    Simulate batch of trades

    Args:
        entry_indices: Array of entry bar indices
        features: Feature dict
        fusion_scores: Fusion score at each entry
        side: +1 for long, -1 for short
        config: Exit config

    Returns:
        List of Trade objects
    """
    trades = []

    for i, entry_idx in enumerate(entry_indices):
        trade = simulate_trade(
            entry_idx,
            features,
            side,
            fusion_scores[i],
            config
        )

        if trade:
            trades.append(trade)

    return trades


def calculate_metrics(trades: List[Trade]) -> Dict:
    """Calculate performance metrics from trade list"""
    if len(trades) == 0:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 1.0,
            'max_drawdown': 0.0,
            'avg_r': 0.0,
            'winners_avg': 0.0,
            'losers_avg': 0.0,
            'avg_bars_held': 0.0,
            'tp1_hit_rate': 0.0
    }

    pnls = np.array([t.pnl_pct for t in trades])
    r_multiples = np.array([t.pnl_r for t in trades])

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]

    # Calculate equity curve
    equity_curve = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Sharpe (annualized for 1H bars)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(8760 / len(trades))
    else:
        sharpe = 0.0

    # Profit factor
    total_wins = np.sum(winners) if len(winners) > 0 else 0
    total_losses = abs(np.sum(losers)) if len(losers) > 0 else 1
    pf = total_wins / total_losses if total_losses > 0 else 1.0

    # TP1 hit rate
    tp1_hits = sum(1 for t in trades if t.tp1_hit)

    return {
        'trades': len(trades),
        'win_rate': len(winners) / len(trades) * 100,
        'total_return': np.sum(pnls),
        'sharpe_ratio': sharpe,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_r': np.mean(r_multiples),
        'winners_avg': np.mean(winners) if len(winners) > 0 else 0.0,
        'losers_avg': np.mean(losers) if len(losers) > 0 else 0.0,
        'avg_bars_held': np.mean([t.bars_held for t in trades]),
        'tp1_hit_rate': tp1_hits / len(trades) * 100
    }
