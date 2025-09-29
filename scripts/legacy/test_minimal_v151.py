#!/usr/bin/env python3
"""
Minimal v1.5.1 test with working signal generation
Focus on core functionality: entry signals + profit ladder exits
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from bull_machine.strategy.atr_exits import enhanced_exit_check
from bull_machine.core.telemetry import log_telemetry

def load_eth_data(timeframe: str) -> pd.DataFrame:
    """Load ETH data"""
    data_path = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
    })

    if 'BUY+SELL V' in df.columns:
        df['volume'] = df['BUY+SELL V']
    else:
        df['volume'] = (df['high'] - df['low']) * df['close'] * 1000

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Resample to 1D if needed
    if timeframe == '1D':
        df.set_index('timestamp', inplace=True)
        df_1d = df.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_1d = df_1d.reset_index()
        df = df_1d

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def simple_momentum_signal(df: pd.DataFrame, lookback: int = 10) -> float:
    """Simple momentum-based signal"""
    if len(df) < lookback + 5:
        return 0.0

    # RSI-like momentum
    closes = df['close'].iloc[-lookback:].values
    gains = np.maximum(np.diff(closes), 0)
    losses = np.maximum(-np.diff(closes), 0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0.01
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.01

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Convert to 0-1 signal strength
    if rsi > 70:
        return 0.8  # Strong bullish
    elif rsi > 60:
        return 0.6  # Moderate bullish
    elif rsi < 30:
        return 0.8  # Strong bearish (for short)
    elif rsi < 40:
        return 0.6  # Moderate bearish
    else:
        return 0.3  # Neutral

def simple_entry_check(df: pd.DataFrame, config: dict, equity: float) -> dict:
    """Simplified entry logic that actually generates signals"""
    if len(df) < 50:
        return None

    # Calculate momentum signal
    momentum = simple_momentum_signal(df)
    entry_threshold = config.get('entry_threshold', 0.35)

    if momentum < entry_threshold:
        return None

    # Determine side (simplified)
    rsi_val = simple_momentum_signal(df, 14)
    side = "long" if rsi_val > 0.5 else "short"

    # Position sizing (simplified ATR-based)
    atr_window = config.get('risk', {}).get('atr_window', 14)
    if len(df) >= atr_window:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(atr_window).mean().iloc[-1]

        risk_pct = config.get('risk', {}).get('risk_pct', 0.01)
        sl_atr = config.get('risk', {}).get('sl_atr', 2.0)

        risk_amount = equity * risk_pct
        stop_distance = atr * sl_atr
        quantity = risk_amount / stop_distance if stop_distance > 0 else equity * 0.02
    else:
        quantity = equity * 0.02

    current_price = df['close'].iloc[-1]

    # Set stops
    if side == "long":
        stop_loss = current_price - (atr * config.get('risk', {}).get('sl_atr', 2.0)) if 'atr' in locals() else current_price * 0.95
        take_profit = current_price + (atr * config.get('risk', {}).get('tp_atr', 4.0)) if 'atr' in locals() else current_price * 1.08
    else:
        stop_loss = current_price + (atr * config.get('risk', {}).get('sl_atr', 2.0)) if 'atr' in locals() else current_price * 1.05
        take_profit = current_price - (atr * config.get('risk', {}).get('tp_atr', 4.0)) if 'atr' in locals() else current_price * 0.92

    return {
        'side': side,
        'quantity': quantity,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'entry_price': current_price,
        'entry_bar': len(df) - 1,
        'weighted_score': momentum,
        'layer_scores': {'momentum': momentum}
    }

def backtest_minimal(profile_name: str) -> dict:
    """Minimal backtest with working signals"""
    print(f"\n=== Testing {profile_name} (Minimal) ===")

    # Load config and data
    config_path = Path(f"configs/v150/assets/{profile_name}.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    timeframe = config['timeframe']
    df = load_eth_data(timeframe)

    print(f"Loaded {len(df)} {timeframe} candles")

    # Simulation
    initial_equity = 10000
    current_equity = initial_equity
    trades = []
    current_position = None
    last_trade_bar = -999

    wins = 0
    losses = 0
    total_pnl = 0.0
    max_dd = 0.0
    equity_high = initial_equity

    # Main loop
    for i in range(50, len(df)):
        bar_df = df.iloc[:i+1].copy()
        current_bar = i
        current_price = df.iloc[i]['close']

        # Position management
        if current_position:
            # Create position dict for enhanced exit check
            position = {
                'side': current_position['side'],
                'entry_price': current_position['entry_price'],
                'stop_loss': current_position.get('stop_loss'),
                'take_profit': current_position.get('take_profit'),
                'entry_bar': current_position.get('entry_bar', i-1),
                'closed_percent': current_position.get('closed_percent', 0.0),
                'high_price': current_position.get('high_price', current_position['entry_price']),
                'low_price': current_position.get('low_price', current_position['entry_price'])
            }

            # Copy ladder flags
            for key in current_position:
                if key.startswith('ladder_'):
                    position[key] = current_position[key]

            # Check exit with enhanced logic (profit ladders + dynamic trailing)
            exit_signal = enhanced_exit_check(bar_df, position, config)

            # Update position tracking
            for key in ['closed_percent', 'high_price', 'low_price']:
                if key in position:
                    current_position[key] = position[key]

            # Copy ladder flags back
            for key in position:
                if key.startswith('ladder_'):
                    current_position[key] = position[key]

            if exit_signal['close_position']:
                # Calculate PnL for closed portion
                side = current_position['side']
                entry_price = current_position['entry_price']
                quantity = current_position['quantity']
                exit_price = exit_signal.get('exit_price', current_price)
                closed_percent = exit_signal.get('closed_percent', 1.0)

                if side == "long":
                    pnl = (exit_price - entry_price) * quantity * closed_percent
                else:
                    pnl = (entry_price - exit_price) * quantity * closed_percent

                total_pnl += pnl
                current_equity += pnl

                # Track win/loss
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                # Track drawdown
                equity_high = max(equity_high, current_equity)
                drawdown = (equity_high - current_equity) / equity_high
                max_dd = max(max_dd, drawdown)

                # Record trade
                trade_record = {
                    'entry_bar': current_position['entry_bar'],
                    'exit_bar': current_bar,
                    'entry_date': df.iloc[current_position['entry_bar']]['timestamp'],
                    'exit_date': df.iloc[current_bar]['timestamp'],
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'closed_percent': closed_percent,
                    'pnl': pnl,
                    'exit_reason': exit_signal['exit_reason'],
                    'bars_held': current_bar - current_position['entry_bar']
                }
                trades.append(trade_record)

                exit_type = "partial" if closed_percent < 1.0 else "full"
                print(f"Exit #{len(trades)} ({exit_type}): {side} {exit_signal['exit_reason']} | "
                      f"PnL: ${pnl:.2f} | Equity: ${current_equity:.2f}")

                # Handle partial vs full exits
                if closed_percent < 1.0:
                    # Reduce position size for partial exit
                    remaining_percent = 1.0 - current_position.get('closed_percent', 0.0) - closed_percent
                    if remaining_percent > 0.01:  # Keep position if >1% remains
                        current_position['closed_percent'] = current_position.get('closed_percent', 0.0) + closed_percent
                        current_position['quantity'] *= (1.0 - closed_percent)
                    else:
                        current_position = None
                        last_trade_bar = current_bar
                else:
                    current_position = None
                    last_trade_bar = current_bar

        # Entry check (if no position)
        if not current_position:
            # Cooldown check
            cooldown_bars = config.get('cooldown_bars', 0)
            if current_bar - last_trade_bar >= cooldown_bars:
                trade_plan = simple_entry_check(bar_df, config, current_equity)
                if trade_plan:
                    current_position = trade_plan.copy()
                    print(f"Entry #{len(trades)+1}: {trade_plan['side']} @ ${current_price:.2f} | "
                          f"Quantity: ${trade_plan['quantity']:.2f}")

    # Calculate results
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in trades) else float('inf')
    total_return = (current_equity - initial_equity) / initial_equity

    # Monthly frequency
    if trades:
        first_date = pd.to_datetime(trades[0]['entry_date'])
        last_date = pd.to_datetime(trades[-1]['entry_date'])
        months = (last_date - first_date).days / 30.44
        monthly_frequency = total_trades / max(months, 1)
    else:
        monthly_frequency = 0

    results = {
        'profile': profile_name,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'monthly_frequency': monthly_frequency,
        'final_equity': current_equity,
        'trades': trades
    }

    print(f"\n=== {profile_name} Results ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total Return: {total_return:.1%}")
    print(f"Max Drawdown: {max_dd:.1%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Monthly Frequency: {monthly_frequency:.1f}")
    print(f"Final Equity: ${current_equity:.2f}")

    return results

def main():
    """Test minimal working version with profit ladders"""
    print("Bull Machine v1.5.1 Minimal Working Test")
    print("=" * 50)

    profiles = ['ETH', 'ETH_4H']
    all_results = {}

    for profile in profiles:
        all_results[profile] = backtest_minimal(profile)

    # Overall assessment
    print(f"\n{'='*60}")
    print("MINIMAL TEST ASSESSMENT")
    print(f"{'='*60}")

    for profile, results in all_results.items():
        trades = results['total_trades']
        win_rate = results['win_rate']
        returns = results['total_return']
        monthly_freq = results['monthly_frequency']

        status = "✓ WORKING" if trades > 0 else "✗ NO TRADES"
        print(f"{profile}: {status} | {trades} trades | {win_rate:.1%} WR | {returns:.1%} returns | {monthly_freq:.1f}/mo")

    # Save results
    log_telemetry("v151_minimal_test.json", {
        "minimal_test_complete": True,
        "results": all_results
    })

if __name__ == "__main__":
    main()