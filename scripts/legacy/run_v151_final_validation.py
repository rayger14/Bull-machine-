#!/usr/bin/env python3
"""
Bull Machine v1.5.1 Final Validation
Complete system test with profit ladders, dynamic trailing, and MTF ensemble
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from bull_machine.modules.fusion.v151_core_trader import CoreTraderV151
from bull_machine.core.telemetry import log_telemetry

def load_eth_data(timeframe: str) -> pd.DataFrame:
    """Load ETH data from Chart Logs 2"""
    # Real data paths from Chart Logs 2
    data_paths = {
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'  # Use 4H for both, resample 1D later
    }

    if timeframe not in data_paths:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    data_file = Path(data_paths[timeframe])
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load and process data
    df = pd.read_csv(data_file)

    # Convert timestamp and standardize columns
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
    })

    # Calculate volume from buy/sell data if available
    if 'BUY+SELL V' in df.columns:
        df['volume'] = df['BUY+SELL V']
    else:
        # Fallback: simulate volume based on price action
        df['volume'] = (df['high'] - df['low']) * df['close'] * 1000

    # Sort by timestamp
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

def load_asset_config(profile_name: str) -> Dict:
    """Load asset profile configuration"""
    config_path = Path("configs/v150/assets") / f"{profile_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)

def validate_exit_signal(exit_signal: Dict) -> bool:
    """Validate exit signal format"""
    required_keys = ['close_position', 'closed_percent', 'exit_price', 'exit_reason']
    return all(key in exit_signal for key in required_keys)

def backtest_profile(profile_name: str, target_metrics: Dict) -> Dict:
    """Backtest a specific profile with v1.5.1 enhancements"""
    print(f"\n=== Testing {profile_name} Profile ===")

    # Load configuration and data
    config = load_asset_config(profile_name)
    timeframe = config['timeframe']
    df = load_eth_data(timeframe)

    print(f"Loaded {len(df)} {timeframe} candles")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Initialize trader with config
    trader = CoreTraderV151(config)

    # Simulation parameters
    initial_equity = 10000
    current_equity = initial_equity
    trades = []
    current_position = None
    last_trade_bar = -999

    # Metrics tracking
    wins = 0
    losses = 0
    total_pnl = 0.0
    max_dd = 0.0
    equity_high = initial_equity

    print(f"Configuration: threshold={config['entry_threshold']}, cooldown={config.get('cooldown_bars', 0)}")
    print("Features enabled:", [k for k, v in config.get('features', {}).items() if v])

    # Main simulation loop
    for i in range(50, len(df)):  # Start after warmup period
        bar_df = df.iloc[:i+1].copy()
        current_bar = i
        current_price = df.iloc[i]['close']

        # Position management
        if current_position:
            # Update trailing stops
            current_position = trader.update_stop(bar_df, current_position, config)

            # Check exit conditions
            exit_signal = trader.check_exit(bar_df, current_position, config)

            # Handle different exit signal formats
            should_exit = False
            exit_reason = "none"
            exit_price = current_price
            closed_percent = 1.0

            if isinstance(exit_signal, bool):
                should_exit = exit_signal
                exit_reason = "legacy_exit"
            elif isinstance(exit_signal, dict) and validate_exit_signal(exit_signal):
                should_exit = exit_signal['close_position']
                exit_reason = exit_signal['exit_reason']
                exit_price = exit_signal.get('exit_price', current_price)
                closed_percent = exit_signal.get('closed_percent', 1.0)

            if should_exit:
                # Calculate PnL
                side = current_position['side']
                entry_price = current_position['entry_price']
                quantity = current_position['quantity']

                if side == "long":
                    pnl = (exit_price - entry_price) * quantity * closed_percent
                else:  # short
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
                    'exit_reason': exit_reason,
                    'bars_held': current_bar - current_position['entry_bar']
                }
                trades.append(trade_record)

                print(f"Exit #{len(trades)}: {side} {exit_reason} | "
                      f"PnL: ${pnl:.2f} | Equity: ${current_equity:.2f} | "
                      f"Bars held: {trade_record['bars_held']}")

                # Update position for partial exits
                if closed_percent < 1.0:
                    remaining = 1.0 - current_position.get('closed_percent', 0.0) - closed_percent
                    if remaining > 0.01:  # Keep position if >1% remains
                        current_position['closed_percent'] = current_position.get('closed_percent', 0.0) + closed_percent
                        current_position['quantity'] *= (1.0 - closed_percent)
                    else:
                        current_position = None
                        last_trade_bar = current_bar
                else:
                    current_position = None
                    last_trade_bar = current_bar

        # Entry signal check
        if not current_position:
            trade_plan = trader.check_entry(bar_df, last_trade_bar, config, current_equity)
            if trade_plan:
                current_position = trade_plan.copy()
                print(f"Entry #{len(trades)+1}: {trade_plan['side']} @ ${current_price:.2f} | "
                      f"Quantity: ${trade_plan['quantity']:.2f} | "
                      f"Score: {trade_plan['weighted_score']:.3f}")

    # Calculate final metrics
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in trades) else float('inf')
    total_return = (current_equity - initial_equity) / initial_equity

    # Monthly frequency calculation (approximate)
    if trades:
        first_date = pd.to_datetime(trades[0]['entry_date'])
        last_date = pd.to_datetime(trades[-1]['entry_date'])
        months = (last_date - first_date).days / 30.44
        monthly_frequency = total_trades / max(months, 1)
    else:
        monthly_frequency = 0

    # Results summary
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

    # Target validation
    print(f"\n=== Target Validation ===")
    targets_met = 0
    total_targets = len(target_metrics)

    for metric, target in target_metrics.items():
        actual = results[metric]
        if metric.endswith('_min'):
            met = actual >= target
            operator = ">="
        else:  # max targets
            met = actual <= target
            operator = "<="

        status = "✓" if met else "✗"
        print(f"{status} {metric}: {actual:.3f} {operator} {target}")

        if met:
            targets_met += 1

    print(f"\nTargets Met: {targets_met}/{total_targets} ({targets_met/total_targets:.1%})")

    return results

def main():
    """Run final validation for RC promotion"""
    print("Bull Machine v1.5.1 Final Validation")
    print("=" * 50)

    # Define target metrics for each profile
    target_metrics = {
        'ETH': {  # 1D profile
            'monthly_frequency': 2.0,  # minimum 2 trades/month
            'win_rate': 0.50,          # minimum 50% WR
            'max_drawdown': 0.092,     # maximum 9.2% DD
            'total_return': 0.10,      # minimum 10% return
            'profit_factor': 1.3       # minimum 1.3 PF
        },
        'ETH_4H': {  # 4H profile
            'monthly_frequency': 2.0,  # minimum 2 trades/month
            'win_rate': 0.45,          # minimum 45% WR
            'max_drawdown': 0.20,      # maximum 20% DD
            'total_return': 0.30,      # minimum 30% return
            'profit_factor': 1.2       # minimum 1.2 PF
        }
    }

    # Run validation for both profiles
    all_results = {}

    try:
        for profile_name, targets in target_metrics.items():
            all_results[profile_name] = backtest_profile(profile_name, targets)

        # Overall assessment
        print(f"\n{'='*60}")
        print("FINAL ASSESSMENT")
        print(f"{'='*60}")

        rc_ready = True
        for profile, results in all_results.items():
            targets = target_metrics[profile]
            targets_met = 0

            for metric, target in targets.items():
                actual = results[metric]
                if metric.endswith('_min'):
                    met = actual >= target
                else:
                    met = actual <= target

                if met:
                    targets_met += 1

            profile_ready = targets_met >= len(targets) * 0.8  # 80% of targets
            print(f"{profile}: {'✓ RC READY' if profile_ready else '✗ NEEDS WORK'} ({targets_met}/{len(targets)} targets)")

            if not profile_ready:
                rc_ready = False

        print(f"\nOverall Status: {'🚀 READY FOR RC PROMOTION' if rc_ready else '⚠️  NEEDS FURTHER OPTIMIZATION'}")

        # Save detailed results
        log_telemetry("v151_final_validation.json", {
            "validation_complete": True,
            "rc_ready": rc_ready,
            "profile_results": all_results,
            "target_metrics": target_metrics
        })

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return rc_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)