#!/usr/bin/env python3
"""
Bull Machine v1.4.2 - ETH Comprehensive Backtest
Using production-ready v1.3 engine with v1.4.2 enhancements
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import Bull Machine v1.3 modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from bull_machine.version import __version__, get_version_banner
from bull_machine.app.main_v13 import (
    load_data,
    merge_configs,
    run_wyckoff_analysis,
    run_liquidity_analysis,
    run_structure_analysis,
    run_momentum_analysis,
    run_volume_analysis,
    run_context_analysis,
    run_advanced_fusion_engine,
    run_mtf_analysis
)
from bull_machine.strategy.risk_manager import calculate_position_size, calculate_stop_loss

# ETH data files configuration
ETH_DATA_FILES = {
    "1D": {
        "files": [
            "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
            "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
        ],
        "bars_per_day": 1
    },
    "4H": {
        "files": [
            "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
            "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
        ],
        "bars_per_day": 6
    },
    "6H": {
        "files": [
            "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 360_3c856.csv"
        ],
        "bars_per_day": 4
    },
    "12H": {
        "files": [
            "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv"
        ],
        "bars_per_day": 2
    },
    "1H": {
        "files": [
            "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
        ],
        "bars_per_day": 24
    }
}

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load ETH data with proper formatting."""
    try:
        df = pd.read_csv(file_path)

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            logging.error(f"Missing required columns in {file_path}")
            return pd.DataFrame()

        # Handle timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        elif 'timestamp' not in df.columns:
            # Generate timestamps
            df['timestamp'] = pd.date_range(
                start='2020-06-01',
                periods=len(df),
                freq='D'
            )

        # Handle volume
        if 'volume' not in df.columns:
            if 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            elif 'Total Buy Volume' in df.columns and 'Total Sell Volume' in df.columns:
                df['volume'] = df['Total Buy Volume'] + df['Total Sell Volume']
            else:
                df['volume'] = 100000  # Default

        # Select and return clean data
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def simulate_trades(df: pd.DataFrame, config: dict, timeframe: str) -> dict:
    """Simulate trades using Bull Machine v1.4.2 logic."""

    results = {
        'timeframe': timeframe,
        'total_bars': len(df),
        'trades': [],
        'signals': 0,
        'entries': 0
    }

    balance = config['initial_balance']
    peak_balance = balance
    equity_curve = []

    # Track active position
    position = None

    # Need sufficient history
    min_bars = 200
    if len(df) < min_bars:
        logging.warning(f"Insufficient data: {len(df)} bars (need {min_bars})")
        return results

    # Process bars
    for i in range(min_bars, len(df)):
        current_bar = df.iloc[i]

        # Get history windows
        htf_window = df.iloc[max(0, i-100):i]
        mtf_window = df.iloc[max(0, i-50):i]
        ltf_window = df.iloc[max(0, i-20):i]

        # Exit check
        if position:
            bars_held = i - position['entry_bar']
            current_pnl_pct = ((current_bar['close'] - position['entry']) / position['entry']) * 100

            # Exit conditions (v1.4.2 rules)
            should_exit = False
            exit_reason = ""

            # Stop loss
            if position['direction'] == 'long':
                if current_bar['low'] <= position['stop']:
                    should_exit = True
                    exit_reason = "Stop Loss"
            else:
                if current_bar['high'] >= position['stop']:
                    should_exit = True
                    exit_reason = "Stop Loss"

            # Time-based exit
            if bars_held >= 20:
                should_exit = True
                exit_reason = "Time Limit"

            # Profit target
            if current_pnl_pct >= 8:  # 8% profit target
                should_exit = True
                exit_reason = "Profit Target"

            # Volume spike exit (v1.4.2 distribution detection)
            if i > 0:
                vol_ratio = current_bar['volume'] / df.iloc[i-20:i]['volume'].mean()
                if vol_ratio > 2.0 and bars_held >= 5:
                    should_exit = True
                    exit_reason = "Volume Spike"

            if should_exit:
                # Calculate final PnL
                exit_price = current_bar['close']
                if position['direction'] == 'long':
                    pnl_pct = ((exit_price - position['entry']) / position['entry']) * 100
                else:
                    pnl_pct = ((position['entry'] - exit_price) / position['entry']) * 100

                pnl_dollar = (balance * config['risk_per_trade']) * (pnl_pct / 100)
                balance += pnl_dollar

                # Record trade
                position['exit'] = exit_price
                position['exit_bar'] = i
                position['exit_reason'] = exit_reason
                position['pnl_pct'] = pnl_pct
                position['pnl_dollar'] = pnl_dollar
                position['bars_held'] = bars_held

                results['trades'].append(position)
                position = None

                logging.info(f"  EXIT @ {exit_price:.2f} ({exit_reason}) | PnL: {pnl_pct:+.2f}%")

        # Entry logic (only if no position)
        if not position and i % config.get('check_frequency', 4) == 0:

            # Run MTF analysis
            mtf_result = run_mtf_analysis(htf_window, mtf_window, ltf_window, config)

            if mtf_result['decision'] != 'VETO':
                # Run layer analyses
                wyckoff = run_wyckoff_analysis(ltf_window)
                liquidity = run_liquidity_analysis(ltf_window, config)
                structure = run_structure_analysis(ltf_window)
                momentum = run_momentum_analysis(ltf_window)
                volume = run_volume_analysis(ltf_window)
                context = run_context_analysis(ltf_window)

                # Fusion scoring
                fusion = run_advanced_fusion_engine(
                    wyckoff, liquidity, structure,
                    momentum, volume, context, config
                )

                # Adjust threshold based on MTF
                threshold = config['enter_threshold']
                if mtf_result['decision'] == 'RAISE':
                    threshold += 0.10

                results['signals'] += 1

                # Check for entry
                if fusion['final_score'] >= threshold:
                    # Determine direction
                    direction = 'long' if fusion['signal'] == 'long' else 'short'

                    # Risk management
                    entry_price = current_bar['close']
                    atr = (htf_window['high'] - htf_window['low']).rolling(14).mean().iloc[-1]

                    if direction == 'long':
                        stop_price = entry_price - (atr * 1.5)
                    else:
                        stop_price = entry_price + (atr * 1.5)

                    # Enter position
                    position = {
                        'timestamp': current_bar['timestamp'],
                        'entry': entry_price,
                        'entry_bar': i,
                        'direction': direction,
                        'stop': stop_price,
                        'fusion_score': fusion['final_score'],
                        'wyckoff_phase': wyckoff.get('phase', 'unknown'),
                        'mtf_alignment': mtf_result.get('alignment_score', 0)
                    }

                    results['entries'] += 1

                    logging.info(f"\n[{timeframe}] {current_bar['timestamp']}")
                    logging.info(f"  ENTER {direction.upper()} @ {entry_price:.2f} | "
                               f"Stop: {stop_price:.2f} | Score: {fusion['final_score']:.3f}")

        # Track equity
        equity_curve.append(balance)

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance

    # Close any open position at end
    if position:
        exit_price = df.iloc[-1]['close']
        if position['direction'] == 'long':
            pnl_pct = ((exit_price - position['entry']) / position['entry']) * 100
        else:
            pnl_pct = ((position['entry'] - exit_price) / position['entry']) * 100

        position['exit'] = exit_price
        position['pnl_pct'] = pnl_pct
        position['exit_reason'] = 'End of Data'
        results['trades'].append(position)

    # Calculate statistics
    if results['trades']:
        wins = [t for t in results['trades'] if t['pnl_pct'] > 0]
        losses = [t for t in results['trades'] if t['pnl_pct'] <= 0]

        results['total_trades'] = len(results['trades'])
        results['winning_trades'] = len(wins)
        results['losing_trades'] = len(losses)
        results['win_rate'] = (len(wins) / len(results['trades'])) * 100
        results['total_pnl'] = sum(t['pnl_pct'] for t in results['trades'])
        results['avg_win'] = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        results['avg_loss'] = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        results['max_drawdown'] = ((peak_balance - min(equity_curve)) / peak_balance * 100) if equity_curve else 0

        # Profit factor
        gross_profit = sum(t['pnl_pct'] for t in wins)
        gross_loss = abs(sum(t['pnl_pct'] for t in losses))
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0

    results['final_balance'] = balance
    results['total_return'] = ((balance - config['initial_balance']) / config['initial_balance']) * 100

    return results

def main():
    """Run ETH comprehensive backtest."""
    print("=" * 80)
    print(f"🚀 {get_version_banner()} - ETH Comprehensive Backtest")
    print("=" * 80)

    # Configuration
    config = merge_configs()
    config['initial_balance'] = 10000
    config['risk_per_trade'] = 0.01
    config['enter_threshold'] = 0.35
    config['mtf_enabled'] = True
    config['check_frequency'] = 4  # Check every 4 bars

    all_results = {}

    # Process each timeframe
    for timeframe, tf_config in ETH_DATA_FILES.items():
        print(f"\n{'='*60}")
        print(f"📊 Testing ETH {timeframe} Timeframe")
        print(f"{'='*60}")

        for file_path in tf_config['files']:
            if not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
                continue

            print(f"\n📂 Processing: {Path(file_path).name}")

            # Load data
            df = load_eth_data(file_path)
            if df.empty:
                print("❌ Failed to load data")
                continue

            print(f"✅ Loaded {len(df)} bars")
            print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Run simulation
            print(f"🔄 Running backtest...")
            results = simulate_trades(df, config, timeframe)

            # Store results
            key = f"{timeframe}_{Path(file_path).stem[:20]}"
            all_results[key] = results

            # Display results
            if results['trades']:
                print(f"\n📈 Performance Summary:")
                print(f"  Signals: {results['signals']}")
                print(f"  Entries: {results['entries']}")
                print(f"  Completed Trades: {results['total_trades']}")
                print(f"  Win Rate: {results['win_rate']:.1f}%")
                print(f"  Total PnL: {results['total_pnl']:+.2f}%")
                print(f"  Avg Win: {results['avg_win']:+.2f}%")
                print(f"  Avg Loss: {results['avg_loss']:+.2f}%")
                print(f"  Profit Factor: {results['profit_factor']:.2f}")
                print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
                print(f"  Final Balance: ${results['final_balance']:,.2f}")
                print(f"  Total Return: {results['total_return']:+.1f}%")

                # Show sample trades
                print(f"\n  Recent Trades:")
                for trade in results['trades'][-5:]:
                    print(f"    {trade['timestamp']} | {trade['direction'].upper()} | "
                          f"Entry: {trade['entry']:.2f} | Exit: {trade.get('exit', 'Open'):.2f} | "
                          f"PnL: {trade.get('pnl_pct', 0):+.2f}% | {trade.get('exit_reason', '')}")
            else:
                print(f"\n📊 No trades taken (Signals: {results['signals']})")

    # Overall summary
    print(f"\n{'='*80}")
    print(f"📊 OVERALL SUMMARY - ALL TIMEFRAMES")
    print(f"{'='*80}")

    if all_results:
        total_trades = sum(r.get('total_trades', 0) for r in all_results.values())
        total_signals = sum(r.get('signals', 0) for r in all_results.values())

        print(f"\n🎯 Aggregate Statistics:")
        print(f"  Total Signals: {total_signals}")
        print(f"  Total Trades: {total_trades}")

        if total_trades > 0:
            total_wins = sum(r.get('winning_trades', 0) for r in all_results.values())
            avg_return = np.mean([r.get('total_return', 0) for r in all_results.values()])

            print(f"  Overall Win Rate: {(total_wins/total_trades)*100:.1f}%")
            print(f"  Average Return: {avg_return:+.1f}%")

            # Best performing
            best = max(all_results.items(), key=lambda x: x[1].get('total_return', -999))
            print(f"\n🏆 Best Performing: {best[0]}")
            print(f"  Return: {best[1].get('total_return', 0):+.1f}%")
            print(f"  Win Rate: {best[1].get('win_rate', 0):.1f}%")
            print(f"  Trades: {best[1].get('total_trades', 0)}")

    # Save results
    output_file = f"eth_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n{'='*80}")
    print(f"✅ ETH Backtest Complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()