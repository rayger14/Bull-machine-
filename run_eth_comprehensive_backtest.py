#!/usr/bin/env python3
"""
Bull Machine v1.4.2 - Comprehensive ETH Backtest
Multi-timeframe analysis across available ETH data
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import Bull Machine modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bull_machine.version import __version__, get_version_banner

# Simple dataclass for bar data
@dataclass
class BarData:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

# ETH data files configuration
ETH_DATA_FILES = {
    "1D": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
    ],
    "4H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    ],
    "6H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 360_3c856.csv"
    ],
    "12H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv"
    ],
    "22H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1320_f91da.csv"
    ],
    "1H": [
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
    ]
}

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load and prepare ETH data from CSV."""
    try:
        df = pd.read_csv(file_path)

        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                logging.error(f"Missing required column: {col}")
                return pd.DataFrame()

        # Add timestamp if missing
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        # Add volume if missing
        if 'volume' not in df.columns:
            if 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            else:
                df['volume'] = 100000  # Default volume

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def run_backtest(df: pd.DataFrame, timeframe: str, config: dict) -> dict:
    """Run Bull Machine v1.4.2 backtest on ETH data."""

    results = {
        'timeframe': timeframe,
        'total_bars': len(df),
        'trades': [],
        'signals_generated': 0,
        'trades_taken': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
        'sharpe_ratio': 0.0
    }

    # Initialize modules
    mtf_sync = MTFSyncEngine(config)
    fusion = FusionEngine(config)
    wyckoff = WyckoffAnalyzer()
    liquidity = LiquidityAnalyzer(config)
    structure = StructureAnalyzer()
    momentum = MomentumAnalyzer()
    volume = VolumeAnalyzer()
    context = ContextAnalyzer()
    risk_mgr = RiskManager(config)
    exit_eval = AdvancedExitEvaluator(config)
    telemetry = TelemetryTracker()

    # Track account balance
    balance = config['initial_balance']
    peak_balance = balance
    equity_curve = [balance]

    # Active trade tracking
    active_trade = None
    trade_history = []

    # Process each bar
    for i in range(200, len(df)):  # Need history for indicators
        current_bar = df.iloc[i]
        history = df.iloc[max(0, i-200):i]

        # Create BarData
        bar_data = BarData(
            timestamp=current_bar['timestamp'],
            open=current_bar['open'],
            high=current_bar['high'],
            low=current_bar['low'],
            close=current_bar['close'],
            volume=current_bar['volume']
        )

        # Check exit conditions for active trade
        if active_trade:
            exit_signal = exit_eval.evaluate(
                trade=active_trade,
                current_bar=bar_data,
                bars_held=i - active_trade['entry_bar'],
                current_pnl=(current_bar['close'] - active_trade['entry']) / active_trade['entry']
            )

            if exit_signal['should_exit']:
                # Close trade
                exit_price = current_bar['close']
                pnl = 0.0

                if active_trade['direction'] == 'long':
                    pnl = ((exit_price - active_trade['entry']) / active_trade['entry']) * 100
                else:
                    pnl = ((active_trade['entry'] - exit_price) / active_trade['entry']) * 100

                # Update balance
                trade_return = pnl / 100
                position_value = balance * active_trade['size_pct']
                trade_pnl = position_value * trade_return
                balance += trade_pnl

                # Record trade
                active_trade['exit'] = exit_price
                active_trade['exit_bar'] = i
                active_trade['exit_reason'] = exit_signal['reason']
                active_trade['pnl'] = pnl
                active_trade['pnl_dollar'] = trade_pnl

                results['trades'].append(active_trade)
                results['total_pnl'] += pnl

                if pnl > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1

                logging.info(f"  EXIT {active_trade['direction'].upper()} @ {exit_price:.2f} "
                           f"({exit_signal['reason']}) | PnL: {pnl:+.2f}% (${trade_pnl:+.2f})")

                active_trade = None

        # Generate signals if no active trade
        if not active_trade and i % config.get('signal_frequency', 1) == 0:
            # MTF Analysis
            htf_data = history.tail(100)
            mtf_decision = mtf_sync.analyze(htf_data, history.tail(50), history.tail(25))

            if mtf_decision['decision'] != 'VETO':
                # Run layer analysis
                wyckoff_score = wyckoff.analyze(history)
                liquidity_score = liquidity.analyze(history)
                structure_score = structure.analyze(history)
                momentum_score = momentum.analyze(history)
                volume_score = volume.analyze(history)
                context_score = context.analyze(history)

                # Fusion scoring
                layer_scores = {
                    'wyckoff': wyckoff_score.get('score', 0),
                    'liquidity': liquidity_score.get('score', 0),
                    'structure': structure_score.get('score', 0),
                    'momentum': momentum_score.get('score', 0),
                    'volume': volume_score.get('score', 0),
                    'context': context_score.get('score', 0),
                    'mtf': mtf_decision.get('alignment_score', 0) / 100
                }

                fusion_result = fusion.fuse(layer_scores)

                # Check signal threshold
                threshold = config['enter_threshold']
                if mtf_decision['decision'] == 'RAISE':
                    threshold += 0.10

                if fusion_result['score'] >= threshold:
                    results['signals_generated'] += 1

                    # Determine direction
                    direction = fusion_result.get('signal', 'neutral')
                    if direction != 'neutral':
                        # Risk calculation
                        entry_price = current_bar['close']
                        stop_price = risk_mgr.calculate_stop(
                            entry=entry_price,
                            direction=direction,
                            atr=history['high'].tail(14).values - history['low'].tail(14).values
                        )

                        size_pct = min(0.02, config.get('position_size', 0.01))  # Max 2% per trade

                        # Enter trade
                        active_trade = {
                            'entry': entry_price,
                            'entry_bar': i,
                            'direction': direction,
                            'stop': stop_price['price'],
                            'size_pct': size_pct,
                            'fusion_score': fusion_result['score'],
                            'wyckoff_phase': wyckoff_score.get('phase', 'unknown'),
                            'timestamp': current_bar['timestamp']
                        }

                        results['trades_taken'] += 1

                        logging.info(f"[{timeframe}] Bar {i} | {current_bar['timestamp']}")
                        logging.info(f"  ENTER {direction.upper()} @ {entry_price:.2f} | "
                                   f"Stop: {stop_price['price']:.2f} | "
                                   f"Score: {fusion_result['score']:.3f}")

        # Update equity curve
        equity_curve.append(balance)

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        else:
            drawdown = (peak_balance - balance) / peak_balance * 100
            results['max_drawdown'] = max(results['max_drawdown'], drawdown)

    # Calculate final statistics
    if results['trades']:
        wins = [t['pnl'] for t in results['trades'] if t['pnl'] > 0]
        losses = [t['pnl'] for t in results['trades'] if t['pnl'] <= 0]

        results['win_rate'] = (results['winning_trades'] / len(results['trades'])) * 100
        results['avg_win'] = np.mean(wins) if wins else 0
        results['avg_loss'] = np.mean(losses) if losses else 0

        if results['avg_loss'] != 0:
            results['profit_factor'] = abs(results['avg_win'] * len(wins)) / abs(results['avg_loss'] * len(losses))

        # Calculate Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            results['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

    results['final_balance'] = balance
    results['total_return'] = ((balance - config['initial_balance']) / config['initial_balance']) * 100

    return results

def main():
    """Run comprehensive ETH backtests."""
    print(f"{'='*70}")
    print(f"🚀 {get_version_banner()} - Comprehensive ETH Backtest")
    print(f"{'='*70}")

    # Load configuration
    config = {
        'initial_balance': 10000,
        'enter_threshold': 0.35,
        'position_size': 0.01,
        'signal_frequency': 1,
        'mtf_enabled': True,
        'quality_floor': 0.25,
        'max_risk_per_trade': 0.02
    }

    all_results = {}

    # Process each timeframe
    for timeframe, file_list in ETH_DATA_FILES.items():
        print(f"\n{'='*70}")
        print(f"📊 Processing ETH {timeframe} Data")
        print(f"{'='*70}")

        for file_path in file_list:
            if not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
                continue

            print(f"\n📂 Loading: {Path(file_path).name}")
            df = load_eth_data(file_path)

            if df.empty:
                print(f"❌ Failed to load data")
                continue

            print(f"✅ Loaded {len(df)} bars")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Run backtest
            print(f"\n🔄 Running backtest...")
            results = run_backtest(df, timeframe, config)

            # Store results
            key = f"{timeframe}_{Path(file_path).stem}"
            all_results[key] = results

            # Display results
            print(f"\n📈 Results for {timeframe}:")
            print(f"  Signals Generated: {results['signals_generated']}")
            print(f"  Trades Taken: {results['trades_taken']}")
            print(f"  Win Rate: {results['win_rate']:.1f}%")
            print(f"  Total PnL: {results['total_pnl']:+.2f}%")
            print(f"  Final Balance: ${results['final_balance']:,.2f}")
            print(f"  Total Return: {results['total_return']:+.2f}%")
            print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")

            if results['trades']:
                print(f"\n  Trade Details:")
                for trade in results['trades'][:5]:  # Show first 5 trades
                    print(f"    {trade['timestamp']} | {trade['direction'].upper()} | "
                          f"Entry: {trade['entry']:.2f} | Exit: {trade.get('exit', 'Active'):.2f} | "
                          f"PnL: {trade.get('pnl', 0):+.2f}%")

    # Summary across all timeframes
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY ACROSS ALL TIMEFRAMES")
    print(f"{'='*70}")

    total_trades = sum(r['trades_taken'] for r in all_results.values())
    total_wins = sum(r['winning_trades'] for r in all_results.values())
    avg_return = np.mean([r['total_return'] for r in all_results.values()])

    print(f"\n🎯 Overall Performance:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total Wins: {total_wins}")
    print(f"  Overall Win Rate: {(total_wins/total_trades*100) if total_trades > 0 else 0:.1f}%")
    print(f"  Average Return: {avg_return:+.2f}%")

    # Best performing timeframe
    if all_results:
        best_tf = max(all_results.items(), key=lambda x: x[1]['total_return'])
        print(f"\n🏆 Best Performing Timeframe: {best_tf[0]}")
        print(f"  Return: {best_tf[1]['total_return']:+.2f}%")
        print(f"  Win Rate: {best_tf[1]['win_rate']:.1f}%")

    # Save detailed results
    output_file = f"eth_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ ETH Comprehensive Backtest Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()