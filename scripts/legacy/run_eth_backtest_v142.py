#!/usr/bin/env python3
"""
Bull Machine v1.4.2 - Comprehensive ETH Backtest
Using the latest v1.4.2 engine with all enhancements
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Add Bull Machine to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.version import __version__, get_version_banner
from bull_machine.scoring.fusion import FusionEngineV141 as FusionEngine
from bull_machine.modules.wyckoff.analyzer import WyckoffAnalyzer
from bull_machine.modules.liquidity.analyzer import LiquidityAnalyzer
from bull_machine.modules.structure.analyzer import StructureAnalyzer
from bull_machine.modules.momentum.analyzer import MomentumAnalyzer
from bull_machine.modules.volume.analyzer import VolumeAnalyzer
from bull_machine.modules.context.analyzer import ContextAnalyzer
from bull_machine.modules.bojan.candle_logic import analyze_bojan_patterns
from bull_machine.strategy.exits.advanced_rules import AdvancedExitRules
from bull_machine.strategy.exits.advanced_evaluator import AdvancedExitEvaluator
from bull_machine.telemetry.tracker import TelemetryTracker

# ETH Data Files
ETH_DATA = {
    "1D": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv", "Desktop_1D"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv", "Downloads_1D")
    ],
    "4H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv", "Desktop_4H"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv", "Downloads_4H")
    ],
    "6H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 360_3c856.csv", "Desktop_6H")
    ],
    "12H": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv", "Desktop_12H")
    ],
    "1H": [
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv", "Downloads_1H")
    ]
}

@dataclass
class Trade:
    """Trade data structure."""
    timestamp: pd.Timestamp
    direction: str
    entry: float
    stop: float
    exit: float = 0
    pnl_pct: float = 0
    pnl_dollar: float = 0
    bars_held: int = 0
    exit_reason: str = ""
    fusion_score: float = 0
    wyckoff_phase: str = ""
    quality_score: float = 0

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load and prepare ETH data."""
    try:
        df = pd.read_csv(file_path)

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            logging.error(f"Missing required columns")
            return pd.DataFrame()

        # Handle timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        else:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        # Handle volume
        if 'volume' not in df.columns:
            if 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            elif 'Total Buy Volume' in df.columns:
                df['volume'] = df['Total Buy Volume'] + df.get('Total Sell Volume', 0)
            else:
                df['volume'] = 100000  # Default

        # Calculate additional indicators
        df['atr'] = calculate_atr(df)
        df['rsi'] = calculate_rsi(df['close'])

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'atr', 'rsi']]

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(period).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_v142_backtest(df: pd.DataFrame, config: dict, timeframe: str) -> dict:
    """
    Run Bull Machine v1.4.2 backtest with all enhancements:
    - Advanced exit system
    - Quality floor enforcement
    - Phase-aware stops
    - Bojan candle patterns (soft-gated)
    - Enhanced telemetry
    """

    results = {
        'timeframe': timeframe,
        'total_bars': len(df),
        'trades': [],
        'signals_generated': 0,
        'signals_filtered': 0,
        'trades_taken': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'max_drawdown': 0.0,
        'telemetry': {}
    }

    # Initialize v1.4.2 modules
    fusion = FusionEngine(config)
    wyckoff = WyckoffAnalyzer()
    liquidity = LiquidityAnalyzer(config)
    structure = StructureAnalyzer()
    momentum = MomentumAnalyzer()
    volume = VolumeAnalyzer()
    context = ContextAnalyzer()
    exit_eval = AdvancedExitEvaluator(config)
    telemetry = TelemetryTracker()

    # Track performance
    balance = config['initial_balance']
    peak_balance = balance
    equity_curve = [balance]
    active_trade = None

    # Quality floor from v1.4.2
    quality_floor = config.get('quality_floor', 0.25)

    # Process bars (need history for indicators)
    min_bars = 200
    if len(df) < min_bars:
        logging.warning(f"Insufficient data: {len(df)} bars")
        return results

    for i in range(min_bars, len(df)):
        current_bar = df.iloc[i]
        history = df.iloc[max(0, i-200):i]

        # Check exit for active trade
        if active_trade:
            bars_held = i - active_trade.entry_bar
            current_price = current_bar['close']

            # Calculate current PnL
            if active_trade.direction == 'long':
                current_pnl = (current_price - active_trade.entry) / active_trade.entry
            else:
                current_pnl = (active_trade.entry - current_price) / active_trade.entry

            # v1.4.2 Advanced Exit Evaluation
            exit_signal = exit_eval.evaluate(
                trade={
                    'direction': active_trade.direction,
                    'entry': active_trade.entry,
                    'phase': active_trade.wyckoff_phase,
                    'quality': active_trade.quality_score
                },
                current_bar={
                    'close': current_price,
                    'volume': current_bar['volume'],
                    'atr': current_bar['atr']
                },
                bars_held=bars_held,
                current_pnl=current_pnl
            )

            if exit_signal['should_exit']:
                # Exit trade
                active_trade.exit = current_price
                active_trade.pnl_pct = current_pnl * 100
                active_trade.pnl_dollar = balance * config['risk_per_trade'] * current_pnl
                active_trade.bars_held = bars_held
                active_trade.exit_reason = exit_signal['reason']

                # Update balance
                balance += active_trade.pnl_dollar

                # Track results
                results['trades'].append(vars(active_trade))
                results['trades_taken'] += 1

                if active_trade.pnl_pct > 0:
                    results['winning_trades'] += 1
                else:
                    results['losing_trades'] += 1

                results['total_pnl'] += active_trade.pnl_pct

                # Log exit
                logging.info(f"  EXIT {active_trade.direction.upper()} @ {current_price:.2f} "
                           f"({exit_signal['reason']}) | PnL: {active_trade.pnl_pct:+.2f}%")

                active_trade = None

        # Generate signals (only if no active trade)
        if not active_trade and i % config.get('signal_frequency', 4) == 0:

            # Run layer analyses
            wyckoff_result = wyckoff.analyze(history)
            liquidity_result = liquidity.analyze(history)
            structure_result = structure.analyze(history)
            momentum_result = momentum.analyze(history)
            volume_result = volume.analyze(history)
            context_result = context.analyze(history)

            # v1.4.2: Bojan patterns (soft-gated)
            bojan_patterns = analyze_bojan_patterns(history)
            bojan_score = 0.0
            if bojan_patterns:
                # Cap at 0.6 for v1.4.1 phase gate
                bojan_score = min(0.6, sum(p.get('confidence', 0) for p in bojan_patterns) / len(bojan_patterns))

            # Prepare scores for fusion
            layer_scores = {
                'wyckoff': wyckoff_result.get('score', 0),
                'liquidity': liquidity_result.get('score', 0),
                'structure': structure_result.get('score', 0),
                'momentum': momentum_result.get('score', 0),
                'volume': volume_result.get('score', 0),
                'context': context_result.get('score', 0),
                'bojan': bojan_score
            }

            # Run fusion engine
            fusion_result = fusion.fuse(layer_scores)

            results['signals_generated'] += 1

            # v1.4.2: Quality floor enforcement
            if fusion_result['score'] < quality_floor:
                results['signals_filtered'] += 1
                telemetry.record_quality_gate(fusion_result['score'], quality_floor)
                continue

            # Check entry threshold
            threshold = config['enter_threshold']

            # v1.4.2: Dynamic threshold adjustment based on phase
            wyckoff_phase = wyckoff_result.get('phase', 'unknown')
            if wyckoff_phase in ['accumulation_A', 'accumulation_C']:
                threshold += 0.10  # Stricter in early accumulation
            elif wyckoff_phase in ['distribution_D', 'distribution_E']:
                threshold += 0.15  # Very strict in distribution

            if fusion_result['score'] >= threshold:
                # Determine direction
                direction = fusion_result.get('signal', 'neutral')

                if direction != 'neutral':
                    entry_price = current_bar['close']

                    # v1.4.2: Phase-aware ATR stops
                    atr = current_bar['atr']
                    stop_multiplier = 1.5  # Default

                    if wyckoff_phase in ['accumulation_B', 'accumulation_D']:
                        stop_multiplier = 2.0  # Wider stops in volatile phases
                    elif wyckoff_phase in ['accumulation_E', 'distribution_A']:
                        stop_multiplier = 1.2  # Tighter stops in trending phases

                    if direction == 'long':
                        stop_price = entry_price - (atr * stop_multiplier)
                    else:
                        stop_price = entry_price + (atr * stop_multiplier)

                    # Create trade
                    active_trade = Trade(
                        timestamp=current_bar['timestamp'],
                        direction=direction,
                        entry=entry_price,
                        stop=stop_price,
                        fusion_score=fusion_result['score'],
                        wyckoff_phase=wyckoff_phase,
                        quality_score=fusion_result.get('quality', fusion_result['score'])
                    )
                    active_trade.entry_bar = i

                    # Log entry
                    logging.info(f"\n[{timeframe}] Bar {i} | {current_bar['timestamp']}")
                    logging.info(f"  SIGNAL: Score={fusion_result['score']:.3f} | Phase={wyckoff_phase}")
                    logging.info(f"  ENTER {direction.upper()} @ {entry_price:.2f} | "
                               f"Stop: {stop_price:.2f} (ATR×{stop_multiplier:.1f})")

        # Update equity
        equity_curve.append(balance)

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        else:
            drawdown = (peak_balance - balance) / peak_balance * 100
            results['max_drawdown'] = max(results['max_drawdown'], drawdown)

    # Close any open position
    if active_trade:
        current_price = df.iloc[-1]['close']
        if active_trade.direction == 'long':
            active_trade.pnl_pct = ((current_price - active_trade.entry) / active_trade.entry) * 100
        else:
            active_trade.pnl_pct = ((active_trade.entry - current_price) / active_trade.entry) * 100

        active_trade.exit = current_price
        active_trade.exit_reason = "End of Data"
        results['trades'].append(vars(active_trade))

    # Calculate final statistics
    results['final_balance'] = balance
    results['total_return'] = ((balance - config['initial_balance']) / config['initial_balance']) * 100

    if results['trades']:
        wins = [t['pnl_pct'] for t in results['trades'] if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in results['trades'] if t['pnl_pct'] <= 0]

        results['win_rate'] = (results['winning_trades'] / len(results['trades'])) * 100
        results['avg_win'] = np.mean(wins) if wins else 0
        results['avg_loss'] = np.mean(losses) if losses else 0

        # Profit factor
        if losses:
            gross_wins = sum(wins)
            gross_losses = abs(sum(losses))
            results['profit_factor'] = gross_wins / gross_losses if gross_losses > 0 else 0

        # Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            results['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            results['sharpe_ratio'] = 0

    # v1.4.2 Telemetry
    results['telemetry'] = telemetry.get_summary()

    return results

def main():
    """Run comprehensive ETH backtest with Bull Machine v1.4.2."""
    print("=" * 80)
    print(f"🚀 {get_version_banner()} - Comprehensive ETH Backtest")
    print("=" * 80)
    print(f"📅 Timestamp: {datetime.now()}")
    print("=" * 80)

    # v1.4.2 Configuration
    config = {
        'initial_balance': 10000,
        'risk_per_trade': 0.01,  # 1% risk per trade
        'enter_threshold': 0.35,
        'quality_floor': 0.25,  # v1.4.2 quality gate
        'signal_frequency': 4,  # Check every 4 bars
        'phase_aware_stops': True,  # v1.4.2 feature
        'exit_params': {  # v1.4.2 advanced exits
            'use_advanced': True,
            'phase_multipliers': True,
            'distribution_exit': True,
            'momentum_exit': True
        }
    }

    print(f"\n⚙️  Configuration:")
    print(f"  Enter Threshold: {config['enter_threshold']}")
    print(f"  Quality Floor: {config['quality_floor']}")
    print(f"  Phase-Aware Stops: {config['phase_aware_stops']}")
    print(f"  Risk Per Trade: {config['risk_per_trade']*100}%")

    all_results = {}
    summary = {
        'total_signals': 0,
        'total_trades': 0,
        'total_wins': 0,
        'best_trade': None,
        'worst_trade': None,
        'best_timeframe': None
    }

    # Process each timeframe
    for timeframe, file_list in ETH_DATA.items():
        print(f"\n{'='*70}")
        print(f"📊 Processing ETH {timeframe} Data")
        print(f"{'='*70}")

        for file_path, label in file_list:
            if not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
                continue

            print(f"\n📂 Loading: {Path(file_path).name}")
            df = load_eth_data(file_path)

            if df.empty:
                print(f"❌ Failed to load data")
                continue

            print(f"✅ Loaded {len(df)} bars")
            print(f"📅 Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            print(f"💰 Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

            # Run v1.4.2 backtest
            print(f"\n🔄 Running Bull Machine v1.4.2 Analysis...")
            results = run_v142_backtest(df, config, timeframe)

            # Store results
            key = f"{timeframe}_{label}"
            all_results[key] = results

            # Update summary
            summary['total_signals'] += results['signals_generated']
            summary['total_trades'] += results['trades_taken']
            summary['total_wins'] += results['winning_trades']

            # Display results
            print(f"\n📈 Results Summary:")
            print(f"  Signals Generated: {results['signals_generated']}")
            print(f"  Signals Filtered (Quality): {results['signals_filtered']}")
            print(f"  Trades Taken: {results['trades_taken']}")

            if results['trades']:
                print(f"  Win Rate: {results['win_rate']:.1f}%")
                print(f"  Total PnL: {results['total_pnl']:+.2f}%")
                print(f"  Avg Win: {results['avg_win']:+.2f}%")
                print(f"  Avg Loss: {results['avg_loss']:.2f}%")
                print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
                print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
                print(f"  Final Balance: ${results['final_balance']:,.2f}")
                print(f"  Total Return: {results['total_return']:+.1f}%")

                # Track best/worst
                for trade in results['trades']:
                    if summary['best_trade'] is None or trade['pnl_pct'] > summary['best_trade']['pnl_pct']:
                        summary['best_trade'] = trade
                        summary['best_trade']['timeframe'] = timeframe
                    if summary['worst_trade'] is None or trade['pnl_pct'] < summary['worst_trade']['pnl_pct']:
                        summary['worst_trade'] = trade
                        summary['worst_trade']['timeframe'] = timeframe

                # Sample trades
                print(f"\n  Sample Trades:")
                for trade in results['trades'][:3]:
                    print(f"    {trade['timestamp']} | {trade['direction'].upper()} | "
                          f"Entry: ${trade['entry']:.2f} | Exit: ${trade.get('exit', 0):.2f} | "
                          f"PnL: {trade.get('pnl_pct', 0):+.2f}% | {trade.get('exit_reason', '')}")
            else:
                print(f"  No trades taken (below quality/threshold)")

    # Overall Summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE ETH BACKTEST SUMMARY")
    print(f"{'='*80}")

    print(f"\n🎯 Overall Statistics:")
    print(f"  Total Signals Generated: {summary['total_signals']}")
    print(f"  Total Trades Taken: {summary['total_trades']}")
    print(f"  Total Winning Trades: {summary['total_wins']}")

    if summary['total_trades'] > 0:
        overall_win_rate = (summary['total_wins'] / summary['total_trades']) * 100
        print(f"  Overall Win Rate: {overall_win_rate:.1f}%")

    if summary['best_trade']:
        print(f"\n🏆 Best Trade:")
        print(f"  Timeframe: {summary['best_trade']['timeframe']}")
        print(f"  Direction: {summary['best_trade']['direction'].upper()}")
        print(f"  Entry: ${summary['best_trade']['entry']:.2f}")
        print(f"  Exit: ${summary['best_trade']['exit']:.2f}")
        print(f"  PnL: {summary['best_trade']['pnl_pct']:+.2f}%")
        print(f"  Exit Reason: {summary['best_trade']['exit_reason']}")

    if summary['worst_trade']:
        print(f"\n📉 Worst Trade:")
        print(f"  Timeframe: {summary['worst_trade']['timeframe']}")
        print(f"  Direction: {summary['worst_trade']['direction'].upper()}")
        print(f"  Entry: ${summary['worst_trade']['entry']:.2f}")
        print(f"  Exit: ${summary['worst_trade']['exit']:.2f}")
        print(f"  PnL: {summary['worst_trade']['pnl_pct']:.2f}%")
        print(f"  Exit Reason: {summary['worst_trade']['exit_reason']}")

    # Best performing timeframe
    if all_results:
        best_tf = max(all_results.items(), key=lambda x: x[1]['total_return'])
        if best_tf[1]['trades']:
            print(f"\n🥇 Best Performing Timeframe: {best_tf[0]}")
            print(f"  Total Return: {best_tf[1]['total_return']:+.1f}%")
            print(f"  Win Rate: {best_tf[1]['win_rate']:.1f}%")
            print(f"  Trades: {best_tf[1]['trades_taken']}")

    # Save results
    output_file = f"eth_backtest_v142_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'version': __version__,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'summary': summary,
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ ETH Comprehensive Backtest Complete!")
    print(f"   {get_version_banner()}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()