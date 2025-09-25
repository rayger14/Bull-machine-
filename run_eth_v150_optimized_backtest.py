#!/usr/bin/env python3
"""
Bull Machine v1.5.0 OPTIMIZED - ETH Backtest
Tests the optimized fusion engine with relaxed floors and reduced alpha impact
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

# Import optimized v1.5.0 modules
from bull_machine.core.config_loader import load_config
from bull_machine.core.fusion_enhanced import create_fusion_engine
from bull_machine.core.telemetry import log_telemetry

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: str = "long"
    size: float = 1.0
    pnl: float = 0.0
    win: bool = False
    confidence: float = 0.0
    signals: Optional[dict] = None

# ETH Data Files
ETH_DATA_FILES = {
    "1D": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv"
    ],
    "4H": [
        "/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    ],
    "1H": [
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv"
    ]
}

def load_eth_data(file_path: str) -> pd.DataFrame:
    """Load and prepare ETH data from CSV."""
    try:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_csv(file_path)

        # Map column names (handle spaces and variations)
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower().strip()
            if 'time' in lower_col and 'timestamp' not in column_mapping:
                column_mapping['timestamp'] = col
            elif lower_col == 'open':
                column_mapping['open'] = col
            elif lower_col == 'high':
                column_mapping['high'] = col
            elif lower_col == 'low':
                column_mapping['low'] = col
            elif lower_col == 'close':
                column_mapping['close'] = col
            elif 'buy+sell v' in lower_col or 'volume' in lower_col:
                column_mapping['volume'] = col

        # Rename columns
        df = df.rename(columns={v: k for k, v in column_mapping.items()})

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing columns: {missing_cols} in {file_path}")
            return pd.DataFrame()

        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')

        # Handle volume
        if 'volume' not in df.columns:
            df['volume'] = 100000  # Default volume

        # Clean data
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[df['high'] >= df['low']]  # Basic sanity check
        df = df[df['close'] > 0]

        result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        return result_df

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def create_mock_layer_scores(df: pd.DataFrame, enhanced: bool = False) -> Dict[str, float]:
    """Create mock layer scores for testing fusion engine."""
    # Use price action to generate realistic scores
    close = df['close'].iloc[-20:]  # Last 20 bars

    # Basic momentum (trend strength)
    momentum_score = 0.5 + (close.iloc[-1] - close.iloc[0]) / (close.iloc[0] * 4)  # Normalized
    momentum_score = max(0.0, min(1.0, momentum_score))

    # Volume trend
    volume = df['volume'].iloc[-10:] if 'volume' in df.columns else [100000] * 10
    volume_score = 0.4 + (np.mean(volume[-3:]) - np.mean(volume[-10:])) / np.mean(volume[-10:]) * 0.3
    volume_score = max(0.0, min(1.0, volume_score))

    # Structure (price in range)
    high_20 = close.max()
    low_20 = close.min()
    current = close.iloc[-1]
    structure_score = 0.3 + (current - low_20) / (high_20 - low_20 + 1e-8) * 0.4
    structure_score = max(0.0, min(1.0, structure_score))

    # Generate base scores with more realistic ranges for relaxed floors
    base_scores = {
        "wyckoff": 0.35 + np.random.normal(0, 0.08),      # Target around 0.35 for relaxed 0.32
        "liquidity": 0.32 + np.random.normal(0, 0.06),   # Target around 0.32 for relaxed 0.28
        "structure": structure_score,
        "momentum": momentum_score,
        "volume": volume_score,
        "context": 0.38 + np.random.normal(0, 0.05),     # Target around 0.38 for relaxed 0.32
        "mtf": 0.40 + np.random.normal(0, 0.07)          # Target around 0.40 for relaxed 0.35
    }

    # Clamp all scores to valid range
    for key in base_scores:
        base_scores[key] = max(0.0, min(1.0, base_scores[key]))

    return base_scores

def run_optimized_fusion_backtest(df: pd.DataFrame, timeframe: str, config_type: str = "baseline") -> Dict:
    """Run backtest using optimized Bull Machine fusion engine."""

    # Load appropriate config
    if config_type == "optimized_1d":
        config = load_config("ETH")
        config["timeframe"] = "1D"
    elif config_type == "optimized_4h":
        config = load_config("ETH_4H")
        config["timeframe"] = "4H"
    elif config_type == "optimized_1h":
        config = load_config("ETH")
        config["timeframe"] = "1H"
    else:  # baseline
        config = load_config()
        config["timeframe"] = timeframe

    # Enable features for optimized configs
    if config_type.startswith("optimized"):
        if config_type == "optimized_4h":
            # 4H selective features (LCA and VIP disabled)
            config["features"] = {
                "mtf_dl2": True,
                "six_candle_leg": True,
                "orderflow_lca": False,  # Disabled for 4H
                "negative_vip": False,   # Disabled for 4H
                "live_data": False
            }
        else:
            # Full features for 1D and 1H
            config["features"] = {
                "mtf_dl2": True,
                "six_candle_leg": True,
                "orderflow_lca": True,
                "negative_vip": True,
                "live_data": False
            }

    fusion_engine = create_fusion_engine(config)

    results = {
        'timeframe': timeframe,
        'config_type': config_type,
        'optimized': config_type.startswith("optimized"),
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

    active_trades = []
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0

    # Need minimum bars for analysis
    start_idx = max(50, len(df) // 4)

    logging.info(f"\n{'='*60}")
    logging.info(f"Running {config_type} backtest on {timeframe}")
    logging.info(f"Data: {len(df)} bars, analyzing from bar {start_idx}")
    logging.info(f"{'='*60}")

    quality_veto_count = 0
    regime_veto_count = 0

    for i in range(start_idx, len(df)):
        current_bar = df.iloc[i]
        window_df = df.iloc[max(0, i-50):i+1].copy()

        # Generate layer scores
        layer_scores = create_mock_layer_scores(window_df, config_type.startswith("optimized"))

        # Create mock wyckoff context
        wyckoff_context = {
            "bias": "long" if layer_scores["momentum"] > 0.5 else "short",
            "phase": "C",
            "regime": "accumulation"
        }

        # Run fusion engine
        try:
            fusion_result = fusion_engine.fuse(
                layer_scores=layer_scores,
                df=window_df,
                wyckoff_context=wyckoff_context
            )

            signal = fusion_result.get("signal", "neutral")
            confidence = fusion_result.get("confidence", 0.0)
            fusion_score = fusion_result.get("score", 0.0)
            quality_passed = fusion_result.get("quality_passed", True)
            regime_passed = fusion_result.get("regime_passed", True)

            if not quality_passed:
                quality_veto_count += 1
            if not regime_passed:
                regime_veto_count += 1

            if signal != "neutral":
                results['signals_generated'] += 1

            # Entry logic with optimized confidence threshold (0.65 instead of 0.7)
            confidence_threshold = 0.65 if config_type.startswith("optimized") else 0.7

            if signal in ["long", "short"] and confidence > confidence_threshold and len(active_trades) < 3:
                entry_price = current_bar['close']

                trade = Trade(
                    entry_time=current_bar['timestamp'],
                    entry_price=entry_price,
                    direction=signal,
                    size=1.0,
                    confidence=confidence,
                    signals={
                        'fusion_score': fusion_score,
                        'layer_scores': layer_scores.copy(),
                        'config_type': config_type,
                        'quality_passed': quality_passed,
                        'regime_passed': regime_passed
                    }
                )

                active_trades.append(trade)
                results['trades_taken'] += 1

                if results['trades_taken'] <= 5:  # Log first 5 trades
                    logging.info(f"TRADE {results['trades_taken']}: {signal.upper()} @ {entry_price:.2f} "
                               f"(confidence: {confidence:.3f}, score: {fusion_score:.3f})")

            # Simple exit logic
            for trade in active_trades[:]:
                bars_held = i - df[df['timestamp'] == trade.entry_time].index[0]

                # Exit conditions
                should_exit = False
                exit_reason = ""

                # Time-based exit (10-20 bars)
                if bars_held >= 15:
                    should_exit = True
                    exit_reason = "time"

                # Price-based exits
                price_change = (current_bar['close'] - trade.entry_price) / trade.entry_price

                if trade.direction == "long":
                    if price_change >= 0.03:  # 3% profit
                        should_exit = True
                        exit_reason = "profit"
                    elif price_change <= -0.015:  # 1.5% loss
                        should_exit = True
                        exit_reason = "stop"
                else:  # short
                    if price_change <= -0.03:
                        should_exit = True
                        exit_reason = "profit"
                    elif price_change >= 0.015:
                        should_exit = True
                        exit_reason = "stop"

                if should_exit:
                    exit_price = current_bar['close']

                    if trade.direction == "long":
                        trade.pnl = (exit_price - trade.entry_price) / trade.entry_price
                    else:
                        trade.pnl = (trade.entry_price - exit_price) / trade.entry_price

                    trade.exit_time = current_bar['timestamp']
                    trade.exit_price = exit_price
                    trade.win = trade.pnl > 0

                    # Update balance
                    balance += balance * trade.pnl * 0.01  # 1% position size
                    peak_balance = max(peak_balance, balance)
                    drawdown = (peak_balance - balance) / peak_balance
                    max_dd = max(max_dd, drawdown)

                    results['trades'].append(trade)
                    active_trades.remove(trade)

                    if trade.win:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1

        except Exception as e:
            if i < start_idx + 5:  # Log first few errors
                logging.warning(f"Fusion error at bar {i}: {e}")
            continue

    # Calculate final statistics
    if results['trades']:
        total_trades = len(results['trades'])
        results['win_rate'] = results['winning_trades'] / total_trades
        results['total_pnl'] = sum(t.pnl for t in results['trades'])
        results['max_drawdown'] = max_dd

        winning_trades = [t for t in results['trades'] if t.win]
        losing_trades = [t for t in results['trades'] if not t.win]

        if winning_trades:
            results['avg_win'] = np.mean([t.pnl for t in winning_trades])
        if losing_trades:
            results['avg_loss'] = np.mean([abs(t.pnl) for t in losing_trades])

        if results['avg_loss'] > 0:
            results['profit_factor'] = (results['winning_trades'] * results['avg_win']) / (results['losing_trades'] * results['avg_loss'])

        # Simple Sharpe approximation
        if results['trades']:
            pnls = [t.pnl for t in results['trades']]
            if np.std(pnls) > 0:
                results['sharpe_ratio'] = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized

    # Log veto statistics
    logging.info(f"Quality vetoes: {quality_veto_count}, Regime vetoes: {regime_veto_count}")

    return results

def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    config_type = results.get('config_type', 'unknown')

    print(f"\n{'='*80}")
    print(f"  {config_type} - {results['timeframe']} Results")
    print(f"{'='*80}")
    print(f"Total Bars Analyzed:     {results['total_bars']:,}")
    print(f"Signals Generated:       {results['signals_generated']}")
    print(f"Trades Taken:           {results['trades_taken']}")
    print(f"Winning Trades:         {results['winning_trades']}")
    print(f"Losing Trades:          {results['losing_trades']}")
    print(f"Win Rate:               {results['win_rate']:.1%}")
    print(f"Total PnL:              {results['total_pnl']:.2%}")
    print(f"Max Drawdown:           {results['max_drawdown']:.2%}")
    print(f"Average Win:            {results['avg_win']:.2%}")
    print(f"Average Loss:           {results['avg_loss']:.2%}")
    print(f"Profit Factor:          {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")

    if results['trades']:
        print(f"\n--- Sample Trades ---")
        for i, trade in enumerate(results['trades'][:3], 1):
            pnl_pct = trade.pnl * 100
            win_loss = "WIN" if trade.win else "LOSS"
            print(f"{i:2d}. {trade.direction.upper():5} | "
                  f"{trade.entry_price:7.2f} -> {trade.exit_price:7.2f} | "
                  f"{pnl_pct:+6.2f}% | {win_loss}")

def main():
    """Run comprehensive ETH v1.5.0 optimized backtest."""

    print("\n" + "="*100)
    print("  Bull Machine v1.5.0 OPTIMIZED - ETH Backtest")
    print("  Testing Relaxed Floors + Reduced Alpha Impact + Selective Features")
    print("="*100)

    all_results = []

    # Test each timeframe with both baseline and optimized configs
    for timeframe, file_paths in ETH_DATA_FILES.items():
        print(f"\n\n{'='*60}")
        print(f"  Processing {timeframe} Timeframe")
        print(f"{'='*60}")

        # Find first available data file
        df = pd.DataFrame()
        for file_path in file_paths:
            df = load_eth_data(file_path)
            if not df.empty:
                print(f"Loaded: {file_path}")
                print(f"Data: {len(df)} bars, from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                break

        if df.empty:
            print(f"❌ No data available for {timeframe}")
            continue

        if len(df) < 100:
            print(f"❌ Insufficient data for {timeframe} ({len(df)} bars)")
            continue

        # Run baseline and optimized backtests
        print(f"\n🔄 Running Baseline backtest...")
        baseline_results = run_optimized_fusion_backtest(df, timeframe, "baseline")

        print(f"\n🚀 Running Optimized backtest...")
        if timeframe == "1D":
            optimized_results = run_optimized_fusion_backtest(df, timeframe, "optimized_1d")
        elif timeframe == "4H":
            optimized_results = run_optimized_fusion_backtest(df, timeframe, "optimized_4h")
        else:  # 1H
            optimized_results = run_optimized_fusion_backtest(df, timeframe, "optimized_1h")

        # Store results
        all_results.extend([baseline_results, optimized_results])

        # Print results
        print_backtest_results(baseline_results)
        print_backtest_results(optimized_results)

        # Performance comparison
        if baseline_results['trades_taken'] > 0 and optimized_results['trades_taken'] > 0:
            pnl_improvement = optimized_results['total_pnl'] - baseline_results['total_pnl']
            wr_improvement = optimized_results['win_rate'] - baseline_results['win_rate']
            trade_change = optimized_results['trades_taken'] - baseline_results['trades_taken']

            print(f"\n{'='*80}")
            print(f"  OPTIMIZATION IMPACT - {timeframe}")
            print(f"{'='*80}")
            print(f"PnL Change:             {pnl_improvement:+.2%}")
            print(f"Win Rate Change:        {wr_improvement:+.1%}")
            print(f"Trade Count Change:     {trade_change:+d}")
            print(f"Optimization Success:   {pnl_improvement > 0}")

    # Overall summary
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"  OPTIMIZATION SUMMARY")
        print(f"{'='*100}")

        baseline_results = [r for r in all_results if not r.get('optimized', False)]
        optimized_results = [r for r in all_results if r.get('optimized', False)]

        # Aggregate stats
        total_baseline_trades = sum(r['trades_taken'] for r in baseline_results)
        total_optimized_trades = sum(r['trades_taken'] for r in optimized_results)

        avg_baseline_wr = np.mean([r['win_rate'] for r in baseline_results if r['trades_taken'] > 0])
        avg_optimized_wr = np.mean([r['win_rate'] for r in optimized_results if r['trades_taken'] > 0])

        total_baseline_pnl = sum(r['total_pnl'] for r in baseline_results)
        total_optimized_pnl = sum(r['total_pnl'] for r in optimized_results)

        print(f"Baseline Trades:         {total_baseline_trades}")
        print(f"Optimized Trades:        {total_optimized_trades}")
        print(f"Trade Count Change:      {total_optimized_trades - total_baseline_trades:+d}")
        print(f"Baseline Avg Win Rate:   {avg_baseline_wr:.1%}")
        print(f"Optimized Avg Win Rate:  {avg_optimized_wr:.1%}")
        print(f"Baseline Total PnL:      {total_baseline_pnl:.2%}")
        print(f"Optimized Total PnL:     {total_optimized_pnl:.2%}")

        improvement = total_optimized_pnl - total_baseline_pnl
        print(f"\n🎯 OPTIMIZATION IMPACT:  {improvement:+.2%}")
        print(f"🏆 WINNER: {'Optimized v1.5.0' if improvement > 0 else 'Baseline'}")

        # Check acceptance criteria
        print(f"\n📋 ACCEPTANCE CRITERIA CHECK:")

        # Find ETH 1D results
        eth_1d_results = [r for r in optimized_results if r['timeframe'] == '1D']
        if eth_1d_results:
            eth_1d = eth_1d_results[0]
            trades_per_month = (eth_1d['trades_taken'] / (len(df) / 30)) if len(df) > 30 else 0
            print(f"ETH 1D: {trades_per_month:.1f} trades/mo (target: 2-4), WR: {eth_1d['win_rate']:.1%} (target: ≥50%)")

        # Find ETH 4H results
        eth_4h_results = [r for r in optimized_results if r['timeframe'] == '4H']
        if eth_4h_results:
            eth_4h = eth_4h_results[0]
            trades_per_month = (eth_4h['trades_taken'] / (len(df) / 180)) if len(df) > 180 else 0  # 4H bars per month
            print(f"ETH 4H: {trades_per_month:.1f} trades/mo (target: 2-4), WR: {eth_4h['win_rate']:.1%} (target: ≥45%)")

        # Log results for analysis
        log_telemetry("eth_v150_optimized_results.json", {
            "baseline_results": baseline_results,
            "optimized_results": optimized_results,
            "improvement": improvement,
            "total_trades": total_baseline_trades + total_optimized_trades,
            "optimization_success": improvement > 0
        })

    print(f"\n✅ ETH v1.5.0 Optimized Backtest Complete!")

if __name__ == "__main__":
    main()