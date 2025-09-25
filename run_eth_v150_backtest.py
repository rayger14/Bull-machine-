#!/usr/bin/env python3
"""
Bull Machine v1.5.0 Enhanced - ETH Comprehensive Backtest
Tests the enhanced fusion engine with new alphas against baseline v1.4.2
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

# Import enhanced v1.5.0 modules
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
        logging.info(f"Raw columns: {list(df.columns)}")

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
        logging.info(f"Processed {len(result_df)} bars")
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

    base_scores = {
        "wyckoff": 0.4 + np.random.normal(0, 0.1),
        "liquidity": 0.35 + np.random.normal(0, 0.08),
        "structure": structure_score,
        "momentum": momentum_score,
        "volume": volume_score,
        "context": 0.45 + np.random.normal(0, 0.07),
        "mtf": 0.5  # Base MTF score
    }

    # Clamp all scores to valid range
    for key in base_scores:
        base_scores[key] = max(0.0, min(1.0, base_scores[key]))

    return base_scores

def run_fusion_backtest(df: pd.DataFrame, timeframe: str, enhanced: bool = False) -> Dict:
    """Run backtest using Bull Machine fusion engine."""

    config_name = "ETH" if enhanced else None
    config = load_config(config_name)

    # Enable v1.5.0 features if enhanced
    if enhanced:
        config["features"] = {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "orderflow_lca": True,
            "negative_vip": True,
            "live_data": False  # Keep disabled
        }

    fusion_engine = create_fusion_engine(config)

    results = {
        'timeframe': timeframe,
        'enhanced': enhanced,
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
    logging.info(f"Running {'Enhanced v1.5.0' if enhanced else 'Baseline v1.4.2'} backtest on {timeframe}")
    logging.info(f"Data: {len(df)} bars, analyzing from bar {start_idx}")
    logging.info(f"{'='*60}")

    for i in range(start_idx, len(df)):
        current_bar = df.iloc[i]
        window_df = df.iloc[max(0, i-50):i+1].copy()

        # Generate layer scores
        layer_scores = create_mock_layer_scores(window_df, enhanced)

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

            if signal != "neutral":
                results['signals_generated'] += 1

            # Simple entry logic
            if signal in ["long", "short"] and confidence > 0.7 and len(active_trades) < 3:
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
                        'enhanced': enhanced
                    }
                )

                active_trades.append(trade)
                results['trades_taken'] += 1

                if results['trades_taken'] <= 10:  # Log first 10 trades
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
            if i < start_idx + 10:  # Log first few errors
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

    return results

def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    enhanced = results.get('enhanced', False)
    system_name = "Enhanced v1.5.0" if enhanced else "Baseline v1.4.2"

    print(f"\n{'='*80}")
    print(f"  {system_name} - {results['timeframe']} Results")
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
        for i, trade in enumerate(results['trades'][:5], 1):
            pnl_pct = trade.pnl * 100
            win_loss = "WIN" if trade.win else "LOSS"
            print(f"{i:2d}. {trade.direction.upper():5} | "
                  f"{trade.entry_price:7.2f} -> {trade.exit_price:7.2f} | "
                  f"{pnl_pct:+6.2f}% | {win_loss}")

def main():
    """Run comprehensive ETH v1.5.0 enhanced backtest."""

    print("\n" + "="*100)
    print("  Bull Machine v1.5.0 Enhanced - ETH Comprehensive Backtest")
    print("  Comparing Baseline v1.4.2 vs Enhanced v1.5.0 with New Alphas")
    print("="*100)

    all_results = []

    # Test each timeframe
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

        # Run both baseline and enhanced backtests
        print(f"\n🔄 Running Baseline v1.4.2 backtest...")
        baseline_results = run_fusion_backtest(df, timeframe, enhanced=False)

        print(f"\n🚀 Running Enhanced v1.5.0 backtest...")
        enhanced_results = run_fusion_backtest(df, timeframe, enhanced=True)

        # Store results
        all_results.extend([baseline_results, enhanced_results])

        # Print results
        print_backtest_results(baseline_results)
        print_backtest_results(enhanced_results)

        # Performance comparison
        if baseline_results['trades_taken'] > 0 and enhanced_results['trades_taken'] > 0:
            pnl_improvement = enhanced_results['total_pnl'] - baseline_results['total_pnl']
            wr_improvement = enhanced_results['win_rate'] - baseline_results['win_rate']

            print(f"\n{'='*80}")
            print(f"  Performance Comparison - {timeframe}")
            print(f"{'='*80}")
            print(f"PnL Improvement:        {pnl_improvement:+.2%}")
            print(f"Win Rate Improvement:   {wr_improvement:+.1%}")
            print(f"Enhanced Advantage:     {pnl_improvement > 0 and wr_improvement > 0}")

    # Overall summary
    if all_results:
        print(f"\n\n{'='*100}")
        print(f"  OVERALL BACKTEST SUMMARY")
        print(f"{'='*100}")

        baseline_results = [r for r in all_results if not r.get('enhanced', False)]
        enhanced_results = [r for r in all_results if r.get('enhanced', False)]

        # Aggregate stats
        total_baseline_trades = sum(r['trades_taken'] for r in baseline_results)
        total_enhanced_trades = sum(r['trades_taken'] for r in enhanced_results)

        avg_baseline_wr = np.mean([r['win_rate'] for r in baseline_results if r['trades_taken'] > 0])
        avg_enhanced_wr = np.mean([r['win_rate'] for r in enhanced_results if r['trades_taken'] > 0])

        total_baseline_pnl = sum(r['total_pnl'] for r in baseline_results)
        total_enhanced_pnl = sum(r['total_pnl'] for r in enhanced_results)

        print(f"Baseline v1.4.2 Trades:    {total_baseline_trades}")
        print(f"Enhanced v1.5.0 Trades:    {total_enhanced_trades}")
        print(f"Baseline Avg Win Rate:     {avg_baseline_wr:.1%}")
        print(f"Enhanced Avg Win Rate:     {avg_enhanced_wr:.1%}")
        print(f"Baseline Total PnL:        {total_baseline_pnl:.2%}")
        print(f"Enhanced Total PnL:        {total_enhanced_pnl:.2%}")

        improvement = total_enhanced_pnl - total_baseline_pnl
        print(f"\n🎯 ENHANCEMENT IMPACT:     {improvement:+.2%}")
        print(f"🏆 WINNER: {'Enhanced v1.5.0' if improvement > 0 else 'Baseline v1.4.2'}")

        # Log results for analysis
        log_telemetry("eth_v150_backtest_results.json", {
            "baseline_results": baseline_results,
            "enhanced_results": enhanced_results,
            "improvement": improvement,
            "total_trades": total_baseline_trades + total_enhanced_trades
        })

    print(f"\n✅ ETH v1.5.0 Enhanced Backtest Complete!")

if __name__ == "__main__":
    main()