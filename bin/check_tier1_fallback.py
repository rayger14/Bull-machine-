#!/usr/bin/env python3
"""
STEP 4: Confirm Archetype NOT Falling Back to Tier1

Analyzes backtest logs or trade data to determine if trades originate
from archetype logic vs Tier1 fallback.

Key indicators of CORRECT operation:
- Fusion Score present and non-zero
- Liquidity Score present
- RuntimeContext features populated
- Signal correlation between S1/S4/S5 is LOW (they fire on different events)

Key indicators of FALLBACK (WRONG):
- All trades from generic Tier1 logic
- Fusion scores all zero or missing
- High signal correlation (all systems firing identically)

Usage:
    python bin/check_tier1_fallback.py --test-period 2022-05-01:2022-08-01
    python bin/check_tier1_fallback.py --trades data/trades_s4.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backtesting.backtest_engine import BacktestEngine


def analyze_trade_origin(trades_df: pd.DataFrame) -> dict:
    """
    Analyze trade data to determine origin (archetype vs fallback).

    Returns:
        {
            'total_trades': int,
            'archetype_trades': int,
            'fallback_trades': int,
            'fusion_score_present': bool,
            'avg_fusion_score': float,
            'liquidity_score_present': bool
        }
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'archetype_trades': 0,
            'fallback_trades': 0,
            'fusion_score_present': False,
            'avg_fusion_score': 0.0,
            'liquidity_score_present': False
        }

    total = len(trades_df)

    # Check for fusion scores
    fusion_present = 'fusion_score' in trades_df.columns
    fusion_nonzero = 0
    avg_fusion = 0.0

    if fusion_present:
        fusion_scores = trades_df['fusion_score'].dropna()
        fusion_nonzero = (fusion_scores > 0).sum()
        avg_fusion = fusion_scores.mean() if len(fusion_scores) > 0 else 0.0

    # Check for liquidity scores
    liquidity_present = 'liquidity_score' in trades_df.columns

    # Trades with valid fusion scores are archetype trades
    archetype_trades = fusion_nonzero if fusion_present else 0

    # Everything else is fallback
    fallback_trades = total - archetype_trades

    return {
        'total_trades': total,
        'archetype_trades': archetype_trades,
        'fallback_trades': fallback_trades,
        'fusion_score_present': fusion_present,
        'avg_fusion_score': avg_fusion,
        'liquidity_score_present': liquidity_present
    }


def run_backtest_and_analyze(
    config_path: Path,
    start_date: str,
    end_date: str
) -> dict:
    """
    Run backtest and analyze trade origins.
    """
    try:
        # Load config
        import json
        with open(config_path) as f:
            config = json.load(f)

        # Run backtest
        print(f"Running backtest: {start_date} to {end_date}...")

        engine = BacktestEngine(config)

        # Check if data exists
        data_path = Path(config.get('data_path', 'data/features_1h.parquet'))
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            return None

        # Load data
        df = pd.read_parquet(data_path)

        # Filter to test period
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        if len(df) == 0:
            print(f"Error: No data in period {start_date} to {end_date}")
            return None

        # Run backtest (simplified)
        # In reality, would call engine.run() and extract trades
        # For now, return placeholder
        print("Warning: Backtest execution stub - implement full backtest")

        trades_df = pd.DataFrame()  # Placeholder

        return analyze_trade_origin(trades_df)

    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Check for Tier1 fallback behavior"
    )
    parser.add_argument(
        '--test-period',
        type=str,
        help='Test period (format: YYYY-MM-DD:YYYY-MM-DD)'
    )
    parser.add_argument(
        '--trades',
        type=str,
        help='Path to trades CSV file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/s4_optimized_oos_test.json',
        help='Config file to test'
    )

    args = parser.parse_args()

    if args.trades:
        # Analyze existing trades file
        trades_path = Path(args.trades)
        if not trades_path.exists():
            print(f"Error: Trades file not found: {trades_path}")
            return 1

        trades_df = pd.read_csv(trades_path)
        result = analyze_trade_origin(trades_df)

    elif args.test_period:
        # Run backtest for period
        start, end = args.test_period.split(':')

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config not found: {config_path}")
            return 1

        result = run_backtest_and_analyze(config_path, start, end)

    else:
        print("Error: Must specify --test-period or --trades")
        return 1

    if result is None:
        print("Analysis failed")
        return 1

    # Print results
    print("\n" + "="*50)
    print("TIER1 FALLBACK ANALYSIS")
    print("="*50)

    total = result['total_trades']
    archetype = result['archetype_trades']
    fallback = result['fallback_trades']

    print(f"\nTotal Trades:     {total}")
    print(f"Archetype Trades: {archetype} ({100*archetype/total if total > 0 else 0:.1f}%)")
    print(f"Fallback Trades:  {fallback} ({100*fallback/total if total > 0 else 0:.1f}%)")

    print(f"\nFusion Score Present:    {result['fusion_score_present']}")
    print(f"Avg Fusion Score:        {result['avg_fusion_score']:.3f}")
    print(f"Liquidity Score Present: {result['liquidity_score_present']}")

    # Determine pass/fail
    fallback_pct = 100 * fallback / total if total > 0 else 100

    print("\n" + "="*50)

    if fallback_pct < 30 and result['fusion_score_present']:
        print("\033[0;32m✓ PASS\033[0m: Archetype logic dominating")
        print(f"Fallback: {fallback_pct:.1f}%")
        return 0
    else:
        print("\033[0;31m✗ FAIL\033[0m: Too much fallback behavior")
        print(f"Fallback: {fallback_pct:.1f}%")
        print("\nPossible causes:")
        print("  - Feature access issues (check Steps 1-2)")
        print("  - Thresholds too strict (no signals generated)")
        print("  - Domain engines not activated (check Step 3)")
        return 1


if __name__ == '__main__':
    exit(main())
