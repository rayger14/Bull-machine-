#!/usr/bin/env python3
"""
What-If Replay Script for Max-Hold Analysis

Quantifies foregone profits from capping winners early by replaying
max_hold exits with extended holding periods.

Usage:
    python3 bin/what_if_max_hold.py \
        --trades_csv reports/optuna_results/BTC_v3_full_year_trades.csv \
        --prices_parquet data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet \
        --extra_hours 96 \
        --out reports/what_if_max_hold_BTC_+96h.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_trades(path: Path) -> pd.DataFrame:
    """Load trades CSV and validate required columns."""
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])

    needed = {"entry_time", "exit_time", "direction", "entry_price",
              "exit_price", "exit_reason", "position_size"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Trades CSV missing columns: {missing}")

    return df


def load_prices(path: Path) -> pd.DataFrame:
    """Load price parquet and ensure datetime index."""
    df = pd.read_parquet(path)

    if "close" not in df.columns:
        raise ValueError("Price parquet must contain a 'close' column.")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"]))
        else:
            raise ValueError("Parquet must have datetime index or a 'time' column.")

    df = df.sort_index()
    return df[["close"]]


def pnl(direction: str, entry_price: float, exit_price: float, size_dollars: float) -> float:
    """Calculate PNL for a trade."""
    ret = (exit_price - entry_price) / entry_price if direction == "long" \
          else (entry_price - exit_price) / entry_price
    return ret * size_dollars


def what_if_extend(trades: pd.DataFrame, prices: pd.DataFrame, extra_hours: int) -> pd.DataFrame:
    """
    Extend max_hold trades by extra_hours and compute hypothetical PNL.

    For non-max_hold trades, keep original exit.
    For max_hold trades, extend holding period and recompute PNL.
    """
    trades = trades.copy()
    new_rows = []

    for _, t in trades.iterrows():
        # If not a max_hold exit, keep original
        if str(t.exit_reason).lower() != "max_hold":
            new_rows.append({
                **t,
                "what_if_exit_price": t.exit_price,
                "what_if_exit_time": t.exit_time,
                "what_if_pnl": t.pnl
            })
            continue

        # Extend the exit time by extra_hours
        later_time = t.exit_time + pd.Timedelta(hours=extra_hours)
        later_time = min(later_time, prices.index.max())

        # Find the closest bar using backfill
        later_bar_idx = prices.index.get_indexer([later_time], method="backfill")[0]
        later_bar = prices.index[later_bar_idx]
        later_price = float(prices.loc[later_bar, "close"])

        # Compute hypothetical PNL
        size = float(t.position_size)
        hyp_pnl = pnl(str(t.direction).lower(), float(t.entry_price), later_price, size)

        new_rows.append({
            **t,
            "what_if_exit_price": later_price,
            "what_if_exit_time": later_bar,
            "what_if_pnl": hyp_pnl
        })

    return pd.DataFrame(new_rows)


def analyze_results(df: pd.DataFrame, extra_hours: int) -> dict:
    """Generate summary statistics for what-if analysis."""
    max_hold_trades = df[df['exit_reason'].str.lower() == 'max_hold'].copy()

    if len(max_hold_trades) == 0:
        return {
            'extra_hours': extra_hours,
            'max_hold_count': 0,
            'total_delta_pnl': 0.0,
            'avg_delta_pnl': 0.0,
            'winners_extended': 0,
            'losers_extended': 0,
            'trades': []
        }

    # Calculate delta PNL for each max_hold trade
    max_hold_trades['delta_pnl'] = max_hold_trades['what_if_pnl'] - max_hold_trades['pnl']

    total_delta = max_hold_trades['delta_pnl'].sum()
    avg_delta = max_hold_trades['delta_pnl'].mean()

    # Count winners vs losers
    winners = (max_hold_trades['delta_pnl'] > 0).sum()
    losers = (max_hold_trades['delta_pnl'] < 0).sum()

    # Build detailed trade list
    trades_detail = []
    for _, t in max_hold_trades.iterrows():
        trades_detail.append({
            'entry_time': str(t.entry_time),
            'exit_time': str(t.exit_time),
            'direction': t.direction,
            'entry_price': float(t.entry_price),
            'original_exit_price': float(t.exit_price),
            'original_pnl': float(t.pnl),
            'what_if_exit_time': str(t.what_if_exit_time),
            'what_if_exit_price': float(t.what_if_exit_price),
            'what_if_pnl': float(t.what_if_pnl),
            'delta_pnl': float(t.delta_pnl)
        })

    return {
        'extra_hours': extra_hours,
        'max_hold_count': len(max_hold_trades),
        'total_delta_pnl': float(total_delta),
        'avg_delta_pnl': float(avg_delta),
        'winners_extended': int(winners),
        'losers_extended': int(losers),
        'trades': trades_detail
    }


def main():
    parser = argparse.ArgumentParser(description='What-if analysis for max_hold exits')
    parser.add_argument('--trades_csv', type=str, required=True,
                       help='Path to trades CSV file')
    parser.add_argument('--prices_parquet', type=str, required=True,
                       help='Path to price data parquet')
    parser.add_argument('--extra_hours', type=int, required=True,
                       help='Number of hours to extend max_hold trades')
    parser.add_argument('--out', type=str, required=True,
                       help='Output JSON file path')

    args = parser.parse_args()

    print("=" * 80)
    print("MAX-HOLD WHAT-IF ANALYSIS")
    print("=" * 80)
    print(f"Trades CSV: {args.trades_csv}")
    print(f"Price Data: {args.prices_parquet}")
    print(f"Extension: +{args.extra_hours} hours")
    print()

    # Load data
    print("Loading trades and price data...")
    trades = load_trades(Path(args.trades_csv))
    prices = load_prices(Path(args.prices_parquet))

    print(f"Loaded {len(trades)} trades")
    print(f"Price data: {prices.index[0]} to {prices.index[-1]}")
    print()

    # Run what-if analysis
    print("Running what-if replay...")
    extended_trades = what_if_extend(trades, prices, args.extra_hours)

    # Analyze results
    results = analyze_results(extended_trades, args.extra_hours)

    # Display summary
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Max-hold trades found: {results['max_hold_count']}")
    print(f"Total delta PNL: ${results['total_delta_pnl']:,.2f}")
    print(f"Average delta PNL: ${results['avg_delta_pnl']:,.2f}")
    print(f"Better outcomes: {results['winners_extended']}")
    print(f"Worse outcomes: {results['losers_extended']}")
    print()

    # Save to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
