#!/usr/bin/env python3
"""
Example: Using Regime Ground Truth Labels

Demonstrates common use cases for the ground truth regime labels:
1. Loading and inspecting the data
2. Mapping timestamps to regimes
3. Analyzing performance by regime
4. Creating regime-based filters
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def load_ground_truth(path: str = "data/regime_ground_truth_2020_2024.json") -> dict:
    """Load ground truth regime labels"""
    with open(path, 'r') as f:
        return json.load(f)


def create_timestamp_regime_map(ground_truth: dict) -> pd.Series:
    """
    Create a pandas Series that maps any timestamp to its regime

    Returns:
        Series indexed by timestamp with regime labels
    """
    monthly = ground_truth['monthly']

    # Build a series with one entry per month
    regime_series = []
    for month_str, regime in monthly.items():
        # Parse YYYY-MM format
        dt = pd.to_datetime(month_str + "-01")
        regime_series.append((dt, regime))

    # Create Series
    df = pd.DataFrame(regime_series, columns=['date', 'regime'])
    df = df.set_index('date')

    # Forward fill to cover all days in each month
    date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    full_df = pd.DataFrame(index=date_range)
    full_df['regime'] = df['regime']
    full_df['regime'] = full_df['regime'].ffill()

    return full_df['regime']


def analyze_by_regime(df: pd.DataFrame, regime_col: str = 'regime', returns_col: str = 'returns'):
    """
    Analyze trading performance by regime

    Args:
        df: DataFrame with regime labels and returns
        regime_col: Name of column containing regime labels
        returns_col: Name of column containing returns
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE BY REGIME")
    print("=" * 80)

    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        regime_df = df[df[regime_col] == regime]

        if len(regime_df) == 0:
            continue

        returns = regime_df[returns_col].dropna()

        if len(returns) == 0:
            print(f"\n{regime.upper():12s}: No data")
            continue

        # Compute stats
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        win_rate = (returns > 0).mean()
        n_trades = len(returns)

        print(f"\n{regime.upper():12s}:")
        print(f"  Trades:     {n_trades:6d}")
        print(f"  Mean Return: {mean_ret:7.2%}")
        print(f"  Std Dev:     {std_ret:7.2%}")
        print(f"  Sharpe:      {sharpe:7.2f}")
        print(f"  Win Rate:    {win_rate:7.1%}")


def example_1_inspect_ground_truth():
    """Example 1: Load and inspect ground truth data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Inspect Ground Truth")
    print("=" * 80)

    gt = load_ground_truth()

    # Show metadata
    print("\nMetadata:")
    print(f"  Version: {gt['metadata']['version']}")
    print(f"  Created: {gt['metadata']['created']}")

    # Show regime definitions
    print("\nRegime Definitions:")
    for regime, desc in gt['metadata']['regimes'].items():
        print(f"  {regime:12s}: {desc}")

    # Show some key periods
    print("\nKey Periods:")
    periods = [
        ("COVID Crash", "2020-03"),
        ("Bull Peak", "2021-11"),
        ("Luna Crash", "2022-05"),
        ("FTX Collapse", "2022-11"),
        ("ETF Rally", "2024-03"),
    ]

    for name, month in periods:
        regime = gt['monthly'].get(month, "N/A")
        event = gt['key_events'].get(month, "")
        print(f"  {name:15s} ({month}): {regime:12s}")
        if event:
            print(f"    → {event}")


def example_2_map_timestamps():
    """Example 2: Map arbitrary timestamps to regimes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Map Timestamps to Regimes")
    print("=" * 80)

    gt = load_ground_truth()
    regime_map = create_timestamp_regime_map(gt)

    # Test some dates
    test_dates = [
        "2020-03-15",  # COVID crash
        "2021-11-10",  # ATH
        "2022-05-15",  # Luna
        "2022-11-11",  # FTX
        "2024-03-14",  # New ATH
    ]

    print("\nTimestamp → Regime Mapping:")
    for date_str in test_dates:
        dt = pd.to_datetime(date_str)
        regime = regime_map.loc[dt]
        print(f"  {date_str}: {regime}")


def example_3_filter_backtest():
    """Example 3: Filter backtest results by regime"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Filter Backtest by Regime")
    print("=" * 80)

    gt = load_ground_truth()

    # Create synthetic backtest data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')

    # Simulate returns (positive for risk_on, negative for risk_off)
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.02

    # Create DataFrame
    backtest_df = pd.DataFrame({
        'returns': returns
    }, index=dates)

    # Map to regimes
    regime_map = create_timestamp_regime_map(gt)
    backtest_df['regime'] = regime_map

    # Analyze
    analyze_by_regime(backtest_df)


def example_4_regime_aware_strategy():
    """Example 4: Implement regime-aware strategy sizing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Regime-Aware Position Sizing")
    print("=" * 80)

    gt = load_ground_truth()
    regime_map = create_timestamp_regime_map(gt)

    # Define position sizing by regime
    regime_sizing = {
        'risk_on': 1.0,      # Full size
        'neutral': 0.5,      # Half size
        'risk_off': 0.25,    # Quarter size
        'crisis': 0.0        # No positions
    }

    print("\nPosition Sizing Rules:")
    for regime, size in regime_sizing.items():
        print(f"  {regime:12s}: {size:5.0%} of base position")

    # Example: Size position for today
    today = pd.Timestamp.now().normalize()

    # Find most recent regime
    available_dates = regime_map[regime_map.index <= today].index
    if len(available_dates) > 0:
        latest_date = available_dates[-1]
        current_regime = regime_map.loc[latest_date]
        position_size = regime_sizing[current_regime]

        print(f"\nCurrent State:")
        print(f"  Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"  Regime: {current_regime}")
        print(f"  Position Size: {position_size:.0%}")

        # Example position sizing
        base_position = 10000  # $10k base
        actual_position = base_position * position_size
        print(f"\n  Base Position: ${base_position:,.0f}")
        print(f"  Actual Position: ${actual_position:,.0f}")


def main():
    """Run all examples"""
    example_1_inspect_ground_truth()
    example_2_map_timestamps()
    example_3_filter_backtest()
    example_4_regime_aware_strategy()

    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
