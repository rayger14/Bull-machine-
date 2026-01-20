#!/usr/bin/env python3
"""
Extract regime-specific thresholds from winning trades.

This script analyzes backtest results to calculate optimal entry thresholds
for each macro regime (neutral, risk_on, risk_off, crisis).

Usage:
    python bin/extract_thresholds.py
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_parquet_data(path: str) -> pd.DataFrame:
    """Load feature parquet file with regime column."""
    try:
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} rows from {path}")
        print(f"Columns: {df.columns.tolist()}")

        if 'macro_regime' not in df.columns:
            raise ValueError(f"No 'macro_regime' column found in {path}")

        print(f"\nRegime distribution:")
        print(df['macro_regime'].value_counts())

        return df
    except Exception as e:
        print(f"Error loading parquet: {e}")
        raise


def load_backtest_results(bear_path: str, bull_path: str) -> pd.DataFrame:
    """Load and combine backtest results from bear and bull markets."""
    try:
        bear_df = pd.read_csv(bear_path)
        bull_df = pd.read_csv(bull_path)

        print(f"\nLoaded {len(bear_df)} bear trades, {len(bull_df)} bull trades")

        # Combine
        combined = pd.concat([bear_df, bull_df], ignore_index=True)

        # Parse entry_time to datetime
        combined['entry_time'] = pd.to_datetime(combined['entry_time'])

        print(f"Total trades: {len(combined)}")
        print(f"Winning trades: {combined['trade_won'].sum()}")
        print(f"Win rate: {combined['trade_won'].mean():.2%}")

        return combined
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        raise


def map_regime_to_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map regime from one-hot encoded columns to regime string.

    Backtest results have columns: macro_regime_risk_on, macro_regime_neutral, etc.
    We need to convert these to a single 'regime' column.
    """
    regime_cols = [
        'macro_regime_risk_on',
        'macro_regime_neutral',
        'macro_regime_risk_off',
        'macro_regime_crisis'
    ]

    # Check which columns exist
    existing_cols = [col for col in regime_cols if col in trades_df.columns]

    if not existing_cols:
        print("WARNING: No regime columns found in backtest results")
        trades_df['regime'] = 'neutral'  # Default
        return trades_df

    # Convert one-hot to regime string
    def get_regime(row):
        if row.get('macro_regime_risk_on', 0) == 1:
            return 'risk_on'
        elif row.get('macro_regime_neutral', 0) == 1:
            return 'neutral'
        elif row.get('macro_regime_risk_off', 0) == 1:
            return 'risk_off'
        elif row.get('macro_regime_crisis', 0) == 1:
            return 'crisis'
        else:
            return 'neutral'  # Default if no flag set

    trades_df['regime'] = trades_df.apply(get_regime, axis=1)

    print(f"\nTrade regime distribution:")
    print(trades_df['regime'].value_counts())

    return trades_df


def calculate_regime_thresholds(trades_df: pd.DataFrame, regime: str) -> dict:
    """
    Calculate quantile-based thresholds for a specific regime.

    Args:
        trades_df: DataFrame with backtest results
        regime: Regime name ('neutral', 'risk_on', 'risk_off', 'crisis')

    Returns:
        Dictionary of threshold values
    """
    # Filter for regime and winning trades
    regime_winners = trades_df[
        (trades_df['regime'] == regime) &
        (trades_df['trade_won'] == 1)
    ]

    n_winners = len(regime_winners)

    if n_winners == 0:
        print(f"WARNING: No winning trades for regime {regime}")
        return {
            'min_liquidity': 0.0,
            'fusion_threshold': 0.4,
            'volume_z_min': -1.0,
            'funding_z_min': 1.5,
            'sample_size': 0
        }

    print(f"\n{regime.upper()}: {n_winners} winning trades")

    # Calculate quantiles
    thresholds = {}

    # Liquidity score (10th percentile - minimum acceptable)
    if 'entry_liquidity_score' in regime_winners.columns:
        liquidity_values = regime_winners['entry_liquidity_score'].dropna()
        if len(liquidity_values) > 0:
            thresholds['min_liquidity'] = float(np.percentile(liquidity_values, 10))
        else:
            thresholds['min_liquidity'] = 0.0
    else:
        thresholds['min_liquidity'] = 0.0

    # Fusion score (15th percentile)
    if 'entry_fusion_score' in regime_winners.columns:
        fusion_values = regime_winners['entry_fusion_score'].dropna()
        if len(fusion_values) > 0:
            thresholds['fusion_threshold'] = float(np.percentile(fusion_values, 15))
        else:
            thresholds['fusion_threshold'] = 0.4
    else:
        thresholds['fusion_threshold'] = 0.4

    # Volume Z-score (20th percentile)
    if 'volume_zscore' in regime_winners.columns:
        volume_values = regime_winners['volume_zscore'].dropna()
        if len(volume_values) > 0:
            thresholds['volume_z_min'] = float(np.percentile(volume_values, 20))
        else:
            thresholds['volume_z_min'] = -1.0
    else:
        thresholds['volume_z_min'] = -1.0

    # Funding Z-score (80th percentile for shorts - higher is more extreme)
    # Note: backtest results might not have this, so we'll set a default
    thresholds['funding_z_min'] = 1.5  # Conservative default for short entries

    # Add sample size for validation
    thresholds['sample_size'] = n_winners

    print(f"  Liquidity (p10): {thresholds['min_liquidity']:.3f}")
    print(f"  Fusion (p15): {thresholds['fusion_threshold']:.3f}")
    print(f"  Volume Z (p20): {thresholds['volume_z_min']:.3f}")
    print(f"  Funding Z (default): {thresholds['funding_z_min']:.1f}")

    return thresholds


def main():
    """Main execution function."""
    print("=" * 80)
    print("REGIME-SPECIFIC THRESHOLD EXTRACTION")
    print("=" * 80)

    # Paths
    parquet_path = PROJECT_ROOT / "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"
    bear_results = PROJECT_ROOT / "results/validation/bear_2022_separated.csv"
    bull_results = PROJECT_ROOT / "results/validation/bull_2024_separated.csv"
    output_path = PROJECT_ROOT / "configs/auto/hmm_thresholds.json"

    # Verify files exist
    if not parquet_path.exists():
        print(f"ERROR: Parquet file not found: {parquet_path}")
        sys.exit(1)

    if not bear_results.exists():
        print(f"ERROR: Bear results not found: {bear_results}")
        sys.exit(1)

    if not bull_results.exists():
        print(f"ERROR: Bull results not found: {bull_results}")
        sys.exit(1)

    # Load data
    print("\n1. Loading data...")
    parquet_df = load_parquet_data(str(parquet_path))
    trades_df = load_backtest_results(str(bear_results), str(bull_results))

    # Map regimes to trades
    print("\n2. Mapping regimes to trades...")
    trades_df = map_regime_to_trades(trades_df)

    # Calculate thresholds for each regime
    print("\n3. Calculating regime-specific thresholds...")

    regimes = ['neutral', 'risk_on', 'risk_off', 'crisis']
    regime_map = {
        'neutral': 0,
        'risk_on': 1,
        'risk_off': 2,
        'crisis': 3
    }

    thresholds = {}

    for regime in regimes:
        regime_thresholds = calculate_regime_thresholds(trades_df, regime)
        regime_id = regime_map[regime]
        thresholds[str(regime_id)] = {
            'regime_name': regime,
            **regime_thresholds
        }

    # Save to JSON
    print(f"\n4. Saving thresholds to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'version': '1.0',
        'description': 'Regime-specific thresholds extracted from winning trades (2022-2024)',
        'extraction_date': pd.Timestamp.now().isoformat(),
        'total_trades': len(trades_df),
        'total_winners': int(trades_df['trade_won'].sum()),
        'regimes': thresholds
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nThresholds saved to: {output_path}")
    print("\nSummary:")
    for regime_id, data in thresholds.items():
        regime_name = data['regime_name']
        n_samples = data['sample_size']
        fusion = data['fusion_threshold']
        print(f"  Regime {regime_id} ({regime_name}): {n_samples} winners, fusion={fusion:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
