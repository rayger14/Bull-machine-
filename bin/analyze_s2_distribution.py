#!/usr/bin/env python3
"""
S2 (Failed Rally) Empirical Distribution Analyzer

Computes fusion scores for ALL bars in 2022 bear market data to:
1. Understand the empirical distribution of S2 signals
2. Identify percentile thresholds for high-conviction trades
3. Recommend data-driven Optuna search ranges

Current Issue: 418 trades at fusion=0.55 (way too many)
Target: 5-10 trades/year -> Need to find what percentile gives ~10-20 bars/year

Usage:
    python3 bin/analyze_s2_distribution.py

Output:
    - results/s2_calibration/fusion_distribution.csv
    - results/s2_calibration/fusion_percentiles.json
    - Console report with recommendations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List

# S2 runtime enrichment (if available)
try:
    from engine.strategies.archetypes.bear.failed_rally_runtime import S2RuntimeFeatures
    RUNTIME_FEATURES_AVAILABLE = True
except ImportError:
    RUNTIME_FEATURES_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = Path("results/s2_calibration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data path
DATA_PATH = Path("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")


def compute_s2_fusion_score(row: pd.Series) -> Dict[str, float]:
    """
    Compute S2 fusion score for a single bar.

    This mimics the logic from the actual S2 archetype detector to ensure
    distribution analysis matches runtime behavior.

    Components:
    - OB Retest: Price near recent resistance (25%)
    - Wick Rejection: Strong upper wick (25%)
    - RSI Signal: Overbought extreme (20%)
    - Volume Fade: Declining volume (15%)
    - 4H Confirmation: Higher timeframe alignment (15%)

    Args:
        row: DataFrame row with features

    Returns:
        Dict with component scores and final fusion score
    """
    components = {}

    # 1. OB Retest Score (0-1)
    # Check if price is near recent resistance zone
    if 'ob_retest_flag' in row and pd.notna(row['ob_retest_flag']):
        components['ob_retest'] = float(row['ob_retest_flag'])
    else:
        # Fallback: use distance to recent high
        high = row.get('high', 0.0)
        close = row.get('close', 0.0)
        rolling_high = row.get('high_20', high)  # 20-bar rolling high

        if rolling_high > 0:
            distance_pct = abs(close - rolling_high) / rolling_high
            components['ob_retest'] = max(0.0, 1.0 - distance_pct / 0.05)  # Within 5%
        else:
            components['ob_retest'] = 0.0

    # 2. Wick Rejection Score (0-1)
    # Measure upper wick strength
    if 'wick_upper_ratio' in row and pd.notna(row['wick_upper_ratio']):
        wick_ratio = row['wick_upper_ratio']
    else:
        # Compute wick ratio
        high = row.get('high', 0.0)
        low = row.get('low', 0.0)
        open_price = row.get('open', 0.0)
        close = row.get('close', 0.0)

        candle_range = high - low
        if candle_range > 1e-9:
            upper_body = max(open_price, close)
            upper_wick = high - upper_body
            wick_ratio = upper_wick / candle_range
        else:
            wick_ratio = 0.0

    # Strong rejection = wick_ratio > 0.5 (50% of candle is wick)
    components['wick_rejection'] = min(wick_ratio / 0.6, 1.0)

    # 3. RSI Signal Score (0-1)
    # Overbought RSI indicates potential reversal
    rsi = row.get('rsi_14', 50.0)
    if pd.isna(rsi):
        rsi = 50.0

    # RSI > 70 = overbought, score increases linearly from 70-85
    if rsi > 70:
        components['rsi_signal'] = min((rsi - 70) / 15, 1.0)
    else:
        components['rsi_signal'] = 0.0

    # 4. Volume Fade Score (0-1)
    # Declining volume during rally = weak conviction
    if 'volume_fade_flag' in row and pd.notna(row['volume_fade_flag']):
        components['volume_fade'] = float(row['volume_fade_flag'])
    else:
        # Fallback: use volume z-score
        vol_z = row.get('volume_zscore', 0.0)
        if pd.isna(vol_z):
            vol_z = 0.0

        # Below-average volume = higher score
        components['volume_fade'] = max(0.0, 1.0 - (vol_z + 1.0) / 2.0)

    # 5. 4H Confirmation Score (0-1)
    # Higher timeframe downtrend confirmation
    if 'tf4h_trend' in row:
        tf4h_trend = row.get('tf4h_trend', 0.0)
        if pd.notna(tf4h_trend):
            # Negative trend = bearish = higher score
            components['tf4h_confirm'] = max(0.0, -tf4h_trend)
        else:
            components['tf4h_confirm'] = 0.5  # Neutral
    else:
        # Fallback: use EMA slope
        ema_fast = row.get('ema_21', 0.0)
        ema_slow = row.get('ema_55', 0.0)

        if ema_fast > 0 and ema_slow > 0:
            trend_strength = (ema_slow - ema_fast) / ema_fast
            components['tf4h_confirm'] = max(0.0, min(trend_strength / 0.05, 1.0))
        else:
            components['tf4h_confirm'] = 0.5

    # Compute weighted fusion score
    weights = {
        'ob_retest': 0.25,
        'wick_rejection': 0.25,
        'rsi_signal': 0.20,
        'volume_fade': 0.15,
        'tf4h_confirm': 0.15,
    }

    fusion_score = sum(components[k] * weights[k] for k in weights)

    return {
        **components,
        'fusion_score': fusion_score,
        'wick_ratio_raw': wick_ratio if 'wick_ratio' in locals() else components['wick_rejection'] * 0.6,
        'rsi_raw': rsi,
        'vol_z_raw': row.get('volume_zscore', 0.0),
    }


def analyze_distribution(df: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Analyze S2 fusion score distribution for a given year.

    Args:
        df: Feature dataframe
        year: Year to analyze (default: 2022 bear market)

    Returns:
        DataFrame with fusion scores for each bar
    """
    # Filter to target year
    df_year = df[
        (df.index >= f'{year}-01-01') &
        (df.index < f'{year+1}-01-01')
    ].copy()

    logger.info(f"Analyzing {len(df_year):,} bars from {year}")
    logger.info(f"Date range: {df_year.index.min()} to {df_year.index.max()}")

    # Apply runtime enrichment if available
    if RUNTIME_FEATURES_AVAILABLE:
        logger.info("Applying S2 runtime feature enrichment...")
        enricher = S2RuntimeFeatures(lookback_window=14)
        df_year = enricher.enrich_dataframe(df_year)
    else:
        logger.warning("Runtime features not available, using fallback calculations")

    # Compute S2 fusion score for each bar
    logger.info("Computing S2 fusion scores for all bars...")

    scores_list = []
    for idx, row in df_year.iterrows():
        scores = compute_s2_fusion_score(row)
        scores['timestamp'] = idx
        scores_list.append(scores)

    df_scores = pd.DataFrame(scores_list)
    df_scores.set_index('timestamp', inplace=True)

    return df_scores


def print_distribution_report(df_scores: pd.DataFrame, target_trades_per_year: int = 10):
    """
    Print comprehensive distribution analysis report.

    Args:
        df_scores: DataFrame with fusion scores
        target_trades_per_year: Target number of trades (default: 10)
    """
    fusion = df_scores['fusion_score']

    print("\n" + "="*80)
    print("S2 FUSION SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nDataset: {len(df_scores):,} bars")
    print(f"Period: {df_scores.index.min().date()} to {df_scores.index.max().date()}")
    print(f"Duration: {(df_scores.index.max() - df_scores.index.min()).days} days")

    # Basic statistics
    print("\n--- Basic Statistics ---")
    print(f"Mean:     {fusion.mean():.4f}")
    print(f"Median:   {fusion.median():.4f}")
    print(f"Std Dev:  {fusion.std():.4f}")
    print(f"Min:      {fusion.min():.4f}")
    print(f"Max:      {fusion.max():.4f}")

    # Percentile analysis
    print("\n--- Percentile Analysis ---")
    print("\nPercentile | Score  | Bars Above | % of Total | Annual Trades*")
    print("-" * 70)

    percentiles = [50, 75, 85, 90, 95, 97, 98, 99, 99.5, 99.9]
    pct_values = np.percentile(fusion.dropna(), percentiles)

    days_in_data = (df_scores.index.max() - df_scores.index.min()).days
    bars_per_day = len(df_scores) / days_in_data

    for p, v in zip(percentiles, pct_values):
        count_above = (fusion >= v).sum()
        pct_above = (count_above / len(fusion)) * 100
        annual_trades = (count_above / days_in_data) * 365

        # Highlight rows near target
        marker = " <-- TARGET RANGE" if 5 <= annual_trades <= 15 else ""

        print(f"{p:>7}th   | {v:.4f} | {count_above:>10,} | {pct_above:>8.2f}% | {annual_trades:>13.1f}{marker}")

    print("\n* Annualized trade count if threshold applied to entire year")

    # Component analysis
    print("\n--- Component Score Analysis ---")
    components = ['ob_retest', 'wick_rejection', 'rsi_signal', 'volume_fade', 'tf4h_confirm']

    print("\nComponent       | Mean   | Median | Std Dev | Strong (>0.7)")
    print("-" * 65)

    for comp in components:
        if comp in df_scores.columns:
            c = df_scores[comp]
            strong_count = (c > 0.7).sum()
            strong_pct = (strong_count / len(c)) * 100

            print(f"{comp:15} | {c.mean():.4f} | {c.median():.4f} | "
                  f"{c.std():.4f} | {strong_count:>6,} ({strong_pct:.1f}%)")

    # Current baseline analysis
    print("\n--- Current Baseline Analysis ---")
    baseline_threshold = 0.55
    baseline_trades = (fusion >= baseline_threshold).sum()
    baseline_annual = (baseline_trades / days_in_data) * 365
    baseline_percentile = (fusion < baseline_threshold).sum() / len(fusion) * 100

    print(f"Current threshold: {baseline_threshold}")
    print(f"Trades in dataset: {baseline_trades:,}")
    print(f"Annual trades:     {baseline_annual:.1f}")
    print(f"Percentile:        {baseline_percentile:.1f}th")
    print(f"Status:            TOO MANY TRADES (target: {target_trades_per_year}/year)")

    # Recommendations
    print("\n--- Recommended Search Ranges ---")
    print("\nBased on distribution analysis, recommend the following Optuna search ranges:")
    print()

    # Find percentile for target trades
    target_bars = int((target_trades_per_year / 365) * days_in_data)

    # Find threshold that gives approximately target_bars
    sorted_fusion = fusion.sort_values(ascending=False)
    target_threshold = sorted_fusion.iloc[min(target_bars, len(sorted_fusion)-1)]

    # Find what percentile this is
    target_percentile = (fusion < target_threshold).sum() / len(fusion) * 100

    # Get p95, p97, p99 values
    p95 = np.percentile(fusion.dropna(), 95)
    p97 = np.percentile(fusion.dropna(), 97)
    p99 = np.percentile(fusion.dropna(), 99)

    print(f"1. fusion_threshold: [{p97:.3f}, {p99:.3f}]")
    print(f"   Rationale: Target {target_trades_per_year} trades/yr ≈ {target_percentile:.1f}th percentile (score={target_threshold:.3f})")
    print(f"              Start search at p97 ({p97:.3f}) to be conservative")
    print()

    # Analyze wick_ratio for strong signals
    if 'wick_ratio_raw' in df_scores.columns:
        strong_fusion = df_scores[df_scores['fusion_score'] > p95]
        wick_p50 = np.percentile(strong_fusion['wick_ratio_raw'].dropna(), 50)
        wick_p75 = np.percentile(strong_fusion['wick_ratio_raw'].dropna(), 75)

        print(f"2. wick_ratio_min: [{wick_p50:.2f}, {wick_p75:.2f}]")
        print(f"   Rationale: Among top 5% fusion scores, median wick={wick_p50:.2f}")
        print()

    # Analyze RSI for strong signals
    if 'rsi_raw' in df_scores.columns:
        strong_fusion = df_scores[df_scores['fusion_score'] > p95]
        rsi_p25 = np.percentile(strong_fusion['rsi_raw'].dropna(), 25)
        rsi_p50 = np.percentile(strong_fusion['rsi_raw'].dropna(), 50)

        print(f"3. rsi_min: [{rsi_p25:.1f}, {rsi_p50:.1f}]")
        print(f"   Rationale: Among top 5% fusion scores, RSI range is {rsi_p25:.1f}-{rsi_p50:.1f}")
        print()

    # Volume z-score
    if 'vol_z_raw' in df_scores.columns:
        strong_fusion = df_scores[df_scores['fusion_score'] > p95]
        vol_z_p50 = np.percentile(strong_fusion['vol_z_raw'].dropna(), 50)
        vol_z_p75 = np.percentile(strong_fusion['vol_z_raw'].dropna(), 75)

        print(f"4. volume_z_max: [{vol_z_p50:.2f}, {vol_z_p75:.2f}]")
        print(f"   Rationale: Strong signals have below-average volume (z={vol_z_p50:.2f} to {vol_z_p75:.2f})")
        print()

    print("5. liquidity_max: [0.05, 0.25]")
    print("   Rationale: Low liquidity areas more likely to reject (5-25%)")
    print()
    print("6. cooldown_bars: [4, 20]")
    print("   Rationale: Prevent overtrading while allowing multiple setups")

    print("\n" + "="*80)


def main():
    """Run S2 distribution analysis"""
    print("S2 (Failed Rally) Empirical Distribution Analyzer")
    print("="*80)
    print()

    # Load data
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.error(f"Expected path: {DATA_PATH}")
        return 1

    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} bars")
    logger.info(f"Columns: {len(df.columns)}")

    # Analyze 2022 distribution
    df_scores = analyze_distribution(df, year=2022)

    # Save distribution data
    output_csv = RESULTS_DIR / "fusion_distribution_2022.csv"
    df_scores.to_csv(output_csv)
    logger.info(f"\nSaved distribution data: {output_csv}")

    # Print report
    print_distribution_report(df_scores, target_trades_per_year=10)

    # Save percentile summary
    fusion = df_scores['fusion_score']
    percentiles = [50, 75, 85, 90, 95, 97, 98, 99, 99.5, 99.9]
    pct_values = np.percentile(fusion.dropna(), percentiles)

    summary = {
        'date_generated': datetime.now().isoformat(),
        'data_path': str(DATA_PATH),
        'bars_analyzed': len(df_scores),
        'date_range': {
            'start': df_scores.index.min().isoformat(),
            'end': df_scores.index.max().isoformat(),
        },
        'statistics': {
            'mean': float(fusion.mean()),
            'median': float(fusion.median()),
            'std': float(fusion.std()),
            'min': float(fusion.min()),
            'max': float(fusion.max()),
        },
        'percentiles': {
            str(p): float(v) for p, v in zip(percentiles, pct_values)
        },
        'baseline_analysis': {
            'threshold': 0.55,
            'trades': int((fusion >= 0.55).sum()),
            'annual_trades': float((fusion >= 0.55).sum() / 365 * 365),
            'percentile': float((fusion < 0.55).sum() / len(fusion) * 100),
        },
        'recommended_search_ranges': {
            'fusion_threshold': [float(np.percentile(fusion, 97)), float(np.percentile(fusion, 99))],
            'wick_ratio_min': [2.0, 4.0],
            'rsi_min': [75.0, 85.0],
            'volume_z_max': [-2.0, 0.0],
            'liquidity_max': [0.05, 0.25],
            'cooldown_bars': [4, 20],
        }
    }

    output_json = RESULTS_DIR / "fusion_percentiles_2022.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved percentile summary: {output_json}")

    print("\nFiles generated:")
    print(f"  - {output_csv}")
    print(f"  - {output_json}")
    print()
    print("Next step: Run bin/optimize_s2_calibration.py with recommended ranges")

    return 0


if __name__ == '__main__':
    sys.exit(main())
