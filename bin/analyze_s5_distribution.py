#!/usr/bin/env python3
"""
S5 (Long Squeeze) Fusion Score Distribution Analyzer

Analyzes the distribution of S5 fusion scores across 2022-2024 data to inform
Optuna optimization search ranges. Critical for setting realistic thresholds.

ARCHITECTURE:
- Loads historical feature data (2022-2024)
- Applies S5 runtime enrichment to compute fusion scores
- Outputs percentile distribution (p50-p99.9)
- Recommends Optuna search ranges based on target trade frequency

TARGET METRICS:
- Trade frequency: 7-12 trades/year
- Percentile range: Typically p97-p99.5 for rare squeeze events
- Expected fusion scores: 0.5-0.8 for high-conviction signals

OUTPUTS:
1. Percentile distribution table
2. Recommended Optuna search ranges
3. CSV export with all fusion scores for further analysis

Author: Claude Code (Backend Architect)
Date: 2025-11-20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Tuple, List

from engine.strategies.archetypes.bear.long_squeeze_runtime import apply_s5_enrichment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_feature_data(start_date: str = '2022-01-01', end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Load feature store data for analysis.

    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV + features
    """
    feature_file = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature store file not found: {feature_file}\n"
            "Run bin/feature_store.py to generate features."
        )

    logger.info(f"Loading feature data from {feature_file}")
    df = pd.read_parquet(feature_file)

    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    logger.info(f"Features available: {len(df.columns)} columns")

    return df


def analyze_fusion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute S5 fusion scores and analyze distribution.

    Args:
        df: Feature dataframe

    Returns:
        DataFrame with fusion scores and component features
    """
    logger.info("Applying S5 runtime enrichment...")

    # Apply S5 enrichment
    df_enriched = apply_s5_enrichment(
        df,
        funding_lookback=24,
        oi_lookback=12,
        rsi_threshold=70.0
    )

    # Extract relevant columns for analysis
    analysis_cols = [
        's5_fusion_score',
        'funding_z_score',
        'oi_change',
        'rsi_overbought',
        'liquidity_score'
    ]

    available_cols = [col for col in analysis_cols if col in df_enriched.columns]
    df_analysis = df_enriched[available_cols].copy()

    logger.info(f"Computed fusion scores for {len(df_analysis)} bars")

    return df_analysis


def compute_percentiles(scores: pd.Series, percentiles: list = None) -> pd.DataFrame:
    """
    Compute percentile distribution of fusion scores.

    Args:
        scores: Series of fusion scores
        percentiles: List of percentiles to compute (default: 50-99.9)

    Returns:
        DataFrame with percentile analysis
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 97, 98, 99, 99.5, 99.7, 99.9]

    results = []

    for p in percentiles:
        val = np.percentile(scores.dropna(), p)
        count_above = (scores > val).sum()
        pct_above = count_above / len(scores) * 100

        results.append({
            'percentile': f'p{p}',
            'percentile_value': p,
            'fusion_threshold': val,
            'bars_above': count_above,
            'pct_above': pct_above,
            'trades_per_year': count_above / 3.0  # Approx 3 years of data
        })

    return pd.DataFrame(results)


def recommend_search_ranges(percentile_df: pd.DataFrame, target_trades_per_year: Tuple[int, int] = (7, 12)) -> Dict:
    """
    Recommend Optuna search ranges based on target trade frequency.

    Args:
        percentile_df: Percentile distribution dataframe
        target_trades_per_year: Target range for trades per year

    Returns:
        Dictionary with recommended search ranges
    """
    # Find percentiles that yield target trade frequency
    target_min, target_max = target_trades_per_year

    # Find bounds
    lower_percentile = percentile_df[percentile_df['trades_per_year'] <= target_max].iloc[-1]
    upper_percentile = percentile_df[percentile_df['trades_per_year'] >= target_min].iloc[0]

    recommendations = {
        'fusion_threshold': {
            'min': lower_percentile['fusion_threshold'],
            'max': upper_percentile['fusion_threshold'],
            'recommended_start': np.percentile(
                percentile_df['fusion_threshold'].values,
                75  # Start at median of range
            )
        },
        'funding_z_min': {
            'min': 1.0,
            'max': 3.0,
            'comment': 'Positive funding > 1σ indicates long bias, > 2σ extreme'
        },
        'rsi_min': {
            'min': 70,
            'max': 85,
            'comment': 'RSI overbought threshold, higher = more selective'
        },
        'liquidity_max': {
            'min': 0.05,
            'max': 0.25,
            'comment': 'Low liquidity amplifies squeeze risk, lower = more selective'
        },
        'oi_change_min': {
            'min': 0.05,
            'max': 0.20,
            'comment': 'Rising OI threshold (if available), higher = more selective'
        },
        'cooldown_bars': {
            'min': 4,
            'max': 20,
            'comment': 'Prevent overtrading after signals'
        }
    }

    return recommendations


def main():
    """Main analysis routine"""

    print("="*80)
    print("S5 (LONG SQUEEZE) FUSION SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    print()

    try:
        # Load feature data
        df = load_feature_data(start_date='2022-01-01', end_date='2024-12-31')

        # Analyze fusion distribution
        df_analysis = analyze_fusion_distribution(df)

        # Get fusion scores
        fusion_scores = df_analysis['s5_fusion_score']

        # Basic statistics
        print("\n" + "-"*80)
        print("BASIC STATISTICS")
        print("-"*80)
        print(f"Total bars: {len(fusion_scores)}")
        print(f"Valid scores: {fusion_scores.notna().sum()}")
        print(f"Mean fusion score: {fusion_scores.mean():.4f}")
        print(f"Std dev: {fusion_scores.std():.4f}")
        print(f"Min: {fusion_scores.min():.4f}")
        print(f"Max: {fusion_scores.max():.4f}")

        # Compute percentiles
        print("\n" + "-"*80)
        print("PERCENTILE DISTRIBUTION")
        print("-"*80)

        percentile_df = compute_percentiles(fusion_scores)
        print()
        print(percentile_df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if x < 100 else f'{x:.1f}'))

        # Recommend search ranges
        print("\n" + "-"*80)
        print("RECOMMENDED OPTUNA SEARCH RANGES")
        print("-"*80)

        recommendations = recommend_search_ranges(percentile_df, target_trades_per_year=(7, 12))

        print("\nFor target: 7-12 trades/year")
        print()
        for param, config in recommendations.items():
            if 'comment' in config:
                print(f"{param:20s}: [{config['min']:.4f}, {config['max']:.4f}]  # {config['comment']}")
            else:
                print(f"{param:20s}: [{config['min']:.4f}, {config['max']:.4f}]")
                if 'recommended_start' in config:
                    print(f"{'':20s}  (recommended start: {config['recommended_start']:.4f})")

        # Component feature statistics
        print("\n" + "-"*80)
        print("COMPONENT FEATURE STATISTICS")
        print("-"*80)

        print("\nFunding Z-Score:")
        funding_z = df_analysis['funding_z_score']
        print(f"  Mean: {funding_z.mean():.2f}")
        print(f"  p95: {np.percentile(funding_z.dropna(), 95):.2f}")
        print(f"  p99: {np.percentile(funding_z.dropna(), 99):.2f}")
        print(f"  Extreme positive (>2σ): {(funding_z > 2.0).sum()} bars ({(funding_z > 2.0).sum()/len(funding_z)*100:.1f}%)")

        print("\nOI Change:")
        oi_change = df_analysis['oi_change']
        if oi_change.abs().sum() > 0:  # Check if OI data available
            print(f"  Mean: {oi_change.mean():.3f}")
            print(f"  p95: {np.percentile(oi_change.dropna(), 95):.3f}")
            print(f"  Rising >10%: {(oi_change > 0.10).sum()} bars ({(oi_change > 0.10).sum()/len(oi_change)*100:.1f}%)")
        else:
            print("  [WARNING] OI data not available - will use fallback in optimization")

        print("\nRSI Overbought:")
        rsi_ob = df_analysis['rsi_overbought']
        print(f"  Overbought bars: {rsi_ob.sum()} ({rsi_ob.sum()/len(rsi_ob)*100:.1f}%)")

        print("\nLiquidity Score:")
        liq = df_analysis['liquidity_score']
        print(f"  Mean: {liq.mean():.3f}")
        print(f"  Low liquidity (<0.25): {(liq < 0.25).sum()} bars ({(liq < 0.25).sum()/len(liq)*100:.1f}%)")

        # Save results
        output_dir = Path('results/optimization')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full analysis
        output_file = output_dir / 's5_fusion_distribution.csv'
        df_analysis.to_csv(output_file)
        logger.info(f"\nFull analysis saved to: {output_file}")

        # Save percentile summary
        percentile_file = output_dir / 's5_percentile_distribution.csv'
        percentile_df.to_csv(percentile_file, index=False)
        logger.info(f"Percentile distribution saved to: {percentile_file}")

        # Save recommendations as JSON
        import json
        rec_file = output_dir / 's5_optuna_search_ranges.json'
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        logger.info(f"Search ranges saved to: {rec_file}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nNext step: Run bin/optimize_s5_calibration.py with recommended ranges")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
