#!/usr/bin/env python3
"""
Validate Archetype M (Coil Break) with New Coil Features

Tests the improved coil detection logic and compares signal quality
before/after the coil feature implementation.

Usage:
    python bin/validate_archetype_m_coil.py [--period 2023-01-01:2023-03-31]
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.models.archetype_model import RuntimeContext


def create_mock_context(row: pd.Series, row_prev: pd.Series = None) -> RuntimeContext:
    """Create a mock RuntimeContext for testing"""

    # Simple mock that just provides row access
    class MockContext:
        def __init__(self, row_data):
            self.row = row_data
            self.regime_config = {"use_soft_controls": False}

        def get_threshold(self, archetype: str, param: str, default):
            return default

    return MockContext(row)


def analyze_archetype_m_signals(df: pd.DataFrame, period: str = None):
    """
    Analyze Archetype M signal quality with coil features

    Args:
        df: DataFrame with OHLCV and coil features
        period: Optional date range filter (e.g., "2023-01-01:2023-03-31")
    """
    print(f"\n{'='*80}")
    print("ARCHETYPE M (COIL BREAK) VALIDATION")
    print(f"{'='*80}")

    # Filter by period
    if period:
        try:
            start, end = period.split(':')
            df = df.loc[start:end]
            print(f"\nFiltered to period: {start} to {end}")
        except Exception as e:
            print(f"\nWARNING: Invalid period format '{period}': {e}")

    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df):,}")

    # Initialize archetype logic
    config = {}  # Empty config for testing
    adapter = ArchetypeLogic(config)

    # ===========================================================================
    # COIL FEATURE ANALYSIS
    # ===========================================================================
    print(f"\n{'='*80}")
    print("COIL FEATURE STATISTICS")
    print(f"{'='*80}")

    coil_cols = ['tf1h_coil_score', 'tf4h_coil_score', 'tf1d_coil_score']

    for col in coil_cols:
        if col in df.columns:
            series = df[col].dropna()
            print(f"\n{col}:")
            print(f"   Mean: {series.mean():.4f}")
            print(f"   Std: {series.std():.4f}")
            print(f"   Min: {series.min():.4f}")
            print(f"   25%: {series.quantile(0.25):.4f}")
            print(f"   50%: {series.quantile(0.50):.4f}")
            print(f"   75%: {series.quantile(0.75):.4f}")
            print(f"   Max: {series.max():.4f}")

            # High compression periods (score > 0.55)
            high_compression = (series > 0.55).sum()
            high_compression_pct = (high_compression / len(series)) * 100
            print(f"   High compression (>0.55): {high_compression:,} ({high_compression_pct:.1f}%)")

    # Coiling flags
    for col in ['tf1h_is_coiling', 'tf4h_is_coiling', 'tf1d_is_coiling']:
        if col in df.columns:
            coiling_count = df[col].sum()
            coiling_pct = (coiling_count / len(df)) * 100
            print(f"\n{col}: {coiling_count:,} ({coiling_pct:.1f}%)")

    # Breakout events
    print(f"\n{'='*80}")
    print("BREAKOUT EVENTS")
    print(f"{'='*80}")

    for col in ['tf1h_coil_breakout', 'tf4h_coil_breakout', 'tf1d_coil_breakout']:
        if col in df.columns:
            breakout_count = df[col].sum()
            print(f"   {col}: {breakout_count:,}")

    # ===========================================================================
    # ARCHETYPE M SIGNAL DETECTION
    # ===========================================================================
    print(f"\n{'='*80}")
    print("ARCHETYPE M SIGNAL DETECTION")
    print(f"{'='*80}")

    signals = []
    signal_details = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        row_prev = df.iloc[i - 1]

        # Create context
        context = create_mock_context(row, row_prev)

        # Check pattern
        result = adapter._pattern_M(context)

        if result:
            matched, score, tags = result

            signal_info = {
                'timestamp': row.name,
                'close': row.get('close', 0),
                'score': score,
                'tags': tags,
                'coil_score_1h': row.get('tf1h_coil_score', 0),
                'coil_score_4h': row.get('tf4h_coil_score', 0),
                'coil_score_1d': row.get('tf1d_coil_score', 0),
                'is_coiling_4h': row.get('tf4h_is_coiling', False),
                'coil_breakout_4h': row.get('tf4h_coil_breakout', False),
                'tf4h_bos_bullish': row.get('tf4h_bos_bullish', False),
            }

            signals.append(row.name)
            signal_details.append(signal_info)

    print(f"\nTotal Archetype M signals detected: {len(signals)}")

    if len(signals) == 0:
        print("\nNo signals detected. Checking potential reasons:")

        # Debug: Check if coil features exist
        has_coil_features = 'tf4h_coil_score' in df.columns
        print(f"   - Coil features present: {has_coil_features}")

        if has_coil_features:
            # Check high compression + BOS
            high_coil = df['tf4h_coil_score'] > 0.55
            has_bos = df.get('tf4h_bos_bullish', False)

            print(f"   - Periods with high compression (>0.55): {high_coil.sum()}")
            print(f"   - Periods with 4H BOS bullish: {has_bos.sum()}")
            print(f"   - Periods with BOTH: {(high_coil & has_bos).sum()}")

            # Check breakout events with BOS
            if 'tf4h_coil_breakout' in df.columns:
                breakout_with_bos = df['tf4h_coil_breakout'] & has_bos
                print(f"   - Breakout events with BOS: {breakout_with_bos.sum()}")

        return

    # ===========================================================================
    # SIGNAL QUALITY ANALYSIS
    # ===========================================================================
    print(f"\n{'='*80}")
    print("SIGNAL QUALITY ANALYSIS")
    print(f"{'='*80}")

    df_signals = pd.DataFrame(signal_details)

    # Score distribution
    print(f"\nSignal scores:")
    print(f"   Mean: {df_signals['score'].mean():.4f}")
    print(f"   Std: {df_signals['score'].std():.4f}")
    print(f"   Min: {df_signals['score'].min():.4f}")
    print(f"   Max: {df_signals['score'].max():.4f}")

    # Coil intensity distribution
    print(f"\n4H Coil scores at signal time:")
    print(f"   Mean: {df_signals['coil_score_4h'].mean():.4f}")
    print(f"   Min: {df_signals['coil_score_4h'].min():.4f}")
    print(f"   Max: {df_signals['coil_score_4h'].max():.4f}")

    # Multi-TF alignment
    mtf_aligned = (df_signals['coil_score_1h'] > 0.55) & (df_signals['coil_score_4h'] > 0.55) & (df_signals['coil_score_1d'] > 0.55)
    print(f"\nMulti-TF alignment (all 3 TFs coiling): {mtf_aligned.sum()} ({mtf_aligned.sum() / len(df_signals) * 100:.1f}%)")

    # Fallback vs proper detection
    fallback_signals = df_signals['tags'].apply(lambda x: 'fallback' in x).sum()
    print(f"\nFallback (legacy ATR) signals: {fallback_signals}")
    print(f"Proper coil detection signals: {len(df_signals) - fallback_signals}")

    # ===========================================================================
    # SIGNAL LIST (SAMPLE)
    # ===========================================================================
    print(f"\n{'='*80}")
    print("SIGNAL DETAILS (First 20)")
    print(f"{'='*80}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    display_cols = ['timestamp', 'close', 'score', 'coil_score_4h', 'coil_score_1h',
                    'coil_breakout_4h', 'tf4h_bos_bullish']

    print(f"\n{df_signals[display_cols].head(20).to_string()}")

    # ===========================================================================
    # RECOMMENDATIONS
    # ===========================================================================
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    signals_per_year = len(signals) / ((df.index.max() - df.index.min()).days / 365.25)
    print(f"\nSignal frequency: {signals_per_year:.1f} per year")

    if signals_per_year < 5:
        print("   - Consider lowering min_coil_score threshold (currently 0.55)")
        print("   - Or disable require_breakout_flag to capture more coiling states")
    elif signals_per_year > 50:
        print("   - Consider raising min_coil_score threshold for better quality")
        print("   - Or enable require_breakout_flag for stricter detection")
    else:
        print("   - Signal frequency looks reasonable for coil breakout strategy")

    if fallback_signals > 0:
        print(f"\n   - {fallback_signals} signals using fallback logic (no coil features)")
        print("   - This indicates coil_score_4h was 0.0 at those timestamps")

    return df_signals


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Validate Archetype M with coil features')
    parser.add_argument('--period', type=str, help='Date range (YYYY-MM-DD:YYYY-MM-DD)')
    args = parser.parse_args()

    # Load canonical feature store
    canonical_path = PROJECT_ROOT / "data/features_mtf/BTC_1H_CANONICAL_20260202.parquet"

    if not canonical_path.exists():
        print(f"\nERROR: Canonical feature store not found at:")
        print(f"   {canonical_path}")
        return 1

    print(f"\nLoading canonical features from:")
    print(f"   {canonical_path}")

    df = pd.read_parquet(canonical_path)

    # Check for coil features
    coil_features = [c for c in df.columns if 'coil' in c.lower()]
    if not coil_features:
        print("\nERROR: No coil features found in canonical store")
        print("Run: python bin/add_coil_features.py && python bin/merge_coil_features.py")
        return 1

    print(f"   Coil features found: {len(coil_features)}")

    # Run analysis
    df_signals = analyze_archetype_m_signals(df, period=args.period)

    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
