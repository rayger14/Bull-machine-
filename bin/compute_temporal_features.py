#!/usr/bin/env python3
"""
Compute Temporal Fusion Features for Feature Store

Adds temporal confluence features to historical data for backtesting and validation.

Usage:
    python bin/compute_temporal_features.py \
        --input data/features/btc_1h_features.parquet \
        --output data/features/btc_1h_temporal.parquet \
        --config configs/temporal_fusion_config.json

Output Features:
    - temporal_fib_score: Fibonacci time cluster score [0-1]
    - temporal_gann_score: Gann cycle vibration score [0-1]
    - temporal_vol_score: Volatility cycle score [0-1]
    - temporal_emotional_score: Emotional cycle score [0-1]
    - temporal_confluence: Combined temporal score [0-1]
    - bars_since_sc: Bars since Selling Climax
    - bars_since_ar: Bars since Automatic Rally
    - bars_since_st: Bars since Secondary Test
    - bars_since_sos_long: Bars since Sign of Strength
    - bars_since_sos_short: Bars since Sign of Weakness

Author: Bull Machine v2.0 - Temporal Intelligence
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.temporal.temporal_fusion import (
    TemporalFusionEngine,
    compute_temporal_features_batch,
    compute_bars_since_wyckoff_events
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load temporal fusion configuration."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> dict:
    """Get default temporal fusion configuration."""
    return {
        'enabled': True,
        'temporal_weights': {
            'fib_time': 0.40,
            'gann_cycles': 0.30,
            'volatility_cycles': 0.20,
            'emotional_cycles': 0.10
        },
        'temporal_adjustment_range': [0.85, 1.15],
        'fib_levels': [13, 21, 34, 55, 89, 144],
        'gann_vibrations': [3, 7, 9, 12, 21, 36, 45, 72, 90, 144],
        'fib_tolerance_bars': 3,
        'gann_tolerance_bars': 2,
        'vol_compression_threshold': 0.75,
        'vol_expansion_threshold': 1.25,
        'emotional_rsi_thresholds': {
            'extreme_fear': 25,
            'hope_lower': 35,
            'hope_upper': 45,
            'greed': 65,
            'extreme_greed': 75
        }
    }


def validate_input_features(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate that input dataframe has required features.

    Returns:
        (is_valid, missing_features)
    """
    required_features = [
        'close',
        'volume',
        'atr',  # or atr_14
        'rsi',  # or rsi_14
    ]

    optional_features = [
        'wyckoff_sc',
        'wyckoff_ar',
        'wyckoff_st',
        'wyckoff_sos',
        'wyckoff_sow',
        'funding',
        'atr_ma_20'
    ]

    missing = []
    for feat in required_features:
        # Check for alternate names
        if feat == 'atr' and 'atr_14' in df.columns:
            continue
        if feat == 'rsi' and 'rsi_14' in df.columns:
            continue
        if feat not in df.columns:
            missing.append(feat)

    if missing:
        logger.error(f"Missing required features: {missing}")
        return False, missing

    # Warn about missing optional features
    missing_optional = [f for f in optional_features if f not in df.columns]
    if missing_optional:
        logger.warning(f"Missing optional features (will use defaults): {missing_optional}")

    return True, []


def compute_bars_since_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bars_since_* features for Wyckoff events.

    This is a critical step for temporal confluence scoring.
    """
    logger.info("Computing bars_since_* features...")

    # Map of event columns to bars_since columns
    event_mappings = {
        'wyckoff_sc': 'bars_since_sc',
        'wyckoff_ar': 'bars_since_ar',
        'wyckoff_st': 'bars_since_st',
        'wyckoff_sos': 'bars_since_sos_long',
        'wyckoff_sow': 'bars_since_sos_short'
    }

    events_found = []

    for event_col, bars_col in event_mappings.items():
        if event_col in df.columns:
            events_found.append(event_col)

            # Initialize with large value
            df[bars_col] = 999

            # Find all event occurrences
            event_indices = df[df[event_col]].index.tolist()

            if event_indices:
                logger.info(f"  Found {len(event_indices)} {event_col} events")

                # For each bar, compute bars since last event
                for idx in range(len(df)):
                    # Find most recent event before or at this bar
                    recent_events = [e for e in event_indices if e <= idx]
                    if recent_events:
                        last_event_idx = df.index.get_loc(recent_events[-1])
                        df.at[df.index[idx], bars_col] = idx - last_event_idx

    if not events_found:
        logger.warning("No Wyckoff event columns found! Temporal scoring will be limited.")
        logger.warning("Run Wyckoff event detection first for better results.")
    else:
        logger.info(f"Computed bars_since features for: {events_found}")

    return df


def compute_atr_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ATR moving average if not present."""
    if 'atr_ma_20' not in df.columns:
        if 'atr' in df.columns:
            df['atr_ma_20'] = df['atr'].rolling(20).mean()
            logger.info("Computed atr_ma_20 from atr")
        elif 'atr_14' in df.columns:
            df['atr_ma_20'] = df['atr_14'].rolling(20).mean()
            logger.info("Computed atr_ma_20 from atr_14")
    return df


def add_default_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add default values for missing optional features."""
    # Default funding rate to 0 if missing
    if 'funding' not in df.columns:
        df['funding'] = 0.0
        logger.info("Added default funding column (0.0)")

    # Ensure RSI column name
    if 'rsi' not in df.columns and 'rsi_14' in df.columns:
        df['rsi'] = df['rsi_14']

    # Ensure ATR column name
    if 'atr' not in df.columns and 'atr_14' in df.columns:
        df['atr'] = df['atr_14']

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute temporal fusion features for feature store'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input feature file (parquet or csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output file path (parquet or csv)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Temporal fusion config JSON (optional, uses defaults if not provided)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default=None,
        help='Output format (auto-detected from extension if not specified)'
    )

    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading input data: {args.input}")
    if args.input.suffix == '.parquet':
        df = pd.read_parquet(args.input)
    elif args.input.suffix == '.csv':
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    else:
        logger.error(f"Unsupported input format: {args.input.suffix}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Validate input features
    is_valid, missing = validate_input_features(df)
    if not is_valid:
        logger.error("Input validation failed. Aborting.")
        sys.exit(1)

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    logger.info(f"Using temporal weights: {config['temporal_weights']}")

    # Add missing optional features with defaults
    df = add_default_features(df)

    # Compute ATR MA if needed
    df = compute_atr_ma(df)

    # Compute bars_since_* features
    df = compute_bars_since_features(df)

    # Compute temporal fusion features
    logger.info("Computing temporal fusion features...")
    df = compute_temporal_features_batch(df, config, add_to_df=True)

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL FEATURE STATISTICS")
    logger.info("="*60)

    temporal_cols = [
        'temporal_fib_score',
        'temporal_gann_score',
        'temporal_vol_score',
        'temporal_emotional_score',
        'temporal_confluence'
    ]

    for col in temporal_cols:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"\n{col}:")
            logger.info(f"  Mean:   {stats['mean']:.3f}")
            logger.info(f"  Median: {stats['50%']:.3f}")
            logger.info(f"  Min:    {stats['min']:.3f}")
            logger.info(f"  Max:    {stats['max']:.3f}")

    # Confluence distribution
    if 'temporal_confluence' in df.columns:
        confluence = df['temporal_confluence']
        high_conf = (confluence >= 0.70).sum()
        low_conf = (confluence <= 0.30).sum()
        neutral = ((confluence > 0.30) & (confluence < 0.70)).sum()

        logger.info(f"\nTemporal Confluence Distribution:")
        logger.info(f"  High (≥0.70):    {high_conf:6d} bars ({high_conf/len(df)*100:5.2f}%)")
        logger.info(f"  Neutral:         {neutral:6d} bars ({neutral/len(df)*100:5.2f}%)")
        logger.info(f"  Low (≤0.30):     {low_conf:6d} bars ({low_conf/len(df)*100:5.2f}%)")

    # Save output
    output_format = args.format or args.output.suffix.lstrip('.')

    logger.info(f"\nSaving output to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'parquet':
        df.to_parquet(args.output)
    elif output_format == 'csv':
        df.to_csv(args.output)
    else:
        logger.error(f"Unsupported output format: {output_format}")
        sys.exit(1)

    logger.info(f"✓ Successfully computed temporal features for {len(df)} bars")
    logger.info(f"✓ Added {len(temporal_cols)} temporal feature columns")


if __name__ == '__main__':
    main()
