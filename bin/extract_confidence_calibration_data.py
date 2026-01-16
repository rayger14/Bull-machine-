#!/usr/bin/env python3
"""
Extract Confidence Calibration Data

Purpose: Generate training data for confidence calibration by running ensemble
predictions over historical data and computing forward outcomes.

Process:
1. Load historical features (2018-2024)
2. Run ensemble regime detection (batch mode)
3. Calculate forward outcomes for each bar:
   - Forward returns (1h, 24h)
   - Forward volatility (1h, 24h)
   - Regime stability (did regime stay same?)
   - Transition events
4. Save calibration dataset for training

Output: data/confidence_calibration_data.parquet

Author: Claude Code
Date: 2026-01-15
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context.regime_service import RegimeService, REGIME_MODE_DYNAMIC_ENSEMBLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_features() -> pd.DataFrame:
    """Load complete feature dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'features_2018_2024_complete.parquet'

    if not data_path.exists():
        raise FileNotFoundError(f"Feature data not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"✅ Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    return df


def run_ensemble_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ensemble predictions to get raw confidence scores.

    We disable hysteresis and event override to get raw ensemble predictions.
    This gives us the true ensemble agreement scores before smoothing.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Running Ensemble Predictions")
    logger.info(f"{'='*80}")

    model_path = Path(__file__).parent.parent / 'models' / 'ensemble_regime_v1.pkl'

    # Initialize RegimeService with raw ensemble (no smoothing)
    service = RegimeService(
        mode=REGIME_MODE_DYNAMIC_ENSEMBLE,
        model_path=str(model_path),
        enable_event_override=False,  # Get raw ensemble predictions
        enable_hysteresis=False,      # No smoothing
        enable_ema_smoothing=False    # No EMA smoothing
    )

    logger.info("✓ RegimeService initialized (raw mode)")
    logger.info("  - Event Override: DISABLED (raw predictions)")
    logger.info("  - Hysteresis: DISABLED (no smoothing)")
    logger.info("  - EMA Smoothing: DISABLED")

    # Classify all bars in batch mode
    logger.info(f"\nClassifying {len(df):,} bars...")
    results = service.classify_batch(df)

    logger.info("\n✓ Classification complete")
    logger.info(f"  Mean confidence: {results['regime_confidence'].mean():.3f}")
    logger.info(f"  Median confidence: {results['regime_confidence'].median():.3f}")
    logger.info(f"  Min confidence: {results['regime_confidence'].min():.3f}")
    logger.info(f"  Max confidence: {results['regime_confidence'].max():.3f}")

    # Log regime distribution
    logger.info("\nRegime distribution:")
    for regime, count in results['regime_label'].value_counts().items():
        pct = count / len(results) * 100
        logger.info(f"  {regime:12s}: {count:6,} ({pct:5.1f}%)")

    return results


def calculate_forward_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward-looking outcomes for each bar.

    These outcomes will be used to train the confidence calibrator.
    We measure what actually happened after each prediction.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Calculating Forward Outcomes")
    logger.info(f"{'='*80}")

    # Ensure we have close price
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    # 1. Forward returns (shifted backward so we can see future)
    logger.info("\n1. Computing forward returns...")
    df['forward_return_1h'] = df['close'].pct_change(1).shift(-1)
    df['forward_return_24h'] = df['close'].pct_change(24).shift(-24)

    logger.info(f"   ✓ Forward 1h return: mean={df['forward_return_1h'].mean()*100:.4f}%, "
                f"std={df['forward_return_1h'].std()*100:.3f}%")
    logger.info(f"   ✓ Forward 24h return: mean={df['forward_return_24h'].mean()*100:.3f}%, "
                f"std={df['forward_return_24h'].std()*100:.3f}%")

    # 2. Forward volatility (rolling std of returns over next N hours)
    logger.info("\n2. Computing forward volatility...")

    # Calculate hourly returns
    hourly_returns = df['close'].pct_change()

    # Forward volatility = std of next 24h of returns
    def forward_volatility(series, horizon=24):
        """Calculate volatility over next horizon bars."""
        result = []
        for i in range(len(series)):
            if i + horizon >= len(series):
                result.append(np.nan)
            else:
                window = series.iloc[i:i+horizon]
                result.append(window.std() * np.sqrt(horizon))
        return pd.Series(result, index=series.index)

    df['forward_volatility_1h'] = hourly_returns.shift(-1).abs()  # Next 1h abs return
    df['forward_volatility_24h'] = forward_volatility(hourly_returns, horizon=24)

    logger.info(f"   ✓ Forward 1h volatility: mean={df['forward_volatility_1h'].mean()*100:.3f}%")
    logger.info(f"   ✓ Forward 24h volatility: mean={df['forward_volatility_24h'].mean()*100:.3f}%")

    # 3. Forward max drawdown (worst drawdown over next 24h)
    logger.info("\n3. Computing forward max drawdown...")

    def forward_max_drawdown(prices, horizon=24):
        """Calculate max drawdown over next horizon bars."""
        result = []
        for i in range(len(prices)):
            if i + horizon >= len(prices):
                result.append(np.nan)
            else:
                window = prices.iloc[i:i+horizon]
                peak = window.iloc[0]
                max_dd = 0
                for price in window:
                    if price > peak:
                        peak = price
                    dd = (price - peak) / peak
                    max_dd = min(max_dd, dd)
                result.append(max_dd)
        return pd.Series(result, index=prices.index)

    df['forward_max_dd_24h'] = forward_max_drawdown(df['close'], horizon=24)

    logger.info(f"   ✓ Forward 24h max DD: mean={df['forward_max_dd_24h'].mean()*100:.2f}%, "
                f"worst={df['forward_max_dd_24h'].min()*100:.2f}%")

    # 4. Regime stability (did regime stay the same?)
    logger.info("\n4. Computing regime stability...")

    df['regime_stable_1h'] = (df['regime_label'] == df['regime_label'].shift(-1)).astype(float)
    df['regime_stable_24h'] = (df['regime_label'] == df['regime_label'].shift(-24)).astype(float)

    stability_1h = df['regime_stable_1h'].mean()
    stability_24h = df['regime_stable_24h'].mean()

    logger.info(f"   ✓ Regime stability 1h: {stability_1h*100:.1f}% (bars with same regime)")
    logger.info(f"   ✓ Regime stability 24h: {stability_24h*100:.1f}% (bars with same regime)")

    # 5. Transition events (did regime change in next N hours?)
    logger.info("\n5. Computing transition events...")

    df['transition_next_1h'] = (df['regime_label'] != df['regime_label'].shift(-1)).astype(float)

    # For 24h transitions, check if regime at t+24 differs from regime at t
    # (simpler and avoids rolling on string columns)
    df['transition_next_24h'] = (df['regime_label'] != df['regime_label'].shift(-24)).astype(float)

    transitions_1h = df['transition_next_1h'].sum()
    transitions_24h = df['transition_next_24h'].sum()

    logger.info(f"   ✓ Transitions in next 1h: {transitions_1h:,} ({transitions_1h/len(df)*100:.2f}% of bars)")
    logger.info(f"   ✓ Transitions in next 24h: {transitions_24h:,} ({transitions_24h/len(df)*100:.2f}% of bars)")

    # 6. Outcome quality score (composite metric)
    logger.info("\n6. Computing outcome quality score...")

    # Normalize metrics to [0, 1] range
    # Good outcome = high return, low volatility, high stability
    return_score = (df['forward_return_24h'] - df['forward_return_24h'].mean()) / df['forward_return_24h'].std()
    return_score = (return_score + 3) / 6  # Clip to ~[0, 1]
    return_score = return_score.clip(0, 1)

    vol_score = 1 - (df['forward_volatility_24h'] - df['forward_volatility_24h'].min()) / (
        df['forward_volatility_24h'].max() - df['forward_volatility_24h'].min()
    )

    stability_score = df['regime_stable_24h']

    # Weighted composite (return 40%, volatility 30%, stability 30%)
    df['outcome_quality_score'] = (
        0.4 * return_score +
        0.3 * vol_score +
        0.3 * stability_score
    )

    logger.info(f"   ✓ Outcome quality score: mean={df['outcome_quality_score'].mean():.3f}, "
                f"std={df['outcome_quality_score'].std():.3f}")

    return df


def prepare_calibration_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final calibration dataset with relevant features.

    This dataset will be used to train the confidence calibrator.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Preparing Calibration Dataset")
    logger.info(f"{'='*80}")

    # Select relevant columns
    calibration_cols = [
        # Ensemble predictions
        'regime_label',
        'regime_confidence',  # This is ensemble_agreement (raw)
        'regime_source',

        # Forward outcomes (targets for calibration)
        'forward_return_1h',
        'forward_return_24h',
        'forward_volatility_1h',
        'forward_volatility_24h',
        'forward_max_dd_24h',
        'regime_stable_1h',
        'regime_stable_24h',
        'transition_next_1h',
        'transition_next_24h',
        'outcome_quality_score',

        # Keep close for reference
        'close'
    ]

    # Check which columns exist
    available_cols = [col for col in calibration_cols if col in df.columns]
    missing_cols = [col for col in calibration_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"⚠️  Missing columns: {missing_cols}")

    calibration_df = df[available_cols].copy()

    # Drop rows with NaN in forward outcomes (last 24 hours won't have forward data)
    initial_len = len(calibration_df)
    calibration_df = calibration_df.dropna(subset=[
        'forward_return_24h',
        'forward_volatility_24h',
        'forward_max_dd_24h',
        'regime_stable_24h'
    ])
    dropped = initial_len - len(calibration_df)

    logger.info(f"\n✓ Dataset prepared")
    logger.info(f"  Total bars: {len(calibration_df):,}")
    logger.info(f"  Dropped (missing forward data): {dropped:,}")
    logger.info(f"  Columns: {len(calibration_df.columns)}")

    # Log confidence distribution
    logger.info(f"\nConfidence distribution:")
    logger.info(f"  Mean: {calibration_df['regime_confidence'].mean():.3f}")
    logger.info(f"  Median: {calibration_df['regime_confidence'].median():.3f}")
    logger.info(f"  P10: {calibration_df['regime_confidence'].quantile(0.1):.3f}")
    logger.info(f"  P25: {calibration_df['regime_confidence'].quantile(0.25):.3f}")
    logger.info(f"  P75: {calibration_df['regime_confidence'].quantile(0.75):.3f}")
    logger.info(f"  P90: {calibration_df['regime_confidence'].quantile(0.9):.3f}")

    return calibration_df


def main():
    """Main execution flow."""
    print("="*80)
    print("Phase 1.1: Extract Confidence Calibration Data")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()

    try:
        # Step 1: Load features
        logger.info("Step 1: Loading historical features...")
        df = load_features()

        # Step 2: Run ensemble predictions
        logger.info("\nStep 2: Running ensemble predictions...")
        df_with_predictions = run_ensemble_predictions(df)

        # Step 3: Calculate forward outcomes
        logger.info("\nStep 3: Calculating forward outcomes...")
        df_with_outcomes = calculate_forward_outcomes(df_with_predictions)

        # Step 4: Prepare calibration dataset
        logger.info("\nStep 4: Preparing calibration dataset...")
        calibration_df = prepare_calibration_dataset(df_with_outcomes)

        # Step 5: Save calibration dataset
        output_path = Path(__file__).parent.parent / 'data' / 'confidence_calibration_data.parquet'
        logger.info(f"\nStep 5: Saving calibration dataset...")
        logger.info(f"  Output: {output_path}")

        calibration_df.to_parquet(output_path)

        logger.info(f"  ✅ Saved {len(calibration_df):,} rows")

        # Summary
        print()
        print("="*80)
        print("CALIBRATION DATA EXTRACTION COMPLETE")
        print("="*80)
        print(f"\nDataset: {output_path}")
        print(f"Total bars: {len(calibration_df):,}")
        print(f"Date range: {calibration_df.index[0]} to {calibration_df.index[-1]}")
        print(f"Duration: {(calibration_df.index[-1] - calibration_df.index[0]).days} days")

        print(f"\nConfidence statistics:")
        print(f"  Mean: {calibration_df['regime_confidence'].mean():.3f}")
        print(f"  Median: {calibration_df['regime_confidence'].median():.3f}")
        print(f"  Range: [{calibration_df['regime_confidence'].min():.3f}, "
              f"{calibration_df['regime_confidence'].max():.3f}]")

        print(f"\nForward outcome statistics:")
        print(f"  Mean 24h return: {calibration_df['forward_return_24h'].mean()*100:.3f}%")
        print(f"  Mean 24h volatility: {calibration_df['forward_volatility_24h'].mean()*100:.3f}%")
        print(f"  Mean 24h max DD: {calibration_df['forward_max_dd_24h'].mean()*100:.2f}%")
        print(f"  Regime stability 24h: {calibration_df['regime_stable_24h'].mean()*100:.1f}%")

        print(f"\n✅ Ready for Phase 1.2: Build calibration curves")
        print(f"\nNext steps:")
        print(f"  1. Run: python bin/build_confidence_calibrator.py")
        print(f"  2. Validate calibration on OOS data")
        print(f"  3. Integrate into RegimeService")

        return 0

    except Exception as e:
        logger.error(f"❌ Calibration data extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
