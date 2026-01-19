#!/usr/bin/env python3
"""
Build Confidence Calibrator

Purpose: Train isotonic regression calibrators that map ensemble agreement
to calibrated confidence based on empirical outcomes.

Process:
1. Load calibration data (from extract_confidence_calibration_data.py)
2. Split into train (2018-2022) and OOS test (2023-2024)
3. Train isotonic regression calibrators:
   - Return calibrator: agreement → normalized forward returns
   - Volatility calibrator: agreement → inverse volatility (high conf = low vol)
   - Stability calibrator: agreement → regime stability
   - Composite calibrator: weighted combination
4. Validate on OOS data
5. Save calibrators to pickle

Output: models/confidence_calibrator_v1.pkl

Author: Claude Code
Date: 2026-01-15
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pickle
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error
from engine.context.confidence_calibrator import CompositeCalibrator
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_data() -> pd.DataFrame:
    """Load calibration dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'confidence_calibration_data.parquet'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {data_path}\n"
            f"Run bin/extract_confidence_calibration_data.py first"
        )

    df = pd.read_parquet(data_path)
    logger.info(f"✅ Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    return df


def split_train_test(df: pd.DataFrame, split_date: str = '2023-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train and OOS test sets.

    Train: 2018-2022 (5 years)
    Test: 2023-2024 (2 years)
    """
    logger.info(f"\n{'='*80}")
    logger.info("Splitting Train/Test Sets")
    logger.info(f"{'='*80}")

    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    logger.info(f"\nTrain set (2018-2022):")
    logger.info(f"  Bars: {len(train):,}")
    logger.info(f"  Date range: {train.index[0]} to {train.index[-1]}")
    logger.info(f"  Duration: {(train.index[-1] - train.index[0]).days} days")

    logger.info(f"\nTest set (2023-2024):")
    logger.info(f"  Bars: {len(test):,}")
    logger.info(f"  Date range: {test.index[0]} to {test.index[-1]}")
    logger.info(f"  Duration: {(test.index[-1] - test.index[0]).days} days")

    logger.info(f"\nSplit ratio: {len(train)/len(df)*100:.1f}% train, {len(test)/len(df)*100:.1f}% test")

    return train, test


def normalize_target(series: pd.Series, method: str = 'rank') -> np.ndarray:
    """
    Normalize target variable to [0, 1] range.

    Methods:
    - 'rank': Rank normalization (preserves monotonicity best)
    - 'minmax': Min-max scaling
    - 'zscore': Z-score then clip to [0, 1]
    """
    if method == 'rank':
        # Rank normalization: best for isotonic regression
        # Maps values to percentile ranks
        return series.rank(pct=True).values

    elif method == 'minmax':
        # Min-max scaling
        min_val = series.min()
        max_val = series.max()
        return ((series - min_val) / (max_val - min_val)).values

    elif method == 'zscore':
        # Z-score then clip
        z = (series - series.mean()) / series.std()
        return np.clip((z + 3) / 6, 0, 1)  # Map ~[-3, 3] to [0, 1]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def train_return_calibrator(train: pd.DataFrame) -> IsotonicRegression:
    """
    Train calibrator for forward returns.

    Maps: ensemble_agreement → expected return quality
    """
    logger.info(f"\n1. Training Return Calibrator...")

    X = train['regime_confidence'].values  # ensemble agreement
    y = normalize_target(train['forward_return_24h'], method='rank')

    # Train isotonic regression (monotonic increasing)
    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,  # Higher confidence → better returns
        out_of_bounds='clip'
    )

    calibrator.fit(X, y)

    # Validate on train set
    y_pred = calibrator.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    logger.info(f"   ✓ Return calibrator trained")
    logger.info(f"   Train R²: {r2:.4f}")
    logger.info(f"   Train MAE: {mae:.4f}")

    return calibrator


def train_volatility_calibrator(train: pd.DataFrame) -> IsotonicRegression:
    """
    Train calibrator for forward volatility.

    Maps: ensemble_agreement → volatility confidence (inverse of vol)
    High confidence should predict LOW volatility.
    """
    logger.info(f"\n2. Training Volatility Calibrator...")

    X = train['regime_confidence'].values
    # Inverse volatility: lower vol = higher confidence
    y_vol = train['forward_volatility_24h']
    y = 1.0 - normalize_target(y_vol, method='rank')  # Invert so high conf = high score

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,  # Higher confidence → lower volatility → higher score
        out_of_bounds='clip'
    )

    calibrator.fit(X, y)

    y_pred = calibrator.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    logger.info(f"   ✓ Volatility calibrator trained")
    logger.info(f"   Train R²: {r2:.4f}")
    logger.info(f"   Train MAE: {mae:.4f}")

    return calibrator


def train_stability_calibrator(train: pd.DataFrame) -> IsotonicRegression:
    """
    Train calibrator for regime stability.

    Maps: ensemble_agreement → regime stability
    High confidence should predict stable regimes (few transitions).
    """
    logger.info(f"\n3. Training Stability Calibrator...")

    X = train['regime_confidence'].values
    y = train['regime_stable_24h'].values  # Already binary [0, 1]

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,  # Higher confidence → more stable regimes
        out_of_bounds='clip'
    )

    calibrator.fit(X, y)

    y_pred = calibrator.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    logger.info(f"   ✓ Stability calibrator trained")
    logger.info(f"   Train R²: {r2:.4f}")
    logger.info(f"   Train MAE: {mae:.4f}")

    return calibrator


def create_composite_calibrator(
    return_cal: IsotonicRegression,
    volatility_cal: IsotonicRegression,
    stability_cal: IsotonicRegression,
    weights: Dict[str, float] = None
) -> CompositeCalibrator:
    """
    Create composite calibrator as weighted combination.

    Adjusted weights based on performance:
    - Stability: 70% (best R²=0.17)
    - Volatility: 30% (weak R²=0.06)
    - Returns: 0% (no signal)
    """
    logger.info(f"\n4. Creating Composite Calibrator...")

    if weights is None:
        # Weight by performance: stability dominates
        weights = {'return': 0.0, 'volatility': 0.3, 'stability': 0.7}

    logger.info(f"   Weights (performance-based): {weights}")
    logger.info(f"     Stability: {weights['stability']*100:.0f}% (R²=0.17 on train)")
    logger.info(f"     Volatility: {weights['volatility']*100:.0f}% (R²=0.06 on train)")
    logger.info(f"     Return: {weights['return']*100:.0f}% (R²=0.00 on train)")

    composite = CompositeCalibrator(return_cal, volatility_cal, stability_cal, weights)

    logger.info(f"   ✓ Composite calibrator created (class-based, pickleable)")

    return composite


def validate_on_test(
    test: pd.DataFrame,
    calibrators: Dict[str, callable]
) -> Dict[str, float]:
    """
    Validate calibrators on OOS test set.

    Measures:
    - R² score (how well calibrated conf predicts outcomes)
    - MAE (average error)
    - Monotonicity (is calibrated conf monotonic with agreement?)
    """
    logger.info(f"\n{'='*80}")
    logger.info("Validating on OOS Test Set (2023-2024)")
    logger.info(f"{'='*80}")

    X_test = test['regime_confidence'].values

    results = {}

    # Test each calibrator
    for name, calibrator in calibrators.items():
        logger.info(f"\n{name.upper()} Calibrator:")

        # Predict on test set
        y_pred = calibrator.predict(X_test) if hasattr(calibrator, 'predict') else calibrator(X_test)

        # Calculate target (same normalization as training)
        if name == 'return':
            y_true = normalize_target(test['forward_return_24h'], method='rank')
        elif name == 'volatility':
            y_vol = test['forward_volatility_24h']
            y_true = 1.0 - normalize_target(y_vol, method='rank')
        elif name == 'stability':
            y_true = test['regime_stable_24h'].values
        elif name == 'composite':
            # Composite target: weighted combination of individual targets
            # (matches composite predictor's weighting)
            weights = calibrator.weights if hasattr(calibrator, 'weights') else {'return': 0.0, 'volatility': 0.3, 'stability': 0.7}

            return_target = normalize_target(test['forward_return_24h'], method='rank')
            vol_target = 1.0 - normalize_target(test['forward_volatility_24h'], method='rank')
            stability_target = test['regime_stable_24h'].values

            y_true = (
                weights['return'] * return_target +
                weights['volatility'] * vol_target +
                weights['stability'] * stability_target
            )
        else:
            continue

        # Metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Monotonicity check (correlation with raw agreement)
        correlation = np.corrcoef(X_test, y_pred)[0, 1]

        logger.info(f"  OOS R²: {r2:.4f}")
        logger.info(f"  OOS MAE: {mae:.4f}")
        logger.info(f"  Correlation with agreement: {correlation:.4f}")

        results[name] = {
            'r2': r2,
            'mae': mae,
            'correlation': correlation
        }

        # Quality check
        if r2 > 0.1:
            logger.info(f"  ✅ PASS - Calibrator has predictive power")
        elif r2 > 0.0:
            logger.info(f"  ⚠️  WEAK - Calibrator has weak signal")
        else:
            logger.info(f"  ❌ FAIL - Calibrator has no predictive power")

    return results


def plot_calibration_curves(
    train: pd.DataFrame,
    calibrators: Dict[str, callable],
    output_dir: Path
):
    """
    Plot calibration curves for visualization.

    Creates plots showing:
    - Raw confidence vs calibrated confidence
    - Calibration curves per calibrator type
    """
    logger.info(f"\n{'='*80}")
    logger.info("Generating Calibration Curve Plots")
    logger.info(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate confidence range
    X_plot = np.linspace(0, 1, 100)

    # Plot each calibrator
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, calibrator) in enumerate(calibrators.items()):
        ax = axes[idx]

        # Predict on range
        if hasattr(calibrator, 'predict'):
            y_plot = calibrator.predict(X_plot)
        else:
            y_plot = calibrator(X_plot)

        # Plot calibration curve
        ax.plot(X_plot, y_plot, 'b-', linewidth=2, label='Calibration curve')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5, label='Perfect calibration')

        ax.set_xlabel('Raw Confidence (Ensemble Agreement)', fontsize=10)
        ax.set_ylabel('Calibrated Confidence', fontsize=10)
        ax.set_title(f'{name.capitalize()} Calibrator', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = output_dir / 'calibration_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved calibration curves: {plot_path}")
    plt.close()

    # Scatter plot: raw vs calibrated (composite)
    fig, ax = plt.subplots(figsize=(10, 8))

    X_train = train['regime_confidence'].values
    y_composite = calibrators['composite'](X_train) if callable(calibrators['composite']) else calibrators['composite'].predict(X_train)

    # Hexbin plot (better for large datasets)
    hb = ax.hexbin(X_train, y_composite, gridsize=50, cmap='Blues', mincnt=1)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='No calibration (y=x)')

    ax.set_xlabel('Raw Confidence (Ensemble Agreement)', fontsize=12)
    ax.set_ylabel('Calibrated Confidence', fontsize=12)
    ax.set_title('Raw vs Calibrated Confidence (Composite)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(hb, ax=ax, label='Count')

    scatter_path = output_dir / 'raw_vs_calibrated_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved scatter plot: {scatter_path}")
    plt.close()


def save_calibrators(
    calibrators: Dict[str, callable],
    validation_results: Dict[str, Dict],
    output_path: Path
):
    """Save calibrators to pickle file."""
    logger.info(f"\n{'='*80}")
    logger.info("Saving Calibrators")
    logger.info(f"{'='*80}")

    calibrator_data = {
        'version': 'v1',
        'train_date': datetime.now().isoformat(),
        'calibrators': calibrators,
        'validation_results': validation_results,
        'config': {
            'normalization_method': 'rank',
            'weights': {'return': 0.4, 'volatility': 0.3, 'stability': 0.3},
            'isotonic_params': {
                'y_min': 0.0,
                'y_max': 1.0,
                'increasing': True,
                'out_of_bounds': 'clip'
            }
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(calibrator_data, f)

    logger.info(f"  ✓ Saved calibrators to: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main execution flow."""
    print("="*80)
    print("Phase 1.2: Build Confidence Calibrator")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()

    try:
        # Step 1: Load calibration data
        logger.info("Step 1: Loading calibration data...")
        df = load_calibration_data()

        # Step 2: Split train/test
        logger.info("\nStep 2: Splitting train/test sets...")
        train, test = split_train_test(df, split_date='2023-01-01')

        # Step 3: Train calibrators
        logger.info("\nStep 3: Training calibrators...")
        logger.info(f"{'='*80}")
        logger.info("Training Individual Calibrators")
        logger.info(f"{'='*80}")

        return_cal = train_return_calibrator(train)
        volatility_cal = train_volatility_calibrator(train)
        stability_cal = train_stability_calibrator(train)
        composite_predict = create_composite_calibrator(return_cal, volatility_cal, stability_cal)

        calibrators = {
            'return': return_cal,
            'volatility': volatility_cal,
            'stability': stability_cal,
            'composite': composite_predict
        }

        # Step 4: Validate on test set
        logger.info("\nStep 4: Validating on OOS test set...")
        validation_results = validate_on_test(test, calibrators)

        # Step 5: Plot calibration curves
        logger.info("\nStep 5: Generating calibration curve plots...")
        output_dir = Path(__file__).parent.parent / 'models'
        plot_calibration_curves(train, calibrators, output_dir)

        # Step 6: Save calibrators
        logger.info("\nStep 6: Saving calibrators...")
        output_path = output_dir / 'confidence_calibrator_v1.pkl'
        save_calibrators(calibrators, validation_results, output_path)

        # Summary
        print()
        print("="*80)
        print("CALIBRATION TRAINING COMPLETE")
        print("="*80)

        print(f"\nOutput: {output_path}")
        print(f"\nValidation Results (OOS 2023-2024):")

        for name, metrics in validation_results.items():
            print(f"\n{name.upper()}:")
            print(f"  R²:          {metrics['r2']:.4f}")
            print(f"  MAE:         {metrics['mae']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")

            if metrics['r2'] > 0.1:
                print(f"  Status: ✅ PASS")
            elif metrics['r2'] > 0.0:
                print(f"  Status: ⚠️  WEAK")
            else:
                print(f"  Status: ❌ FAIL")

        # Overall pass/fail
        composite_r2 = validation_results['composite']['r2']

        print(f"\n{'='*80}")
        if composite_r2 > 0.1:
            print("✅ CALIBRATOR READY FOR PRODUCTION")
            print("\nComposite calibrator has predictive power (R² > 0.1)")
            print("Calibrated confidence will improve upon raw ensemble agreement.")
            print(f"\nNext steps:")
            print(f"  1. Run: python bin/validate_confidence_calibration.py")
            print(f"  2. Integrate into RegimeService (Phase 2)")
            return 0
        elif composite_r2 > 0.0:
            print("⚠️  CALIBRATOR WEAK BUT USABLE")
            print(f"\nComposite R² = {composite_r2:.4f} (weak signal)")
            print("Calibrator may provide marginal improvement.")
            print(f"\nRecommendation: Proceed with caution, monitor in production")
            return 0
        else:
            print("❌ CALIBRATOR FAILED VALIDATION")
            print(f"\nComposite R² = {composite_r2:.4f} (no predictive power)")
            print("Calibrator would not improve upon raw agreement.")
            print(f"\nRecommendation: Debug data quality or use raw agreement")
            return 1

    except Exception as e:
        logger.error(f"❌ Calibrator training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
