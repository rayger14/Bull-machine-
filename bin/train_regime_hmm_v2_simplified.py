#!/usr/bin/env python3
"""
Train Rolling Regime Classifier V2 - HMM Implementation (Simplified)
====================================================================

Simplified 4-state HMM using only the 9 features with 100% coverage.

This uses Path A from the research: 9 features with complete data availability.

Features Used:
  Tier 1 (Crypto-native):
    - funding_Z: 30-day z-score of funding rate
    - oi_change_pct_24h: 24h OI % change
    - rv_20d: 20-day realized volatility (proxy for RV_21)

  Tier 2 (Market structure):
    - USDT.D: USDT dominance
    - BTC.D: BTC dominance
    - TOTAL_RET: Total market cap return

  Tier 3 (Macro):
    - VIX_Z: VIX z-score
    - DXY_Z: DXY z-score
    - YC_SPREAD: 10Y - 2Y yield spread

Usage:
    python bin/train_regime_hmm_v2_simplified.py

Outputs:
    - models/hmm_regime_v2_simplified.pkl: Trained HMM model
    - data/regime_labels_v2.parquet: Historical regime labels
    - results/regime_v2_training_report.txt: Training metrics
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 9 features with 100% coverage (Path A)
REGIME_FEATURES_SIMPLIFIED = [
    'funding_Z',
    'oi_change_pct_24h',
    'rv_20d',
    'USDT.D',
    'BTC.D',
    'TOTAL_RET',
    'VIX_Z',
    'DXY_Z',
    'YC_SPREAD'
]

# Known capitulation events for validation
KNOWN_EVENTS = {
    'LUNA May-12': ('2022-05-12', '2022-05-13', 'crisis'),
    'June 18 Bottom': ('2022-06-18', '2022-06-20', 'crisis'),
    'FTX Collapse': ('2022-11-09', '2022-11-10', 'crisis'),
    'March 2023 Banking Crisis': ('2023-03-10', '2023-03-13', 'risk_off'),
    'Japan Carry Unwind': ('2024-08-05', '2024-08-06', 'risk_off'),
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and prepare the 9 regime features from feature store.

    Args:
        df: Feature store DataFrame

    Returns:
        DataFrame with 9 regime features, forward-filled NaNs
    """
    logger.info("Preparing regime features...")

    # Check all required columns exist
    missing = [col for col in REGIME_FEATURES_SIMPLIFIED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract features
    features_df = df[REGIME_FEATURES_SIMPLIFIED].copy()

    # Report coverage
    coverage = {}
    for col in REGIME_FEATURES_SIMPLIFIED:
        non_null = features_df[col].notna().sum()
        total = len(features_df)
        pct = (non_null / total) * 100
        coverage[col] = pct
        logger.info(f"  {col}: {pct:.1f}% coverage ({non_null}/{total})")

    # Forward fill NaNs (maintain time-series continuity)
    features_df = features_df.ffill()

    # Backfill any remaining NaNs at start
    features_df = features_df.bfill()

    # Final check
    remaining_nulls = features_df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"  {remaining_nulls} NaN values remain after filling")
        # Fill with 0 as last resort
        features_df = features_df.fillna(0)

    logger.info(f"  Final shape: {features_df.shape}")

    return features_df


def train_hmm(X: np.ndarray, n_states: int = 4, n_iter: int = 1000) -> GaussianHMM:
    """
    Train 4-state Gaussian HMM.

    Args:
        X: Feature matrix (T x 9)
        n_states: Number of hidden states (default 4)
        n_iter: Max EM iterations

    Returns:
        Trained HMM model
    """
    logger.info(f"Training HMM with {n_states} states...")
    logger.info(f"  Samples: {X.shape[0]:,}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Max iterations: {n_iter}")

    # Initialize HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',  # Diagonal covariance (assumes feature independence)
        n_iter=n_iter,
        random_state=42,
        verbose=True
    )

    # Fit model
    model.fit(X)

    logger.info(f"  Converged: {model.monitor_.converged}")
    logger.info(f"  Log-likelihood: {model.score(X):.2f}")

    return model


def interpret_states(model: GaussianHMM, X: np.ndarray, scaler: StandardScaler) -> Dict[int, str]:
    """
    Interpret HMM states as regime labels based on feature means.

    State interpretation logic:
    - crisis: High volatility (rv_20d), extreme funding (funding_Z), high VIX_Z
    - risk_off: Moderate volatility, negative funding, elevated VIX
    - neutral: Average volatility, balanced funding, normal VIX
    - risk_on: Low volatility, positive funding, low VIX

    Args:
        model: Trained HMM
        X: Scaled feature matrix
        scaler: StandardScaler used for normalization

    Returns:
        state_map: {state_idx: regime_name}
    """
    logger.info("Interpreting HMM states...")

    # Get mean feature values for each state (in scaled space)
    means = model.means_

    # Transform back to original space for interpretation
    means_original = scaler.inverse_transform(means)

    # Feature indices
    funding_idx = REGIME_FEATURES_SIMPLIFIED.index('funding_Z')
    rv_idx = REGIME_FEATURES_SIMPLIFIED.index('rv_20d')
    vix_idx = REGIME_FEATURES_SIMPLIFIED.index('VIX_Z')

    # Score each state for "crisis-ness"
    crisis_scores = []
    for i in range(means.shape[0]):
        # Crisis = high vol + extreme funding + high VIX
        score = (
            means_original[i, rv_idx] * 0.4 +      # High realized vol
            abs(means_original[i, funding_idx]) * 0.3 +  # Extreme funding (pos or neg)
            means_original[i, vix_idx] * 0.3       # High VIX
        )
        crisis_scores.append((i, score))

    # Sort by crisis score (descending)
    crisis_scores.sort(key=lambda x: x[1], reverse=True)

    # Assign labels
    state_map = {
        crisis_scores[0][0]: 'crisis',      # Highest crisis score
        crisis_scores[1][0]: 'risk_off',    # Second highest
        crisis_scores[2][0]: 'neutral',     # Third
        crisis_scores[3][0]: 'risk_on'      # Lowest (most calm)
    }

    # Log interpretation
    for state_idx, regime in state_map.items():
        logger.info(f"  State {state_idx} → {regime}")
        logger.info(f"    funding_Z: {means_original[state_idx, funding_idx]:.3f}")
        logger.info(f"    rv_20d: {means_original[state_idx, rv_idx]:.3f}")
        logger.info(f"    VIX_Z: {means_original[state_idx, vix_idx]:.3f}")

    return state_map


def validate_event_detection(states: pd.Series, state_map: Dict[int, str]) -> Dict[str, bool]:
    """
    Validate regime detection on known capitulation events.

    Args:
        states: Series of regime labels (index = datetime)
        state_map: {state_idx: regime_name}

    Returns:
        event_results: {event_name: detected}
    """
    logger.info("Validating event detection...")

    results = {}
    for event_name, (start_date, end_date, expected_regime) in KNOWN_EVENTS.items():
        # Get regimes during event window
        event_mask = (states.index >= start_date) & (states.index <= end_date)
        event_regimes = states[event_mask]

        if len(event_regimes) == 0:
            logger.warning(f"  {event_name}: NO DATA in window")
            results[event_name] = False
            continue

        # Check if expected regime appears in window
        detected = expected_regime in event_regimes.values

        # Get most common regime in window
        most_common = event_regimes.mode()[0] if len(event_regimes) > 0 else None

        status = "✓" if detected else "✗"
        logger.info(f"  {status} {event_name}: expected={expected_regime}, detected={most_common}")

        results[event_name] = detected

    accuracy = sum(results.values()) / len(results) * 100
    logger.info(f"\nEvent Detection Accuracy: {accuracy:.1f}% ({sum(results.values())}/{len(results)})")

    return results


def compute_validation_metrics(X: np.ndarray, states: np.ndarray) -> Dict[str, float]:
    """
    Compute validation metrics for trained HMM.

    Args:
        X: Feature matrix
        states: Predicted state sequence

    Returns:
        metrics: Dictionary of validation metrics
    """
    logger.info("Computing validation metrics...")

    metrics = {}

    # Silhouette score (cluster quality)
    silhouette = silhouette_score(X, states)
    metrics['silhouette_score'] = silhouette
    logger.info(f"  Silhouette score: {silhouette:.3f} {'✓ GOOD' if silhouette > 0.5 else '✗ POOR'}")

    # Transition frequency (are we thrashing?)
    transitions = np.sum(states[1:] != states[:-1])
    hours_total = len(states)
    days_total = hours_total / 24
    transitions_per_year = (transitions / days_total) * 365
    metrics['transitions_per_year'] = transitions_per_year
    logger.info(f"  Transitions/year: {transitions_per_year:.1f} {'✓ GOOD' if 10 <= transitions_per_year <= 20 else '~ CHECK'}")

    # State distribution
    unique, counts = np.unique(states, return_counts=True)
    logger.info("  State distribution:")
    for state_idx, count in zip(unique, counts):
        pct = (count / len(states)) * 100
        logger.info(f"    State {state_idx}: {pct:.1f}%")
        metrics[f'state_{state_idx}_pct'] = pct

    # Duration statistics
    durations = []
    current_state = states[0]
    current_duration = 1
    for i in range(1, len(states)):
        if states[i] == current_state:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_state = states[i]
            current_duration = 1
    durations.append(current_duration)

    avg_duration_hours = np.mean(durations)
    avg_duration_days = avg_duration_hours / 24
    metrics['avg_regime_duration_days'] = avg_duration_days
    logger.info(f"  Avg regime duration: {avg_duration_days:.1f} days")

    return metrics


def main():
    """Main training pipeline."""

    logger.info("="*80)
    logger.info("HMM REGIME CLASSIFIER V2 - SIMPLIFIED TRAINING")
    logger.info("="*80)
    logger.info(f"Path A: 9 features with 100% coverage")
    logger.info(f"Target: 4 states (risk_on, neutral, risk_off, crisis)")
    logger.info("="*80)

    # Load feature store
    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    logger.info(f"\nLoading data: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Prepare features
    features_df = prepare_features(df)

    # Standardize features (0 mean, unit variance)
    logger.info("\nStandardizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)
    logger.info(f"  Feature matrix shape: {X.shape}")

    # Train HMM
    logger.info("\nTraining HMM...")
    model = train_hmm(X, n_states=4, n_iter=1000)

    # Predict states (Viterbi decoding)
    logger.info("\nPredicting regime states...")
    state_sequence = model.predict(X)

    # Interpret states
    state_map = interpret_states(model, X, scaler)

    # Map states to regime labels
    regime_labels = pd.Series(
        [state_map[s] for s in state_sequence],
        index=features_df.index,
        name='regime_label'
    )

    # Compute validation metrics
    metrics = compute_validation_metrics(X, state_sequence)

    # Validate event detection
    event_results = validate_event_detection(regime_labels, state_map)
    metrics['event_detection_accuracy'] = sum(event_results.values()) / len(event_results) * 100

    # Save model
    logger.info("\nSaving model...")
    Path('models').mkdir(exist_ok=True)
    model_path = Path('models/hmm_regime_v2_simplified.pkl')

    model_data = {
        'model': model,
        'scaler': scaler,
        'state_map': state_map,
        'features': REGIME_FEATURES_SIMPLIFIED,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"  Saved model: {model_path}")

    # Save regime labels
    logger.info("\nSaving regime labels...")
    Path('data').mkdir(exist_ok=True)
    labels_path = Path('data/regime_labels_v2.parquet')
    regime_labels_df = pd.DataFrame({'regime_label': regime_labels})
    regime_labels_df.to_parquet(labels_path)
    logger.info(f"  Saved labels: {labels_path}")

    # Save training report
    logger.info("\nSaving training report...")
    Path('results').mkdir(exist_ok=True)
    report_path = Path('results/regime_v2_training_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HMM REGIME CLASSIFIER V2 - TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: GaussianHMM (4 states, diagonal covariance)\n")
        f.write(f"Features: {len(REGIME_FEATURES_SIMPLIFIED)}\n")
        f.write(f"Samples: {len(df):,} hours ({len(df)/24:.1f} days)\n\n")

        f.write("FEATURES:\n")
        for i, feat in enumerate(REGIME_FEATURES_SIMPLIFIED, 1):
            f.write(f"  {i}. {feat}\n")
        f.write("\n")

        f.write("STATE MAPPING:\n")
        for state_idx in sorted(state_map.keys()):
            f.write(f"  State {state_idx} → {state_map[state_idx]}\n")
        f.write("\n")

        f.write("VALIDATION METRICS:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.3f}\n")
        f.write("\n")

        f.write("EVENT DETECTION:\n")
        for event_name, detected in event_results.items():
            status = "✓ DETECTED" if detected else "✗ MISSED"
            f.write(f"  {status} {event_name}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("VERDICT:\n")
        f.write("="*80 + "\n")

        if metrics['silhouette_score'] > 0.5:
            f.write("✓ Cluster quality: GOOD\n")
        else:
            f.write("✗ Cluster quality: POOR (silhouette < 0.5)\n")

        if 10 <= metrics['transitions_per_year'] <= 20:
            f.write("✓ Transition frequency: GOOD\n")
        else:
            f.write("~ Transition frequency: CHECK (outside 10-20/year)\n")

        if metrics['event_detection_accuracy'] >= 80:
            f.write("✓ Event detection: GOOD\n")
        else:
            f.write("✗ Event detection: NEEDS IMPROVEMENT (<80%)\n")

        f.write("\n" + "="*80 + "\n")

    logger.info(f"  Saved report: {report_path}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Labels: {labels_path}")
    logger.info(f"Report: {report_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Review training report")
    logger.info("  2. Run validation: python bin/validate_regime_hmm.py")
    logger.info("  3. Integrate with engine: Update configs to use 'hmm_v2'")
    logger.info("="*80)


if __name__ == '__main__':
    main()
