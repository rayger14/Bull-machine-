#!/usr/bin/env python3
"""
Train HMM with Real-Time Crisis Features - PRODUCTION READY
============================================================

This script integrates real-time crisis indicators from Agents 1 & 2 into
the HMM training pipeline. It supports STAGED deployment:

Stage 1: Baseline (existing 9 features) - READY NOW
Stage 2: + Existing crisis features (5 features) - READY NOW
Stage 3: + Agent 2 features (flash crash, OI cascade, funding extreme) - PENDING
Stage 4: Full suite (15+ features) - PENDING

Usage:
    # Stage 1: Baseline (reproduce current results)
    python bin/train_hmm_with_crisis_features.py --stage baseline

    # Stage 2: Add existing crisis features
    python bin/train_hmm_with_crisis_features.py --stage existing_crisis

    # Stage 3: Add Agent 2 features (after delivery)
    python bin/train_hmm_with_crisis_features.py --stage agent2

    # Stage 4: Full suite
    python bin/train_hmm_with_crisis_features.py --stage full

Features by Stage:
    BASELINE (9 features):
        - funding_Z, oi_change_pct_24h, rv_20d
        - USDT.D, BTC.D, TOTAL_RET
        - VIX_Z, DXY_Z, YC_SPREAD

    EXISTING_CRISIS (+5 features):
        - crisis_composite, crisis_context
        - volume_z, volume_zscore
        - volatility_spike

    AGENT2 (+7-10 features, pending delivery):
        - flash_crash_1h, flash_crash_4h, flash_crash_1d
        - oi_delta_1h_z, oi_cascade
        - funding_extreme, funding_flip
        - [additional features from Agent 2]

    FULL (20+ features):
        - All of the above

Validation Targets:
    Crisis Detection: >80% accuracy on LUNA, FTX, June 2022
    Silhouette Score: >0.50
    Transitions/year: 10-20
    Regime Distribution: crisis 5-15%, risk_off 15-30%, neutral 20-35%, risk_on 25-45%
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
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature sets by stage
FEATURES_BASELINE = [
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

FEATURES_EXISTING_CRISIS = [
    'crisis_composite',
    'crisis_context',
    'volume_z',
    'volume_zscore',
    'volatility_spike'
]

FEATURES_AGENT2 = [
    # From Agent 2 (will be available after delivery)
    'flash_crash_1h',
    'flash_crash_4h',
    'flash_crash_1d',
    'oi_delta_1h_z',
    'oi_cascade',
    'funding_extreme',
    'funding_flip'
]

# Known crisis events for validation
CRISIS_EVENTS = {
    'LUNA May-12': ('2022-05-09', '2022-05-12', 'crisis', 80),
    'June 18 Bottom': ('2022-06-13', '2022-06-18', 'crisis', 70),
    'FTX Collapse': ('2022-11-08', '2022-11-11', 'crisis', 80),
    'March 2023 Banking Crisis': ('2023-03-10', '2023-03-13', 'risk_off', 60),
    'Japan Carry Unwind': ('2024-08-05', '2024-08-06', 'risk_off', 70),
}

BULL_EVENTS = {
    'Q1 2023 Rally Start': ('2023-01-10', '2023-01-15', 'risk_on', 70),
    'Q1 2023 Mid-Rally': ('2023-02-15', '2023-02-20', 'risk_on', 70),
    'Q1 2023 Continuation': ('2023-03-20', '2023-03-25', 'risk_on', 60),
}


def get_feature_set(stage: str) -> Tuple[List[str], str]:
    """
    Get feature list for specified training stage.

    Args:
        stage: One of 'baseline', 'existing_crisis', 'agent2', 'full'

    Returns:
        (feature_list, description)
    """
    if stage == 'baseline':
        return FEATURES_BASELINE, "Baseline (9 features - lagging indicators)"

    elif stage == 'existing_crisis':
        features = FEATURES_BASELINE + FEATURES_EXISTING_CRISIS
        return features, f"Baseline + Existing Crisis ({len(features)} features)"

    elif stage == 'agent2':
        features = FEATURES_BASELINE + FEATURES_EXISTING_CRISIS + FEATURES_AGENT2
        return features, f"Baseline + Existing + Agent2 ({len(features)} features)"

    elif stage == 'full':
        features = FEATURES_BASELINE + FEATURES_EXISTING_CRISIS + FEATURES_AGENT2
        return features, f"Full Suite ({len(features)} features)"

    else:
        raise ValueError(f"Unknown stage: {stage}. Use baseline|existing_crisis|agent2|full")


def prepare_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Extract and prepare features from feature store.

    Args:
        df: Feature store DataFrame
        feature_list: List of feature names to extract

    Returns:
        DataFrame with selected features, NaNs filled
    """
    logger.info("Preparing features...")

    # Check which features exist
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]

    if missing:
        logger.warning(f"Missing features ({len(missing)}):")
        for feat in missing:
            logger.warning(f"  ❌ {feat}")
        logger.info(f"Proceeding with {len(available)} available features")

    # Extract available features
    features_df = df[available].copy()

    # Report coverage
    coverage = {}
    for col in available:
        non_null = features_df[col].notna().sum()
        total = len(features_df)
        pct = (non_null / total) * 100
        coverage[col] = pct

        if pct < 99:
            logger.warning(f"  {col}: {pct:.1f}% coverage ({non_null}/{total})")
        else:
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

    return features_df, coverage


def train_hmm_ensemble(X: np.ndarray, n_states: int = 4, n_iter: int = 1000,
                       n_init: int = 10) -> Tuple[GaussianHMM, float]:
    """
    Train HMM with multiple random initializations and select best model.

    Args:
        X: Feature matrix (T x D)
        n_states: Number of hidden states
        n_iter: Max EM iterations per initialization
        n_init: Number of random initializations

    Returns:
        (best_model, best_score)
    """
    logger.info(f"Training HMM ensemble with {n_init} initializations...")
    logger.info(f"  Samples: {X.shape[0]:,}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  States: {n_states}")
    logger.info(f"  Max iterations per init: {n_iter}")

    best_model = None
    best_score = -np.inf
    best_seed = None

    for i in range(n_init):
        seed = 42 + i
        logger.info(f"\n  Init {i+1}/{n_init} (seed={seed})...")

        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=n_iter,
            random_state=seed,
            verbose=False
        )

        try:
            model.fit(X)
            score = model.score(X)

            logger.info(f"    Converged: {model.monitor_.converged}")
            logger.info(f"    Log-likelihood: {score:.2f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_seed = seed
                logger.info(f"    ⭐ NEW BEST (seed={seed}, LL={score:.2f})")

        except Exception as e:
            logger.warning(f"    ❌ Failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All HMM initializations failed")

    logger.info(f"\n✅ Best model: seed={best_seed}, LL={best_score:.2f}")

    return best_model, best_score


def interpret_states(model: GaussianHMM, X: np.ndarray, scaler: StandardScaler,
                     feature_names: List[str]) -> Dict[int, str]:
    """
    Interpret HMM states as regime labels based on feature means.

    For models WITH crisis features:
        - crisis: High crisis_composite + high volatility_spike + high volume_z
        - risk_off: Moderate crisis signals + negative funding
        - neutral: Low crisis signals + balanced conditions
        - risk_on: No crisis signals + positive funding + low volatility

    For baseline models (NO crisis features):
        - crisis: High volatility + extreme funding + high VIX
        - risk_off: Moderate volatility + negative funding + elevated VIX
        - neutral: Average volatility + balanced funding + normal VIX
        - risk_on: Low volatility + positive funding + low VIX

    Args:
        model: Trained HMM
        X: Scaled feature matrix
        scaler: StandardScaler used for normalization
        feature_names: List of feature names in order

    Returns:
        state_map: {state_idx: regime_name}
    """
    logger.info("Interpreting HMM states...")

    # Get mean feature values for each state (in scaled space)
    means = model.means_

    # Transform back to original space for interpretation
    means_original = scaler.inverse_transform(means)

    # Determine if we have crisis features
    has_crisis_features = 'crisis_composite' in feature_names

    if has_crisis_features:
        # NEW: Crisis-aware interpretation
        logger.info("  Using CRISIS-AWARE state interpretation")

        crisis_idx = feature_names.index('crisis_composite')
        vol_spike_idx = feature_names.index('volatility_spike')
        volume_z_idx = feature_names.index('volume_z')

        # Score each state for "crisis-ness"
        crisis_scores = []
        for i in range(means.shape[0]):
            score = (
                means_original[i, crisis_idx] * 0.5 +      # Crisis composite (primary)
                means_original[i, vol_spike_idx] * 0.3 +   # Volatility spike
                means_original[i, volume_z_idx] * 0.2      # Volume surge
            )
            crisis_scores.append((i, score))

    else:
        # OLD: Lagging indicator interpretation
        logger.info("  Using BASELINE state interpretation (lagging indicators)")

        funding_idx = feature_names.index('funding_Z')
        rv_idx = feature_names.index('rv_20d')
        vix_idx = feature_names.index('VIX_Z')

        # Score each state for "crisis-ness" using lagging indicators
        crisis_scores = []
        for i in range(means.shape[0]):
            score = (
                means_original[i, rv_idx] * 0.4 +           # High realized vol
                abs(means_original[i, funding_idx]) * 0.3 + # Extreme funding
                means_original[i, vix_idx] * 0.3            # High VIX
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
        if has_crisis_features:
            logger.info(f"    crisis_composite: {means_original[state_idx, crisis_idx]:.3f}")
            logger.info(f"    volatility_spike: {means_original[state_idx, vol_spike_idx]:.3f}")
            logger.info(f"    volume_z: {means_original[state_idx, volume_z_idx]:.3f}")
        else:
            logger.info(f"    funding_Z: {means_original[state_idx, funding_idx]:.3f}")
            logger.info(f"    rv_20d: {means_original[state_idx, rv_idx]:.3f}")
            logger.info(f"    VIX_Z: {means_original[state_idx, vix_idx]:.3f}")

    return state_map


def validate_event_detection(states: pd.Series, all_events: Dict) -> Dict:
    """
    Validate regime detection on known events.

    Args:
        states: Series of regime labels (index = datetime)
        all_events: Dictionary of {name: (start, end, expected_regime, target_pct)}

    Returns:
        results: {event_name: {'detected': bool, 'accuracy': float, 'details': str}}
    """
    logger.info("Validating event detection...")

    results = {}
    for event_name, (start_date, end_date, expected_regime, target_pct) in all_events.items():
        # Get regimes during event window
        event_mask = (states.index >= start_date) & (states.index <= end_date)
        event_regimes = states[event_mask]

        if len(event_regimes) == 0:
            logger.warning(f"  {event_name}: NO DATA in window")
            results[event_name] = {
                'detected': False,
                'accuracy': 0.0,
                'details': 'No data in window'
            }
            continue

        # Calculate detection accuracy
        regime_pct = (event_regimes == expected_regime).mean() * 100
        detected = regime_pct >= target_pct

        # Get most common regime
        most_common = event_regimes.mode()[0] if len(event_regimes) > 0 else None

        status = "✅" if detected else "❌"
        logger.info(f"  {status} {event_name}:")
        logger.info(f"      Expected: {expected_regime} (target {target_pct}%)")
        logger.info(f"      Detected: {regime_pct:.1f}% as {expected_regime}")
        logger.info(f"      Most common: {most_common}")

        results[event_name] = {
            'detected': detected,
            'accuracy': regime_pct,
            'details': f"{regime_pct:.1f}% as {expected_regime}, most_common={most_common}"
        }

    return results


def compute_validation_metrics(X: np.ndarray, states: np.ndarray) -> Dict[str, float]:
    """Compute validation metrics for trained HMM."""
    logger.info("Computing validation metrics...")

    metrics = {}

    # Silhouette score (cluster quality)
    silhouette = silhouette_score(X, states)
    metrics['silhouette_score'] = silhouette
    logger.info(f"  Silhouette score: {silhouette:.3f} {'✅ GOOD' if silhouette > 0.5 else '❌ POOR'}")

    # Transition frequency
    transitions = np.sum(states[1:] != states[:-1])
    hours_total = len(states)
    days_total = hours_total / 24
    transitions_per_year = (transitions / days_total) * 365
    metrics['transitions_per_year'] = transitions_per_year
    logger.info(f"  Transitions/year: {transitions_per_year:.1f} {'✅ GOOD' if 10 <= transitions_per_year <= 20 else '⚠️  CHECK'}")

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
    parser = argparse.ArgumentParser(description='Train HMM with crisis features')
    parser.add_argument('--stage', type=str, default='existing_crisis',
                       choices=['baseline', 'existing_crisis', 'agent2', 'full'],
                       help='Training stage: baseline|existing_crisis|agent2|full')
    parser.add_argument('--n_init', type=int, default=10,
                       help='Number of random initializations (default: 10)')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='Max EM iterations per init (default: 1000)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("HMM REGIME CLASSIFIER - CRISIS FEATURE TRAINING")
    logger.info("="*80)

    # Get feature set for this stage
    feature_list, stage_description = get_feature_set(args.stage)
    logger.info(f"Stage: {args.stage.upper()}")
    logger.info(f"Description: {stage_description}")
    logger.info(f"Features: {len(feature_list)}")
    logger.info("="*80)

    # Load feature store
    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    logger.info(f"\nLoading data: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Prepare features
    features_df, coverage = prepare_features(df, feature_list)

    # Standardize features
    logger.info("\nStandardizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)
    logger.info(f"  Feature matrix shape: {X.shape}")

    # Train HMM ensemble
    logger.info("\nTraining HMM ensemble...")
    model, best_score = train_hmm_ensemble(X, n_states=4, n_iter=args.n_iter, n_init=args.n_init)

    # Predict states
    logger.info("\nPredicting regime states...")
    state_sequence = model.predict(X)

    # Interpret states
    state_map = interpret_states(model, X, scaler, list(features_df.columns))

    # Map states to regime labels
    regime_labels = pd.Series(
        [state_map[s] for s in state_sequence],
        index=features_df.index,
        name='regime_label'
    )

    # Compute validation metrics
    metrics = compute_validation_metrics(X, state_sequence)

    # Validate event detection
    all_events = {**CRISIS_EVENTS, **BULL_EVENTS}
    event_results = validate_event_detection(regime_labels, all_events)

    # Calculate detection rates
    crisis_events = {k: v for k, v in event_results.items() if k in CRISIS_EVENTS}
    bull_events = {k: v for k, v in event_results.items() if k in BULL_EVENTS}

    crisis_accuracy = np.mean([v['detected'] for v in crisis_events.values()]) * 100
    bull_accuracy = np.mean([v['detected'] for v in bull_events.values()]) * 100
    overall_accuracy = np.mean([v['detected'] for v in event_results.values()]) * 100

    metrics['crisis_event_accuracy'] = crisis_accuracy
    metrics['bull_event_accuracy'] = bull_accuracy
    metrics['overall_event_accuracy'] = overall_accuracy

    logger.info("\n" + "="*80)
    logger.info("EVENT DETECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Crisis events: {crisis_accuracy:.1f}% ({sum(v['detected'] for v in crisis_events.values())}/{len(crisis_events)})")
    logger.info(f"Bull events:   {bull_accuracy:.1f}% ({sum(v['detected'] for v in bull_events.values())}/{len(bull_events)})")
    logger.info(f"Overall:       {overall_accuracy:.1f}% ({sum(v['detected'] for v in event_results.values())}/{len(event_results)})")

    # Deployment readiness assessment
    deploy_ready = (
        crisis_accuracy >= 80 and
        metrics['silhouette_score'] > 0.5 and
        10 <= metrics['transitions_per_year'] <= 20
    )

    # Save model
    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL")
    logger.info("="*80)

    Path('models').mkdir(exist_ok=True)
    model_filename = f"hmm_regime_{args.stage}.pkl"
    model_path = Path('models') / model_filename

    model_data = {
        'model': model,
        'scaler': scaler,
        'state_map': state_map,
        'features': list(features_df.columns),
        'stage': args.stage,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'event_results': event_results,
        'deploy_ready': deploy_ready,
        'best_score': best_score
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"✅ Saved model: {model_path}")

    # Save regime labels
    labels_filename = f"regime_labels_{args.stage}.parquet"
    labels_path = Path('data') / labels_filename
    regime_labels_df = pd.DataFrame({'regime_label': regime_labels})
    regime_labels_df.to_parquet(labels_path)
    logger.info(f"✅ Saved labels: {labels_path}")

    # Save training report
    report_filename = f"hmm_training_report_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = Path('results') / report_filename
    Path('results').mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"HMM REGIME CLASSIFIER - {args.stage.upper()} TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Description: {stage_description}\n")
        f.write(f"Model: GaussianHMM (4 states, diagonal covariance)\n")
        f.write(f"Features: {len(features_df.columns)}\n")
        f.write(f"Samples: {len(df):,} hours ({len(df)/24:.1f} days)\n")
        f.write(f"Random initializations: {args.n_init}\n")
        f.write(f"Best log-likelihood: {best_score:.2f}\n\n")

        f.write("FEATURES:\n")
        for i, feat in enumerate(features_df.columns, 1):
            cov = coverage.get(feat, 100)
            f.write(f"  {i:2d}. {feat:25s} ({cov:.1f}% coverage)\n")
        f.write("\n")

        f.write("STATE MAPPING:\n")
        for state_idx in sorted(state_map.keys()):
            f.write(f"  State {state_idx} → {state_map[state_idx]}\n")
        f.write("\n")

        f.write("VALIDATION METRICS:\n")
        for key, value in metrics.items():
            if not key.startswith('state_'):
                f.write(f"  {key}: {value:.3f}\n")
        f.write("\n")

        f.write("STATE DISTRIBUTION:\n")
        for key, value in metrics.items():
            if key.startswith('state_'):
                f.write(f"  {key}: {value:.1f}%\n")
        f.write("\n")

        f.write("EVENT DETECTION RESULTS:\n")
        f.write("\nCrisis Events:\n")
        for event_name, result in crisis_events.items():
            status = "✅ PASS" if result['detected'] else "❌ FAIL"
            f.write(f"  {status} {event_name}\n")
            f.write(f"      {result['details']}\n")

        f.write("\nBull Events:\n")
        for event_name, result in bull_events.items():
            status = "✅ PASS" if result['detected'] else "❌ FAIL"
            f.write(f"  {status} {event_name}\n")
            f.write(f"      {result['details']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DEPLOYMENT READINESS ASSESSMENT\n")
        f.write("="*80 + "\n\n")

        # Check criteria
        f.write("Criteria:\n")

        sil_pass = metrics['silhouette_score'] > 0.5
        f.write(f"  {'✅' if sil_pass else '❌'} Silhouette score > 0.50: {metrics['silhouette_score']:.3f}\n")

        trans_pass = 10 <= metrics['transitions_per_year'] <= 20
        f.write(f"  {'✅' if trans_pass else '❌'} Transitions 10-20/year: {metrics['transitions_per_year']:.1f}\n")

        crisis_pass = crisis_accuracy >= 80
        f.write(f"  {'✅' if crisis_pass else '❌'} Crisis detection ≥80%: {crisis_accuracy:.1f}%\n")

        bull_pass = bull_accuracy >= 70
        f.write(f"  {'✅' if bull_pass else '⚠️ '} Bull detection ≥70%: {bull_accuracy:.1f}%\n")

        f.write("\n")
        if deploy_ready:
            f.write("✅ DEPLOYMENT READY\n\n")
            f.write("This model meets all production criteria and is ready for deployment.\n")
            f.write("\nNext steps:\n")
            f.write("  1. Run comprehensive validation: python bin/quick_hmm_validation.py\n")
            f.write("  2. Create deployment guide\n")
            f.write("  3. Update production configs to use this model\n")
            f.write("  4. Run smoke tests\n")
        else:
            f.write("❌ NOT DEPLOYMENT READY\n\n")
            f.write("This model does not meet production criteria.\n")
            f.write("\nRequired improvements:\n")
            if not crisis_pass:
                f.write(f"  - Improve crisis detection from {crisis_accuracy:.1f}% to ≥80%\n")
            if not sil_pass:
                f.write(f"  - Improve cluster quality from {metrics['silhouette_score']:.3f} to >0.50\n")
            if not trans_pass:
                f.write(f"  - Adjust transition frequency from {metrics['transitions_per_year']:.1f} to 10-20/year\n")
            f.write("\nConsider:\n")
            f.write("  - Try next stage with more features\n")
            f.write("  - Adjust n_init for more robust training\n")
            f.write("  - Review feature engineering\n")

        f.write("\n" + "="*80 + "\n")

    logger.info(f"✅ Saved report: {report_path}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Report: {report_path}")
    logger.info(f"\nDeployment Status: {'✅ READY' if deploy_ready else '❌ NOT READY'}")
    logger.info(f"  Crisis detection: {crisis_accuracy:.1f}% (target: ≥80%)")
    logger.info(f"  Silhouette score: {metrics['silhouette_score']:.3f} (target: >0.50)")
    logger.info(f"  Transitions/year: {metrics['transitions_per_year']:.1f} (target: 10-20)")

    if deploy_ready:
        logger.info("\n🚀 Next steps:")
        logger.info("  1. python bin/quick_hmm_validation.py")
        logger.info("  2. Review deployment guide")
        logger.info("  3. Run smoke tests")
    else:
        logger.info("\n📋 Next steps:")
        logger.info("  1. Review training report")
        logger.info("  2. Try next stage: --stage agent2 (when available)")
        logger.info("  3. Consider feature engineering improvements")

    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
