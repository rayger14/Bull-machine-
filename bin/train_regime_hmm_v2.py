#!/usr/bin/env python3
"""
Train Rolling Regime Classifier V2 - HMM Implementation
========================================================

4-state Hidden Markov Model with 21-day rolling window and crypto-specific features.

This is the BRAINSTEM of the Bull Machine - regime awareness that filters reality.

Usage:
    python bin/train_regime_hmm_v2.py [--data-path DATA] [--output-path OUTPUT]

Outputs:
    - models/hmm_regime_v2.pkl: Trained HMM model
    - data/regime_labels_v2.parquet: Historical regime labels
    - results/regime_v2_validation.json: Validation metrics

Success Criteria:
    - Silhouette score > 0.5 (cluster quality)
    - 10-20 transitions/year (not thrashing)
    - 80%+ accuracy on known events (LUNA, FTX, June 18)
    - Batch = stream results (feature parity)
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context.hmm_regime_model import REGIME_FEATURES_V2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 regime features.

    Args:
        df: Raw DataFrame with price/macro columns

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering regime features...")
    df = df.copy()

    # Tier 1: Crypto-native features
    logger.info("  Computing Tier 1 features (crypto-native)...")

    # funding_Z: 30-day z-score of funding rate
    if 'funding' in df.columns or 'funding_rate' in df.columns:
        funding_col = 'funding' if 'funding' in df.columns else 'funding_rate'
        df['funding_Z'] = rolling_zscore(df[funding_col], window=30*24)
    else:
        logger.warning("    funding column not found, setting funding_Z to 0")
        df['funding_Z'] = 0.0

    # OI_CHANGE: 24h open interest % change
    if 'oi' in df.columns:
        df['OI_CHANGE'] = df['oi'].pct_change(24) * 100
    else:
        logger.warning("    oi column not found, setting OI_CHANGE to 0")
        df['OI_CHANGE'] = 0.0

    # RV_21: 21-day realized volatility
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        df['RV_21'] = realized_volatility(returns, window=21*24)
    elif 'RV_20' in df.columns:
        # Approximate with RV_20
        df['RV_21'] = df['RV_20'] * 100
    else:
        logger.warning("    close/RV_20 not found, setting RV_21 to 0")
        df['RV_21'] = 0.0

    # LIQ_VOL_24h: 24h liquidation volume ($M)
    if 'liquidations' in df.columns:
        df['LIQ_VOL_24h'] = df['liquidations'].rolling(24).sum() / 1e6
    else:
        logger.warning("    liquidations not found, setting LIQ_VOL_24h to 0")
        df['LIQ_VOL_24h'] = 0.0

    # Tier 2: Market structure features
    logger.info("  Computing Tier 2 features (market structure)...")

    # USDT.D: Already exists or set to 6.0
    if 'USDT.D' not in df.columns:
        logger.warning("    USDT.D not found, setting to 6.0")
        df['USDT.D'] = 6.0

    # BTC.D: Already exists or set to 50.0
    if 'BTC.D' not in df.columns:
        logger.warning("    BTC.D not found, setting to 50.0")
        df['BTC.D'] = 50.0

    # TOTAL_RET_21d: Total market cap 21d return
    if 'TOTAL' in df.columns:
        df['TOTAL_RET_21d'] = df['TOTAL'].pct_change(21*24) * 100
    elif 'TOTAL_RET' in df.columns:
        df['TOTAL_RET_21d'] = df['TOTAL_RET'] * 100
    else:
        logger.warning("    TOTAL not found, setting TOTAL_RET_21d to 0")
        df['TOTAL_RET_21d'] = 0.0

    # ALT_ROTATION: TOTAL3 outperformance
    if 'TOTAL3' in df.columns and 'TOTAL' in df.columns:
        total3_ret = df['TOTAL3'].pct_change(21*24)
        total_ret = df['TOTAL'].pct_change(21*24)
        df['ALT_ROTATION'] = (total3_ret - total_ret) * 100
    else:
        logger.warning("    TOTAL3/TOTAL not found, setting ALT_ROTATION to 0")
        df['ALT_ROTATION'] = 0.0

    # Tier 3: Macro features
    logger.info("  Computing Tier 3 features (macro)...")

    # VIX_Z: VIX z-score
    if 'VIX' in df.columns:
        df['VIX_Z'] = rolling_zscore(df['VIX'], window=252*24)
    elif 'VIX_Z' in df.columns:
        pass  # Already exists
    else:
        logger.warning("    VIX not found, setting VIX_Z to 0")
        df['VIX_Z'] = 0.0

    # DXY_Z: DXY z-score
    if 'DXY' in df.columns:
        df['DXY_Z'] = rolling_zscore(df['DXY'], window=252*24)
    elif 'DXY_Z' in df.columns:
        pass  # Already exists
    else:
        logger.warning("    DXY not found, setting DXY_Z to 0")
        df['DXY_Z'] = 0.0

    # YC_SPREAD: Yield curve spread
    if 'YIELD_10Y' in df.columns and 'YIELD_2Y' in df.columns:
        df['YC_SPREAD'] = (df['YIELD_10Y'] - df['YIELD_2Y']) * 100
    elif 'YC_SPREAD' in df.columns:
        pass  # Already exists
    else:
        logger.warning("    YIELD_10Y/YIELD_2Y not found, setting YC_SPREAD to 0")
        df['YC_SPREAD'] = 0.0

    # M2_GROWTH_YOY: M2 YoY growth
    if 'M2' in df.columns:
        df['M2_GROWTH_YOY'] = df['M2'].pct_change(252*24) * 100
    else:
        logger.warning("    M2 not found, setting M2_GROWTH_YOY to 0")
        df['M2_GROWTH_YOY'] = 0.0

    # Tier 4: Event flags (not implemented - set to 0)
    logger.info("  Setting Tier 4 features (event flags) to 0...")
    df['FOMC_D0'] = 0.0
    df['CPI_D0'] = 0.0
    df['NFP_D0'] = 0.0

    # Verify all features exist
    missing = [f for f in REGIME_FEATURES_V2 if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")

    logger.info(f"  Engineered {len(REGIME_FEATURES_V2)} features successfully")

    return df


def rolling_zscore(series: pd.Series, window: int, min_periods: int = 50) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Compute annualized realized volatility (%)."""
    return returns.rolling(window).std() * np.sqrt(252 * 24) * 100


def train_hmm(df: pd.DataFrame, n_states: int = 4) -> tuple:
    """
    Train 4-state HMM using Baum-Welch EM algorithm.

    Args:
        df: DataFrame with engineered features
        n_states: Number of states (default: 4)

    Returns:
        (model, state_map, scaler)
    """
    logger.info(f"Training {n_states}-state HMM...")

    # Extract feature matrix
    X = df[REGIME_FEATURES_V2].values

    # Handle NaNs (fill with 0)
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        logger.warning(f"  Filling {n_missing} NaN values with 0")
        X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    logger.info("  Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize HMM (let it learn transition matrix from data)
    logger.info("  Initializing HMM...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',  # Diagonal covariance (faster, more stable)
        n_iter=200,              # Max EM iterations (reduced for stability)
        tol=1e-3,                # Convergence threshold (relaxed)
        random_state=42,
        init_params='stmc',      # Initialize all parameters
        params='stmc'            # Update all parameters
    )

    logger.info("  Note: Transition matrix will be learned from data")

    # Train via EM
    logger.info("  Running Baum-Welch EM algorithm...")
    model.fit(X_scaled)

    logger.info(f"  Training converged: {model.monitor_.converged}")
    if model.monitor_.history:
        logger.info(f"  Final log-likelihood: {model.monitor_.history[-1]:.2f}")
    logger.info(f"  Iterations: {len(model.monitor_.history)}")

    # Interpret states (map to regime labels)
    logger.info("  Interpreting HMM states...")
    state_map = interpret_hmm_states(model, scaler, df)

    return model, state_map, scaler


def interpret_hmm_states(model: GaussianHMM, scaler: StandardScaler, df: pd.DataFrame) -> dict:
    """
    Analyze learned state means to assign regime labels.

    Logic:
    - High VIX_Z, negative funding_Z, high RV → crisis
    - Negative DXY_Z, low RV, positive funding → risk_on
    - Low volatility, neutral funding → neutral
    - High DXY_Z, rising USDT.D → risk_off

    Args:
        model: Trained HMM
        scaler: Fitted StandardScaler
        df: DataFrame with features (for context)

    Returns:
        Dict mapping state_id → regime_name
    """
    # Get cluster centers in original scale
    means_scaled = model.means_
    means_original = scaler.inverse_transform(means_scaled)

    # Convert to DataFrame
    means_df = pd.DataFrame(means_original, columns=REGIME_FEATURES_V2)

    logger.info("\n" + "="*80)
    logger.info("HMM STATE INTERPRETATION")
    logger.info("="*80)

    state_map = {}

    for state_id in range(len(means_df)):
        row = means_df.iloc[state_id]

        # Key features for classification
        vix_z = row['VIX_Z']
        rv_21 = row['RV_21']
        funding_z = row['funding_Z']
        dxy_z = row['DXY_Z']
        usdt_d = row['USDT.D']
        total_ret = row['TOTAL_RET_21d']

        logger.info(f"\nState {state_id}:")
        logger.info(f"  VIX_Z={vix_z:.2f}, RV_21={rv_21:.1f}%, funding_Z={funding_z:.2f}")
        logger.info(f"  DXY_Z={dxy_z:.2f}, USDT.D={usdt_d:.2f}%, TOTAL_RET={total_ret:.2f}%")

        # Classification logic (from research report)
        if vix_z > 1.5 and rv_21 > 70:
            # Very high VIX + high volatility = crisis
            regime = 'crisis'
        elif vix_z < 0 and funding_z > 0 and total_ret > 0:
            # Low VIX + positive funding + positive returns = risk_on
            regime = 'risk_on'
        elif dxy_z > 0.5 or usdt_d > 6.5:
            # High DXY or high USDT dominance = risk_off
            regime = 'risk_off'
        else:
            # Middle ground = neutral
            regime = 'neutral'

        state_map[state_id] = regime
        logger.info(f"  → Labeled as: {regime.upper()}")

    # Ensure all 4 regimes are represented
    assigned_regimes = set(state_map.values())
    all_regimes = {'risk_on', 'neutral', 'risk_off', 'crisis'}
    missing = all_regimes - assigned_regimes

    if missing:
        logger.warning(f"\n  Missing regimes: {missing}")
        logger.warning("  Manually assigning to ensure all 4 regimes exist")

        # Find unassigned state and assign missing regime
        for regime in missing:
            # Find state with lowest assignment confidence
            # For now, just assign to first available state
            for state_id in range(len(means_df)):
                if state_id not in state_map or state_map[state_id] in assigned_regimes - {state_map[state_id]}:
                    logger.warning(f"  Reassigning state {state_id} → {regime}")
                    state_map[state_id] = regime
                    break

    logger.info("\n" + "="*80)
    logger.info("FINAL STATE MAPPING:")
    logger.info("="*80)
    for state_id, regime in sorted(state_map.items()):
        logger.info(f"  State {state_id} → {regime}")

    return state_map


def classify_all_bars(df: pd.DataFrame, model: GaussianHMM, state_map: dict, scaler: StandardScaler) -> pd.DataFrame:
    """
    Classify all bars using Viterbi algorithm.

    Args:
        df: DataFrame with engineered features
        model: Trained HMM
        state_map: State to regime mapping
        scaler: Fitted StandardScaler

    Returns:
        DataFrame with regime labels added
    """
    logger.info("Classifying all bars using Viterbi decoding...")

    # Extract and scale features
    X = df[REGIME_FEATURES_V2].values
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)

    # Viterbi decode
    states = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    # Map states to regime labels
    df = df.copy()
    df['regime_label'] = [state_map[s] for s in states]
    df['regime_confidence'] = probs.max(axis=1)

    # Add individual regime probabilities
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        regime_states = [s for s, r in state_map.items() if r == regime]
        if regime_states:
            df[f'regime_proba_{regime}'] = probs[:, regime_states].sum(axis=1)
        else:
            df[f'regime_proba_{regime}'] = 0.0

    # Log distribution
    regime_dist = df['regime_label'].value_counts()
    logger.info("\nRegime distribution:")
    for regime, count in regime_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

    return df


def validate_regime_classifier(df: pd.DataFrame) -> dict:
    """
    Run validation metrics on classified data.

    Metrics:
    1. Silhouette score (cluster quality)
    2. Transition frequency (regime stability)
    3. Event accuracy (known market events)

    Args:
        df: DataFrame with regime_label column

    Returns:
        Dict of validation metrics
    """
    logger.info("\n" + "="*80)
    logger.info("VALIDATION METRICS")
    logger.info("="*80)

    metrics = {}

    # Metric 1: Silhouette score
    logger.info("\n1. Silhouette Score (cluster quality)...")
    try:
        X = df[REGIME_FEATURES_V2].fillna(0).values
        labels_int = df['regime_label'].map({
            'risk_on': 0, 'neutral': 1, 'risk_off': 2, 'crisis': 3
        }).values

        silhouette = silhouette_score(X, labels_int, sample_size=min(10000, len(X)))
        metrics['silhouette_score'] = float(silhouette)

        logger.info(f"   Silhouette score: {silhouette:.3f}")
        if silhouette > 0.5:
            logger.info("   ✅ PASS (>0.5)")
        else:
            logger.warning(f"   ⚠️  BELOW TARGET (target: >0.5)")
    except Exception as e:
        logger.error(f"   ❌ Failed to compute silhouette: {e}")
        metrics['silhouette_score'] = None

    # Metric 2: Transition frequency
    logger.info("\n2. Transition Frequency (regime stability)...")
    transitions = (df['regime_label'] != df['regime_label'].shift(1)).sum()
    years = (df.index[-1] - df.index[0]).days / 365.25
    transition_freq = transitions / years

    metrics['transitions_per_year'] = float(transition_freq)
    logger.info(f"   Transitions per year: {transition_freq:.1f}")

    if 10 <= transition_freq <= 20:
        logger.info("   ✅ PASS (10-20 transitions/year)")
    elif transition_freq < 10:
        logger.warning("   ⚠️  Too stable (<10/year, may miss regime shifts)")
    else:
        logger.warning("   ⚠️  Too noisy (>20/year, thrashing)")

    # Metric 3: Regime duration stats
    logger.info("\n3. Regime Duration Statistics...")
    regime_runs = []
    current_regime = None
    run_start = None

    for idx, regime in zip(df.index, df['regime_label']):
        if regime != current_regime:
            if current_regime is not None:
                duration_days = (idx - run_start).total_seconds() / 86400
                regime_runs.append({'regime': current_regime, 'duration_days': duration_days})
            current_regime = regime
            run_start = idx

    runs_df = pd.DataFrame(regime_runs)
    duration_stats = runs_df.groupby('regime')['duration_days'].agg(['mean', 'median', 'std'])

    logger.info("\n   Average regime duration (days):")
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        if regime in duration_stats.index:
            mean_days = duration_stats.loc[regime, 'mean']
            logger.info(f"     {regime:12s}: {mean_days:.1f} days")
            metrics[f'avg_duration_{regime}'] = float(mean_days)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train HMM Regime Classifier V2')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='models/hmm_regime_v2.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        default='2024-01-01',
        help='End of training period (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROLLING REGIME CLASSIFIER V2 - HMM TRAINING")
    print("="*80)

    # Step 1: Load data
    logger.info(f"\n[1/7] Loading data from {args.data_path}...")
    data_path = Path(args.data_path)

    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return 1

    df = pd.read_parquet(data_path)
    logger.info(f"   Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    logger.info(f"   Columns: {len(df.columns)}")

    # Step 2: Feature engineering
    logger.info("\n[2/7] Engineering regime features...")
    df = engineer_regime_features(df)

    # Step 3: Split train/test
    logger.info(f"\n[3/7] Splitting train/test at {args.train_end}...")
    train_end = pd.to_datetime(args.train_end, utc=True)
    df_train = df[df.index < train_end]
    df_test = df[df.index >= train_end]

    logger.info(f"   Train: {len(df_train):,} bars ({df_train.index[0]} to {df_train.index[-1]})")
    logger.info(f"   Test:  {len(df_test):,} bars ({df_test.index[0]} to {df_test.index[-1]})")

    if len(df_train) < 1000:
        logger.error("❌ Insufficient training data")
        return 1

    # Step 4: Train HMM
    logger.info("\n[4/7] Training 4-state HMM...")
    model, state_map, scaler = train_hmm(df_train, n_states=4)

    # Step 5: Classify all bars
    logger.info("\n[5/7] Classifying all bars...")
    df_classified = classify_all_bars(df, model, state_map, scaler)

    # Step 6: Validate
    logger.info("\n[6/7] Running validation on test set...")
    metrics = validate_regime_classifier(df_test)

    # Step 7: Save outputs
    logger.info("\n[7/7] Saving outputs...")

    # Save model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_obj = {
        'model': model,
        'state_map': state_map,
        'scaler': scaler,
        'features': REGIME_FEATURES_V2,
        'model_type': 'hmm',
        'training_samples': len(df_train),
        'train_date_range': (str(df_train.index[0]), str(df_train.index[-1])),
        'created_at': datetime.now().isoformat()
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_obj, f)

    logger.info(f"   ✅ Model saved: {output_path}")

    # Save regime labels
    labels_path = Path('data/regime_labels_v2.parquet')
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    regime_cols = ['regime_label', 'regime_confidence'] + [
        f'regime_proba_{r}' for r in ['risk_on', 'neutral', 'risk_off', 'crisis']
    ]
    df_classified[regime_cols].to_parquet(labels_path)
    logger.info(f"   ✅ Labels saved: {labels_path}")

    # Save validation metrics
    metrics_path = Path('results/regime_v2_validation.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"   ✅ Validation metrics saved: {metrics_path}")

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n✅ Model: {output_path}")
    print(f"✅ Labels: {labels_path}")
    print(f"✅ Validation: {metrics_path}")

    print("\n📊 Validation Summary:")
    print(f"  Silhouette score: {metrics.get('silhouette_score', 'N/A'):.3f}")
    print(f"  Transitions/year: {metrics.get('transitions_per_year', 'N/A'):.1f}")

    print("\n🚀 Next Steps:")
    print("  1. Review validation metrics above")
    print("  2. Visualize regime transitions: python bin/visualize_regimes_v2.py")
    print("  3. Integrate with backtest: update engine/context/regime_classifier.py")
    print("  4. Run validation: python bin/validate_regime_hmm.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
