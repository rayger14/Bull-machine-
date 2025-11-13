#!/usr/bin/env python3
"""
GMM v3.2: Balanced Regime Classifier with Manual Period Labeling

FIXES FROM v3.1:
- Includes 2024 bull market data (v3.1 trained only on 2022-2023 bear)
- Manual time-based labeling to ensure balanced regime distribution
- Prevents over-classification as "crisis"

Changes:
1. Load data from both 2022-2023 (bear) and 2024 (bull) periods
2. Manually label known market conditions:
   - 2022 Q1-Q2: crisis (Luna crash, rate hikes)
   - 2022 Q3-Q4: risk_off (FTX collapse, capitulation)
   - 2023 Q1-Q2: neutral (recovery, consolidation)
   - 2023 Q3-Q4: risk_on (ETF speculation)
   - 2024 Q1-Q2: risk_on (ETF approval, new ATH)
   - 2024 Q3-Q4: neutral/risk_on (consolidation, re-accumulation)
3. Train GMM with balanced examples of all 4 regimes
4. Validate classification on holdout 2024 data

Target Distribution: 25% risk_on, 40% neutral, 25% risk_off, 10% crisis

Output: models/regime_gmm_v3.2_balanced.pkl

Usage:
    python3 bin/train_gmm_v3.2_balanced.py
    python3 bin/train_gmm_v3.2_balanced.py --k 4
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Feature Set (19 P1 features - same as v3.1)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_FEATURES = [
    # Core macro z-scores
    'VIX_Z', 'DXY_Z', 'YC_SPREAD', 'YC_Z',
    # Bitcoin dominance & stablecoins
    'BTC.D_Z', 'USDT.D_Z',
    # Realized volatility (multi-period)
    'RV_7', 'RV_20', 'RV_30', 'RV_60',
    # Funding & open interest
    'funding_Z', 'OI_CHANGE',
    # Breadth & rotation
    'TOTAL_RET', 'TOTAL2_RET', 'TOTAL3_RET', 'ALT_ROTATION',
    # Derivatives
    'VOL_TERM', 'SKEW_25D', 'PERP_BASIS'
]

# ─────────────────────────────────────────────────────────────────────────────
# Manual Period Labeling (based on known market conditions)
# ─────────────────────────────────────────────────────────────────────────────
MANUAL_LABELS = [
    # 2022 Bear Market
    ('2022-01-01', '2022-06-30', 'crisis'),      # Luna crash, Fed rate hikes
    ('2022-07-01', '2022-11-30', 'risk_off'),    # FTX collapse, capitulation
    ('2022-12-01', '2022-12-31', 'risk_off'),    # Year-end washout

    # 2023 Recovery
    ('2023-01-01', '2023-03-31', 'neutral'),     # Banking crisis, SVB
    ('2023-04-01', '2023-06-30', 'neutral'),     # Consolidation
    ('2023-07-01', '2023-09-30', 'risk_on'),     # Summer rally
    ('2023-10-01', '2023-12-31', 'risk_on'),     # ETF speculation

    # 2024 Bull Market
    ('2024-01-01', '2024-03-31', 'risk_on'),     # ETF approval, new ATH
    ('2024-04-01', '2024-06-30', 'neutral'),     # Consolidation, halving
    ('2024-07-01', '2024-09-30', 'neutral'),     # Summer chop
    ('2024-10-01', '2024-12-31', 'risk_on'),     # Q4 rally
]

def label_time_periods(df, manual_labels):
    """
    Apply manual time-based regime labels to dataframe.

    Args:
        df: DataFrame with DatetimeIndex
        manual_labels: List of (start_date, end_date, regime) tuples

    Returns:
        df with 'regime_label' column
    """
    df = df.copy()
    df['regime_label'] = 'neutral'  # Default

    for start_date, end_date, regime in manual_labels:
        mask = (df.index >= start_date) & (df.index <= end_date)
        df.loc[mask, 'regime_label'] = regime
        print(f"  {start_date} to {end_date}: {regime:10s} ({mask.sum():,} samples)")

    return df

def balance_training_set(df, target_dist={'risk_on': 0.25, 'neutral': 0.40, 'risk_off': 0.25, 'crisis': 0.10}, max_samples=5000):
    """
    Balance training set to match target distribution.

    Args:
        df: DataFrame with 'regime_label' column
        target_dist: Target distribution dict {regime: proportion}
        max_samples: Maximum total samples

    Returns:
        Balanced DataFrame
    """
    print("\n" + "="*80)
    print("BALANCING TRAINING SET")
    print("="*80)

    # Count current distribution
    regime_counts = df['regime_label'].value_counts()
    print(f"\nOriginal distribution ({len(df):,} samples):")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"  {regime:12s}: {count:6,} ({pct:5.1f}%)")

    # Calculate target samples per regime
    target_samples = {}
    for regime, proportion in target_dist.items():
        target_samples[regime] = int(max_samples * proportion)

    print(f"\nTarget distribution ({max_samples:,} samples):")
    for regime, target in target_samples.items():
        pct = target / max_samples * 100
        print(f"  {regime:12s}: {target:6,} ({pct:5.1f}%)")

    # Sample from each regime
    balanced_dfs = []
    for regime, target_n in target_samples.items():
        regime_df = df[df['regime_label'] == regime]

        if len(regime_df) == 0:
            print(f"\n⚠️  WARNING: No samples for {regime}, skipping")
            continue

        if len(regime_df) < target_n:
            print(f"\n⚠️  WARNING: {regime} has only {len(regime_df)} samples (wanted {target_n}), using all")
            sampled = regime_df
        else:
            sampled = regime_df.sample(n=target_n, random_state=42)

        balanced_dfs.append(sampled)

    balanced_df = pd.concat(balanced_dfs, axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=42)  # Shuffle

    # Final distribution
    final_counts = balanced_df['regime_label'].value_counts()
    print(f"\nFinal balanced distribution ({len(balanced_df):,} samples):")
    for regime, count in final_counts.items():
        pct = count / len(balanced_df) * 100
        print(f"  {regime:12s}: {count:6,} ({pct:5.1f}%)")

    return balanced_df

def label_clusters_supervised(gmm, scaler, feature_names, X_labeled, y_labels):
    """
    Label clusters using supervised approach: assign each cluster to the most common regime.

    Args:
        gmm: Trained GaussianMixture model
        scaler: Fitted RobustScaler
        feature_names: List of feature names
        X_labeled: Training feature matrix (original scale)
        y_labels: Training regime labels

    Returns:
        Dict mapping cluster_id → regime_name
    """
    print("\n" + "="*80)
    print("CLUSTER LABELING: Supervised Majority Voting")
    print("="*80)

    # Scale features and predict clusters
    X_scaled = scaler.transform(X_labeled)
    cluster_assignments = gmm.predict(X_scaled)

    n_components = gmm.n_components
    label_map = {}

    print(f"\n{'Cluster':>8} {'Size':>8} {'Label':>12} | Regime Distribution")
    print("-" * 80)

    for cluster_id in range(n_components):
        # Find all samples in this cluster
        cluster_mask = (cluster_assignments == cluster_id)
        cluster_labels = y_labels[cluster_mask]

        if len(cluster_labels) == 0:
            print(f"{cluster_id:>8d} {0:>8d} {'EMPTY':>12s} | No samples")
            label_map[cluster_id] = 'neutral'  # Fallback
            continue

        # Majority vote
        regime_counts = pd.Series(cluster_labels).value_counts()
        most_common_regime = regime_counts.index[0]
        label_map[cluster_id] = most_common_regime

        # Distribution
        dist_str = " | ".join([f"{r}={c}" for r, c in regime_counts.items()])
        print(f"{cluster_id:>8d} {len(cluster_labels):>8,} {most_common_regime.upper():>12s} | {dist_str}")

    # Summary
    unique_labels = set(label_map.values())
    print(f"\n{'='*80}")
    print(f"LABEL DISTRIBUTION:")
    print(f"  Unique regimes: {len(unique_labels)} / {n_components} clusters")
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        count = list(label_map.values()).count(regime)
        if count > 0:
            print(f"    {regime:12s}: {count} cluster(s)")

    return label_map

def validate_on_holdout(gmm, scaler, label_map, df_holdout, features):
    """
    Validate model on holdout data.

    Args:
        gmm: Trained model
        scaler: Fitted scaler
        label_map: Cluster to regime mapping
        df_holdout: Holdout DataFrame with 'regime_label' column
        features: Feature list

    Returns:
        Classification report
    """
    print("\n" + "="*80)
    print("VALIDATION ON HOLDOUT DATA")
    print("="*80)

    X_holdout = df_holdout[features].values
    y_true = df_holdout['regime_label'].values

    # Predict
    X_scaled = scaler.transform(X_holdout)
    cluster_preds = gmm.predict(X_scaled)
    y_pred = np.array([label_map[c] for c in cluster_preds])

    # Distribution
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\nPredicted regime distribution ({len(y_pred):,} samples):")
    for regime, count in pred_counts.items():
        pct = count / len(y_pred) * 100
        print(f"  {regime:12s}: {count:6,} ({pct:5.1f}%)")

    # Accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nAccuracy: {accuracy:.1%}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Key validation: Check if 2024 is NOT 100% crisis
    crisis_pct = (y_pred == 'crisis').mean() * 100
    print(f"\n{'='*80}")
    if crisis_pct > 50:
        print(f"⚠️  WARNING: Model classifies {crisis_pct:.1f}% of 2024 as 'crisis'")
        print(f"   This suggests over-fitting to bear market conditions")
    else:
        print(f"✅ PASS: Model classifies {crisis_pct:.1f}% of 2024 as 'crisis' (expected <50%)")

    return accuracy, pred_counts

def main():
    parser = argparse.ArgumentParser(description="GMM v3.2: Balanced regime trainer")
    parser.add_argument('--k', type=int, default=4, help="Number of clusters (4 recommended)")
    parser.add_argument('--max-samples', type=int, default=5000, help="Max training samples")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GMM V3.2: BALANCED REGIME CLASSIFIER TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: K={args.k} clusters with balanced 2022-2024 data")

    # ─── Load Macro Data ───
    macro_path = Path('data/macro/macro_history.parquet')
    if not macro_path.exists():
        print(f"\n❌ Macro history not found: {macro_path}")
        print("   Run: python3 bin/add_p1_features_to_macro.py first")
        return 1

    print(f"\n📖 Loading macro history: {macro_path}")
    df = pd.read_parquet(macro_path)

    # Handle RangeIndex - set timestamp as index
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        print(f"   Set 'timestamp' column as index")

    print(f"   Loaded: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # ─── Select Features ───
    available_features = [f for f in REGIME_FEATURES if f in df.columns]
    missing_features = [f for f in REGIME_FEATURES if f not in df.columns]

    print(f"\n✅ Available features: {len(available_features)} / {len(REGIME_FEATURES)}")
    if missing_features:
        print(f"⚠️  Missing features ({len(missing_features)}): {missing_features}")
        # Set missing features to 0
        for feat in missing_features:
            df[feat] = 0.0
        available_features = REGIME_FEATURES  # Use all features (zero-filled)

    if len(available_features) < 10:
        print(f"\n❌ Too few features available ({len(available_features)}). Need at least 10.")
        return 1

    # ─── Apply Manual Labels ───
    print(f"\n{'='*80}")
    print("APPLYING MANUAL TIME-BASED LABELS")
    print("="*80)

    df = label_time_periods(df, MANUAL_LABELS)

    # ─── Split Train (2022-2024 Q1-Q3) and Holdout (2024 Q4) ───
    print(f"\n{'='*80}")
    print("SPLITTING TRAIN/HOLDOUT")
    print("="*80)

    # Train on 2022-2024 Q1-Q3 (includes bull market data!)
    df_train = df[(df.index >= '2022-01-01') & (df.index <= '2024-09-30')]
    # Holdout on 2024 Q4 (out-of-sample)
    df_holdout = df[(df.index >= '2024-10-01') & (df.index <= '2024-12-31')]

    print(f"\nTrain set (2022-2024 Q1-Q3): {len(df_train):,} samples")
    print(f"Holdout set (2024 Q4):       {len(df_holdout):,} samples")

    # ─── Balance Training Set ───
    # Fill NaN values with 0 for features, then select needed columns
    df_train_clean = df_train[available_features + ['regime_label']].fillna(0)
    df_train_balanced = balance_training_set(
        df_train_clean,
        target_dist={'risk_on': 0.20, 'neutral': 0.30, 'risk_off': 0.35, 'crisis': 0.15},
        max_samples=args.max_samples
    )

    X_train = df_train_balanced[available_features].values
    y_train = df_train_balanced['regime_label'].values

    # ─── Scale Features ───
    print(f"\n{'='*80}")
    print("SCALING FEATURES (RobustScaler)")
    print("="*80)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    print(f"  Scaled shape: {X_scaled.shape}")
    print(f"  Median: {np.median(X_scaled, axis=0).mean():.3f}")
    print(f"  IQR:    {np.percentile(X_scaled, 75, axis=0).mean() - np.percentile(X_scaled, 25, axis=0).mean():.3f}")

    # ─── Train GMM ───
    print(f"\n{'='*80}")
    print(f"TRAINING GMM (K={args.k})")
    print("="*80)

    gmm = GaussianMixture(
        n_components=args.k,
        covariance_type='full',
        random_state=42,
        max_iter=200,
        n_init=10
    )
    gmm.fit(X_scaled)

    print(f"\n  Converged:  {gmm.converged_}")
    print(f"  BIC:        {gmm.bic(X_scaled):,.2f}")
    print(f"  AIC:        {gmm.aic(X_scaled):,.2f}")

    # Silhouette score
    if args.k > 1:
        labels = gmm.predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
        print(f"  Silhouette: {silhouette:.3f}")

    # ─── Label Clusters (Supervised) ───
    label_map = label_clusters_supervised(gmm, scaler, available_features, X_train, y_train)

    # ─── Validate on 2024 Q4 Holdout ───
    df_holdout_clean = df_holdout[available_features + ['regime_label']].fillna(0)
    accuracy, pred_dist = validate_on_holdout(gmm, scaler, label_map, df_holdout_clean, available_features)

    # ─── Save Model ───
    print(f"\n{'='*80}")
    print("SAVING MODEL")
    print("="*80)

    model_path = Path('models/regime_gmm_v3.2_balanced.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'gmm': gmm,
        'scaler': scaler,
        'features': available_features,
        'label_map': label_map,
        'best_k': args.k,
        'confidence_threshold': 0.6,
        'trained_at': datetime.now().isoformat(),
        'version': '3.2',
        'notes': 'Balanced version with 2022-2024 data, manual time-based labels, supervised clustering',
        'validation_accuracy': accuracy,
        'validation_2024_distribution': dict(pred_dist)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved: {model_path}")
    print(f"   K = {args.k}")
    print(f"   Validation accuracy: {accuracy:.1%}")

    # ─── Summary ───
    print(f"\n{'='*80}")
    print("✅ GMM V3.2 TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"  Features:  {len(available_features)}")
    print(f"  Clusters:  {args.k}")
    print(f"  Training: 2022-2024 Q1-Q3 (includes bull market!)")
    print(f"  Validation accuracy: {accuracy:.1%}")
    print(f"\nHoldout (2024 Q4) regime distribution:")
    for regime, count in pred_dist.items():
        pct = count / pred_dist.sum() * 100
        print(f"  {regime:12s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nNext steps:")
    print(f"  1. Update config to use: models/regime_gmm_v3.2_balanced.pkl")
    print(f"  2. Run full backtest with new model")
    print(f"  3. Compare to baseline and v3.1 results")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
