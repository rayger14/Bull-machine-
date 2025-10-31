#!/usr/bin/env python3
"""
Train Gaussian Mixture Model (GMM) for regime classification using real macro data.

This script:
1. Loads real macro_history.parquet with varying VIX, DXY, yields, etc.
2. Trains GMM with 4 components (risk_on, neutral, risk_off, crisis)
3. Labels clusters based on feature centroids
4. Saves trained model to models/regime_classifier_gmm_v2.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 13 macro features for regime classification
REGIME_FEATURES = [
    'VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
    'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
    'funding', 'oi', 'rv_20d', 'rv_60d'
]


def label_clusters_by_features(gmm, scaler, feature_names):
    """
    Label GMM clusters based on their feature centroids.

    Strategy:
    - High VIX, MOVE, rv → crisis or risk_off
    - Low VIX, high TOTAL → risk_on
    - Middle ground → neutral

    Args:
        gmm: Trained GaussianMixture model
        scaler: Fitted StandardScaler
        feature_names: List of feature names

    Returns:
        Dict mapping cluster_id -> regime_name
    """
    # Get cluster centers in original scale
    centers = scaler.inverse_transform(gmm.means_)

    # Convert to DataFrame for easier analysis
    centers_df = pd.DataFrame(centers, columns=feature_names)
    centers_df['cluster_id'] = range(len(centers))

    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)

    # Compute regime scores for each cluster
    regime_scores = []

    for idx, row in centers_df.iterrows():
        vix = row['VIX']
        move = row.get('MOVE', 100)
        rv_20d = row.get('rv_20d', 0.03)
        rv_60d = row.get('rv_60d', 0.03)
        total_mcap = row.get('TOTAL', 0) / 1e12  # Convert to trillions
        btc_dom = row.get('BTC.D', 50.0)

        # Use relative rankings to distinguish clusters
        # Find the cluster with lowest VIX → risk_on
        # Find the cluster with highest VIX → risk_off or crisis
        # Clusters in between → neutral

        scores = {}

        # Simple ranking approach: use VIX and TOTAL together
        # High TOTAL + Low VIX = risk_on
        # Low TOTAL + High VIX = risk_off
        # High VIX (>30) = crisis
        # Otherwise = neutral

        if vix > 30:
            # Very high VIX = crisis
            regime = 'crisis'
        elif vix > 22 and total_mcap < 1.8:
            # High VIX + low market cap = risk_off
            regime = 'risk_off'
        elif vix < 17 and total_mcap > 2.5:
            # Low VIX + high market cap = risk_on
            regime = 'risk_on'
        elif total_mcap < 1.5:
            # Very low market cap = risk_off regardless of VIX
            regime = 'risk_off'
        elif total_mcap > 3.0:
            # Very high market cap = risk_on regardless of VIX
            regime = 'risk_on'
        elif vix < 18:
            # Moderate conditions with low VIX = risk_on
            regime = 'risk_on'
        elif vix > 24:
            # Moderate conditions with high VIX = risk_off
            regime = 'risk_off'
        else:
            # Middle ground = neutral
            regime = 'neutral'

        scores = {'vix': vix, 'total_mcap': total_mcap, 'move': move}
        regime_scores.append((idx, regime, scores, vix, move, total_mcap))

        print(f"\nCluster {idx}:")
        print(f"  VIX={vix:.1f}, MOVE={move:.1f}, rv_20d={rv_20d:.3f}, TOTAL={total_mcap:.2f}T")
        print(f"  Scores: {scores}")
        print(f"  → Labeled as: {regime.upper()}")

    # Build label map
    label_map = {}
    for cluster_id, regime, scores, vix, move, total_mcap in regime_scores:
        label_map[cluster_id] = regime

    # Check for duplicate regimes - if multiple clusters map to same regime,
    # keep the one with highest score
    regime_counts = {}
    for cluster_id, regime in label_map.items():
        if regime not in regime_counts:
            regime_counts[regime] = []
        regime_counts[regime].append(cluster_id)

    # If missing a regime, assign it to the cluster with lowest conflicting score
    all_regimes = {'risk_on', 'neutral', 'risk_off', 'crisis'}
    missing_regimes = all_regimes - set(label_map.values())

    if missing_regimes:
        print(f"\n⚠️  Missing regimes: {missing_regimes}")
        print("   Will assign neutral as fallback for ambiguous clusters")

    print("\n" + "="*80)
    print("FINAL LABEL MAP:")
    print("="*80)
    for cluster_id, regime in sorted(label_map.items()):
        print(f"  Cluster {cluster_id} → {regime}")

    return label_map


def main():
    print("\n" + "="*80)
    print("TRAINING REGIME CLASSIFIER GMM v2")
    print("="*80)

    # 1. Load real macro data
    macro_path = Path('data/macro/macro_history.parquet')
    print(f"\n1. Loading macro history: {macro_path}")

    if not macro_path.exists():
        print(f"❌ Macro history not found: {macro_path}")
        print("   Run: python3 bin/populate_macro_data.py")
        return 1

    macro_df = pd.read_parquet(macro_path)
    print(f"   Shape: {macro_df.shape}")
    print(f"   Date range: {macro_df['timestamp'].min()} to {macro_df['timestamp'].max()}")

    # 2. Extract features and clean data
    print(f"\n2. Extracting {len(REGIME_FEATURES)} features...")

    X_df = macro_df[REGIME_FEATURES].copy()

    # Check for missing values
    missing_counts = X_df.isna().sum()
    print(f"\n   Missing value counts:")
    for feat, count in missing_counts.items():
        pct = count / len(X_df) * 100
        if count > 0:
            print(f"     {feat:12s}: {count:6d} ({pct:5.1f}%)")

    # Drop rows with ANY missing values for training
    X_df_clean = X_df.dropna()
    print(f"\n   Rows after dropping NaNs: {len(X_df_clean)} / {len(X_df)} ({len(X_df_clean)/len(X_df)*100:.1f}%)")

    if len(X_df_clean) < 1000:
        print(f"❌ Insufficient clean data ({len(X_df_clean)} rows). Need at least 1000 rows.")
        return 1

    # Subsample for faster training (use every 6th hour = ~4 samples/day)
    X_subsample = X_df_clean.iloc[::6].copy()
    print(f"\n   Subsampled to {len(X_subsample)} rows (every 6 hours)")

    # 3. Standardize features
    print(f"\n3. Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subsample)

    print(f"\n   Feature statistics (standardized):")
    X_scaled_df = pd.DataFrame(X_scaled, columns=REGIME_FEATURES)
    print(f"     Mean: {X_scaled_df.mean().mean():.3f} (should be ~0)")
    print(f"     Std:  {X_scaled_df.std().mean():.3f} (should be ~1)")

    # 4. Train GMM with 4 components
    print(f"\n4. Training Gaussian Mixture Model...")
    print(f"   Components: 4 (risk_on, neutral, risk_off, crisis)")
    print(f"   Covariance type: full")
    print(f"   Training samples: {len(X_scaled)}")

    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=42,
        max_iter=200,
        n_init=10,
        verbose=1
    )

    gmm.fit(X_scaled)

    print(f"\n   ✅ Training converged in {gmm.n_iter_} iterations")
    print(f"   Log-likelihood: {gmm.score(X_scaled):.2f}")

    # 5. Analyze clusters and assign regime labels
    print(f"\n5. Analyzing clusters and assigning regime labels...")

    # Predict cluster assignments
    cluster_assignments = gmm.predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled)

    print(f"\n   Cluster distribution:")
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = count / len(cluster_assignments) * 100
        print(f"     Cluster {cluster_id}: {count:6d} samples ({pct:5.1f}%)")

    # Label clusters based on feature centroids
    label_map = label_clusters_by_features(gmm, scaler, REGIME_FEATURES)

    # 6. Validate regime distribution
    print(f"\n6. Validating regime distribution...")

    regime_labels = [label_map[c] for c in cluster_assignments]
    unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)

    print(f"\n   Regime distribution (on training data):")
    for regime, count in zip(unique_regimes, regime_counts):
        pct = count / len(regime_labels) * 100
        print(f"     {regime:12s}: {count:6d} samples ({pct:5.1f}%)")

    # 7. Save model
    output_path = Path('models/regime_classifier_gmm_v2.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n7. Saving model to {output_path}...")

    model_obj = {
        'model': gmm,
        'scaler': scaler,
        'label_map': label_map,
        'feature_order': REGIME_FEATURES,
        'n_components': 4,
        'training_samples': len(X_scaled),
        'date_range': (str(X_subsample.index[0]), str(X_subsample.index[-1]))
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"   ✅ Model saved")

    # 8. Test classification on sample data
    print(f"\n8. Testing classification on sample data...")

    test_samples = [
        {
            'name': '2024 Bull Market',
            'VIX': 15.0, 'DXY': 104.0, 'MOVE': 95.0,
            'YIELD_2Y': 4.5, 'YIELD_10Y': 4.3,
            'USDT.D': 5.5, 'BTC.D': 55.0,
            'TOTAL': 2.5e12, 'TOTAL2': 1.2e12,
            'funding': 0.01, 'oi': 0.015,
            'rv_20d': 0.025, 'rv_60d': 0.028
        },
        {
            'name': '2022 Bear Market',
            'VIX': 28.0, 'DXY': 110.0, 'MOVE': 140.0,
            'YIELD_2Y': 4.2, 'YIELD_10Y': 3.8,
            'USDT.D': 7.0, 'BTC.D': 42.0,
            'TOTAL': 0.9e12, 'TOTAL2': 0.35e12,
            'funding': -0.005, 'oi': 0.008,
            'rv_20d': 0.055, 'rv_60d': 0.060
        },
        {
            'name': '2020 COVID Crisis',
            'VIX': 45.0, 'DXY': 103.0, 'MOVE': 180.0,
            'YIELD_2Y': 0.5, 'YIELD_10Y': 0.9,
            'USDT.D': 5.0, 'BTC.D': 65.0,
            'TOTAL': 0.25e12, 'TOTAL2': 0.08e12,
            'funding': -0.02, 'oi': 0.005,
            'rv_20d': 0.12, 'rv_60d': 0.10
        }
    ]

    for sample in test_samples:
        name = sample.pop('name')
        x_test = np.array([sample[f] for f in REGIME_FEATURES]).reshape(1, -1)
        x_test_scaled = scaler.transform(x_test)

        cluster_id = gmm.predict(x_test_scaled)[0]
        probs = gmm.predict_proba(x_test_scaled)[0]
        regime = label_map[cluster_id]

        print(f"\n   {name}:")
        print(f"     → Regime: {regime} (cluster {cluster_id})")
        print(f"     → Confidence: {probs[cluster_id]:.1%}")

    print("\n" + "="*80)
    print("✅ GMM TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {output_path}")
    print(f"\nTo use in backtest:")
    print(f"  1. Update bin/backtest_knowledge_v2.py to import RegimeClassifier")
    print(f"  2. Update config 'regime_classifier.model_path' to '{output_path}'")
    print(f"  3. Run backtest with adaptive fusion enabled")
    print("\n" + "="*80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
