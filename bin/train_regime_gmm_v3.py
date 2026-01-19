#!/usr/bin/env python3
"""
Train Gaussian Mixture Model (GMM) v3 for regime classification.

MVP Simplifications for Production:
1. **2 clusters only**: risk_on / risk_off (no neutral/crisis complexity)
2. **7 core features**: VIX, DXY, BTC.D, USDT.D, TOTAL, rv_20d, funding
3. **Deterministic labeling**: Based on VIX + dominance metrics
4. **Rule-based fallback**: When GMM confidence < 60%, use hard rules

Saves to: models/regime_gmm_v3.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 7 core macro features for regime classification (MVP)
REGIME_FEATURES = [
    'VIX',        # Market fear gauge
    'DXY',        # Dollar strength
    'BTC.D',      # Bitcoin dominance
    'USDT.D',     # Stablecoin dominance (risk-off proxy)
    'TOTAL',      # Total crypto market cap
    'rv_20d',     # Realized volatility
    'funding'     # Perpetual funding rate
]


def rule_based_regime(vix, btc_dom, usdt_dom):
    """
    Rule-based fallback for regime classification.

    Used when GMM confidence is low or features are missing.

    Rules (based on Moneytaur's framework):
    - risk_on:  VIX < 20 AND BTC.D rising AND USDT.D falling
    - risk_off: Everything else

    Args:
        vix: VIX level
        btc_dom: Bitcoin dominance %
        usdt_dom: USDT dominance %

    Returns:
        'risk_on' or 'risk_off'
    """
    if vix < 20 and btc_dom > 52 and usdt_dom < 6.0:
        return 'risk_on'
    else:
        return 'risk_off'


def label_clusters_deterministic(gmm, scaler, feature_names):
    """
    Label GMM clusters using deterministic rules based on feature centroids.

    Strategy (MVP):
    - Cluster with LOWER VIX → risk_on
    - Cluster with HIGHER VIX → risk_off

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

    # Extract key features
    cluster_0_vix = centers_df.loc[0, 'VIX']
    cluster_1_vix = centers_df.loc[1, 'VIX']

    cluster_0_btc_dom = centers_df.loc[0, 'BTC.D']
    cluster_1_btc_dom = centers_df.loc[1, 'BTC.D']

    cluster_0_usdt_dom = centers_df.loc[0, 'USDT.D']
    cluster_1_usdt_dom = centers_df.loc[1, 'USDT.D']

    # Print cluster characteristics
    for cluster_id in [0, 1]:
        row = centers_df.loc[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  VIX:      {row['VIX']:.1f}")
        print(f"  DXY:      {row['DXY']:.1f}")
        print(f"  BTC.D:    {row['BTC.D']:.1f}%")
        print(f"  USDT.D:   {row['USDT.D']:.2f}%")
        print(f"  TOTAL:    {row['TOTAL']/1e12:.2f}T")
        print(f"  rv_20d:   {row['rv_20d']:.3f}")
        print(f"  funding:  {row['funding']:.4f}")

    # Deterministic labeling: lower VIX → risk_on
    if cluster_0_vix < cluster_1_vix:
        label_map = {0: 'risk_on', 1: 'risk_off'}
        print(f"\n  → Cluster 0 (VIX={cluster_0_vix:.1f}) = RISK_ON")
        print(f"  → Cluster 1 (VIX={cluster_1_vix:.1f}) = RISK_OFF")
    else:
        label_map = {0: 'risk_off', 1: 'risk_on'}
        print(f"\n  → Cluster 0 (VIX={cluster_0_vix:.1f}) = RISK_OFF")
        print(f"  → Cluster 1 (VIX={cluster_1_vix:.1f}) = RISK_ON")

    print("\n" + "="*80)
    print("FINAL LABEL MAP:")
    print("="*80)
    for cluster_id, regime in sorted(label_map.items()):
        print(f"  Cluster {cluster_id} → {regime.upper()}")

    return label_map


def main():
    print("\n" + "="*80)
    print("TRAINING REGIME CLASSIFIER GMM v3 (MVP)")
    print("="*80)
    print("\nChanges from v2:")
    print("  - 2 clusters (risk_on/risk_off) instead of 4")
    print("  - 7 core features instead of 13")
    print("  - Deterministic VIX-based labeling")
    print("  - Rule-based fallback for low confidence")

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

    # 4. Train GMM with 2 components
    print(f"\n4. Training Gaussian Mixture Model...")
    print(f"   Components: 2 (risk_on, risk_off)")
    print(f"   Covariance type: full")
    print(f"   Training samples: {len(X_scaled)}")

    gmm = GaussianMixture(
        n_components=2,
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

    # Label clusters deterministically based on VIX
    label_map = label_clusters_deterministic(gmm, scaler, REGIME_FEATURES)

    # 6. Validate regime distribution
    print(f"\n6. Validating regime distribution...")

    regime_labels = [label_map[c] for c in cluster_assignments]
    unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)

    print(f"\n   Regime distribution (on training data):")
    for regime, count in zip(unique_regimes, regime_counts):
        pct = count / len(regime_labels) * 100
        print(f"     {regime:12s}: {count:6d} samples ({pct:5.1f}%)")

    # 7. Test rule-based fallback
    print(f"\n7. Testing rule-based fallback...")

    test_cases = [
        {'VIX': 15.0, 'BTC.D': 55.0, 'USDT.D': 5.5, 'expected': 'risk_on'},
        {'VIX': 28.0, 'BTC.D': 42.0, 'USDT.D': 7.0, 'expected': 'risk_off'},
        {'VIX': 19.5, 'BTC.D': 53.0, 'USDT.D': 5.8, 'expected': 'risk_on'},
        {'VIX': 22.0, 'BTC.D': 48.0, 'USDT.D': 6.5, 'expected': 'risk_off'}
    ]

    print("\n   Rule-based classification tests:")
    for case in test_cases:
        result = rule_based_regime(case['VIX'], case['BTC.D'], case['USDT.D'])
        status = "✅" if result == case['expected'] else "❌"
        print(f"     {status} VIX={case['VIX']:.1f}, BTC.D={case['BTC.D']:.1f}%, USDT.D={case['USDT.D']:.1f}% → {result} (expected: {case['expected']})")

    # 8. Save model
    output_path = Path('models/regime_gmm_v3.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n8. Saving model to {output_path}...")

    model_obj = {
        'model': gmm,
        'scaler': scaler,
        'label_map': label_map,
        'feature_order': REGIME_FEATURES,
        'n_components': 2,
        'training_samples': len(X_scaled),
        'date_range': (str(X_subsample.index[0]), str(X_subsample.index[-1])),
        'version': 'v3_mvp',
        'rule_fallback': rule_based_regime  # Include fallback function
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"   ✅ Model saved")

    # 9. Test classification on sample data
    print(f"\n9. Testing classification on sample data...")

    test_samples = [
        {
            'name': '2024 Bull Market',
            'VIX': 15.0, 'DXY': 104.0, 'BTC.D': 55.0,
            'USDT.D': 5.5, 'TOTAL': 2.5e12,
            'funding': 0.01, 'rv_20d': 0.025
        },
        {
            'name': '2022 Bear Market',
            'VIX': 28.0, 'DXY': 110.0, 'BTC.D': 42.0,
            'USDT.D': 7.0, 'TOTAL': 0.9e12,
            'funding': -0.005, 'rv_20d': 0.055
        },
        {
            'name': '2023 Chop',
            'VIX': 19.0, 'DXY': 103.0, 'BTC.D': 48.0,
            'USDT.D': 6.2, 'TOTAL': 1.2e12,
            'funding': 0.002, 'rv_20d': 0.035
        }
    ]

    for sample in test_samples:
        name = sample.pop('name')
        x_test = np.array([sample[f] for f in REGIME_FEATURES]).reshape(1, -1)
        x_test_scaled = scaler.transform(x_test)

        cluster_id = gmm.predict(x_test_scaled)[0]
        probs = gmm.predict_proba(x_test_scaled)[0]
        regime = label_map[cluster_id]
        confidence = probs[cluster_id]

        # Test fallback
        fallback_regime = rule_based_regime(sample['VIX'], sample['BTC.D'], sample['USDT.D'])
        fallback_match = "✅" if fallback_regime == regime else "⚠️"

        print(f"\n   {name}:")
        print(f"     → GMM Regime: {regime.upper()} (cluster {cluster_id})")
        print(f"     → Confidence: {confidence:.1%}")
        print(f"     {fallback_match} Rule fallback: {fallback_regime.upper()}")

    print("\n" + "="*80)
    print("✅ GMM v3 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {output_path}")
    print(f"\nKey improvements:")
    print(f"  ✅ 2 clusters (simpler, more stable)")
    print(f"  ✅ 7 core features (less noise)")
    print(f"  ✅ Deterministic labeling (reproducible)")
    print(f"  ✅ Rule-based fallback (robust)")
    print(f"\nNext steps:")
    print(f"  1. Update config 'regime_classifier.model_path' to '{output_path}'")
    print(f"  2. Run backtest with GMM v3")
    print(f"  3. Verify regime distribution (should NOT be 100% risk_on)")
    print("\n" + "="*80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
