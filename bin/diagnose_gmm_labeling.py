#!/usr/bin/env python3
"""
Diagnose why GMM v3 full training resulted in 100% neutral classification.

This script will:
1. Load the trained model and macro data
2. Examine cluster centroids and feature distributions
3. Test rule-based labeling on actual cluster members
4. Identify why crisis/risk_off/risk_on labels weren't assigned
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def rule_based_regime(row):
    """
    Rule-based regime classification using z-scores and macro features.
    (Replicated from train_regime_gmm_v3_full.py to allow pickle loading)
    """
    vix_z = row.get('VIX_Z', 0)
    dxy_z = row.get('DXY_Z', 0)
    btc_d_z = row.get('BTC.D_Z', 0)
    usdt_d_z = row.get('USDT.D_Z', 0)
    alt_rot = row.get('ALT_ROTATION', 0)
    yc_z = row.get('YC_Z', 0)

    # Crisis: Extreme fear + strong dollar
    if vix_z > 1.5 and dxy_z > 1.0:
        return 'crisis'

    # Risk-off: Elevated fear + stablecoin flight
    if vix_z > 0.5 and usdt_d_z > 0.5:
        return 'risk_off'

    # Risk-on: Low fear + BTC strength or alt rotation + positive yield environment
    if vix_z < -0.5 and (btc_d_z > 0 or usdt_d_z < 0) and alt_rot > 0:
        return 'risk_on'

    # Neutral: Default
    return 'neutral'

def load_model_and_data():
    """Load trained GMM v3 model and macro history."""
    print("\n" + "="*80)
    print("LOADING GMM V3 MODEL AND DATA")
    print("="*80)

    # Load model
    model_path = Path('models/regime_gmm_v3_full.pkl')
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"\n✅ Loaded model from: {model_path}")
    print(f"   K = {model_data['gmm'].n_components}")
    print(f"   Features: {len(model_data['features'])}")
    print(f"   Label map: {model_data['label_map']}")

    # Load macro history
    macro_path = Path('data/macro/macro_history.parquet')
    df = pd.read_parquet(macro_path)
    print(f"\n✅ Loaded macro history: {df.shape}")

    return model_data, df

def test_rule_on_cluster_samples(model_data, df, cluster_id, n_samples=10):
    """Test rule-based labeling on actual cluster members."""
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} SAMPLE ANALYSIS")
    print(f"{'='*80}")

    # Get cluster assignments
    features = model_data['features']
    scaler = model_data['scaler']
    gmm = model_data['gmm']
    rule_fallback = model_data['rule_fallback']

    # Prepare data
    df_clean = df[features].dropna()
    X_scaled = scaler.transform(df_clean)
    labels = gmm.predict(X_scaled)

    # Get samples from this cluster
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) == 0:
        print(f"   ⚠️  No samples in cluster {cluster_id}")
        return

    print(f"\n   Cluster size: {len(cluster_indices)} samples ({100*len(cluster_indices)/len(labels):.1f}%)")

    # Sample random members
    sample_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)

    # Test rule on each sample
    regime_votes = {'risk_on': 0, 'neutral': 0, 'risk_off': 0, 'crisis': 0}

    print(f"\n   Sample member analysis:")
    print(f"   {'VIX_Z':>8} {'DXY_Z':>8} {'BTC.D_Z':>8} {'USDT.D_Z':>8} {'ALT_ROT':>8} | Regime")
    print(f"   {'-'*60}")

    for idx in sample_indices:
        row = df_clean.iloc[idx]
        regime = rule_fallback(row)
        regime_votes[regime] += 1

        print(f"   {row.get('VIX_Z', 0):>8.2f} {row.get('DXY_Z', 0):>8.2f} "
              f"{row.get('BTC.D_Z', 0):>8.2f} {row.get('USDT.D_Z', 0):>8.2f} "
              f"{row.get('ALT_ROTATION', 0):>8.2f} | {regime}")

    print(f"\n   Regime vote distribution:")
    for regime, count in regime_votes.items():
        pct = 100 * count / len(sample_indices)
        print(f"      {regime:12s}: {count:2d} / {len(sample_indices)} ({pct:5.1f}%)")

    return regime_votes

def analyze_cluster_centroids(model_data, df):
    """Analyze cluster centroids to understand feature distributions."""
    print(f"\n{'='*80}")
    print("CLUSTER CENTROID ANALYSIS")
    print(f"{'='*80}")

    features = model_data['features']
    scaler = model_data['scaler']
    gmm = model_data['gmm']

    # Get centroids in original feature space
    centroids_scaled = gmm.means_
    centroids = scaler.inverse_transform(centroids_scaled)

    # Key features for regime detection
    key_features = ['VIX_Z', 'DXY_Z', 'BTC.D_Z', 'USDT.D_Z', 'ALT_ROTATION', 'YC_SPREAD', 'YC_Z']
    key_indices = [features.index(f) if f in features else -1 for f in key_features]

    print(f"\n   Cluster centroids (key features):")
    print(f"   {'Cluster':>8} | {' '.join(f'{f:>10}' for f in key_features)}")
    print(f"   {'-'*90}")

    for k in range(gmm.n_components):
        centroid_vals = [centroids[k, idx] if idx >= 0 else 0.0 for idx in key_indices]
        print(f"   {k:>8d} | {' '.join(f'{v:>10.2f}' for v in centroid_vals)}")

    # Check if z-score features are actually z-scores
    print(f"\n   Z-score feature statistics:")
    df_clean = df[features].dropna()
    for feat in ['VIX_Z', 'DXY_Z', 'BTC.D_Z', 'USDT.D_Z', 'funding_Z']:
        if feat in df_clean.columns:
            mean = df_clean[feat].mean()
            std = df_clean[feat].std()
            print(f"      {feat:15s}: mean={mean:7.3f}, std={std:6.3f} "
                  f"{'✅ OK' if abs(mean) < 0.2 and 0.8 < std < 1.2 else '⚠️  NOT Z-SCORE'}")

def test_rule_thresholds(df):
    """Test how many samples each regime rule captures."""
    print(f"\n{'='*80}")
    print("RULE THRESHOLD COVERAGE ANALYSIS")
    print(f"{'='*80}")

    df_clean = df[['VIX_Z', 'DXY_Z', 'BTC.D_Z', 'USDT.D_Z', 'ALT_ROTATION']].dropna()

    print(f"\n   Total samples: {len(df_clean)}")

    # Crisis rule: vix_z > 1.5 and dxy_z > 1.0
    crisis_mask = (df_clean['VIX_Z'] > 1.5) & (df_clean['DXY_Z'] > 1.0)
    print(f"\n   Crisis rule (VIX_Z > 1.5 AND DXY_Z > 1.0):")
    print(f"      Matches: {crisis_mask.sum():5d} / {len(df_clean)} ({100*crisis_mask.sum()/len(df_clean):.2f}%)")

    # Risk-off rule: vix_z > 0.5 and usdt_d_z > 0.5
    risk_off_mask = (df_clean['VIX_Z'] > 0.5) & (df_clean['USDT.D_Z'] > 0.5)
    print(f"\n   Risk-off rule (VIX_Z > 0.5 AND USDT.D_Z > 0.5):")
    print(f"      Matches: {risk_off_mask.sum():5d} / {len(df_clean)} ({100*risk_off_mask.sum()/len(df_clean):.2f}%)")

    # Risk-on rule: vix_z < -0.5 and (btc_d_z > 0 or usdt_d_z < 0) and alt_rot > 0
    risk_on_mask = (df_clean['VIX_Z'] < -0.5) & \
                   ((df_clean['BTC.D_Z'] > 0) | (df_clean['USDT.D_Z'] < 0)) & \
                   (df_clean['ALT_ROTATION'] > 0)
    print(f"\n   Risk-on rule (VIX_Z < -0.5 AND (BTC.D_Z > 0 OR USDT.D_Z < 0) AND ALT_ROT > 0):")
    print(f"      Matches: {risk_on_mask.sum():5d} / {len(df_clean)} ({100*risk_on_mask.sum()/len(df_clean):.2f}%)")

    # Neutral (everything else)
    neutral_mask = ~(crisis_mask | risk_off_mask | risk_on_mask)
    print(f"\n   Neutral (default):")
    print(f"      Matches: {neutral_mask.sum():5d} / {len(df_clean)} ({100*neutral_mask.sum()/len(df_clean):.2f}%)")

    # Show feature distributions
    print(f"\n   Feature distributions (percentiles):")
    print(f"   {'Feature':15s} | {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    print(f"   {'-'*70}")
    for feat in ['VIX_Z', 'DXY_Z', 'BTC.D_Z', 'USDT.D_Z', 'ALT_ROTATION']:
        if feat in df_clean.columns:
            p5, p25, p50, p75, p95 = df_clean[feat].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            print(f"   {feat:15s} | {p5:>8.2f} {p25:>8.2f} {p50:>8.2f} {p75:>8.2f} {p95:>8.2f}")

def main():
    print("\n" + "="*80)
    print("GMM V3 LABELING DIAGNOSTIC")
    print("="*80)

    # Load model and data
    model_data, df = load_model_and_data()
    if model_data is None or df is None:
        return 1

    # Analyze cluster centroids
    analyze_cluster_centroids(model_data, df)

    # Test rule thresholds on full dataset
    test_rule_thresholds(df)

    # Test rule on cluster samples
    for cluster_id in range(model_data['gmm'].n_components):
        test_rule_on_cluster_samples(model_data, df, cluster_id, n_samples=10)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nKey findings to investigate:")
    print("  1. Are z-score features properly normalized? (mean~0, std~1)")
    print("  2. What % of data falls into each regime rule?")
    print("  3. Are cluster centroids actually different enough to distinguish regimes?")
    print("  4. Do cluster members have homogeneous feature values?")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
