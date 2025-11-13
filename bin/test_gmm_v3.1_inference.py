#!/usr/bin/env python3
"""
Test GMM v3.1 Inference

Quick validation script to test the trained GMM v3.1 model:
1. Load model and recent macro data
2. Classify sample periods (2022 bear, 2024 bull, etc.)
3. Show regime distribution and confidence scores
4. Validate output makes sense

Usage:
    python3 bin/test_gmm_v3.1_inference.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

def load_model():
    """Load trained GMM v3.1 model."""
    model_path = Path('models/regime_gmm_v3.1_fixed.pkl')

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python3 bin/train_regime_gmm_v3.1_fixed.py first")
        return None

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"✅ Loaded model: {model_path}")
    print(f"   K = {model_data['best_k']}, Features = {len(model_data['features'])}")
    print(f"   Label map: {model_data['label_map']}")

    return model_data

def classify_period(df, model_data, start_date, end_date, period_name):
    """Classify a specific time period and show results."""
    print(f"\n{'='*80}")
    print(f"{period_name}: {start_date} to {end_date}")
    print('='*80)

    # Filter to period
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    df_period = df[mask]

    if len(df_period) == 0:
        print(f"⚠️  No data for period {start_date} to {end_date}")
        return

    # Extract features
    features = model_data['features']
    X = df_period[features].dropna()

    if len(X) == 0:
        print(f"⚠️  No valid samples (all NaN)")
        return

    print(f"Samples: {len(X):,} hours")

    # Scale and predict
    scaler = model_data['scaler']
    gmm = model_data['gmm']
    label_map = model_data['label_map']

    X_scaled = scaler.transform(X)
    cluster_labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    # Map clusters to regimes
    regime_labels = [label_map[c] for c in cluster_labels]

    # Get max probability per sample (confidence)
    confidences = probabilities.max(axis=1)

    # Regime distribution
    regime_counts = pd.Series(regime_labels).value_counts()
    print(f"\nRegime Distribution:")
    for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
        count = regime_counts.get(regime, 0)
        pct = 100 * count / len(regime_labels)
        print(f"  {regime:12s}: {count:5d} hours ({pct:5.1f}%)")

    # Confidence stats
    print(f"\nConfidence Scores:")
    print(f"  Mean:   {confidences.mean():.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Min:    {confidences.min():.3f}")
    print(f"  Max:    {confidences.max():.3f}")
    print(f"  <0.6:   {(confidences < 0.6).sum()} samples ({100*(confidences < 0.6).sum()/len(confidences):.1f}%)")

    # Show some examples
    print(f"\nSample Classifications (random 10):")
    indices = np.random.choice(len(X), min(10, len(X)), replace=False)
    print(f"{'Date':20s} {'Regime':12s} {'Confidence':>10s} | VIX_Z  DXY_Z  ALT_ROT")
    print('-'*80)
    for idx in indices:
        date_str = df_period.iloc[X.index[idx]]['timestamp'].strftime('%Y-%m-%d %H:%M')
        regime = regime_labels[idx]
        conf = confidences[idx]

        # Get feature values
        vix_z = X.iloc[idx].get('VIX_Z', 0)
        dxy_z = X.iloc[idx].get('DXY_Z', 0)
        alt_rot = X.iloc[idx].get('ALT_ROTATION', 0)

        print(f"{date_str:20s} {regime.upper():12s} {conf:>10.3f} | {vix_z:6.2f} {dxy_z:6.2f} {alt_rot:6.2f}")

def main():
    print("\n" + "="*80)
    print("GMM V3.1 INFERENCE TEST")
    print("="*80)

    # Load model
    model_data = load_model()
    if model_data is None:
        return 1

    # Load macro data
    macro_path = Path('data/macro/macro_history.parquet')
    print(f"\n📖 Loading macro history: {macro_path}")
    df = pd.read_parquet(macro_path)
    print(f"   Loaded: {df.shape}")

    # Test on key periods
    periods = [
        ('2022-05-01', '2022-06-30', '2022 BEAR MARKET (Luna crash)'),
        ('2022-11-01', '2022-11-30', '2022 BEAR MARKET (FTX collapse)'),
        ('2023-01-01', '2023-03-31', '2023 Q1 RECOVERY'),
        ('2024-01-01', '2024-03-31', '2024 Q1 BULL RUN'),
        ('2024-10-01', '2024-10-31', '2024 Q4 (Recent)'),
    ]

    for start, end, name in periods:
        try:
            classify_period(df, model_data, start, end, name)
        except Exception as e:
            print(f"\n⚠️  Error classifying {name}: {e}")

    # Overall distribution (last 12 months)
    print(f"\n{'='*80}")
    print("OVERALL DISTRIBUTION (Last 12 Months)")
    print('='*80)

    try:
        one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
        df_recent = df[df['timestamp'] >= one_year_ago]

        features = model_data['features']
        X = df_recent[features].dropna()

        if len(X) > 0:
            X_scaled = model_data['scaler'].transform(X)
            cluster_labels = model_data['gmm'].predict(X_scaled)
            regime_labels = [model_data['label_map'][c] for c in cluster_labels]

            regime_counts = pd.Series(regime_labels).value_counts()
            print(f"\nTotal samples: {len(regime_labels):,} hours")
            print(f"\nRegime Distribution:")
            for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
                count = regime_counts.get(regime, 0)
                pct = 100 * count / len(regime_labels)
                bar = '█' * int(pct / 2)
                print(f"  {regime:12s}: {count:5d} hours ({pct:5.1f}%) {bar}")
    except Exception as e:
        print(f"⚠️  Error: {e}")

    print("\n" + "="*80)
    print("✅ INFERENCE TEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Temperature calibration on 2024 holdout")
    print("  2. Build config router with regime awareness")
    print("  3. Train bear_v10 optimizer for risk_off/crisis")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
