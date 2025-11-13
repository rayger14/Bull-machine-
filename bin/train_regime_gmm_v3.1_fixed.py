#!/usr/bin/env python3
"""
GMM v3.1: Fixed Regime Classifier with Direct Centroid Labeling

FIXES:
- Replaced majority voting with direct centroid-based labeling
- Relaxed thresholds for better regime separation
- Added comprehensive diagnostics

Changes from v3.0:
1. label_clusters_via_centroids(): Apply rule function directly to cluster centers
2. Relaxed risk_on threshold: VIX_Z < -0.3 (was -0.5)
3. Relaxed risk_off threshold: VIX_Z > 0.3 (was 0.5)
4. Better logging and validation

Output: models/regime_gmm_v3.1_fixed.pkl

Usage:
    python3 bin/train_regime_gmm_v3.1_fixed.py
    python3 bin/train_regime_gmm_v3.1_fixed.py --k-max 4
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Feature Set (19 P1 features)
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
    # Event flags
    'FOMC_D0', 'CPI_D0', 'NFP_D0'
]

def rule_based_regime_relaxed(row):
    """
    RELAXED rule-based regime classification for better separation.

    Changes from v3.0:
    - risk_on:  VIX_Z < -0.3 (was -0.5) - more inclusive
    - risk_off: VIX_Z > +0.3 (was +0.5) - more inclusive
    - crisis:   VIX_Z > +1.2 AND DXY_Z > +0.8 (was 1.5, 1.0) - slightly relaxed

    Args:
        row: dict-like with feature values

    Returns:
        'risk_on', 'neutral', 'risk_off', or 'crisis'
    """
    vix_z = row.get('VIX_Z', 0)
    dxy_z = row.get('DXY_Z', 0)
    btc_d_z = row.get('BTC.D_Z', 0)
    usdt_d_z = row.get('USDT.D_Z', 0)
    alt_rot = row.get('ALT_ROTATION', 0)
    yc_z = row.get('YC_Z', 0)

    # Crisis (extreme stress)
    if vix_z > 1.2 and dxy_z > 0.8:
        return 'crisis'

    # Risk-off (elevated fear + stablecoin flight)
    if vix_z > 0.3 and usdt_d_z > 0.3:
        return 'risk_off'

    # Risk-on (low fear + favorable conditions)
    # More relaxed: VIX < -0.3 and positive signals
    if vix_z < -0.3:
        # Check for bullish crypto signals
        bullish_signals = 0
        if btc_d_z > 0:  # BTC dominance increasing
            bullish_signals += 1
        if usdt_d_z < 0:  # USDT dominance decreasing
            bullish_signals += 1
        if alt_rot > 0:   # Alt rotation positive
            bullish_signals += 1
        if yc_z > -0.5:   # Yield curve not too inverted
            bullish_signals += 1

        if bullish_signals >= 2:  # At least 2 bullish signals
            return 'risk_on'

    # Neutral (default)
    return 'neutral'

def label_clusters_via_ranking(gmm, scaler, feature_names, n_components):
    """
    Label clusters using RELATIVE RANKING instead of absolute thresholds.

    This ensures diverse labels by ranking clusters from most bearish to most bullish
    based on a composite risk score.

    Args:
        gmm: Trained GaussianMixture model
        scaler: Fitted RobustScaler
        feature_names: List of feature names
        n_components: Number of clusters (K)

    Returns:
        Dict mapping cluster_id → regime_name
    """
    # Get cluster centers in original scale
    centers = scaler.inverse_transform(gmm.means_)
    centers_df = pd.DataFrame(centers, columns=feature_names)

    print("\n" + "="*80)
    print("CLUSTER LABELING: Relative Ranking Method")
    print("="*80)

    # Compute risk score for each cluster (high = bearish, low = bullish)
    risk_scores = []
    for cluster_id in range(n_components):
        row = centers_df.iloc[cluster_id]

        # Composite risk score
        # Positive contributors (bearish): VIX_Z, DXY_Z, USDT.D_Z, -ALT_ROTATION
        # Negative contributors (bullish): -VIX_Z, BTC.D_Z, ALT_ROTATION
        vix_z = row.get('VIX_Z', 0)
        dxy_z = row.get('DXY_Z', 0)
        btc_d_z = row.get('BTC.D_Z', 0)
        usdt_d_z = row.get('USDT.D_Z', 0)
        alt_rot = row.get('ALT_ROTATION', 0)

        # Risk score: weighted sum (higher = more bearish/risky)
        risk_score = (
            vix_z * 2.0 +          # Fear (heavy weight)
            dxy_z * 1.0 +          # Dollar strength
            usdt_d_z * 1.5 -       # Stablecoin flight
            btc_d_z * 0.5 -        # BTC dominance (moderate bullish)
            alt_rot * 1.0          # Alt rotation (bullish)
        )

        risk_scores.append((cluster_id, risk_score, row))

    # Sort by risk score (high to low)
    risk_scores.sort(key=lambda x: x[1], reverse=True)

    # Assign labels based on ranking
    label_map = {}

    if n_components == 2:
        # Simple: risk_off vs risk_on
        label_map[risk_scores[0][0]] = 'risk_off'  # Highest risk
        label_map[risk_scores[1][0]] = 'risk_on'   # Lowest risk

    elif n_components == 3:
        # risk_off, neutral, risk_on
        label_map[risk_scores[0][0]] = 'risk_off'  # Highest risk
        label_map[risk_scores[1][0]] = 'neutral'   # Middle
        label_map[risk_scores[2][0]] = 'risk_on'   # Lowest risk

    elif n_components == 4:
        # crisis, risk_off, neutral, risk_on
        label_map[risk_scores[0][0]] = 'crisis'    # Extreme risk
        label_map[risk_scores[1][0]] = 'risk_off'  # High risk
        label_map[risk_scores[2][0]] = 'neutral'   # Moderate
        label_map[risk_scores[3][0]] = 'risk_on'   # Low risk

    else:  # K >= 5
        # crisis, risk_off, neutral (multiple), risk_on
        label_map[risk_scores[0][0]] = 'crisis'    # Extreme risk
        label_map[risk_scores[1][0]] = 'risk_off'  # High risk
        # Middle clusters = neutral
        for i in range(2, n_components - 1):
            label_map[risk_scores[i][0]] = 'neutral'
        label_map[risk_scores[-1][0]] = 'risk_on'  # Lowest risk

    # Print results
    print(f"\n{'Cluster':>8} {'Risk Score':>12} {'Label':>12} | Feature Profile")
    print("-" * 80)
    for cluster_id, risk_score, row in risk_scores:
        regime = label_map[cluster_id]
        print(f"{cluster_id:>8d} {risk_score:>12.2f} {regime.upper():>12s} | "
              f"VIX={row.get('VIX_Z', 0):.2f} DXY={row.get('DXY_Z', 0):.2f} "
              f"ALT_ROT={row.get('ALT_ROTATION', 0):.2f}")

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

def train_and_select_gmm(X_scaled, k_min=2, k_max=5):
    """
    Train GMMs for K ∈ [k_min, k_max] and select best via BIC.

    Args:
        X_scaled: Scaled feature matrix
        k_min: Minimum clusters to try
        k_max: Maximum clusters to try

    Returns:
        (best_gmm, best_k, bic_scores_dict)
    """
    print("\n" + "="*80)
    print(f"BIC-BASED MODEL SELECTION (K ∈ [{k_min}, {k_max}])")
    print("="*80)

    bic_scores = {}
    silhouette_scores = {}
    models = {}

    for k in range(k_min, k_max + 1):
        print(f"\nTraining GMM with K={k} components...")

        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            n_init=10
        )
        gmm.fit(X_scaled)

        bic = gmm.bic(X_scaled)
        bic_scores[k] = bic

        # Silhouette score (only if K > 1)
        if k > 1 and len(X_scaled) > k:
            labels = gmm.predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
            silhouette_scores[k] = silhouette
        else:
            silhouette_scores[k] = 0.0

        models[k] = gmm

        print(f"  BIC:        {bic:,.2f}")
        print(f"  Silhouette: {silhouette_scores[k]:.3f}")
        print(f"  Converged:  {gmm.converged_}")

    # Select best K (lowest BIC)
    best_k = min(bic_scores, key=bic_scores.get)
    best_bic = bic_scores[best_k]

    print(f"\n{'='*80}")
    print(f"BEST MODEL: K = {best_k} (BIC: {best_bic:,.2f})")
    print(f"{'='*80}")

    return models[best_k], best_k, bic_scores, silhouette_scores

def validate_feature_distributions(df, features):
    """Quick validation of z-score features."""
    print("\n" + "="*80)
    print("FEATURE VALIDATION")
    print("="*80)

    z_features = [f for f in features if f.endswith('_Z')]
    print(f"\nZ-score features ({len(z_features)}):")
    for feat in z_features:
        if feat in df.columns:
            mean = df[feat].mean()
            std = df[feat].std()
            is_valid = abs(mean) < 0.3 and 0.7 < std < 1.5
            status = "✅" if is_valid else "⚠️ "
            print(f"  {status} {feat:15s}: mean={mean:7.3f}, std={std:6.3f}")

def main():
    parser = argparse.ArgumentParser(description="GMM v3.1: Fixed regime trainer")
    parser.add_argument('--k-min', type=int, default=2, help="Minimum clusters")
    parser.add_argument('--k-max', type=int, default=5, help="Maximum clusters")
    parser.add_argument('--train-samples', type=int, default=5000, help="Max samples for training (subsampling)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GMM V3.1: FIXED REGIME CLASSIFIER TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ─── Load Macro Data ───
    macro_path = Path('data/macro/macro_history.parquet')
    if not macro_path.exists():
        print(f"\n❌ Macro history not found: {macro_path}")
        print("   Run: python3 bin/add_p1_features_to_macro.py first")
        return 1

    print(f"\n📖 Loading macro history: {macro_path}")
    df = pd.read_parquet(macro_path)
    print(f"   Loaded: {df.shape}")

    # ─── Select Features ───
    available_features = [f for f in REGIME_FEATURES if f in df.columns]
    missing_features = [f for f in REGIME_FEATURES if f not in df.columns]

    print(f"\n✅ Available features: {len(available_features)} / {len(REGIME_FEATURES)}")
    if missing_features:
        print(f"⚠️  Missing features ({len(missing_features)}): {missing_features}")

    if len(available_features) < 10:
        print(f"\n❌ Too few features available ({len(available_features)}). Need at least 10.")
        return 1

    # ─── Prepare Training Data ───
    print(f"\n{'='*80}")
    print("PREPARING TRAINING DATA")
    print("="*80)

    df_clean = df[available_features].dropna()
    print(f"  Clean samples: {len(df_clean):,} / {len(df):,} ({len(df_clean)/len(df)*100:.1f}%)")

    # Validate z-score features
    validate_feature_distributions(df_clean, available_features)

    # Subsample if too large
    if len(df_clean) > args.train_samples:
        df_train = df_clean.sample(n=args.train_samples, random_state=42)
        print(f"\n  Subsampled to {args.train_samples:,} for training")
    else:
        df_train = df_clean

    X = df_train[available_features].values

    # ─── Scale Features ───
    print(f"\n{'='*80}")
    print("SCALING FEATURES (RobustScaler)")
    print("="*80)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Scaled shape: {X_scaled.shape}")
    print(f"  Median: {np.median(X_scaled, axis=0).mean():.3f}")
    print(f"  IQR:    {np.percentile(X_scaled, 75, axis=0).mean() - np.percentile(X_scaled, 25, axis=0).mean():.3f}")

    # ─── Train and Select GMM ───
    gmm, best_k, bic_scores, sil_scores = train_and_select_gmm(X_scaled, args.k_min, args.k_max)

    # ─── Label Clusters (Relative Ranking Method) ───
    label_map = label_clusters_via_ranking(gmm, scaler, available_features, best_k)

    # ─── Save Model ───
    print(f"\n{'='*80}")
    print("SAVING MODEL")
    print("="*80)

    model_path = Path('models/regime_gmm_v3.1_fixed.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'gmm': gmm,
        'scaler': scaler,
        'features': available_features,
        'label_map': label_map,
        'best_k': best_k,
        'bic_scores': bic_scores,
        'silhouette_scores': sil_scores,
        # Don't save function to avoid pickle issues
        # 'rule_fallback': rule_based_regime_relaxed,
        'confidence_threshold': 0.6,
        'trained_at': datetime.now().isoformat(),
        'version': '3.1',
        'notes': 'Fixed version with ranking-based labeling, NO function pickle'
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model saved: {model_path}")
    print(f"   K = {best_k}, BIC = {bic_scores[best_k]:,.2f}")
    print(f"   Label map: {label_map}")

    # ─── Summary ───
    print(f"\n{'='*80}")
    print("✅ GMM V3.1 TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"  Features:  {len(available_features)}")
    print(f"  Clusters:  {best_k}")
    print(f"  BIC:       {bic_scores[best_k]:,.2f}")
    print(f"  Silhouette:{sil_scores[best_k]:.3f}")
    print(f"\nRegime distribution:")
    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        count = list(label_map.values()).count(regime)
        if count > 0:
            print(f"  {regime:12s}: {count} cluster(s)")

    print(f"\nNext steps:")
    print(f"  1. Test inference: from engine.regime.gmm_v3 import RegimeClassifierGMM")
    print(f"  2. Temperature calibration on 2024 holdout")
    print(f"  3. Build config router with regime awareness")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
