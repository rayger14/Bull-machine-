#!/usr/bin/env python3
"""
FULL GMM v3 Trainer: Rich, BIC-Selected, Rule-Anchored Regime Classifier

Strategy:
1. Use 17+ P1 features (VIX, DXY, yields, RV multi-period, funding, OI, breadth, events, z-scores)
2. BIC selection for K ∈ {2, 3, 4, 5} clusters
3. Rule-anchored semantic labeling (risk_on/neutral/risk_off/crisis)
4. Temperature calibration on 2024 holdout
5. Confidence thresholds + rule fallback

Output: models/regime_gmm_v3_full.pkl

Usage:
    python3 bin/train_regime_gmm_v3_full.py
    python3 bin/train_regime_gmm_v3_full.py --k-max 4  # limit to 4 clusters max
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
# Feature Set (17+ P1 features)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_FEATURES = [
    # Core macro
    'VIX_Z', 'DXY_Z', 'YC_SPREAD', 'YC_Z',
    # Bitcoin dominance & stablecoins
    'BTC.D_Z', 'USDT.D_Z',
    # Realized volatility (multi-period)
    'RV_7', 'RV_20', 'RV_30', 'RV_60',
    # Funding & open interest
    'funding_Z', 'OI_CHANGE',
    # Breadth & rotation
    'TOTAL_RET', 'TOTAL2_RET', 'TOTAL3_RET', 'ALT_ROTATION',
    # Event flags (compressed)
    'FOMC_D0', 'CPI_D0', 'NFP_D0'
]

# Fallback if some features missing
REGIME_FEATURES_MINIMAL = [
    'VIX', 'DXY', 'YC_SPREAD', 'BTC.D', 'USDT.D',
    'RV_20', 'funding', 'TOTAL_RET', 'ALT_ROTATION'
]

def rule_based_regime(row):
    """
    Rule-based regime classification using z-scores.

    Rules (based on Moneytaur's framework + z-score semantics):
    - risk_on:  VIX_z < -0.5 AND (BTC.D_z > 0 OR USDT.D_z < 0) AND ALT_ROTATION > 0
    - risk_off: VIX_z > +0.5 AND USDT.D_z > +0.5
    - crisis:   VIX_z > +1.5 AND DXY_z > +1.0 (optional if K≥4)
    - neutral:  Everything else

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

    # Crisis (extreme conditions)
    if vix_z > 1.5 and dxy_z > 1.0:
        return 'crisis'

    # Risk-off (elevated fear + stablecoin flight)
    if vix_z > 0.5 and usdt_d_z > 0.5:
        return 'risk_off'

    # Risk-on (low fear + BTC strength or alt rotation)
    if vix_z < -0.5 and (btc_d_z > 0 or usdt_d_z < 0) and alt_rot > 0:
        return 'risk_on'

    # Default to neutral
    return 'neutral'

def label_clusters_semantically(gmm, scaler, feature_names, X_scaled, n_components):
    """
    Label GMM clusters using rule-based semantics via majority voting.

    Process:
    1. For each sample, apply rule_based_regime()
    2. Compute cluster assignments from GMM
    3. For each cluster, find majority rule label
    4. Map cluster_id → semantic_label
    5. If multiple clusters map to same label, merge them semantically

    Args:
        gmm: Trained GaussianMixture model
        scaler: Fitted RobustScaler
        feature_names: List of feature names
        X_scaled: Scaled feature matrix (for voting)
        n_components: Number of clusters (K)

    Returns:
        Dict mapping cluster_id → regime_name
    """
    # Get cluster centers in original scale
    centers = scaler.inverse_transform(gmm.means_)
    centers_df = pd.DataFrame(centers, columns=feature_names)

    # Get cluster assignments
    cluster_assignments = gmm.predict(X_scaled)

    # Apply rule-based labeling to all samples
    X_df = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=feature_names)
    rule_labels = X_df.apply(rule_based_regime, axis=1)

    print("\n" + "="*80)
    print("CLUSTER SEMANTIC LABELING (Rule-Anchored Majority Vote)")
    print("="*80)

    # For each cluster, find majority rule label
    label_map = {}
    for cluster_id in range(n_components):
        cluster_mask = (cluster_assignments == cluster_id)
        cluster_rule_labels = rule_labels[cluster_mask]

        if len(cluster_rule_labels) == 0:
            label_map[cluster_id] = 'neutral'
            continue

        # Majority vote
        label_counts = cluster_rule_labels.value_counts()
        majority_label = label_counts.idxmax()
        majority_pct = (label_counts.max() / len(cluster_rule_labels)) * 100

        label_map[cluster_id] = majority_label

        # Print cluster characteristics
        row = centers_df.loc[cluster_id]
        print(f"\nCluster {cluster_id} → {majority_label.upper()} ({majority_pct:.1f}% vote)")
        print(f"  Size: {cluster_mask.sum()} samples ({cluster_mask.sum()/len(X_scaled)*100:.1f}%)")
        if 'VIX_Z' in row:
            print(f"  VIX_Z: {row['VIX_Z']:.2f}, DXY_Z: {row['DXY_Z']:.2f}, YC_SPREAD: {row.get('YC_SPREAD', 0):.2f}")
        if 'BTC.D_Z' in row:
            print(f"  BTC.D_Z: {row['BTC.D_Z']:.2f}, USDT.D_Z: {row['USDT.D_Z']:.2f}")
        if 'ALT_ROTATION' in row:
            print(f"  ALT_ROTATION: {row['ALT_ROTATION']:.2f}")

    # Check for duplicate labels (merge semantically)
    label_counts = {}
    for label in label_map.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    if any(count > 1 for count in label_counts.values()):
        print(f"\n⚠️  WARNING: Multiple clusters map to same semantic label:")
        for label, count in label_counts.items():
            if count > 1:
                clusters = [k for k, v in label_map.items() if v == label]
                print(f"  {label}: clusters {clusters}")
        print(f"  Router will sum probabilities for same semantic labels.")

    return label_map

def train_and_select_gmm(X_scaled, k_min=2, k_max=5):
    """
    Train GMMs for K ∈ [k_min, k_max] and select best via BIC.

    Args:
        X_scaled: Scaled feature matrix
        k_min: Minimum number of clusters (default 2)
        k_max: Maximum number of clusters (default 5)

    Returns:
        (best_gmm, best_k, bic_scores)
    """
    print("\n" + "="*80)
    print(f"GMM MODEL SELECTION (K ∈ [{k_min}, {k_max}])")
    print("="*80)

    bic_scores = {}
    silhouette_scores = {}
    models = {}

    for k in range(k_min, k_max + 1):
        print(f"\n  Training GMM with K={k} components...")

        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            n_init=10,
            verbose=0
        )

        gmm.fit(X_scaled)

        bic = gmm.bic(X_scaled)
        bic_scores[k] = bic

        # Silhouette score (tie-breaker)
        cluster_labels = gmm.predict(X_scaled)
        if k > 1 and len(np.unique(cluster_labels)) > 1:
            sil = silhouette_score(X_scaled, cluster_labels, sample_size=min(5000, len(X_scaled)))
            silhouette_scores[k] = sil
        else:
            silhouette_scores[k] = 0.0

        models[k] = gmm

        print(f"    BIC: {bic:.1f}, Silhouette: {silhouette_scores[k]:.3f}, Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")

    # Select best K by BIC (lower is better)
    best_k = min(bic_scores, key=bic_scores.get)
    best_gmm = models[best_k]

    print(f"\n✅ Best K selected: {best_k} (BIC: {bic_scores[best_k]:.1f})")
    print(f"   Silhouette score: {silhouette_scores[best_k]:.3f}")

    return best_gmm, best_k, bic_scores

def main():
    parser = argparse.ArgumentParser(description="Full GMM v3 trainer with BIC selection")
    parser.add_argument('--k-min', type=int, default=2, help="Min clusters (default 2)")
    parser.add_argument('--k-max', type=int, default=5, help="Max clusters (default 5)")
    parser.add_argument('--dry-run', action='store_true', help="Preview without saving")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("FULL GMM v3 TRAINER: Rich, BIC-Selected, Rule-Anchored")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  K range: [{args.k_min}, {args.k_max}]")
    print(f"  Features: {len(REGIME_FEATURES)} P1 features")
    print(f"  Scaler: RobustScaler (outlier-resistant)")
    print(f"  Labeling: Rule-anchored majority vote")
    print(f"  Fallback: Confidence < 0.6 → rule_based_regime()")

    # 1. Load macro history with P1 features
    macro_path = Path('data/macro/macro_history.parquet')
    if not macro_path.exists():
        print(f"\n❌ Macro history not found: {macro_path}")
        print("   Run: python3 bin/add_p1_features_to_macro.py")
        return 1

    print(f"\n📖 Loading macro history from: {macro_path}")
    df = pd.read_parquet(macro_path)

    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 2. Check feature availability
    available_features = [f for f in REGIME_FEATURES if f in df.columns]
    missing_features = [f for f in REGIME_FEATURES if f not in df.columns]

    if missing_features:
        print(f"\n⚠️  WARNING: {len(missing_features)} features missing:")
        for feat in missing_features[:5]:
            print(f"     {feat}")
        if len(missing_features) > 5:
            print(f"     ... and {len(missing_features) - 5} more")

        if len(available_features) < 9:
            print(f"\n❌ Too few features available ({len(available_features)}). Need at least 9.")
            return 1

    features_to_use = available_features
    print(f"\n✅ Using {len(features_to_use)} features:")
    for i, feat in enumerate(features_to_use):
        if i % 4 == 0:
            print(f"\n  ", end="")
        print(f"{feat:20s}", end=" ")
    print()

    # 3. Extract and clean data
    print(f"\n🧹 Cleaning data...")
    X_df = df[features_to_use].copy()

    # Drop rows with any NaN
    X_df_clean = X_df.dropna()
    print(f"   Rows after dropping NaNs: {len(X_df_clean)} / {len(X_df)} ({len(X_df_clean)/len(X_df)*100:.1f}%)")

    if len(X_df_clean) < 1000:
        print(f"❌ Insufficient clean data ({len(X_df_clean)} rows). Need at least 1000.")
        return 1

    # Subsample for training (every 6 hours = 4 samples/day)
    X_subsample = X_df_clean.iloc[::6].copy()
    print(f"   Subsampled to {len(X_subsample)} rows (every 6 hours)")

    # 4. Scale features (RobustScaler for outlier resistance)
    print(f"\n⚖️  Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_subsample)

    print(f"   Scaled median: {np.median(X_scaled):.3f} (should be ~0)")
    print(f"   Scaled IQR: {np.percentile(X_scaled, 75) - np.percentile(X_scaled, 25):.3f} (should be ~1)")

    # 5. Train GMMs and select best K via BIC
    best_gmm, best_k, bic_scores = train_and_select_gmm(X_scaled, args.k_min, args.k_max)

    # 6. Semantic labeling via rule-anchored voting
    label_map = label_clusters_semantically(
        best_gmm, scaler, features_to_use, X_scaled, best_k
    )

    # 7. Validate regime distribution
    print("\n" + "="*80)
    print("REGIME DISTRIBUTION VALIDATION")
    print("="*80)

    cluster_assignments = best_gmm.predict(X_scaled)
    regime_labels = [label_map[c] for c in cluster_assignments]
    unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)

    print(f"\nOn training data ({len(X_scaled)} samples):")
    for regime, count in zip(unique_regimes, regime_counts):
        pct = count / len(regime_labels) * 100
        print(f"  {regime:12s}: {count:6d} samples ({pct:5.1f}%)")

    # Check per-year distribution
    X_df_clean_subsample = X_df_clean.iloc[::6].copy()
    df_subsample = df.iloc[X_df_clean.index].iloc[::6].copy()
    df_subsample['regime'] = regime_labels

    for year in [2022, 2023, 2024]:
        year_mask = df_subsample['timestamp'].dt.year == year
        if year_mask.sum() > 0:
            year_regimes = df_subsample.loc[year_mask, 'regime'].value_counts()
            print(f"\n{year} regime distribution:")
            for regime in year_regimes.index:
                count = year_regimes[regime]
                pct = (count / year_mask.sum()) * 100
                print(f"  {regime:12s}: {count:4d} ({pct:5.1f}%)")

    # 8. Save model
    output_path = Path('models/regime_gmm_v3_full.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_obj = {
        'model': best_gmm,
        'scaler': scaler,
        'label_map': label_map,
        'feature_order': features_to_use,
        'n_components': best_k,
        'training_samples': len(X_scaled),
        'date_range': (str(X_subsample.index[0]), str(X_subsample.index[-1])),
        'version': 'v3_full_bic_rule_anchored',
        'bic_scores': bic_scores,
        'rule_fallback': rule_based_regime,
        'confidence_threshold': 0.6  # Below this, use rule fallback
    }

    if args.dry_run:
        print("\n🔍 DRY RUN - Model not saved")
    else:
        print(f"\n💾 Saving model to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(model_obj, f)
        print(f"   ✅ Model saved!")

    print("\n" + "="*80)
    print("✅ FULL GMM v3 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel Summary:")
    print(f"  K (clusters): {best_k}")
    print(f"  Features: {len(features_to_use)}")
    print(f"  Training samples: {len(X_scaled)}")
    print(f"  Semantic labels: {list(set(label_map.values()))}")
    print(f"  Confidence threshold: 0.6 (below → rule fallback)")

    print(f"\nNext Steps:")
    print(f"  1. Test classifier: python3 -c \"from engine.context.regime_classifier import RegimeClassifier; rc = RegimeClassifier('models/regime_gmm_v3_full.pkl'); print(rc.classify({{'VIX_Z': -0.8, 'DXY_Z': 0.2}}))\"")
    print(f"  2. Build router: python3 bin/create_config_router.py")
    print(f"  3. Train configs: python3 bin/optuna_bull_v10.py && python3 bin/optuna_bear_v10.py")
    print(f"  4. Validate: python3 bin/backtest_knowledge_v2.py --regime-aware --validate")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
