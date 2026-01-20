#!/usr/bin/env python3
"""
Simplified HMM Training - Following Research Document Specifications
====================================================================

Implements the exact approach from DYNAMIC_REGIME_DETECTION_RESEARCH.md:
- 10 random initializations to avoid local optima
- Full covariance matrix
- Uses existing features (no re-computation)
- Picks best model by log-likelihood
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("SIMPLIFIED HMM TRAINING - MULTIPLE INITIALIZATIONS")
print("="*80)

# Step 1: Load data
print("\n[1/5] Loading data...")
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet')
print(f"✅ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# Step 2: Select features (use what's available, avoid NaNs)
print("\n[2/5] Selecting features...")

# UPDATED: Features now fixed by Agent 1!
# rv_20d and oi_change_pct_24h are now available with real values
features = [
    'funding_Z',          # Funding rate z-score
    'oi_change_pct_24h',  # 24h open interest change %
    'rv_20d',             # 20-day realized volatility (FIXED!)
    'USDT.D',             # USDT dominance %
    'BTC.D',              # BTC dominance %
    'VIX_Z',              # VIX z-score (fear index)
    'DXY_Z',              # Dollar index z-score
    'YC_SPREAD',          # Yield curve spread
]

# Check which features actually exist
available_features = [f for f in features if f in df.columns]
print(f"✅ Using {len(available_features)} features: {available_features}")

if len(available_features) < 5:
    print("❌ Insufficient features available")
    sys.exit(1)

# Extract feature matrix
X = df[available_features].fillna(0).values
print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ NaN count after fillna: {np.isnan(X).sum()}")

# Step 3: Standardize features (CRITICAL for HMM convergence)
print("\n[3/5] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✅ Features scaled (mean≈0, std≈1)")

# Step 4: Train HMM with multiple initializations (RESEARCH RECOMMENDATION)
print("\n[4/5] Training HMM with 10 random initializations...")
print("(This avoids local optima in EM algorithm)")

best_model = None
best_score = -np.inf
best_seed = None

for seed in range(10):
    print(f"\n  Trial {seed+1}/10 (seed={seed})...")

    try:
        # Train HMM
        model = GaussianHMM(
            n_components=4,
            covariance_type='full',  # Research recommendation
            n_iter=100,
            tol=1e-4,
            random_state=seed,
            init_params='stmc'
        )

        model.fit(X_scaled)

        # Score model (log-likelihood)
        score = model.score(X_scaled)

        print(f"    Log-likelihood: {score:.2f}")
        print(f"    Converged: {model.monitor_.converged}")

        # Check if this is the best model
        if score > best_score and not np.isnan(score):
            best_score = score
            best_model = model
            best_seed = seed
            print(f"    ✅ NEW BEST MODEL!")

    except Exception as e:
        print(f"    ❌ Failed: {e}")
        continue

if best_model is None:
    print("\n❌ All training attempts failed")
    sys.exit(1)

print(f"\n✅ Best model: seed={best_seed}, log-likelihood={best_score:.2f}")

# Step 5: Interpret states
print("\n[5/5] Interpreting HMM states...")

# Get state means (in original scale)
means_scaled = best_model.means_
means_original = scaler.inverse_transform(means_scaled)
means_df = pd.DataFrame(means_original, columns=available_features)

print("\nState characteristics:")
state_map = {}

for state_id in range(4):
    row = means_df.iloc[state_id]

    # Simple interpretation logic
    vix_z = row.get('VIX_Z', 0)
    rv = row.get('rv_20d', 0)
    funding_z = row.get('funding_Z', 0)

    print(f"\nState {state_id}:")
    print(f"  VIX_Z={vix_z:.2f}, RV={rv:.1f}%, funding_Z={funding_z:.2f}")

    # Classify regime
    if vix_z > 1.0 and rv > 60:
        regime = 'crisis'
    elif vix_z < 0 and funding_z > 0:
        regime = 'risk_on'
    elif vix_z > 0.5:
        regime = 'risk_off'
    else:
        regime = 'neutral'

    state_map[state_id] = regime
    print(f"  → {regime.upper()}")

# Ensure all 4 regimes are assigned (fallback logic)
assigned_regimes = set(state_map.values())
all_regimes = {'risk_on', 'neutral', 'risk_off', 'crisis'}
missing = all_regimes - assigned_regimes

if missing:
    print(f"\n⚠️  Missing regimes: {missing}")
    print("  Assigning to ensure coverage...")

    for missing_regime in missing:
        # Find state with highest mean VIX for crisis, lowest for risk_on, etc.
        if missing_regime == 'crisis':
            # Assign to state with highest VIX_Z
            best_state = means_df['VIX_Z'].idxmax() if 'VIX_Z' in means_df.columns else 3
        elif missing_regime == 'risk_on':
            # Assign to state with lowest VIX_Z
            best_state = means_df['VIX_Z'].idxmin() if 'VIX_Z' in means_df.columns else 0
        else:
            # Assign to any unassigned state
            best_state = next(s for s in range(4) if s not in state_map or state_map[s] not in assigned_regimes)

        state_map[best_state] = missing_regime
        print(f"  State {best_state} → {missing_regime}")

print("\nFinal state mapping:")
for state_id, regime in sorted(state_map.items()):
    print(f"  State {state_id} → {regime}")

# Classify all data
print("\nClassifying all bars...")
states = best_model.predict(X_scaled)
probs = best_model.predict_proba(X_scaled)

df_result = df.copy()
df_result['regime_label'] = [state_map[s] for s in states]
df_result['regime_confidence'] = probs.max(axis=1)

# Show distribution
print("\nRegime distribution:")
regime_dist = df_result['regime_label'].value_counts()
for regime, count in regime_dist.items():
    pct = count / len(df_result) * 100
    print(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

# Validate: Check transitions
transitions = (df_result['regime_label'] != df_result['regime_label'].shift()).sum()
years = (df_result.index[-1] - df_result.index[0]).days / 365.25
trans_per_year = transitions / years
print(f"\nTransitions per year: {trans_per_year:.1f} (target: 10-20)")

# Validate: Silhouette score
try:
    labels_int = df_result['regime_label'].map({
        'risk_on': 0, 'neutral': 1, 'risk_off': 2, 'crisis': 3
    }).values
    silhouette = silhouette_score(X_scaled, labels_int, sample_size=min(10000, len(X_scaled)))
    print(f"Silhouette score: {silhouette:.3f} (target: >0.50)")
except Exception as e:
    print(f"Silhouette score: Failed ({e})")
    silhouette = None

# Save model
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_obj = {
    'model': best_model,
    'scaler': scaler,
    'state_map': state_map,
    'feature_order': available_features,
    'best_seed': best_seed,
    'log_likelihood': best_score,
    'silhouette_score': silhouette,
    'transitions_per_year': trans_per_year,
    'training_samples': len(X),
    'model_type': 'hmm_simplified'
}

output_path = 'models/hmm_regime_v2_simplified.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(model_obj, f)

print(f"\n✅ Model saved: {output_path}")
print(f"✅ Training samples: {len(X):,}")
print(f"✅ Best seed: {best_seed}")
print(f"✅ Log-likelihood: {best_score:.2f}")

# Save regime labels
labels_path = 'data/regime_labels_hmm_v2.parquet'
regime_cols = ['regime_label', 'regime_confidence']
df_result[regime_cols].to_parquet(labels_path)
print(f"✅ Labels saved: {labels_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\n🚀 Next step: Run validation with bin/quick_hmm_validation.py")
