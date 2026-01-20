#!/usr/bin/env python3
"""
Train HMM with State Features Only - Phase 1
=============================================

Phase 1 of hybrid regime detection: Train HMM with ONLY state features
(no binary event flags, no macro features).

Features Used (4 total):
    - crash_frequency_7d: Rolling count of crash events over 7 days
    - crisis_persistence: EWMA of crisis conditions (96H span)
    - aftershock_score: Decay-weighted composite of recent events
    - drawdown_persistence: EWMA of drawdown state (>10% from high)

Success Criteria:
    - Transitions/year: <50 (down from 117)
    - LUNA crisis detection: >60%
    - FTX crisis detection: >40%
    - June crisis detection: >40%

Usage:
    python bin/train_hmm_phase1_state_only.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# Phase 1: State features only
STATE_FEATURES = [
    'crash_frequency_7d',
    'crisis_persistence',
    'aftershock_score',
    'drawdown_persistence'
]

# Crisis events for validation
CRISIS_EVENTS = {
    'LUNA': ('2022-05-09', '2022-05-12'),
    'June_2022': ('2022-06-13', '2022-06-18'),
    'FTX': ('2022-11-08', '2022-11-11'),
}

BULL_PERIODS = {
    'Q1_2023_Rally': ('2023-01-01', '2023-03-31'),
    'Q4_2023_Rally': ('2023-10-01', '2023-12-31'),
}


def load_and_prepare_data():
    """Load feature store and prepare state features."""
    print("="*80)
    print("PHASE 1: TRAIN HMM WITH STATE FEATURES ONLY")
    print("="*80)

    # Load feature store
    print("\n[1/5] Loading feature store...")
    df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    print(f"  ✅ Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Check state features exist
    print("\n[2/5] Checking state features...")
    missing = [f for f in STATE_FEATURES if f not in df.columns]
    if missing:
        print(f"  ❌ Missing features: {missing}")
        print("  Run: python bin/add_crisis_features.py --tier all")
        sys.exit(1)

    for feat in STATE_FEATURES:
        mean = df[feat].mean()
        std = df[feat].std()
        act = (df[feat] > 0.1).sum() / len(df) * 100
        print(f"  ✅ {feat:25s}: mean={mean:.3f}, std={std:.3f}, act={act:.1f}%")

    # Extract feature matrix
    print("\n[3/5] Preparing feature matrix...")
    X = df[STATE_FEATURES].copy()

    # Handle missing values (forward fill)
    X = X.fillna(method='ffill').fillna(0)

    # Check for inf/nan
    if X.isna().any().any() or np.isinf(X.values).any():
        print("  ⚠️  WARNING: Found NaN/inf values, cleaning...")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  ✅ Feature matrix: {X.shape}")
    print(f"  Range: {X.min().min():.3f} to {X.max().max():.3f}")

    return df, X


def train_hmm(X, n_components=4, n_init=10):
    """Train Gaussian HMM with multiple random initializations."""
    print(f"\n[4/5] Training HMM (n_states={n_components}, n_init={n_init})...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Scaled feature statistics:")
    for i, feat in enumerate(STATE_FEATURES):
        print(f"    {feat:25s}: mean={X_scaled[:, i].mean():.3f}, std={X_scaled[:, i].std():.3f}")

    # Train HMM with multiple initializations
    best_model = None
    best_score = -np.inf
    best_idx = -1

    print(f"\n  Training {n_init} random initializations...")
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_components,
            covariance_type='full',
            n_iter=100,
            random_state=i,
            verbose=False
        )

        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)

            if score > best_score:
                best_score = score
                best_model = model
                best_idx = i

            print(f"    Init {i:2d}: log-likelihood={score:.2f}")

        except Exception as e:
            print(f"    Init {i:2d}: FAILED - {e}")

    if best_model is None:
        print("  ❌ All initializations failed!")
        sys.exit(1)

    print(f"\n  ✅ Best model: init {best_idx}, log-likelihood={best_score:.2f}")

    return best_model, scaler, best_score


def interpret_states(model, scaler, df, X):
    """Interpret HMM states based on feature means."""
    print("\n[5/5] Interpreting HMM states...")

    # Get state predictions
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)

    # Compute feature means by state
    state_profiles = []
    for state_id in range(model.n_components):
        mask = states == state_id
        profile = {
            'state_id': state_id,
            'count': mask.sum(),
            'pct': mask.sum() / len(states) * 100
        }

        for i, feat in enumerate(STATE_FEATURES):
            profile[feat] = X[mask][feat].mean()

        state_profiles.append(profile)

    # Sort by crisis_persistence (most crisis-like state = highest persistence)
    state_profiles.sort(key=lambda x: x['crisis_persistence'], reverse=True)

    # Assign labels
    labels = ['crisis', 'risk_off', 'neutral', 'risk_on']
    state_mapping = {}

    print("\n  State Profiles (sorted by crisis_persistence):")
    print("  " + "="*76)
    print(f"  {'State':<10} {'Label':<12} {'Pct':<8} {'crash_freq':<12} {'crisis_pers':<12} {'aftershock':<12} {'drawdown':<12}")
    print("  " + "-"*76)

    for i, profile in enumerate(state_profiles):
        state_id = profile['state_id']
        label = labels[i] if i < len(labels) else f'state_{i}'
        state_mapping[state_id] = label

        print(f"  {state_id:<10} {label:<12} {profile['pct']:>6.1f}%  "
              f"{profile['crash_frequency_7d']:>10.3f}  "
              f"{profile['crisis_persistence']:>10.3f}  "
              f"{profile['aftershock_score']:>10.3f}  "
              f"{profile['drawdown_persistence']:>10.3f}")

    print("  " + "="*76)

    # Apply labels
    regime_labels = pd.Series([state_mapping[s] for s in states], index=df.index)

    # Count transitions
    transitions = (regime_labels != regime_labels.shift()).sum()
    years = (df.index[-1] - df.index[0]).days / 365.25
    trans_per_year = transitions / years

    print(f"\n  Transition Statistics:")
    print(f"    Total transitions: {transitions}")
    print(f"    Years: {years:.2f}")
    print(f"    Transitions/year: {trans_per_year:.1f}")

    return regime_labels, state_mapping, trans_per_year


def validate_crisis_detection(df, regime_labels):
    """Validate crisis detection on known events."""
    print("\n" + "="*80)
    print("CRISIS DETECTION VALIDATION")
    print("="*80)

    results = {}

    for event_name, (start, end) in CRISIS_EVENTS.items():
        window = df.loc[start:end]
        regime_window = regime_labels.loc[start:end]

        crisis_pct = (regime_window == 'crisis').sum() / len(regime_window) * 100
        risk_off_pct = (regime_window == 'risk_off').sum() / len(regime_window) * 100
        combined_pct = crisis_pct + risk_off_pct

        results[event_name] = {
            'crisis': crisis_pct,
            'risk_off': risk_off_pct,
            'combined': combined_pct
        }

        print(f"\n📊 {event_name} ({start} to {end}): {len(window)} bars")
        print(f"  Crisis: {crisis_pct:.1f}%")
        print(f"  Risk-Off: {risk_off_pct:.1f}%")
        print(f"  Combined: {combined_pct:.1f}%")

    # Check for false positives during bull periods
    print("\n" + "="*80)
    print("FALSE POSITIVE CHECK (Bull Periods)")
    print("="*80)

    for period_name, (start, end) in BULL_PERIODS.items():
        window = df.loc[start:end]
        regime_window = regime_labels.loc[start:end]

        crisis_pct = (regime_window == 'crisis').sum() / len(regime_window) * 100
        risk_on_pct = (regime_window == 'risk_on').sum() / len(regime_window) * 100

        print(f"\n📈 {period_name} ({start} to {end}): {len(window)} bars")
        print(f"  Crisis (should be <10%): {crisis_pct:.1f}%")
        print(f"  Risk-On (should be >50%): {risk_on_pct:.1f}%")

    return results


def assess_phase1_results(trans_per_year, crisis_results):
    """Assess if Phase 1 passes success criteria."""
    print("\n" + "="*80)
    print("PHASE 1 SUCCESS CRITERIA ASSESSMENT")
    print("="*80)

    passed = []
    failed = []

    # Criterion 1: Transitions/year < 50
    if trans_per_year < 50:
        passed.append(f"✅ Transitions/year: {trans_per_year:.1f} < 50")
    else:
        failed.append(f"❌ Transitions/year: {trans_per_year:.1f} >= 50")

    # Criterion 2: LUNA > 60%
    luna_pct = crisis_results['LUNA']['crisis']
    if luna_pct > 60:
        passed.append(f"✅ LUNA crisis detection: {luna_pct:.1f}% > 60%")
    else:
        failed.append(f"❌ LUNA crisis detection: {luna_pct:.1f}% < 60%")

    # Criterion 3: FTX > 40%
    ftx_pct = crisis_results['FTX']['crisis']
    if ftx_pct > 40:
        passed.append(f"✅ FTX crisis detection: {ftx_pct:.1f}% > 40%")
    else:
        failed.append(f"❌ FTX crisis detection: {ftx_pct:.1f}% < 40%")

    # Criterion 4: June > 40%
    june_pct = crisis_results['June_2022']['crisis']
    if june_pct > 40:
        passed.append(f"✅ June crisis detection: {june_pct:.1f}% > 40%")
    else:
        failed.append(f"❌ June crisis detection: {june_pct:.1f}% < 40%")

    print("\nPassed Criteria:")
    for p in passed:
        print(f"  {p}")

    if failed:
        print("\nFailed Criteria:")
        for f in failed:
            print(f"  {f}")

    # Overall assessment
    print("\n" + "="*80)
    if len(failed) == 0:
        print("🎯 PHASE 1: PASS - All criteria met!")
        print("✅ Proceed to Phase 2: Add macro features")
        decision = "PASS"
    elif len(passed) >= 2:
        print("⚠️  PHASE 1: PARTIAL - Some improvement, needs tuning")
        print("🔧 Recommend: Adjust EWMA spans or decay rates")
        decision = "PARTIAL"
    else:
        print("❌ PHASE 1: FAIL - Insufficient improvement")
        print("🔄 Recommend: Pivot to supervised learning or rule-based approach")
        decision = "FAIL"
    print("="*80)

    return decision


def main():
    # Load data
    df, X = load_and_prepare_data()

    # Train HMM
    model, scaler, score = train_hmm(X, n_components=4, n_init=10)

    # Interpret states
    regime_labels, state_mapping, trans_per_year = interpret_states(model, scaler, df, X)

    # Validate
    crisis_results = validate_crisis_detection(df, regime_labels)

    # Assess
    decision = assess_phase1_results(trans_per_year, crisis_results)

    # Save model
    print("\n" + "="*80)
    print("SAVING PHASE 1 MODEL")
    print("="*80)

    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / 'hmm_regime_phase1.pkl'
    model_obj = {
        'model': model,
        'scaler': scaler,
        'features': STATE_FEATURES,
        'state_mapping': state_mapping,
        'log_likelihood': score,
        'transitions_per_year': trans_per_year,
        'crisis_detection': crisis_results,
        'decision': decision,
        'trained_at': datetime.now().isoformat()
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_obj, f)

    print(f"  ✅ Model saved: {model_path}")

    # Save regime labels
    labels_path = Path('data/regime_labels_phase1.parquet')
    regime_df = pd.DataFrame({'regime_label': regime_labels})
    regime_df.to_parquet(labels_path)
    print(f"  ✅ Labels saved: {labels_path}")

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"\nDecision: {decision}")
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")

    return 0 if decision == "PASS" else 1


if __name__ == '__main__':
    sys.exit(main())
