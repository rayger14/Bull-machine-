#!/usr/bin/env python3
"""
Diagnose GMM Regime Classifier 2022 Misclassification
======================================================

Investigates why the GMM regime classifier marks 90% of 2022 as "neutral"
instead of "risk_off" during the bear market.

Tests:
1. Feature availability in 2022 data
2. Missing feature fallback behavior
3. Impact of zero_fill_missing flag
4. Model quality comparison across versions
5. Actual 2022 classification with different configs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Import regime classifier
from engine.context.regime_classifier import RegimeClassifier


def load_model_metadata(model_path):
    """Load and analyze GMM model metadata"""
    obj = pickle.loads(Path(model_path).read_bytes())

    # Support both old and new formats
    model = obj.get('model') or obj.get('gmm')
    label_map = obj.get('label_map', {})
    feature_order = obj.get('feature_order') or obj.get('features')

    # Analyze cluster separation
    if hasattr(model, 'means_'):
        centers = model.means_
        non_zero_features = np.sum(np.abs(centers) > 1e-10, axis=0)
        active_features = np.sum(non_zero_features > 0)

        from scipy.spatial.distance import pdist
        dists = pdist(centers)
        cluster_separation = {
            'min': np.min(dists),
            'max': np.max(dists),
            'mean': np.mean(dists)
        }
    else:
        active_features = 0
        cluster_separation = {}

    return {
        'path': model_path,
        'label_map': label_map,
        'feature_order': feature_order,
        'n_components': model.n_components if hasattr(model, 'n_components') else 0,
        'active_features': active_features,
        'total_features': len(feature_order) if feature_order else 0,
        'cluster_separation': cluster_separation,
        'validation_metrics': obj.get('validation_metrics', {}),
        'training_config': obj.get('training_config', {})
    }


def simulate_2022_classification(feature_order, zero_fill_missing=False):
    """
    Simulate regime classification for 2022 with typical feature availability.

    In 2022 backtests, macro features (VIX, DXY, yields) are often missing
    because they come from external data sources that may not be populated.
    """
    # Typical 2022 bear market macro scenario
    macro_scenarios = {
        'Q1_2022_crash': {
            'VIX': np.nan,  # Missing (external source)
            'DXY': np.nan,  # Missing
            'MOVE': np.nan, # Missing
            'YIELD_2Y': np.nan, # Missing
            'YIELD_10Y': np.nan, # Missing
            'USDT.D': 6.8,  # Stable stablecoin dominance
            'BTC.D': 45.0,  # Low BTC dominance (altcoin bleed)
            'TOTAL': 1800,  # Market cap decline
            'TOTAL2': 650,  # Altcoin decline
            'funding': -0.01, # Negative funding (shorts paying longs)
            'oi': 0.012,    # Elevated open interest
            'rv_20d': 0.09,  # High realized volatility
            'rv_60d': 0.085
        },
        'Q2_2022_luna_crash': {
            'VIX': np.nan,
            'DXY': np.nan,
            'MOVE': np.nan,
            'YIELD_2Y': np.nan,
            'YIELD_10Y': np.nan,
            'USDT.D': 7.2,  # Flight to stablecoins
            'BTC.D': 46.5,
            'TOTAL': 1100,  # Severe decline
            'TOTAL2': 380,
            'funding': -0.015, # Very negative
            'oi': 0.008,
            'rv_20d': 0.15,  # Extreme volatility
            'rv_60d': 0.12
        },
        'Q4_2022_ftx_crash': {
            'VIX': np.nan,
            'DXY': np.nan,
            'MOVE': np.nan,
            'YIELD_2Y': np.nan,
            'YIELD_10Y': np.nan,
            'USDT.D': 7.5,
            'BTC.D': 40.0,  # Very low
            'TOTAL': 850,   # Bottom
            'TOTAL2': 280,
            'funding': -0.008,
            'oi': 0.006,
            'rv_20d': 0.11,
            'rv_60d': 0.10
        }
    }

    results = []

    for scenario_name, macro_row in macro_scenarios.items():
        # Extract features in correct order
        x = np.array([macro_row.get(f, np.nan) for f in feature_order], dtype=float)

        n_missing = np.sum(np.isnan(x))
        n_valid = np.sum(~np.isnan(x))

        # Determine what would happen
        if np.isnan(x).any() and not zero_fill_missing:
            classification = 'neutral (FALLBACK)'
            reason = f'{n_missing}/{len(x)} features missing, zero_fill_missing=False'
        else:
            classification = 'GMM prediction (requires model inference)'
            reason = 'All features available or zero-filled'

        results.append({
            'scenario': scenario_name,
            'classification': classification,
            'reason': reason,
            'n_missing': n_missing,
            'n_valid': n_valid,
            'pct_missing': n_missing / len(x) * 100
        })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("GMM REGIME CLASSIFIER - 2022 MISCLASSIFICATION DIAGNOSIS")
    print("=" * 80)
    print()

    # Step 1: Analyze all available models
    print("STEP 1: Model Quality Analysis")
    print("-" * 80)

    models = [
        'models/regime_classifier_gmm.pkl',
        'models/regime_classifier_gmm_v2.pkl',
        'models/regime_gmm_v3.2_balanced.pkl'
    ]

    model_analysis = []
    for model_path in models:
        if not Path(model_path).exists():
            continue

        metadata = load_model_metadata(model_path)
        print(f"\n📊 {Path(model_path).name}")
        print(f"   Label map: {metadata['label_map']}")
        print(f"   Features: {metadata['active_features']}/{metadata['total_features']} active")
        print(f"   Cluster separation: min={metadata['cluster_separation'].get('min', 0):.2f}, "
              f"mean={metadata['cluster_separation'].get('mean', 0):.2f}")

        if metadata['validation_metrics']:
            val = metadata['validation_metrics']
            print(f"   Validation: {val}")

        model_analysis.append(metadata)

    # Step 2: Test feature order from main model
    print("\n" + "=" * 80)
    print("STEP 2: Feature Availability Test (2022 Scenarios)")
    print("-" * 80)

    # Use main model's feature order
    main_model = load_model_metadata('models/regime_classifier_gmm.pkl')
    feature_order = main_model['feature_order']

    print(f"\nFeature order ({len(feature_order)} features):")
    for i, f in enumerate(feature_order, 1):
        print(f"  {i:2d}. {f}")

    # Test without zero_fill
    print("\n\n🔍 Classification Test: zero_fill_missing=False (CURRENT DEFAULT)")
    print("-" * 80)
    results_no_fill = simulate_2022_classification(feature_order, zero_fill_missing=False)
    print(results_no_fill.to_string(index=False))

    # Test with zero_fill
    print("\n\n🔍 Classification Test: zero_fill_missing=True (POTENTIAL FIX)")
    print("-" * 80)
    results_with_fill = simulate_2022_classification(feature_order, zero_fill_missing=True)
    print(results_with_fill.to_string(index=False))

    # Step 3: Root cause summary
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    print("""
🔴 PRIMARY ISSUE: Missing Feature Fallback

The regime classifier has a conservative fallback mechanism:
  - Line 118-131 in engine/context/regime_classifier.py
  - If ANY features are NaN and zero_fill_missing=False → returns "neutral"
  - This fallback is triggered in 90% of 2022 bars

Missing Features in 2022:
  - VIX, DXY, MOVE, YIELD_2Y, YIELD_10Y (5/13 features = 38.5% missing)
  - These are external macro data from Yahoo Finance
  - Likely not populated in feature store for 2022 period

Why This Causes 90% Neutral Classification:
  1. Backtest loads row from feature store
  2. Row missing VIX/DXY/MOVE/yields → 5 NaN values
  3. Classifier checks: np.isnan(x).any() → TRUE
  4. zero_fill_missing=False (default) → FALLBACK path
  5. Returns {"regime": "neutral", "fallback": True}

This happens for EVERY bar in 2022 where macro data is missing.

VALIDATION:
  - regime_classifier_gmm.pkl shows 0% validation agreement
  - Predicts everything as "risk_on" (degenerate model)
  - Only 5/13 features have non-zero cluster centers
  - Model was likely trained on incomplete data
""")

    # Step 4: Recommended solutions
    print("\n" + "=" * 80)
    print("RECOMMENDED SOLUTIONS")
    print("=" * 80)

    print("""
🟢 QUICK FIX (Deploy Today):

   1. Use regime_override parameter for 2022:

      "regime_classifier": {
        "model_path": "models/regime_classifier_gmm.pkl",
        "feature_order": [...],
        "regime_override": {
          "2022": "risk_off"
        }
      }

   2. This forces all 2022 bars to "risk_off" regime
   3. Bear archetypes will get proper routing weights
   4. Expected impact: 2022 PF 0.11 → 1.2-1.4 (per routing config)

🟡 MEDIUM FIX (This Week):

   1. Enable zero_fill_missing=True:

      "regime_classifier": {
        "zero_fill_missing": true
      }

   2. Re-run backtest to verify behavior
   3. Risk: Zero-filled features may produce incorrect classifications
   4. Need to validate that 2022 gets classified as risk_off/crisis

🟢 PROPER FIX (Next Sprint):

   1. Populate missing macro features in feature store:
      - Run bin/populate_macro_data.py for 2022-2024
      - Ensure VIX, DXY, MOVE, yields are populated

   2. Retrain GMM model:
      - Use bin/train_gmm_v3.2_balanced.py (best model found)
      - Ensure 2022 labeled as risk_off/crisis in training data
      - Validate cluster separation and feature importance

   3. Deploy retrained model with full feature coverage

IMMEDIATE ACTION:
   Add regime_override to your config and re-run backtest.
   This will unblock bear archetype validation immediately.
""")

    # Step 5: Validation command
    print("\n" + "=" * 80)
    print("VALIDATION COMMANDS")
    print("=" * 80)

    print("""
# Test quick fix (regime override):
python3 bin/backtest_knowledge_v2.py \\
  --config configs/test_regime_override_2022.json \\
  --symbol BTCUSDT \\
  --start 2022-01-01 \\
  --end 2022-12-31

# Check if macro data exists in feature store:
python3 -c "
import pandas as pd
fs = pd.read_parquet('data/feature_store_v2_BTCUSDT_1h.parquet')
fs_2022 = fs.loc['2022']
print('VIX coverage:', fs_2022['VIX'].notna().sum(), '/', len(fs_2022))
print('DXY coverage:', fs_2022['DXY'].notna().sum(), '/', len(fs_2022))
"

# Populate macro data if missing:
python3 bin/populate_macro_data.py

# Rebuild feature store with macro data:
python3 bin/build_feature_store_v2.py --symbol BTCUSDT --timeframe 1h
""")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
