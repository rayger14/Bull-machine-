#!/usr/bin/env python3
"""
Current State Diagnostic Tool

Quick checks to verify the system state and identify blocking issues.

Usage:
    python3 bin/diagnose_current_state.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime


def print_header(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def check_data_layer():
    """Check feature store completeness"""
    print_header("1. DATA LAYER CHECK")

    data_path = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'

    try:
        df = pd.read_parquet(data_path)
        print(f"✓ Main file loaded: {data_path}")
        print(f"  Shape: {df.shape[0]:,} bars × {df.shape[1]} columns")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Check S1 v2 features
        s1_features = [
            'wick_lower_ratio',
            'liquidity_vacuum_score',
            'volume_panic',
            'crisis_context',
            'liquidity_vacuum_fusion',
            'liquidity_drain_pct',
            'liquidity_velocity',
            'liquidity_persistence',
            'capitulation_depth',
            'crisis_composite',
            'volume_climax_last_3b',
            'wick_exhaustion_last_3b'
        ]

        print("\n  S1 v2 Features:")
        all_present = True
        for feat in s1_features:
            present = feat in df.columns
            status = "✓" if present else "✗"
            print(f"    {status} {feat}")
            if not present:
                all_present = False

        if all_present:
            print("\n  ✓ All S1 v2 features PRESENT")
        else:
            print("\n  ✗ MISSING S1 v2 features - runtime enrichment needed")

        # Check regime column
        print("\n  Macro Regime Column:")
        if 'macro_regime' in df.columns:
            regime_dist = df['macro_regime'].value_counts()
            print(f"    ✓ macro_regime column present")
            print(f"    Distribution:")
            for regime, count in regime_dist.items():
                pct = (count / len(df)) * 100
                print(f"      {regime}: {count:,} ({pct:.1f}%)")

            # Check for all 'neutral'
            if len(regime_dist) == 1 and 'neutral' in regime_dist:
                print(f"    ⚠️  WARNING: All regimes are 'neutral' - may block S1")
        else:
            print(f"    ✗ macro_regime column MISSING")

        # Check funding features
        print("\n  Funding Features:")
        funding_cols = ['funding', 'funding_rate', 'funding_Z']
        for col in funding_cols:
            present = col in df.columns
            status = "✓" if present else "✗"
            if present:
                non_null = df[col].notna().sum()
                pct_coverage = (non_null / len(df)) * 100
                print(f"    {status} {col}: {pct_coverage:.1f}% coverage")
            else:
                print(f"    {status} {col}")

        return True

    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False


def check_configs():
    """Check config file availability and structure"""
    print_header("2. CONFIG FILE CHECK")

    configs = {
        'Bull Market': 'configs/mvp/mvp_bull_market_v1.json',
        'Bear Market': 'configs/mvp/mvp_bear_market_v1.json',
        'S1 v2': 'configs/s1_v2_production.json',
    }

    for name, path in configs.items():
        try:
            with open(path, 'r') as f:
                config = json.load(f)

            print(f"\n✓ {name} Config: {path}")

            # Check key fields
            if 'archetypes' in config:
                archetypes = config['archetypes']
                print(f"  - use_archetypes: {archetypes.get('use_archetypes', False)}")

                # Check enabled archetypes
                enabled = [k[7:] for k, v in archetypes.items()
                          if k.startswith('enable_') and v is True]
                if enabled:
                    print(f"  - Enabled: {', '.join(enabled)}")

                # Check thresholds
                if 'thresholds' in archetypes:
                    thresholds = archetypes['thresholds']
                    print(f"  - Thresholds defined: {list(thresholds.keys())}")

            # Check regime override
            if 'regime_classifier' in config:
                rc = config['regime_classifier']
                if 'regime_override' in rc:
                    print(f"  - Regime override: {rc['regime_override']}")

        except FileNotFoundError:
            print(f"\n✗ {name} Config: NOT FOUND at {path}")
        except Exception as e:
            print(f"\n✗ {name} Config: ERROR loading - {e}")


def check_archetype_wrapper():
    """Check ArchetypeModel wrapper status"""
    print_header("3. ARCHETYPE WRAPPER CHECK")

    try:
        from engine.models.archetype_model import ArchetypeModel
        print("✓ ArchetypeModel imported successfully")

        # Try to instantiate with bear config
        try:
            model = ArchetypeModel(
                config_path='configs/mvp/mvp_bear_market_v1.json',
                archetype_name='long_squeeze',
                name='S5-Test'
            )
            print(f"✓ ArchetypeModel instantiated: {model.name}")
            print(f"  - Direction: {model.direction}")
            print(f"  - Default regime: {model.default_regime}")
            print(f"  - Fusion threshold: {model.fusion_threshold}")

            # Test regime switching
            model.set_regime('neutral')
            print(f"✓ Regime switching works")

        except Exception as e:
            print(f"✗ Failed to instantiate ArchetypeModel: {e}")

    except ImportError as e:
        print(f"✗ Failed to import ArchetypeModel: {e}")


def check_regime_classifier():
    """Check regime classifier availability"""
    print_header("4. REGIME CLASSIFIER CHECK")

    classifier_path = 'models/regime_classifier_gmm.pkl'

    try:
        from engine.context.regime_classifier import RegimeClassifier
        print("✓ RegimeClassifier module imported")

        # Check if model file exists
        from pathlib import Path
        if Path(classifier_path).exists():
            print(f"✓ Model file exists: {classifier_path}")

            # Try to load
            try:
                classifier = RegimeClassifier(model_path=classifier_path)
                print(f"✓ RegimeClassifier loaded successfully")

                # Test with sample data
                df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
                bar = df.iloc[1000]

                regime = classifier.predict_single(bar)
                probs = classifier.predict_proba_single(bar)

                print(f"\n  Sample prediction (bar 1000):")
                print(f"    Regime: {regime}")
                print(f"    Probabilities: {probs}")

            except Exception as e:
                print(f"✗ Failed to load classifier: {e}")

        else:
            print(f"✗ Model file NOT FOUND: {classifier_path}")

    except ImportError as e:
        print(f"✗ Failed to import RegimeClassifier: {e}")


def check_comparison_results():
    """Check if comparison has been run"""
    print_header("5. COMPARISON RESULTS CHECK")

    results_path = 'results/baseline_vs_archetype_comparison.csv'

    try:
        from pathlib import Path
        if Path(results_path).exists():
            df = pd.read_csv(results_path)
            print(f"✓ Comparison results found: {results_path}")
            print(f"\n  Results summary:")
            print(df.to_string(index=False))

            # Check for 0 trades
            if 'Test_Trades' in df.columns:
                zero_trade_models = df[df['Test_Trades'] == 0]['Model'].tolist()
                if zero_trade_models:
                    print(f"\n  ⚠️  Models with 0 trades: {', '.join(zero_trade_models)}")
        else:
            print(f"✗ No comparison results found at {results_path}")
            print(f"  Run: python3 examples/baseline_vs_archetype_comparison.py")

    except Exception as e:
        print(f"✗ Error checking results: {e}")


def check_runtime_enrichment():
    """Check runtime enrichment modules"""
    print_header("6. RUNTIME ENRICHMENT CHECK")

    runtime_modules = {
        'S1 (Liquidity Vacuum)': 'engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py',
        'S2 (Failed Rally)': 'engine/strategies/archetypes/bear/failed_rally_runtime.py',
        'S4 (Funding Divergence)': 'engine/strategies/archetypes/bear/funding_divergence_runtime.py',
        'S5 (Long Squeeze)': 'engine/strategies/archetypes/bear/long_squeeze_runtime.py',
    }

    for name, path in runtime_modules.items():
        from pathlib import Path
        if Path(path).exists():
            print(f"✓ {name}: {path}")

            # Try to import
            try:
                if 'liquidity_vacuum' in path:
                    from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment
                    print(f"  - apply_liquidity_vacuum_enrichment() importable ✓")
            except ImportError as e:
                print(f"  - Import failed: {e}")

        else:
            print(f"✗ {name}: NOT FOUND")


def print_summary():
    """Print diagnostic summary and recommendations"""
    print_header("DIAGNOSTIC SUMMARY")

    print("""
SYSTEM STATUS: 🟡 PARTIALLY FUNCTIONAL

Key Findings:
1. Data Layer: Check S1 v2 features and regime column above
2. Configs: Verify bull/bear/S1 configs present
3. Wrapper: Check ArchetypeModel instantiation
4. Regime: Check classifier availability
5. Comparison: Look for 0-trade models
6. Runtime: Verify enrichment modules exist

Next Steps:
1. If S1 v2 features missing → Data needs enrichment
2. If regime all 'neutral' → Need classifier or force regime
3. If ArchetypeModel fails → Config structure mismatch
4. If 0 trades in comparison → Regime blocking signals

Quick Fixes:
- Force regime: model.set_regime('crisis')
- Try S4/S5: They allow 'neutral' regime
- Check regime column: df['macro_regime'].value_counts()

See: BULL_MACHINE_CURRENT_STATE_REPORT.md for full analysis
    """)


def main():
    """Run all diagnostic checks"""
    print("=" * 80)
    print("BULL MACHINE - CURRENT STATE DIAGNOSTICS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run checks
    check_data_layer()
    check_configs()
    check_archetype_wrapper()
    check_regime_classifier()
    check_comparison_results()
    check_runtime_enrichment()

    # Summary
    print_summary()

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
