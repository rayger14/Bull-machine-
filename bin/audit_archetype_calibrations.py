#!/usr/bin/env python3
"""
Archetype Calibration Audit - Verify Full Knowledge Base and Optimized Parameters

This script audits whether archetypes are being tested with:
1. FULL optimized calibrations (post-optimization parameters)
2. ALL required features enabled (complete feature store)
3. CORRECT tunings (regime-specific thresholds from optimization runs)
4. Full knowledge base (all domain engines: Wyckoff, SMC, temporal, macro)
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def audit_feature_store_domain_coverage():
    """Check if feature store contains ALL archetype domain features."""

    print("\n" + "="*80)
    print("DOMAIN FEATURE COVERAGE AUDIT")
    print("="*80)

    # Load feature store
    feature_path = project_root / "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"

    if not feature_path.exists():
        print(f"\n⚠️  WARNING: Feature store not found at {feature_path}")
        return {}

    df = pd.read_parquet(feature_path)
    print(f"\nFeature Store Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")

    # Define required domain features for archetypes
    domains = {
        "Wyckoff (Structural Events)": [
            "wyckoff_phase", "wyckoff_event",
            "volume_climax_3b", "wick_exhaustion_3b",
            "accumulation_score", "distribution_score"
        ],
        "Smart Money Concepts (SMC)": [
            "order_block_bull", "order_block_bear",
            "fvg_bull", "fvg_bear",
            "bos_bull", "bos_choch",
            "liquidity_sweep_high", "liquidity_sweep_low"
        ],
        "Temporal/Fibonacci Time": [
            "fib_time_cluster", "gann_time_window",
            "temporal_confluence", "time_cycle_alignment"
        ],
        "Macro/Regime": [
            "macro_regime", "regime_v2",
            "usdt_d", "btc_d",
            "risk_sentiment"
        ],
        "Funding/OI (S4/S5)": [
            "funding_rate", "funding_z",
            "oi_change_pct_24h", "oi_delta_z",
            "oi_long_short_ratio"
        ],
        "Liquidity (S1/S4)": [
            "liquidity_score", "liquidity_drain_severity",
            "liquidity_velocity_score", "liquidity_persistence_score"
        ],
        "Core Technical": [
            "rsi_14", "atr_14", "volume_z",
            "price_distance_from_high_30d", "capitulation_depth"
        ]
    }

    results = {}
    total_missing = 0
    total_partial = 0
    total_complete = 0

    for domain, features in domains.items():
        missing = []
        partial = []
        complete = []

        for feat in features:
            if feat not in df.columns:
                missing.append(feat)
                total_missing += 1
            else:
                null_pct = df[feat].isna().mean() * 100
                if null_pct > 50:
                    partial.append((feat, null_pct))
                    total_partial += 1
                else:
                    complete.append(feat)
                    total_complete += 1

        coverage_pct = len(complete) / len(features) * 100 if features else 0

        results[domain] = {
            "complete": complete,
            "partial": partial,
            "missing": missing,
            "coverage": coverage_pct
        }

        # Print domain results
        status = "✅" if coverage_pct == 100 else "⚠️" if coverage_pct >= 80 else "❌"
        print(f"\n{status} {domain}:")
        print(f"   Coverage: {coverage_pct:.1f}% ({len(complete)}/{len(features)} features)")

        if missing:
            print(f"   ❌ Missing: {missing}")
        if partial:
            print(f"   ⚠️  Partial (>50% null): {[(f, f'{p:.1f}%') for f, p in partial]}")
        if coverage_pct == 100:
            print(f"   ✅ All features present and complete")

    # Summary
    print("\n" + "="*80)
    print("DOMAIN COVERAGE SUMMARY")
    print("="*80)
    total_features = total_missing + total_partial + total_complete
    print(f"Total Features Required: {total_features}")
    print(f"✅ Complete (usable): {total_complete} ({total_complete/total_features*100:.1f}%)")
    print(f"⚠️  Partial (>50% null): {total_partial} ({total_partial/total_features*100:.1f}%)")
    print(f"❌ Missing: {total_missing} ({total_missing/total_features*100:.1f}%)")

    if total_missing > 0 or total_partial > 0:
        print(f"\n⚠️  WARNING: {total_missing + total_partial} features unavailable or incomplete")
        print("   This may significantly reduce archetype edge!")
    else:
        print("\n✅ All domain features present and complete!")

    return results


def check_domain_engine_activation(config_path: Path) -> Dict[str, bool]:
    """Check which domain engines are active in config."""

    if not config_path.exists():
        print(f"\n⚠️  Config not found: {config_path}")
        return {}

    with open(config_path) as f:
        cfg = json.load(f)

    # Check feature flags (not all configs have this section)
    feature_flags = cfg.get('feature_flags', {})

    # Check archetype thresholds for implicit feature usage
    archetype_cfg = cfg.get('archetypes', {}).get('thresholds', {})

    # Check for runtime features in archetype configs
    runtime_features_enabled = False
    if archetype_cfg:
        for k, v in archetype_cfg.items():
            if isinstance(v, dict) and v.get('use_runtime_features', False):
                runtime_features_enabled = True
                break

    engines = {
        "Wyckoff": feature_flags.get('use_wyckoff', False),
        "SMC": feature_flags.get('use_smc', False),
        "Temporal Confluence": feature_flags.get('use_temporal_confluence', False),
        "Macro Regime": feature_flags.get('use_macro_regime', True),  # Usually implicit
        "Fusion Layer": feature_flags.get('use_fusion_layer', False),
        "Runtime Features": runtime_features_enabled
    }

    print(f"\n📋 Domain Engine Activation: {config_path.name}")
    print("-" * 60)
    enabled_count = 0
    for engine, enabled in engines.items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"   {engine:25s}: {status}")
        if enabled:
            enabled_count += 1

    print(f"\n   Engines Active: {enabled_count}/{len(engines)}")

    # Check if using optimized parameters
    has_optimized = any([
        'optimized' in config_path.name.lower(),
        'calibration' in config_path.name.lower(),
        'production' in config_path.name.lower()
    ])

    if not has_optimized:
        print(f"   ⚠️  WARNING: Config name suggests vanilla/test parameters (not optimized)")
    else:
        print(f"   ✅ Config name suggests optimized/production parameters")

    return engines


def compare_config_vs_optimized(test_config_path: Path, optimized_config_path: Path, archetype_name: str):
    """Compare test config against optimized parameters."""

    print(f"\n{'='*80}")
    print(f"PARAMETER DRIFT ANALYSIS: {archetype_name}")
    print(f"{'='*80}")

    if not test_config_path.exists():
        print(f"❌ Test config not found: {test_config_path}")
        return False

    if not optimized_config_path.exists():
        print(f"⚠️  Optimized config not found: {optimized_config_path}")
        print(f"   Skipping drift analysis for {archetype_name}")
        return None

    with open(test_config_path) as f:
        test_cfg = json.load(f)

    with open(optimized_config_path) as f:
        opt_cfg = json.load(f)

    # Extract archetype-specific thresholds
    test_thresholds = test_cfg.get('archetypes', {}).get('thresholds', {}).get(archetype_name, {})
    opt_thresholds = opt_cfg.get('archetypes', {}).get('thresholds', {}).get(archetype_name, {})

    if not test_thresholds or not opt_thresholds:
        print(f"⚠️  Archetype thresholds not found in one or both configs")
        return None

    print(f"\nTest Config: {test_config_path.name}")
    print(f"Optimized Config: {optimized_config_path.name}")
    print(f"Archetype: {archetype_name}")

    # Compare key parameters
    key_params = [
        'fusion_threshold', 'final_fusion_gate',
        'funding_z_max', 'funding_z_min',
        'resilience_min', 'liquidity_max',
        'rsi_min', 'rsi_max',
        'cooldown_bars', 'atr_stop_mult', 'max_risk_pct',
        'capitulation_depth_max', 'crisis_composite_min',
        'confluence_threshold'
    ]

    drifts = []
    matches = []

    for param in key_params:
        test_val = test_thresholds.get(param)
        opt_val = opt_thresholds.get(param)

        if test_val is None and opt_val is None:
            continue  # Param not relevant for this archetype

        if test_val is None or opt_val is None:
            drifts.append({
                'param': param,
                'test': test_val,
                'optimized': opt_val,
                'drift_%': 'N/A',
                'status': 'MISSING'
            })
            continue

        if test_val == opt_val:
            matches.append(param)
        else:
            # Calculate drift percentage
            if opt_val != 0:
                drift_pct = abs(test_val - opt_val) / abs(opt_val) * 100
            else:
                drift_pct = float('inf') if test_val != 0 else 0

            drifts.append({
                'param': param,
                'test': test_val,
                'optimized': opt_val,
                'drift_%': drift_pct,
                'status': 'DRIFT'
            })

    print(f"\n✅ Matching Parameters: {len(matches)}")
    if matches:
        print(f"   {matches}")

    if drifts:
        print(f"\n⚠️  DRIFTED/MISSING Parameters: {len(drifts)}")
        print(f"\n{'Parameter':<30} {'Optimized':<15} {'Test':<15} {'Drift':<15} {'Status':<10}")
        print("-" * 95)
        for d in drifts:
            drift_str = f"{d['drift_%']:.1f}%" if isinstance(d['drift_%'], (int, float)) else d['drift_%']
            print(f"{d['param']:<30} {str(d['optimized']):<15} {str(d['test']):<15} {drift_str:<15} {d['status']:<10}")

        return False
    else:
        print(f"\n✅ ALL PARAMETERS MATCH OPTIMIZED CONFIG")
        return True


def query_optuna_best_trial(db_path: Path, study_name: str = None):
    """Query Optuna database for best trial parameters."""

    if not db_path.exists():
        print(f"❌ Optuna DB not found: {db_path}")
        return None

    print(f"\n{'='*80}")
    print(f"OPTUNA OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Database: {db_path.name}")

    try:
        conn = sqlite3.connect(db_path)

        # Get all studies
        studies_df = pd.read_sql_query("SELECT * FROM studies", conn)
        print(f"\nStudies Found: {len(studies_df)}")

        if len(studies_df) == 0:
            print("⚠️  No studies found in database")
            conn.close()
            return None

        for _, study in studies_df.iterrows():
            study_id = study['study_id']
            study_name = study['study_name']

            # Get best trial
            trials_df = pd.read_sql_query(
                f"SELECT * FROM trials WHERE study_id = {study_id} ORDER BY value DESC LIMIT 1",
                conn
            )

            if len(trials_df) == 0:
                continue

            best_trial = trials_df.iloc[0]

            print(f"\n📊 Study: {study_name}")
            print(f"   Best Trial: #{best_trial['number']}")
            print(f"   Objective Value: {best_trial['value']:.4f}")
            print(f"   State: {best_trial['state']}")

            # Get trial parameters
            trial_id = best_trial['trial_id']
            params_df = pd.read_sql_query(
                f"SELECT * FROM trial_params WHERE trial_id = {trial_id}",
                conn
            )

            if len(params_df) > 0:
                print(f"\n   Optimized Parameters:")
                for _, param in params_df.iterrows():
                    print(f"      {param['param_name']:<30}: {param['param_value']}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error querying Optuna DB: {e}")
        return None


def main():
    """Run comprehensive calibration audit."""

    print("\n" + "="*80)
    print("ARCHETYPE CALIBRATION AUDIT")
    print("="*80)
    print("\nPurpose: Verify archetypes are tested with FULL calibrations and knowledge base")
    print("\nAuditing:")
    print("  1. Feature store domain coverage (Wyckoff, SMC, Temporal, Macro, Funding)")
    print("  2. Domain engine activation in configs")
    print("  3. Parameter drift (test configs vs optimized configs)")
    print("  4. Optuna optimization results")
    print("\n" + "="*80)

    # 1. Audit feature store
    domain_coverage = audit_feature_store_domain_coverage()

    # 2. Check domain engine activation for each archetype config
    print("\n" + "="*80)
    print("DOMAIN ENGINE ACTIVATION AUDIT")
    print("="*80)

    configs_to_check = [
        project_root / "configs/s1_v2_production.json",
        project_root / "configs/s4_optimized_oos_test.json",
        project_root / "configs/system_s5_production.json",
    ]

    for config_path in configs_to_check:
        check_domain_engine_activation(config_path)

    # 3. Compare test configs vs optimized configs
    print("\n" + "="*80)
    print("CONFIGURATION DRIFT AUDIT")
    print("="*80)

    comparisons = [
        (
            project_root / "configs/s4_optimized_oos_test.json",
            project_root / "results/s4_calibration/s4_optimized_config.json",
            "funding_divergence"
        ),
    ]

    drift_results = []
    for test_cfg, opt_cfg, archetype in comparisons:
        result = compare_config_vs_optimized(test_cfg, opt_cfg, archetype)
        drift_results.append((archetype, result))

    # 4. Query Optuna databases
    print("\n" + "="*80)
    print("OPTUNA OPTIMIZATION DATABASE AUDIT")
    print("="*80)

    optuna_dbs = [
        project_root / "results/s4_calibration/optuna_s4_calibration.db",
        project_root / "optuna_production_v2_long_squeeze.db",
    ]

    for db_path in optuna_dbs:
        if db_path.exists():
            query_optuna_best_trial(db_path)

    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT: ARE WE TESTING WITH FULL CALIBRATIONS?")
    print("="*80)

    # Check domain coverage
    if domain_coverage:
        total_domains = len(domain_coverage)
        complete_domains = sum(1 for d in domain_coverage.values() if d['coverage'] == 100)

        print(f"\n📊 Domain Coverage: {complete_domains}/{total_domains} domains complete")

        if complete_domains < total_domains:
            print("   ⚠️  INCOMPLETE KNOWLEDGE BASE - Missing domain features may reduce edge")

    # Check parameter drift
    drifted = sum(1 for _, result in drift_results if result is False)
    matched = sum(1 for _, result in drift_results if result is True)
    unknown = sum(1 for _, result in drift_results if result is None)

    print(f"\n📊 Configuration Drift: {matched} matched, {drifted} drifted, {unknown} unknown")

    if drifted > 0:
        print("   ⚠️  CONFIGURATION DRIFT DETECTED - Test configs differ from optimized")

    # Final assessment
    print("\n" + "="*80)

    if (complete_domains == total_domains if domain_coverage else False) and drifted == 0:
        print("✅ VERDICT: Testing with FULL calibrations and complete knowledge base")
        print("\n   All domain features present")
        print("   All parameters match optimized configs")
        print("   Poor performance likely due to:")
        print("     - Legitimate strategy failure")
        print("     - Data quality issues")
        print("     - Regime mismatch (testing in wrong market conditions)")
    else:
        print("⚠️  VERDICT: Testing with INCOMPLETE calibrations or knowledge base")
        print("\n   Poor performance may be due to:")
        if complete_domains < total_domains if domain_coverage else True:
            print("     ❌ Missing domain features (Wyckoff, SMC, Temporal, etc.)")
        if drifted > 0:
            print("     ❌ Test configs using vanilla/drifted parameters (not optimized)")
        print("\n   RECOMMENDATION: Fix calibration issues before accepting poor results!")

    print("="*80)


if __name__ == "__main__":
    main()
