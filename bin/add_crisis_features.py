#!/usr/bin/env python3
"""
Add Real-Time Crisis Features to Feature Store
===============================================

Agent 2 Implementation: Crisis Feature Engineering

Adds 8 crisis indicators from Agent 1's research to the feature store:
1. flash_crash_1h, flash_crash_4h, flash_crash_1d
2. volume_spike, volume_z_7d
3. oi_delta_1h_z, oi_cascade
4. funding_extreme, funding_flip

Expected impact: 8-48x faster crisis detection (2 days → 0-6 hours)

Usage:
    python3 bin/add_crisis_features.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from engine.features.crisis_indicators import engineer_crisis_features
from engine.features.state_features import convert_events_to_states, validate_state_features


def main():
    parser = argparse.ArgumentParser(description='Add crisis features to feature store')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset symbol')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features')
    parser.add_argument('--tier', type=str, default='tier1', choices=['tier1', 'tier2', 'tier3', 'all'],
                        help='State feature tier to implement (default: tier1)')
    args = parser.parse_args()

    print("=" * 80)
    print("ADDING REAL-TIME CRISIS FEATURES TO FEATURE STORE")
    print("=" * 80)
    print(f"\nAsset: {args.asset}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Overwrite: {args.overwrite}")
    print(f"State Feature Tier: {args.tier}")

    # Step 1: Load existing feature store
    print("\n[1/4] Loading feature store...")

    # Try with_macro first, then without
    feature_file = Path(f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}_with_macro.parquet')
    if not feature_file.exists():
        feature_file = Path(f'data/features_mtf/{args.asset}_1H_{args.start}_to_{args.end}.parquet')

    if not feature_file.exists():
        print(f"  ❌ Feature file not found (tried both _with_macro and without)")
        print("\n  Available feature files:")
        for f in sorted(Path('data/features_mtf').glob(f'{args.asset}_*.parquet')):
            print(f"    - {f.name}")
        sys.exit(1)

    df = pd.read_parquet(feature_file)
    print(f"  ✅ Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Columns: {len(df.columns)}")

    # Step 2: Check if crisis features already exist
    print("\n[2/4] Checking existing crisis features...")

    # Event features
    event_features = [
        'flash_crash_1h', 'flash_crash_4h', 'flash_crash_1d',
        'volume_spike', 'volume_z_7d',
        'oi_delta_1h_z', 'oi_cascade',
        'funding_extreme', 'funding_flip',
        'crisis_composite_score', 'crisis_confirmed'
    ]

    # State features (by tier)
    state_features_tier1 = ['crash_stress_24h', 'crash_stress_72h', 'vol_persistence', 'hours_since_crisis']
    state_features_tier2 = ['crash_frequency_7d', 'funding_stress_ewma', 'cascade_risk', 'crisis_persistence']
    state_features_tier3 = ['vol_regime_shift', 'drawdown_persistence', 'aftershock_score']

    # All crisis features (events + states)
    crisis_features = event_features.copy()
    if args.tier in ['tier1', 'all']:
        crisis_features.extend(state_features_tier1)
    if args.tier in ['tier2', 'all']:
        crisis_features.extend(state_features_tier2)
    if args.tier in ['tier3', 'all']:
        crisis_features.extend(state_features_tier3)

    existing = [f for f in crisis_features if f in df.columns]

    if existing and not args.overwrite:
        print(f"  ⚠️  Found {len(existing)} existing crisis features:")
        for feat in existing:
            print(f"    - {feat}")
        print("\n  Use --overwrite to replace them")
        sys.exit(1)
    elif existing:
        print(f"  ⚠️  Found {len(existing)} existing features - will overwrite")
        df = df.drop(columns=existing)
    else:
        print("  ✅ No existing crisis features found")

    # Step 3: Validate required columns
    print("\n[3/4] Validating required columns...")

    required_cols = ['close', 'volume', 'oi', 'funding']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"  ❌ Missing required columns: {missing}")
        print("\n  Available columns:")
        for col in sorted(df.columns):
            print(f"    - {col}")
        sys.exit(1)

    print("  ✅ All required columns present")

    # Check for nulls in required columns
    for col in required_cols:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f"    {col}: {null_count} nulls ({null_pct:.1f}%)")

        if null_pct > 5:
            print(f"    ⚠️  WARNING: {col} has {null_pct:.1f}% nulls - may affect crisis detection")

    # Step 4: Engineer crisis features
    print("\n[4/6] Engineering crisis event features...")

    df_with_crisis = engineer_crisis_features(df)

    # Step 5: Convert events to states
    print(f"\n[5/6] Converting events to state features (tier={args.tier})...")

    df_with_crisis = convert_events_to_states(df_with_crisis, tier=args.tier)

    # Validate new features
    print("\n  Validating new features:")

    for feat in crisis_features:
        if feat in df_with_crisis.columns:
            non_null = df_with_crisis[feat].notna().sum()
            non_null_pct = non_null / len(df_with_crisis) * 100

            # For binary features, show trigger rate
            if feat in ['flash_crash_1h', 'flash_crash_4h', 'flash_crash_1d',
                        'volume_spike', 'oi_cascade', 'funding_extreme',
                        'funding_flip', 'crisis_confirmed']:
                trigger_count = df_with_crisis[feat].sum()
                trigger_pct = trigger_count / non_null * 100
                print(f"    {feat}: {non_null} non-null ({non_null_pct:.1f}%), {trigger_count} triggers ({trigger_pct:.2f}%)")
            else:
                # Continuous features (state features, volume_z_7d, etc)
                mean_val = df_with_crisis[feat].mean()
                std_val = df_with_crisis[feat].std()
                p90_val = df_with_crisis[feat].quantile(0.90)

                # For state features, compute activation rate (>0.2 threshold)
                if feat in state_features_tier1 + state_features_tier2 + state_features_tier3:
                    activation_rate = (df_with_crisis[feat] > 0.2).mean() * 100
                    print(f"    {feat}: mean={mean_val:.3f}, std={std_val:.3f}, p90={p90_val:.3f}, activation={activation_rate:.1f}%")
                else:
                    print(f"    {feat}: {non_null} non-null ({non_null_pct:.1f}%), mean={mean_val:.3f}, std={std_val:.3f}")

    # Step 6: Save updated feature store
    print("\n[6/6] Saving updated feature store...")

    # Backup original
    backup_path = feature_file.with_suffix('.parquet.bak_pre_crisis')
    if not backup_path.exists():
        print(f"  Creating backup: {backup_path.name}")
        df.to_parquet(backup_path, compression='snappy')
    else:
        print(f"  Backup already exists: {backup_path.name}")

    # Save updated file
    df_with_crisis.to_parquet(feature_file, compression='snappy')
    print(f"  ✅ Saved {len(df_with_crisis)} bars with {len(df_with_crisis.columns)} columns")

    # Step 7: Quick validation on known crisis events
    print("\n" + "=" * 80)
    print("QUICK VALIDATION ON KNOWN CRISIS EVENTS")
    print("=" * 80)

    # LUNA collapse (May 9-12, 2022)
    luna_start = pd.Timestamp('2022-05-09', tz='UTC')
    luna_end = pd.Timestamp('2022-05-12', tz='UTC')
    luna_window = df_with_crisis[(df_with_crisis.index >= luna_start) & (df_with_crisis.index <= luna_end)]

    if len(luna_window) > 0:
        print("\n📊 LUNA Collapse (May 9-12, 2022):")
        print(f"  Window: {len(luna_window)} bars")

        # Event features
        print("  Event Features:")
        for feat in ['flash_crash_1h', 'flash_crash_4h', 'volume_spike', 'crisis_composite_score']:
            if feat in luna_window.columns:
                if feat == 'crisis_composite_score':
                    max_score = luna_window[feat].max()
                    mean_score = luna_window[feat].mean()
                    print(f"    {feat}: max={max_score:.0f}, mean={mean_score:.1f}")
                else:
                    trigger_rate = luna_window[feat].mean() * 100
                    print(f"    {feat}: {trigger_rate:.1f}% triggering")

        # State features (if tier1+ enabled)
        if args.tier in ['tier1', 'all']:
            print("  State Features (Tier 1):")
            for feat in state_features_tier1:
                if feat in luna_window.columns:
                    mean_val = luna_window[feat].mean()
                    activation_rate = (luna_window[feat] > 0.2).mean() * 100
                    print(f"    {feat}: mean={mean_val:.3f}, activation={activation_rate:.1f}%")

    # FTX collapse (Nov 8-11, 2022)
    ftx_start = pd.Timestamp('2022-11-08', tz='UTC')
    ftx_end = pd.Timestamp('2022-11-11', tz='UTC')
    ftx_window = df_with_crisis[(df_with_crisis.index >= ftx_start) & (df_with_crisis.index <= ftx_end)]

    if len(ftx_window) > 0:
        print("\n📊 FTX Collapse (Nov 8-11, 2022):")
        print(f"  Window: {len(ftx_window)} bars")

        # Event features
        print("  Event Features:")
        for feat in ['flash_crash_1h', 'flash_crash_4h', 'oi_cascade', 'crisis_composite_score']:
            if feat in ftx_window.columns:
                if feat == 'crisis_composite_score':
                    max_score = ftx_window[feat].max()
                    mean_score = ftx_window[feat].mean()
                    print(f"    {feat}: max={max_score:.0f}, mean={mean_score:.1f}")
                else:
                    trigger_rate = ftx_window[feat].mean() * 100
                    print(f"    {feat}: {trigger_rate:.1f}% triggering")

        # State features (if tier1+ enabled)
        if args.tier in ['tier1', 'all']:
            print("  State Features (Tier 1):")
            for feat in state_features_tier1:
                if feat in ftx_window.columns:
                    mean_val = ftx_window[feat].mean()
                    activation_rate = (ftx_window[feat] > 0.2).mean() * 100
                    print(f"    {feat}: mean={mean_val:.3f}, activation={activation_rate:.1f}%")

    # June 2022 dump (June 13-18, 2022)
    june_start = pd.Timestamp('2022-06-13', tz='UTC')
    june_end = pd.Timestamp('2022-06-18', tz='UTC')
    june_window = df_with_crisis[(df_with_crisis.index >= june_start) & (df_with_crisis.index <= june_end)]

    if len(june_window) > 0:
        print("\n📊 June 2022 Dump (June 13-18, 2022):")
        print(f"  Window: {len(june_window)} bars")
        for feat in ['flash_crash_1h', 'flash_crash_4h', 'crisis_composite_score']:
            if feat in june_window.columns:
                if feat == 'crisis_composite_score':
                    max_score = june_window[feat].max()
                    mean_score = june_window[feat].mean()
                    print(f"  {feat}: max={max_score:.0f}, mean={mean_score:.1f}")
                else:
                    trigger_rate = june_window[feat].mean() * 100
                    print(f"  {feat}: {trigger_rate:.1f}% triggering")

    print("\n" + "=" * 80)
    print("CRISIS FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Added {len(crisis_features)} crisis features")
    print(f"✅ Updated feature store: {feature_file.name}")
    print(f"✅ Backup saved: {backup_path.name}")
    print("\n🚀 Next step: Run Agent 3's HMM retraining pipeline:")
    print("   python3 bin/validate_agent2_crisis_features.py")
    print("   ./bin/execute_hmm_retraining_pipeline.sh")
    print("=" * 80)


if __name__ == '__main__':
    main()
