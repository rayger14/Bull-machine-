#!/usr/bin/env python3
"""
Audit S2 Feature Availability for v2 Confluence Analysis
=========================================================
Research experiment to check which features exist in S2 baseline data.
This is analysis-only - no pipeline changes.
"""

import pandas as pd
import sys
from pathlib import Path

def audit_features():
    """Audit feature availability in S2 enriched trades data"""

    # Load enriched S2 data
    data_path = Path('results/optimization/s2_enriched_2022_trades.csv')
    if not data_path.exists():
        print(f"ERROR: Missing {data_path}")
        print("Expected S2 enriched trades from previous optimization.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} S2 baseline trades from 2022")
    print(f"Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")

    # Define S2 v2 confluence features with fallback alternatives
    confluence_features = {
        'C1_RSI_Divergence': {
            'primary': 'rsi_14',
            'fallback': None,
            'threshold': '> 65 (overbought)',
            'purpose': 'Detect overbought conditions as proxy for divergence'
        },
        'C2_Volume_Fade': {
            'primary': 'volume_zscore',
            'fallback': 'atr_percentile',
            'threshold': '< 0.0 (below average)',
            'purpose': 'Identify weakening momentum'
        },
        'C3_Wick_Rejection': {
            'primary': 'wick_ratio',
            'fallback': None,
            'threshold': '> 2.0 (upper wick > 2x body)',
            'purpose': 'Detect rejection at resistance'
        },
        'C4_MTF_Downtrend': {
            'primary': 'tf4h_trend_aligned',
            'fallback': 'tf4h_fusion',
            'threshold': '< 0 or low fusion',
            'purpose': 'Higher timeframe bearish context'
        },
        'C5_DXY_Strength': {
            'primary': 'dxy_z',
            'fallback': 'macro_regime_risk_off',
            'threshold': '> 0.3 or risk_off=1',
            'purpose': 'Risk-off environment (dollar rising)'
        },
        'C6_Wyckoff_Distribution': {
            'primary': 'wyckoff_phase',
            'fallback': 'wyckoff_phase_score',
            'threshold': '== distribution or score > 0',
            'purpose': 'Distribution phase detection'
        },
        'C7_Liquidity_Trap': {
            'primary': 'liquidity_score',
            'fallback': 'entry_liquidity_score',
            'threshold': '< 0.25 (low liquidity)',
            'purpose': 'Identify trap setups above key levels'
        }
    }

    print("\n" + "="*80)
    print("S2 V2 CONFLUENCE FEATURE AVAILABILITY AUDIT")
    print("="*80)

    available_features = []
    missing_features = []

    for condition_name, config in confluence_features.items():
        print(f"\n{condition_name}:")
        print(f"  Purpose: {config['purpose']}")
        print(f"  Threshold: {config['threshold']}")

        # Check primary feature
        primary = config['primary']
        if primary in df.columns:
            coverage = df[primary].notna().mean() * 100
            mean_val = df[primary].mean()
            std_val = df[primary].std()
            print(f"  ✓ PRIMARY: {primary} ({coverage:.1f}% coverage)")
            print(f"    Stats: mean={mean_val:.3f}, std={std_val:.3f}")
            available_features.append((condition_name, primary, coverage))
        else:
            print(f"  ✗ PRIMARY: {primary} NOT FOUND")

            # Check fallback
            fallback = config['fallback']
            if fallback and fallback in df.columns:
                coverage = df[fallback].notna().mean() * 100
                mean_val = df[fallback].mean()
                std_val = df[fallback].std()
                print(f"  ✓ FALLBACK: {fallback} ({coverage:.1f}% coverage)")
                print(f"    Stats: mean={mean_val:.3f}, std={std_val:.3f}")
                available_features.append((condition_name, fallback, coverage))
            elif fallback:
                print(f"  ✗ FALLBACK: {fallback} NOT FOUND")
                missing_features.append((condition_name, primary, fallback))
            else:
                missing_features.append((condition_name, primary, None))

    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    print(f"\nAvailable Features: {len(available_features)}/7")
    for cond, feat, cov in available_features:
        print(f"  ✓ {cond:30s} → {feat:30s} ({cov:5.1f}%)")

    if missing_features:
        print(f"\nMissing Features: {len(missing_features)}/7")
        for cond, prim, fall in missing_features:
            if fall:
                print(f"  ✗ {cond:30s} → {prim} (fallback: {fall})")
            else:
                print(f"  ✗ {cond:30s} → {prim} (no fallback)")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if len(available_features) >= 5:
        print(f"✓ PROCEED: {len(available_features)}/7 features available")
        print("  Sufficient for S2 v2 confluence analysis")
        print("  Missing features will be treated conservatively (False)")
    else:
        print(f"✗ INSUFFICIENT: Only {len(available_features)}/7 features available")
        print("  Need at least 5/7 for meaningful confluence analysis")
        print("  Consider enriching data with missing features first")
        sys.exit(1)

    # Show all available columns for reference
    print("\n" + "="*80)
    print("ALL AVAILABLE COLUMNS IN S2 DATA")
    print("="*80)
    print(f"Total: {len(df.columns)} columns\n")
    for i, col in enumerate(df.columns, 1):
        coverage = df[col].notna().mean() * 100
        print(f"{i:3d}. {col:40s} ({coverage:5.1f}%)")

if __name__ == '__main__':
    audit_features()
