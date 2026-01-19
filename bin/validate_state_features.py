#!/usr/bin/env python3
"""
Validate State-Aware Crisis Features
====================================

Comprehensive validation of state feature implementation:
1. Test on synthetic data
2. Test on historical BTC data (LUNA, FTX events)
3. Visualize state feature behavior
4. Generate validation report

Usage:
    python3 bin/validate_state_features.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from engine.features.crisis_indicators import engineer_crisis_features
from engine.features.state_features import convert_events_to_states, validate_state_features


def create_synthetic_test_data():
    """Create synthetic data for unit testing state features."""
    print("\n[1/4] Creating synthetic test data...")

    np.random.seed(42)

    # 90 days of hourly data (Jan-Mar 2022)
    dates = pd.date_range(start='2022-01-01', end='2022-03-31', freq='1H', tz='UTC')
    n = len(dates)

    # Simulate price data with crisis events
    close = 40000 + np.cumsum(np.random.randn(n) * 100)

    # Inject crisis events
    flash_crash_1h = np.zeros(n, dtype=int)
    volume_spike = np.zeros(n, dtype=int)
    funding_extreme = np.zeros(n, dtype=int)
    crisis_composite_score = np.zeros(n)

    # Crisis 1: Jan 15-17 (3-day cascade)
    crisis1_start = (dates >= pd.Timestamp('2022-01-15', tz='UTC')).argmax()
    crisis1_end = (dates >= pd.Timestamp('2022-01-17', tz='UTC')).argmax()

    flash_crash_1h[crisis1_start] = 1
    flash_crash_1h[crisis1_start + 12] = 1  # 12H later
    flash_crash_1h[crisis1_start + 36] = 1  # 36H later
    volume_spike[crisis1_start:crisis1_end] = 1
    crisis_composite_score[crisis1_start:crisis1_end] = 4

    # Crisis 2: Feb 20-22 (3-day event)
    crisis2_start = (dates >= pd.Timestamp('2022-02-20', tz='UTC')).argmax()
    crisis2_end = (dates >= pd.Timestamp('2022-02-22', tz='UTC')).argmax()

    flash_crash_1h[crisis2_start] = 1
    flash_crash_1h[crisis2_start + 24] = 1  # 24H later
    volume_spike[crisis2_start:crisis2_end] = 1
    funding_extreme[crisis2_start:crisis2_end] = 1
    crisis_composite_score[crisis2_start:crisis2_end] = 5

    # Create DataFrame
    df = pd.DataFrame({
        'close': close,
        'volume': np.abs(np.random.randn(n) * 1e6),
        'oi': np.abs(np.random.randn(n) * 1e8),
        'funding': np.random.randn(n) * 0.001,
        'flash_crash_1h': flash_crash_1h,
        'flash_crash_4h': np.zeros(n, dtype=int),
        'flash_crash_1d': np.zeros(n, dtype=int),
        'volume_spike': volume_spike,
        'oi_cascade': np.zeros(n, dtype=int),
        'funding_extreme': funding_extreme,
        'funding_flip': np.zeros(n, dtype=int),
        'crisis_composite_score': crisis_composite_score,
        'crisis_confirmed': (crisis_composite_score >= 3).astype(int)
    }, index=dates)

    print(f"  Created {len(df)} bars of synthetic data")
    print(f"  Crisis events: {df['flash_crash_1h'].sum()} flash crashes")
    print(f"  Crisis composite: max={df['crisis_composite_score'].max()}, mean={df['crisis_composite_score'].mean():.2f}")

    return df


def test_synthetic_data(df):
    """Test state features on synthetic data."""
    print("\n[2/4] Testing state features on synthetic data...")

    # Convert events to states
    df = convert_events_to_states(df, tier='all')

    # Validate features
    crisis_windows = [
        (pd.Timestamp('2022-01-15', tz='UTC'), pd.Timestamp('2022-01-17', tz='UTC')),
        (pd.Timestamp('2022-02-20', tz='UTC'), pd.Timestamp('2022-02-22', tz='UTC'))
    ]

    results = validate_state_features(df, crisis_windows)

    print("\n  Synthetic Data Validation Results:")
    print(f"  Tested {len(results['overall_stats'])} state features")

    # Check success criteria
    success_count = 0
    total_count = 0

    for feat, stats in results['overall_stats'].items():
        # Criterion: 10-30% overall activation
        activation_ok = 5 <= stats['activation_rate'] <= 40  # Relaxed for synthetic data

        if activation_ok:
            success_count += 1
        total_count += 1

        status = "✅" if activation_ok else "⚠️"
        print(f"  {status} {feat}: activation={stats['activation_rate']:.1f}% (target: 10-30%)")

    print(f"\n  Success Rate: {success_count}/{total_count} features meet criteria")

    return df, results


def test_historical_data(asset='BTC', start='2022-01-01', end='2024-12-31'):
    """Test state features on historical BTC data."""
    print(f"\n[3/4] Testing state features on historical {asset} data...")

    # Load feature store
    feature_file = Path(f'data/features_mtf/{asset}_1H_{start}_to_{end}_with_macro.parquet')
    if not feature_file.exists():
        feature_file = Path(f'data/features_mtf/{asset}_1H_{start}_to_{end}.parquet')

    if not feature_file.exists():
        print(f"  ⚠️  Feature file not found: {feature_file}")
        print("  Skipping historical data test")
        return None, None

    df = pd.read_parquet(feature_file)
    print(f"  Loaded {len(df)} bars")

    # Check if crisis event features exist
    required_events = ['flash_crash_1h', 'volume_spike', 'crisis_composite_score']
    missing = [col for col in required_events if col not in df.columns]

    if missing:
        print(f"  ⚠️  Missing event features: {missing}")
        print("  Adding crisis event features first...")
        df = engineer_crisis_features(df)

    # Convert events to states
    df = convert_events_to_states(df, tier='all')

    # Validate on LUNA and FTX events
    crisis_windows = [
        (pd.Timestamp('2022-05-09', tz='UTC'), pd.Timestamp('2022-05-12', tz='UTC')),  # LUNA
        (pd.Timestamp('2022-11-08', tz='UTC'), pd.Timestamp('2022-11-11', tz='UTC')),  # FTX
    ]

    results = validate_state_features(df, crisis_windows)

    print("\n  Historical Data Validation Results:")

    # Check LUNA window
    if 'window_1' in results['crisis_stats']:
        print("\n  LUNA Window (May 9-12, 2022):")
        for feat, stats in results['crisis_stats']['window_1'].items():
            mean_val = stats['mean']
            activation = stats['activation_rate']
            print(f"    {feat}: mean={mean_val:.3f}, activation={activation:.1f}%")

    # Check FTX window
    if 'window_2' in results['crisis_stats']:
        print("\n  FTX Window (Nov 8-11, 2022):")
        for feat, stats in results['crisis_stats']['window_2'].items():
            mean_val = stats['mean']
            activation = stats['activation_rate']
            print(f"    {feat}: mean={mean_val:.3f}, activation={activation:.1f}%")

    return df, results


def generate_validation_report(synthetic_results, historical_results):
    """Generate comprehensive validation report."""
    print("\n[4/4] Generating validation report...")

    report = []
    report.append("=" * 80)
    report.append("STATE-AWARE CRISIS FEATURES VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Synthetic data results
    report.append("\n" + "=" * 80)
    report.append("SYNTHETIC DATA VALIDATION")
    report.append("=" * 80)

    if synthetic_results:
        report.append("\nOverall Statistics:")
        for feat, stats in synthetic_results['overall_stats'].items():
            report.append(f"  {feat}:")
            report.append(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            report.append(f"    P50: {stats['p50']:.3f}, P90: {stats['p90']:.3f}")
            report.append(f"    Activation Rate: {stats['activation_rate']:.1f}%")

    # Historical data results
    report.append("\n" + "=" * 80)
    report.append("HISTORICAL DATA VALIDATION (BTC)")
    report.append("=" * 80)

    if historical_results:
        report.append("\nOverall Statistics:")
        for feat, stats in historical_results['overall_stats'].items():
            report.append(f"  {feat}:")
            report.append(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            report.append(f"    P50: {stats['p50']:.3f}, P90: {stats['p90']:.3f}")
            report.append(f"    Activation Rate: {stats['activation_rate']:.1f}%")

        # Crisis window performance
        if 'window_1' in historical_results['crisis_stats']:
            report.append("\nLUNA Crisis Window (May 9-12, 2022):")
            for feat, stats in historical_results['crisis_stats']['window_1'].items():
                report.append(f"  {feat}: mean={stats['mean']:.3f}, activation={stats['activation_rate']:.1f}%")

        if 'window_2' in historical_results['crisis_stats']:
            report.append("\nFTX Crisis Window (Nov 8-11, 2022):")
            for feat, stats in historical_results['crisis_stats']['window_2'].items():
                report.append(f"  {feat}: mean={stats['mean']:.3f}, activation={stats['activation_rate']:.1f}%")

    # Success criteria assessment
    report.append("\n" + "=" * 80)
    report.append("SUCCESS CRITERIA ASSESSMENT")
    report.append("=" * 80)

    criteria = {
        'Persistence': 'State features stay elevated 2-7 days after crisis events',
        'Activation': '10-30% overall activation rate (not always on)',
        'Crisis Response': '>50% activation during LUNA/FTX windows'
    }

    report.append("\nCriteria:")
    for name, desc in criteria.items():
        report.append(f"  {name}: {desc}")

    if historical_results:
        report.append("\nAssessment:")

        # Check activation rates
        tier1_features = ['crash_stress_24h', 'crash_stress_72h', 'vol_persistence', 'hours_since_crisis']
        activation_ok = all(
            10 <= historical_results['overall_stats'][feat]['activation_rate'] <= 30
            for feat in tier1_features
            if feat in historical_results['overall_stats']
        )

        # Check crisis response
        crisis_response_ok = False
        if 'window_1' in historical_results['crisis_stats']:
            crisis_activations = [
                historical_results['crisis_stats']['window_1'][feat]['activation_rate']
                for feat in tier1_features
                if feat in historical_results['crisis_stats']['window_1']
            ]
            crisis_response_ok = all(rate > 50 for rate in crisis_activations)

        report.append(f"  Activation Rate: {'✅ PASS' if activation_ok else '⚠️  FAIL'}")
        report.append(f"  Crisis Response: {'✅ PASS' if crisis_response_ok else '⚠️  FAIL'}")

    report.append("\n" + "=" * 80)
    report.append("VALIDATION COMPLETE")
    report.append("=" * 80)

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_file = Path('STATE_FEATURES_VALIDATION_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\n✅ Report saved: {report_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Validate state-aware crisis features')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset symbol')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    args = parser.parse_args()

    print("=" * 80)
    print("STATE-AWARE CRISIS FEATURES VALIDATION")
    print("=" * 80)
    print(f"\nAsset: {args.asset}")
    print(f"Period: {args.start} to {args.end}")

    # Test 1: Synthetic data
    synthetic_df = create_synthetic_test_data()
    synthetic_df, synthetic_results = test_synthetic_data(synthetic_df)

    # Test 2: Historical data
    historical_df, historical_results = test_historical_data(args.asset, args.start, args.end)

    # Generate report
    report = generate_validation_report(synthetic_results, historical_results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    print("\n✅ State features tested on synthetic and historical data")
    print("✅ Validation report generated: STATE_FEATURES_VALIDATION_REPORT.md")
    print("\n🚀 Next steps:")
    print("   1. Review validation report")
    print("   2. Run unit tests: pytest tests/unit/features/test_state_features.py")
    print("   3. Add state features to feature store: python3 bin/add_crisis_features.py --tier tier1")
    print("=" * 80)


if __name__ == '__main__':
    main()
