#!/usr/bin/env python3
"""
Test Regime Detector v10 on Real Data

Validates GMM v3.1 regime classification on BTC 2022-2024:
1. Load feature stores (2022-2023 bear, 2024 bull)
2. Classify regimes using RegimeDetector
3. Validate regime assignments match reality
4. Export regime labels to file

Usage:
    python3 bin/test_regime_detector_v10.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.regime_detector import RegimeDetector

def load_and_classify(detector, feature_path, period_name):
    """Load feature store and classify regimes."""
    print(f"\n{'='*80}")
    print(f"{period_name}")
    print('='*80)

    # Load feature store
    df = pd.read_parquet(feature_path)
    print(f"✅ Loaded: {df.shape}")

    # Check available columns
    macro_cols = ['VIX', 'DXY', 'YIELD_2Y', 'YIELD_10Y', 'BTC.D', 'USDT.D',
                  'TOTAL', 'TOTAL2', 'funding', 'oi', 'rv_20d', 'rv_60d']
    missing = set(macro_cols) - set(df.columns)
    if missing:
        print(f"⚠️  Missing columns: {missing}")
    else:
        print(f"✅ All required macro columns present")

    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df.columns:
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            print(f"✅ Using DatetimeIndex as timestamp (name: {df.index.name})")
            df['timestamp'] = df.index
        else:
            print(f"⚠️  Warning: No timestamp column, converting index")
            df['timestamp'] = pd.to_datetime(df.index, utc=True)

    # Classify regimes
    print(f"\n🔮 Classifying regimes...")
    df_classified = detector.classify_batch(df)

    # Regime distribution
    regime_counts = df_classified['regime_label'].value_counts()
    total = len(df_classified)

    print(f"\n📊 Regime Distribution:")
    for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
        count = regime_counts.get(regime, 0)
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {regime:12s}: {count:5d} bars ({pct:5.1f}%) {bar}")

    # Confidence stats
    valid_conf = df_classified[df_classified['regime_confidence'] > 0]['regime_confidence']
    if len(valid_conf) > 0:
        print(f"\n📈 Confidence Stats (valid samples: {len(valid_conf):,}):")
        print(f"  Mean:   {valid_conf.mean():.3f}")
        print(f"  Median: {valid_conf.median():.3f}")
        print(f"  Min:    {valid_conf.min():.3f}")
        print(f"  Max:    {valid_conf.max():.3f}")

        # Low confidence bars
        low_conf = (df_classified['regime_confidence'] < 0.6).sum()
        low_conf_pct = 100 * low_conf / total
        print(f"  <0.6:   {low_conf} bars ({low_conf_pct:.1f}%)")
    else:
        print(f"\n⚠️  No valid confidence scores")

    # Sample classifications
    print(f"\n🔍 Sample Classifications (random 10):")
    valid_samples = df_classified[df_classified['regime_confidence'] > 0]
    if len(valid_samples) > 0:
        samples = valid_samples.sample(min(10, len(valid_samples)))
        print(f"{'Date':20s} {'Regime':12s} {'Confidence':>10s} | VIX    DXY")
        print('-'*80)
        for idx, row in samples.iterrows():
            date_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            regime = row['regime_label'].upper()
            conf = row['regime_confidence']
            vix = row.get('VIX', 0)
            dxy = row.get('DXY', 0)
            print(f"{date_str:20s} {regime:12s} {conf:>10.3f} | {vix:6.2f} {dxy:6.2f}")

    return df_classified

def validate_key_periods(df, period_expectations):
    """Validate regime assignments for known key periods."""
    print(f"\n{'='*80}")
    print("VALIDATING KEY PERIODS")
    print('='*80)

    all_pass = True

    for period_name, start_date, end_date, expected_regimes, min_pct in period_expectations:
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        df_period = df[mask]

        if len(df_period) == 0:
            print(f"\n⚠️  {period_name}: No data")
            continue

        regime_counts = df_period['regime_label'].value_counts()
        total = len(df_period)

        # Calculate percentage of expected regimes
        expected_count = sum(regime_counts.get(r, 0) for r in expected_regimes)
        expected_pct = 100 * expected_count / total

        # Validate
        passed = expected_pct >= min_pct
        status = "✅" if passed else "❌"
        all_pass = all_pass and passed

        print(f"\n{status} {period_name} ({start_date} to {end_date}):")
        print(f"   Expected: {expected_regimes} ≥ {min_pct}%")
        print(f"   Got: {expected_pct:.1f}% ({expected_count}/{total} bars)")

        # Show distribution
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            count = regime_counts.get(regime, 0)
            pct = 100 * count / total
            print(f"     {regime:12s}: {count:5d} bars ({pct:5.1f}%)")

    return all_pass

def main():
    print("\n" + "="*80)
    print("REGIME DETECTOR V10 - VALIDATION TEST")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize detector
    detector = RegimeDetector()
    print(f"\n✅ Loaded GMM v3.1 model")
    print(f"   Features: {len(detector.features)}")
    print(f"   Label map: {detector.label_map}")

    # Test periods
    test_periods = [
        ('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet', '2022-2023 Bear Market'),
        ('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet', '2024 Bull Market'),
    ]

    all_results = []

    for feature_path, period_name in test_periods:
        if not Path(feature_path).exists():
            print(f"\n⚠️  Skipping {period_name}: File not found")
            print(f"   {feature_path}")
            continue

        df_classified = load_and_classify(detector, feature_path, period_name)
        all_results.append((period_name, df_classified))

    # Combine all results
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("COMBINED RESULTS (2022-2024)")
        print('='*80)

        df_all = pd.concat([df for _, df in all_results], ignore_index=False)

        # Overall distribution
        regime_counts = df_all['regime_label'].value_counts()
        total = len(df_all)

        print(f"\nTotal samples: {total:,} bars")
        print(f"\nOverall Regime Distribution:")
        for regime in ['crisis', 'risk_off', 'neutral', 'risk_on']:
            count = regime_counts.get(regime, 0)
            pct = 100 * count / total
            bar = '█' * int(pct / 2)
            print(f"  {regime:12s}: {count:5d} bars ({pct:5.1f}%) {bar}")

        # Validate key periods
        key_periods = [
            ('2022 Luna Crash (May-Jun)', '2022-05-01', '2022-06-30', ['crisis', 'risk_off'], 70),
            ('2022 FTX Collapse (Nov)', '2022-11-01', '2022-11-30', ['crisis', 'risk_off'], 70),
            ('2024 Q1 Bull Run', '2024-01-01', '2024-03-31', ['risk_on'], 70),
            ('2024 Q4 Recent', '2024-10-01', '2024-10-31', ['neutral', 'risk_on'], 60),
        ]

        validation_passed = validate_key_periods(df_all, key_periods)

        # Export regime labels
        output_path = Path('data/regime_labels_2022_2024.parquet')
        df_export = df_all[['timestamp', 'regime_label', 'regime_confidence']].copy()
        df_export.to_parquet(output_path)
        print(f"\n💾 Exported regime labels: {output_path}")
        print(f"   Shape: {df_export.shape}")

        # Final status
        print(f"\n{'='*80}")
        if validation_passed:
            print("✅ VALIDATION PASSED")
        else:
            print("⚠️  VALIDATION INCOMPLETE - Some periods don't match expectations")
        print('='*80)

        return 0 if validation_passed else 1

    else:
        print(f"\n❌ No data processed")
        return 1

if __name__ == '__main__':
    exit(main())
