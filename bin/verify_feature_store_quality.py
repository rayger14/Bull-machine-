#!/usr/bin/env python3
"""
Feature Store Final Verification Script
Checks data quality, coverage, and signal statistics for all features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_feature_quality(df, feature_name):
    """Analyze quality metrics for a single feature"""
    series = df[feature_name]

    # Basic stats
    total = len(series)
    non_null = series.notna().sum()
    non_null_pct = (non_null / total * 100)

    # Check if numeric
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # For boolean/binary features
    if series.dtype == bool or (is_numeric and set(series.dropna().unique()).issubset({0, 1, True, False})):
        signals = (series == True) | (series == 1)
        signal_count = signals.sum()
        signal_pct = (signal_count / total * 100)

        # Check if constant
        unique_vals = series.dropna().unique()
        is_constant = len(unique_vals) <= 1

        try:
            min_val = series.min() if non_null > 0 and is_numeric else np.nan
            max_val = series.max() if non_null > 0 and is_numeric else np.nan
        except:
            min_val = np.nan
            max_val = np.nan

        return {
            'feature': feature_name,
            'non_null_pct': non_null_pct,
            'signal_count': signal_count,
            'signal_pct': signal_pct,
            'min': min_val,
            'max': max_val,
            'is_constant': is_constant,
            'dtype': str(series.dtype)
        }
    else:
        # Numeric or other features
        try:
            min_val = series.min() if non_null > 0 and is_numeric else np.nan
            max_val = series.max() if non_null > 0 and is_numeric else np.nan
        except:
            min_val = np.nan
            max_val = np.nan

        return {
            'feature': feature_name,
            'non_null_pct': non_null_pct,
            'signal_count': non_null,
            'signal_pct': non_null_pct,
            'min': min_val,
            'max': max_val,
            'is_constant': series.nunique() <= 1,
            'dtype': str(series.dtype)
        }

def categorize_quality(metrics):
    """Determine quality status"""
    if metrics['non_null_pct'] < 50:
        return 'BROKEN'
    elif metrics['non_null_pct'] < 100:
        return 'POOR'
    elif metrics['is_constant']:
        return 'BROKEN'
    elif metrics['signal_pct'] == 0:
        return 'NEVER_FIRES'
    elif metrics['signal_pct'] == 100:
        return 'ALWAYS_FIRES'
    elif metrics['signal_pct'] < 0.1:
        return 'RARE'
    elif metrics['signal_pct'] > 50:
        return 'FREQUENT'
    else:
        return 'GOOD'

def main():
    # Load feature store
    feature_store_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    print("=" * 80)
    print("FEATURE STORE FINAL VERIFICATION")
    print("=" * 80)
    print(f"\nLoading: {feature_store_path}")

    df = pd.read_parquet(feature_store_path)

    print(f"\n✓ Loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Check date coverage
    print(f"\n{'='*80}")
    print("DATE COVERAGE")
    print("=" * 80)

    if 'timestamp' in df.columns:
        date_col = 'timestamp'
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        date_col = df.index
    else:
        date_col = df.index

    if isinstance(date_col, str):
        dates = pd.to_datetime(df[date_col])
    else:
        dates = pd.to_datetime(date_col)

    print(f"Start date: {dates.min()}")
    print(f"End date: {dates.max()}")
    print(f"Total timespan: {(dates.max() - dates.min()).days} days")

    # Check for gaps (expecting 1H frequency)
    date_diff = dates.diff()
    expected_freq = pd.Timedelta('1h')
    gaps = date_diff[date_diff > expected_freq * 1.5]  # Allow some tolerance

    if len(gaps) > 0:
        print(f"\n⚠ Warning: {len(gaps)} gaps detected in date coverage")
        print(f"  Largest gap: {gaps.max()}")
    else:
        print(f"\n✓ No significant gaps in date coverage")

    # Define feature groups
    v2_features = [
        'oi_change_spike_3h',
        'oi_change_spike_6h',
        'oi_change_spike_12h',
        'oi_change_spike_24h'
    ]

    wyckoff_patterns = [
        'wyckoff_spring_a', 'wyckoff_spring_b', 'wyckoff_spring_c',
        'wyckoff_upthrust_a', 'wyckoff_upthrust_b', 'wyckoff_upthrust_c',
        'wyckoff_sos', 'wyckoff_sow', 'wyckoff_ar', 'wyckoff_st',
        'wyckoff_phase_a', 'wyckoff_phase_b', 'wyckoff_phase_c',
        'wyckoff_phase_d', 'wyckoff_phase_e'
    ]

    smc_patterns = [
        'smc_bos', 'smc_choch', 'smc_fvg', 'smc_order_block',
        'smc_liquidity_sweep', 'smc_displacement', 'smc_consolidation'
    ]

    # Analyze all features
    print(f"\n{'='*80}")
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 80)

    all_metrics = []

    for col in df.columns:
        if col in ['timestamp', 'date', 'datetime']:
            continue

        metrics = analyze_feature_quality(df, col)
        metrics['quality'] = categorize_quality(metrics)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # V2 Features Check
    print(f"\n{'='*80}")
    print("V2 FEATURES (OI CHANGE SPIKES)")
    print("=" * 80)

    v2_found = [f for f in v2_features if f in df.columns]
    v2_missing = [f for f in v2_features if f not in df.columns]

    print(f"\nExpected: {len(v2_features)}")
    print(f"Found: {len(v2_found)}")

    if v2_missing:
        print(f"\n❌ Missing V2 features:")
        for f in v2_missing:
            print(f"  - {f}")
    else:
        print(f"\n✓ All V2 features present")

    if v2_found:
        print(f"\nV2 Feature Quality:")
        for f in v2_found:
            m = metrics_df[metrics_df['feature'] == f].iloc[0]
            print(f"  {f:30s} | Non-null: {m['non_null_pct']:6.2f}% | Signals: {m['signal_count']:6.0f} ({m['signal_pct']:5.2f}%) | {m['quality']}")

    # Wyckoff Features Check
    print(f"\n{'='*80}")
    print("WYCKOFF FEATURES")
    print("=" * 80)

    wyckoff_found = [f for f in wyckoff_patterns if f in df.columns]
    wyckoff_missing = [f for f in wyckoff_patterns if f not in df.columns]

    print(f"\nExpected: {len(wyckoff_patterns)}")
    print(f"Found: {len(wyckoff_found)}")

    if wyckoff_missing:
        print(f"\n⚠ Missing Wyckoff features:")
        for f in wyckoff_missing:
            print(f"  - {f}")

    if wyckoff_found:
        print(f"\nWyckoff Feature Quality:")
        wyckoff_working = 0
        for f in wyckoff_found:
            m = metrics_df[metrics_df['feature'] == f].iloc[0]
            status = "✓" if m['quality'] in ['GOOD', 'RARE', 'FREQUENT'] else "❌"
            if m['quality'] in ['GOOD', 'RARE', 'FREQUENT']:
                wyckoff_working += 1
            print(f"  {status} {f:30s} | Non-null: {m['non_null_pct']:6.2f}% | Signals: {m['signal_count']:6.0f} ({m['signal_pct']:5.2f}%) | {m['quality']}")

        print(f"\nWorking: {wyckoff_working}/{len(wyckoff_found)}")

    # SMC Features Check
    print(f"\n{'='*80}")
    print("SMC FEATURES")
    print("=" * 80)

    smc_found = [f for f in smc_patterns if f in df.columns]
    smc_missing = [f for f in smc_patterns if f not in df.columns]

    print(f"\nExpected: {len(smc_patterns)}")
    print(f"Found: {len(smc_found)}")

    if smc_missing:
        print(f"\n⚠ Missing SMC features:")
        for f in smc_missing:
            print(f"  - {f}")

    if smc_found:
        print(f"\nSMC Feature Quality:")
        smc_working = 0
        for f in smc_found:
            m = metrics_df[metrics_df['feature'] == f].iloc[0]
            status = "✓" if m['quality'] in ['GOOD', 'RARE', 'FREQUENT'] else "❌"
            if m['quality'] in ['GOOD', 'RARE', 'FREQUENT']:
                smc_working += 1
            print(f"  {status} {f:30s} | Non-null: {m['non_null_pct']:6.2f}% | Signals: {m['signal_count']:6.0f} ({m['signal_pct']:5.2f}%) | {m['quality']}")

        print(f"\nWorking: {smc_working}/{len(smc_found)}")

    # Problematic Features
    print(f"\n{'='*80}")
    print("PROBLEMATIC FEATURES")
    print("=" * 80)

    broken = metrics_df[metrics_df['quality'] == 'BROKEN']
    never_fires = metrics_df[metrics_df['quality'] == 'NEVER_FIRES']
    always_fires = metrics_df[metrics_df['quality'] == 'ALWAYS_FIRES']

    if len(broken) > 0:
        print(f"\n❌ BROKEN features ({len(broken)}):")
        for _, m in broken.iterrows():
            print(f"  - {m['feature']:40s} | Reason: Non-null={m['non_null_pct']:.1f}%, Constant={m['is_constant']}")

    if len(never_fires) > 0:
        print(f"\n⚠ NEVER_FIRES features ({len(never_fires)}):")
        for _, m in never_fires.iterrows():
            print(f"  - {m['feature']}")

    if len(always_fires) > 0:
        print(f"\n⚠ ALWAYS_FIRES features ({len(always_fires)}):")
        for _, m in always_fires.iterrows():
            print(f"  - {m['feature']}")

    if len(broken) == 0 and len(never_fires) == 0 and len(always_fires) == 0:
        print(f"\n✓ No problematic features detected")

    # Overall Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_features = len(metrics_df)
    good_features = len(metrics_df[metrics_df['quality'].isin(['GOOD', 'RARE', 'FREQUENT'])])
    quality_pct = (good_features / total_features * 100) if total_features > 0 else 0

    if quality_pct >= 90:
        overall_quality = "EXCELLENT"
    elif quality_pct >= 75:
        overall_quality = "GOOD"
    elif quality_pct >= 50:
        overall_quality = "FAIR"
    else:
        overall_quality = "POOR"

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total analyzable features: {total_features}")
    print(f"Good quality features: {good_features}/{total_features} ({quality_pct:.1f}%)")
    print(f"\nOverall quality: {overall_quality}")

    # Save detailed metrics
    output_path = Path('feature_quality_matrix.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed metrics to: {output_path}")

    # Generate markdown report
    report_path = Path('FEATURE_STORE_FINAL_VERIFICATION.md')

    with open(report_path, 'w') as f:
        f.write("# Feature Store Final Verification Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Feature Store:** `{feature_store_path}`\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total columns:** {len(df.columns)}\n")
        f.write(f"- **Total rows:** {len(df)}\n")
        f.write(f"- **Date range:** {dates.min()} to {dates.max()}\n")
        f.write(f"- **Overall quality:** **{overall_quality}** ({quality_pct:.1f}% good features)\n\n")

        f.write("## Critical Feature Status\n\n")

        # V2 Features
        f.write("### V2 Features (OI Change Spikes)\n\n")
        if v2_missing:
            f.write(f"❌ **Status:** INCOMPLETE ({len(v2_found)}/{len(v2_features)} found)\n\n")
            f.write("**Missing:**\n")
            for feat in v2_missing:
                f.write(f"- {feat}\n")
        else:
            f.write(f"✅ **Status:** COMPLETE ({len(v2_found)}/{len(v2_features)} found)\n\n")

        if v2_found:
            f.write("**Quality:**\n\n")
            f.write("| Feature | Non-Null % | Signal Count | Signal % | Quality |\n")
            f.write("|---------|------------|--------------|----------|----------|\n")
            for feat in v2_found:
                m = metrics_df[metrics_df['feature'] == feat].iloc[0]
                f.write(f"| {feat} | {m['non_null_pct']:.2f}% | {m['signal_count']:.0f} | {m['signal_pct']:.2f}% | {m['quality']} |\n")
            f.write("\n")

        # Wyckoff Features
        f.write("### Wyckoff Features\n\n")
        if wyckoff_missing:
            f.write(f"⚠ **Status:** INCOMPLETE ({len(wyckoff_found)}/{len(wyckoff_patterns)} found)\n\n")
        else:
            f.write(f"✅ **Status:** COMPLETE ({len(wyckoff_found)}/{len(wyckoff_patterns)} found)\n\n")

        if wyckoff_found:
            wyckoff_metrics = metrics_df[metrics_df['feature'].isin(wyckoff_found)]
            wyckoff_working = len(wyckoff_metrics[wyckoff_metrics['quality'].isin(['GOOD', 'RARE', 'FREQUENT'])])
            f.write(f"**Working:** {wyckoff_working}/{len(wyckoff_found)}\n\n")

            f.write("**Quality:**\n\n")
            f.write("| Feature | Non-Null % | Signal Count | Signal % | Quality |\n")
            f.write("|---------|------------|--------------|----------|----------|\n")
            for feat in wyckoff_found:
                m = metrics_df[metrics_df['feature'] == feat].iloc[0]
                status = "✓" if m['quality'] in ['GOOD', 'RARE', 'FREQUENT'] else "❌"
                f.write(f"| {status} {feat} | {m['non_null_pct']:.2f}% | {m['signal_count']:.0f} | {m['signal_pct']:.2f}% | {m['quality']} |\n")
            f.write("\n")

        # SMC Features
        f.write("### SMC Features\n\n")
        if smc_missing:
            f.write(f"⚠ **Status:** INCOMPLETE ({len(smc_found)}/{len(smc_patterns)} found)\n\n")
        else:
            f.write(f"✅ **Status:** COMPLETE ({len(smc_found)}/{len(smc_patterns)} found)\n\n")

        if smc_found:
            smc_metrics = metrics_df[metrics_df['feature'].isin(smc_found)]
            smc_working = len(smc_metrics[smc_metrics['quality'].isin(['GOOD', 'RARE', 'FREQUENT'])])
            f.write(f"**Working:** {smc_working}/{len(smc_found)}\n\n")

            f.write("**Quality:**\n\n")
            f.write("| Feature | Non-Null % | Signal Count | Signal % | Quality |\n")
            f.write("|---------|------------|--------------|----------|----------|\n")
            for feat in smc_found:
                m = metrics_df[metrics_df['feature'] == feat].iloc[0]
                status = "✓" if m['quality'] in ['GOOD', 'RARE', 'FREQUENT'] else "❌"
                f.write(f"| {status} {feat} | {m['non_null_pct']:.2f}% | {m['signal_count']:.0f} | {m['signal_pct']:.2f}% | {m['quality']} |\n")
            f.write("\n")

        # Problematic Features
        f.write("## Problematic Features\n\n")

        if len(broken) > 0:
            f.write(f"### ❌ BROKEN Features ({len(broken)})\n\n")
            f.write("| Feature | Non-Null % | Constant | Issue |\n")
            f.write("|---------|------------|----------|-------|\n")
            for _, m in broken.iterrows():
                issue = "Constant value" if m['is_constant'] else f"Low coverage ({m['non_null_pct']:.1f}%)"
                f.write(f"| {m['feature']} | {m['non_null_pct']:.2f}% | {m['is_constant']} | {issue} |\n")
            f.write("\n")

        if len(never_fires) > 0:
            f.write(f"### ⚠ NEVER_FIRES Features ({len(never_fires)})\n\n")
            f.write("These features exist but never trigger:\n\n")
            for _, m in never_fires.iterrows():
                f.write(f"- {m['feature']}\n")
            f.write("\n")

        if len(always_fires) > 0:
            f.write(f"### ⚠ ALWAYS_FIRES Features ({len(always_fires)})\n\n")
            f.write("These features are always True (may indicate broken logic):\n\n")
            for _, m in always_fires.iterrows():
                f.write(f"- {m['feature']}\n")
            f.write("\n")

        if len(broken) == 0 and len(never_fires) == 0 and len(always_fires) == 0:
            f.write("✅ **No problematic features detected**\n\n")

        # Date Coverage
        f.write("## Date Coverage\n\n")
        f.write(f"- **Start date:** {dates.min()}\n")
        f.write(f"- **End date:** {dates.max()}\n")
        f.write(f"- **Total timespan:** {(dates.max() - dates.min()).days} days\n")
        f.write(f"- **Expected frequency:** 1H\n")

        if len(gaps) > 0:
            f.write(f"- **Gaps detected:** {len(gaps)} (largest: {gaps.max()})\n\n")
            f.write("⚠ **Warning:** Date coverage has gaps. This may affect backtest accuracy.\n\n")
        else:
            f.write(f"- **Gaps detected:** None\n\n")
            f.write("✅ **Continuous date coverage**\n\n")

        # Production Readiness
        f.write("## Production Readiness Assessment\n\n")

        v2_ready = len(v2_missing) == 0 and all(
            metrics_df[metrics_df['feature'] == f].iloc[0]['non_null_pct'] == 100
            for f in v2_found
        )

        critical_systems_ready = (
            len(wyckoff_found) > 0 and wyckoff_working / len(wyckoff_found) >= 0.8 and
            len(smc_found) > 0 and smc_working / len(smc_found) >= 0.8
        )

        if v2_ready:
            f.write("✅ **V2 Features:** Production ready (all present with 100% coverage)\n\n")
        else:
            f.write("❌ **V2 Features:** NOT production ready (missing features or incomplete data)\n\n")

        if critical_systems_ready:
            f.write("✅ **Critical Systems:** Production ready (Wyckoff & SMC >80% working)\n\n")
        else:
            f.write("⚠ **Critical Systems:** Needs review (some features not working optimally)\n\n")

        if overall_quality == "EXCELLENT" and v2_ready and critical_systems_ready:
            f.write("### 🎯 **Overall: PRODUCTION READY**\n\n")
            f.write("Feature store quality is excellent and all critical systems are operational.\n\n")
        elif overall_quality in ["EXCELLENT", "GOOD"]:
            f.write("### ✅ **Overall: GOOD QUALITY**\n\n")
            f.write("Feature store is suitable for production with minor caveats.\n\n")
        else:
            f.write("### ⚠ **Overall: NEEDS IMPROVEMENT**\n\n")
            f.write("Feature store has issues that should be addressed before production deployment.\n\n")

        f.write("## Next Steps\n\n")

        if len(broken) > 0:
            f.write("1. **Fix broken features:** Investigate and repair features with constant values or low coverage\n")
        if len(never_fires) > 0:
            f.write("2. **Review NEVER_FIRES features:** Check if these features are configured correctly\n")
        if len(v2_missing) > 0:
            f.write("3. **Add missing V2 features:** Regenerate feature store with all OI change spike features\n")
        if len(gaps) > 0:
            f.write("4. **Fill date gaps:** Ensure continuous hourly data coverage\n")

        if len(broken) == 0 and len(never_fires) == 0 and len(v2_missing) == 0 and len(gaps) == 0:
            f.write("1. ✅ Feature store is ready for production use\n")
            f.write("2. Monitor feature quality during live trading\n")
            f.write("3. Set up alerts for feature degradation\n")

        f.write("\n---\n\n")
        f.write(f"*Report generated by verify_feature_store_quality.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n✓ Saved verification report to: {report_path}")

    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    exit(main())
