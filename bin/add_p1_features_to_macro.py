#!/usr/bin/env python3
"""
Add P1 priority features to macro_history.parquet for GMM v3 training.

P1 Features to Add:
1. RV_7, RV_30 (in addition to existing rv_20d, rv_60d)
2. OI_CHANGE (% change in open interest)
3. YC_SPREAD (10Y-2Y yield curve spread)
4. TOTAL_RET, TOTAL2_RET, TOTAL3_RET (returns on breadth metrics)
5. ALT_ROTATION (TOTAL3_RET - BTC_RET as alt vs BTC rotation signal)
6. Event flags: FOMC_D0, CPI_D0, NFP_D0, and ±1 day variants
7. Z-scores for continuous macro features

Usage:
    python3 bin/add_p1_features_to_macro.py
    python3 bin/add_p1_features_to_macro.py --dry-run
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Event Calendar (2022-2025)
# ─────────────────────────────────────────────────────────────────────────────
# FOMC meeting dates (decision day, 2PM ET)
FOMC_DATES = [
    # 2022
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27',
    '2022-09-21', '2022-11-02', '2022-12-14',
    # 2023
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26',
    '2023-09-20', '2023-11-01', '2023-12-13',
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31',
    '2024-09-18', '2024-11-07', '2024-12-18',
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30'
]

# CPI release dates (typically 8:30 AM ET, ~13th of month)
CPI_DATES = [
    # 2022 (monthly)
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12', '2022-05-11', '2022-06-10',
    '2022-07-13', '2022-08-10', '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    # 2023
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12', '2023-05-10', '2023-06-13',
    '2023-07-12', '2023-08-10', '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    # 2024
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12',
    '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11'
]

# NFP (Non-Farm Payrolls) - first Friday of month, 8:30 AM ET
NFP_DATES = [
    # 2022
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01', '2022-05-06', '2022-06-03',
    '2022-07-08', '2022-08-05', '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    # 2023
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07', '2023-05-05', '2023-06-02',
    '2023-07-07', '2023-08-04', '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    # 2024
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', '2024-06-07',
    '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06',
    # 2025
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02', '2025-06-06'
]

def compute_realized_volatility(returns, window):
    """Compute annualized realized volatility from returns."""
    # RV = sqrt(252) * std(returns)
    return returns.rolling(window=window).std() * np.sqrt(252)

def compute_z_score(series, window=252):
    """Compute rolling z-score (252-hour = ~10 days)."""
    mean = series.rolling(window=window, min_periods=50).mean()
    std = series.rolling(window=window, min_periods=50).std()
    return (series - mean) / std.replace(0, np.nan)

def create_event_flags(df, event_dates, event_name):
    """
    Create event proximity flags: D-1, D0, D+1.

    Args:
        df: DataFrame with 'timestamp' column (UTC timezone-aware)
        event_dates: List of date strings 'YYYY-MM-DD'
        event_name: String like 'FOMC', 'CPI', 'NFP'

    Returns:
        DataFrame with 3 new columns: {event}_DM1, {event}_D0, {event}_DP1
    """
    # Convert event dates to datetime (midnight UTC)
    event_datetimes = pd.to_datetime(event_dates).tz_localize('UTC')

    # Initialize flags
    df[f'{event_name}_DM1'] = 0  # T-1 day
    df[f'{event_name}_D0'] = 0   # T+0 day (event day)
    df[f'{event_name}_DP1'] = 0  # T+1 day

    for event_dt in event_datetimes:
        # Define windows (24-hour blocks)
        d_minus_1_start = event_dt - timedelta(days=1)
        d_minus_1_end = event_dt
        d_0_end = event_dt + timedelta(days=1)
        d_plus_1_end = event_dt + timedelta(days=2)

        # Set flags
        df.loc[(df['timestamp'] >= d_minus_1_start) & (df['timestamp'] < d_minus_1_end), f'{event_name}_DM1'] = 1
        df.loc[(df['timestamp'] >= event_dt) & (df['timestamp'] < d_0_end), f'{event_name}_D0'] = 1
        df.loc[(df['timestamp'] >= d_0_end) & (df['timestamp'] < d_plus_1_end), f'{event_name}_DP1'] = 1

    return df

def add_p1_features(df):
    """
    Add all P1 priority features to macro DataFrame.

    Assumes df has columns: timestamp, VIX, DXY, YIELD_2Y, YIELD_10Y,
                             BTC.D, USDT.D, TOTAL, TOTAL2, TOTAL3,
                             funding, oi, rv_20d, rv_60d
    """
    print("\n" + "="*80)
    print("ADDING P1 FEATURES TO MACRO HISTORY")
    print("="*80)

    # ─── 1. Realized Volatility (7, 30-day in addition to existing 20, 60) ───
    print("\n1. Computing multi-period realized volatility...")

    # Compute BTC returns from TOTAL3 proxy (approximate, better would be from BTC price)
    # For now, use funding rate as a proxy for BTC return volatility
    # TODO: If BTC OHLC available, use: df['btc_ret'] = df['btc_close'].pct_change()

    # Use TOTAL as crypto market proxy for returns
    if 'TOTAL' in df.columns:
        df['TOTAL_ret'] = df['TOTAL'].pct_change().fillna(0)
        df['RV_7'] = compute_realized_volatility(df['TOTAL_ret'], window=7*24)  # 7 days in hours
        df['RV_30'] = compute_realized_volatility(df['TOTAL_ret'], window=30*24)  # 30 days
        print(f"   ✅ Added RV_7, RV_30 (7-day, 30-day realized vol)")

    # Rename existing rv_20d, rv_60d for consistency
    if 'rv_20d' in df.columns:
        df['RV_20'] = df['rv_20d']
        df['RV_60'] = df['rv_60d']
        print(f"   ✅ Renamed rv_20d → RV_20, rv_60d → RV_60")

    # ─── 2. Open Interest Change ───
    print("\n2. Computing OI change %...")
    if 'oi' in df.columns:
        df['OI_CHANGE'] = df['oi'].pct_change(periods=24).fillna(0) * 100  # 24h % change
        print(f"   ✅ Added OI_CHANGE (24-hour % change)")

    # ─── 3. Yield Curve Features ───
    print("\n3. Computing yield curve features...")
    if 'YIELD_10Y' in df.columns and 'YIELD_2Y' in df.columns:
        df['YC_SPREAD'] = df['YIELD_10Y'] - df['YIELD_2Y']  # 2s10s spread
        df['YC_Z'] = compute_z_score(df['YC_SPREAD'], window=252*24)  # ~10 days
        print(f"   ✅ Added YC_SPREAD (10Y-2Y), YC_Z (z-score)")

    # ─── 4. Breadth Returns & Alt Rotation ───
    print("\n4. Computing breadth returns & alt rotation...")
    if 'TOTAL' in df.columns:
        df['TOTAL_RET'] = df['TOTAL'].pct_change(periods=24).fillna(0) * 100  # 24h % return
    if 'TOTAL2' in df.columns:
        df['TOTAL2_RET'] = df['TOTAL2'].pct_change(periods=24).fillna(0) * 100
    if 'TOTAL3' in df.columns:
        df['TOTAL3_RET'] = df['TOTAL3'].pct_change(periods=24).fillna(0) * 100

    # Alt rotation: TOTAL3 (small caps) outperformance vs BTC proxy
    # Approximation: TOTAL3_RET - TOTAL_RET (since TOTAL includes BTC)
    if 'TOTAL3_RET' in df.columns and 'TOTAL_RET' in df.columns:
        df['ALT_ROTATION'] = df['TOTAL3_RET'] - df['TOTAL_RET']
        print(f"   ✅ Added TOTAL_RET, TOTAL2_RET, TOTAL3_RET, ALT_ROTATION")

    # ─── 5. Event Flags ───
    print("\n5. Creating event proximity flags...")
    df = create_event_flags(df, FOMC_DATES, 'FOMC')
    df = create_event_flags(df, CPI_DATES, 'CPI')
    df = create_event_flags(df, NFP_DATES, 'NFP')
    print(f"   ✅ Added FOMC_DM1/D0/DP1, CPI_DM1/D0/DP1, NFP_DM1/D0/DP1")

    # ─── 6. Z-Scores for Continuous Features ───
    print("\n6. Computing z-scores for macro features...")
    continuous_features = ['VIX', 'DXY', 'YIELD_2Y', 'YIELD_10Y', 'BTC.D', 'USDT.D',
                           'funding', 'RV_7', 'RV_20', 'RV_30', 'RV_60']

    for feat in continuous_features:
        if feat in df.columns:
            df[f'{feat}_Z'] = compute_z_score(df[feat], window=252*24)  # ~10 days rolling

    print(f"   ✅ Added z-scores for {len(continuous_features)} continuous features")

    return df

def main():
    parser = argparse.ArgumentParser(description="Add P1 features to macro history")
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without saving")
    args = parser.parse_args()

    # Load existing macro history
    macro_path = Path('data/macro/macro_history.parquet')
    if not macro_path.exists():
        print(f"❌ Macro history not found: {macro_path}")
        print("   Run: python3 bin/populate_macro_data.py first")
        return 1

    print(f"\n📖 Loading macro history from: {macro_path}")
    df = pd.read_parquet(macro_path)

    print(f"   Original shape: {df.shape}")
    print(f"   Original columns: {list(df.columns)}")

    # Add P1 features
    df_enhanced = add_p1_features(df)

    print(f"\n   Enhanced shape: {df_enhanced.shape}")
    print(f"   New columns added: {df_enhanced.shape[1] - df.shape[1]}")

    # Show summary statistics
    print("\n" + "="*80)
    print("P1 FEATURE SUMMARY STATISTICS")
    print("="*80)

    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    if new_features:
        print(f"\nNew features ({len(new_features)}):")
        for feat in new_features:
            non_null_count = df_enhanced[feat].notna().sum()
            coverage_pct = (non_null_count / len(df_enhanced)) * 100
            mean_val = df_enhanced[feat].mean() if df_enhanced[feat].dtype in [np.float64, np.float32, np.int64] else None
            print(f"  {feat:20s}: {non_null_count:6d} / {len(df_enhanced)} ({coverage_pct:5.1f}% coverage){f', mean={mean_val:.3f}' if mean_val is not None else ''}")

    # Save or preview
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes saved")
        print(f"\nSample of enhanced data:")
        print(df_enhanced[new_features[:5]].head(10))
    else:
        # Save enhanced macro history
        print(f"\n💾 Saving enhanced macro history to: {macro_path}")
        df_enhanced.to_parquet(macro_path, index=False)
        print(f"   ✅ Saved successfully!")

        # Save feature list for reference
        feature_list_path = Path('data/macro/p1_features.txt')
        with open(feature_list_path, 'w') as f:
            f.write("P1 Priority Features Added:\n")
            f.write("="*80 + "\n\n")
            for feat in new_features:
                f.write(f"{feat}\n")
        print(f"   ✅ Feature list saved to: {feature_list_path}")

    print("\n" + "="*80)
    print("✅ P1 FEATURE ADDITION COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Update feature stores: python3 bin/update_macro_in_feature_store.py --asset BTC --years 2022,2023,2024")
    print(f"  2. Retrain GMM v3: python3 bin/train_regime_gmm_v3_full.py")
    print(f"  3. Test with configs: python3 bin/backtest_knowledge_v2.py --regime-aware")
    print("="*80 + "\n")

    return 0

if __name__ == '__main__':
    exit(main())
