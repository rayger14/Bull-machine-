#!/usr/bin/env python3
"""
Quick Fix for Macro Feature Pipeline Failure (2022-2023 data)

PROBLEM:
- RV_20, rv_20d, funding are ALL ZEROS/NULLS in 2022-2023 feature store
- macro_history.parquet has NO DATA for these features before 2024-01-01

SOLUTION (Hybrid Approach):
1. Compute RV_20, RV_60 from BTC OHLCV (we have this data!)
2. Compute funding from available sources or use conservative defaults
3. Update macro_history.parquet with computed features
4. Regenerate the with_macro feature store

Time: ~10 minutes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from engine.io.tradingview_loader import load_tv

print("=" * 80)
print("FIXING MACRO FEATURES FOR 2022-2023")
print("=" * 80)
print("\nThis script will:")
print("  1. Load BTC 1H OHLCV data (2022-2023)")
print("  2. Compute rv_20d, rv_60d from BTC returns")
print("  3. Update macro_history.parquet with computed features")
print("  4. Validate the fix")
print("=" * 80)

# ============================================================================
# Step 1: Load BTC OHLCV data
# ============================================================================
print("\n[1/4] Loading BTC 1H OHLCV data...")

btc_1h = load_tv("BTC_1H")
print(f"  ✅ Loaded {len(btc_1h)} BTC 1H bars")
print(f"  Date range: {btc_1h.index[0]} to {btc_1h.index[-1]}")

# Filter to 2022-2023 range (with buffer for warm-up)
start_date = pd.Timestamp('2021-01-01', tz='UTC')  # Extra year for RV warm-up
end_date = pd.Timestamp('2023-12-31', tz='UTC')
btc_1h = btc_1h[(btc_1h.index >= start_date) & (btc_1h.index <= end_date)].copy()
print(f"  Filtered to 2021-2023: {len(btc_1h)} bars")

# ============================================================================
# Step 2: Compute Realized Volatility Features
# ============================================================================
print("\n[2/4] Computing realized volatility features...")

# Compute returns
btc_1h['returns'] = btc_1h['close'].pct_change()

# Compute RV features (annualized)
# Formula: RV = std(returns) * sqrt(periods_per_year)
# For hourly data: 365 days * 24 hours = 8760 periods/year

def compute_rv(returns, window_hours):
    """Compute annualized realized volatility from hourly returns."""
    # window_hours is in HOURS (e.g., 20 days = 20*24 = 480 hours)
    # Annualization factor for hourly data: sqrt(8760)
    return returns.rolling(window=window_hours, min_periods=int(window_hours*0.5)).std() * np.sqrt(8760)

# Compute multi-period RV
btc_1h['rv_7d'] = compute_rv(btc_1h['returns'], window_hours=7*24)
btc_1h['rv_20d'] = compute_rv(btc_1h['returns'], window_hours=20*24)
btc_1h['rv_30d'] = compute_rv(btc_1h['returns'], window_hours=30*24)
btc_1h['rv_60d'] = compute_rv(btc_1h['returns'], window_hours=60*24)

# Validate computation
print(f"  ✅ Computed RV features:")
print(f"     rv_7d:  mean={btc_1h['rv_7d'].mean():.4f}, std={btc_1h['rv_7d'].std():.4f}, non-null={btc_1h['rv_7d'].notna().sum()}")
print(f"     rv_20d: mean={btc_1h['rv_20d'].mean():.4f}, std={btc_1h['rv_20d'].std():.4f}, non-null={btc_1h['rv_20d'].notna().sum()}")
print(f"     rv_30d: mean={btc_1h['rv_30d'].mean():.4f}, std={btc_1h['rv_30d'].std():.4f}, non-null={btc_1h['rv_30d'].notna().sum()}")
print(f"     rv_60d: mean={btc_1h['rv_60d'].mean():.4f}, std={btc_1h['rv_60d'].std():.4f}, non-null={btc_1h['rv_60d'].notna().sum()}")

# ============================================================================
# Step 3: Update macro_history.parquet
# ============================================================================
print("\n[3/4] Updating macro_history.parquet...")

# Load macro history
macro_path = Path('data/macro/macro_history.parquet')
if not macro_path.exists():
    print(f"  ❌ File not found: {macro_path}")
    sys.exit(1)

# Backup original
backup_path = macro_path.with_suffix('.parquet.bak_pre_rv_fix')
if not backup_path.exists():
    print(f"  Creating backup: {backup_path.name}")
    macro = pd.read_parquet(macro_path)
    macro.to_parquet(backup_path, compression='snappy')
else:
    print(f"  Backup already exists: {backup_path.name}")
    macro = pd.read_parquet(macro_path)

print(f"  Loaded macro_history: {len(macro)} rows")
print(f"  Date range: {macro['timestamp'].min()} to {macro['timestamp'].max()}")

# Check current state
rv_20d_nulls_before = macro['rv_20d'].isna().sum() if 'rv_20d' in macro.columns else len(macro)
print(f"  rv_20d nulls before fix: {rv_20d_nulls_before} ({rv_20d_nulls_before/len(macro)*100:.1f}%)")

# Merge RV features from BTC data
# Use merge_asof to align hourly BTC data with macro timestamps
btc_rv = btc_1h[['rv_7d', 'rv_20d', 'rv_30d', 'rv_60d']].copy()
btc_rv = btc_rv.reset_index()
# Rename the index column to 'timestamp' (handle various index names)
index_col = btc_rv.columns[0]  # First column after reset_index is the old index
btc_rv = btc_rv.rename(columns={index_col: 'timestamp'})

# Ensure macro has timestamp column
if 'timestamp' not in macro.columns:
    macro = macro.reset_index().rename(columns={'index': 'timestamp'})

# Sort both by timestamp
macro = macro.sort_values('timestamp').reset_index(drop=True)
btc_rv = btc_rv.sort_values('timestamp').reset_index(drop=True)

# Merge RV features (backward fill - use most recent BTC RV value)
macro_merged = pd.merge_asof(
    macro,
    btc_rv,
    on='timestamp',
    direction='backward',
    tolerance=pd.Timedelta(hours=2),
    suffixes=('_old', '')
)

# Drop old RV columns if they exist
old_rv_cols = [col for col in macro_merged.columns if col.endswith('_old')]
if old_rv_cols:
    print(f"  Dropping old RV columns: {old_rv_cols}")
    macro_merged = macro_merged.drop(columns=old_rv_cols)

# Fill any remaining nulls with forward-fill then backward-fill
for col in ['rv_7d', 'rv_20d', 'rv_30d', 'rv_60d']:
    if col in macro_merged.columns:
        nulls_before = macro_merged[col].isna().sum()
        macro_merged[col] = macro_merged[col].fillna(method='ffill').fillna(method='bfill')
        nulls_after = macro_merged[col].isna().sum()
        print(f"  {col}: filled {nulls_before - nulls_after} nulls (remaining: {nulls_after})")

# For funding: if still null, use conservative default (0.01% per 8h = ~0.0001 per hour)
if 'funding' in macro_merged.columns:
    funding_nulls_before = macro_merged['funding'].isna().sum()
    macro_merged['funding'] = macro_merged['funding'].fillna(0.0001)
    funding_nulls_filled = funding_nulls_before - macro_merged['funding'].isna().sum()
    print(f"  funding: filled {funding_nulls_filled} nulls with default 0.0001")

# For oi: if still null, use 0
if 'oi' in macro_merged.columns:
    oi_nulls_before = macro_merged['oi'].isna().sum()
    macro_merged['oi'] = macro_merged['oi'].fillna(0.0)
    oi_nulls_filled = oi_nulls_before - macro_merged['oi'].isna().sum()
    print(f"  oi: filled {oi_nulls_filled} nulls with 0.0")

# Save updated macro history
print(f"\n  Saving updated macro_history.parquet...")
macro_merged.to_parquet(macro_path, compression='snappy', index=False)
print(f"  ✅ Saved {len(macro_merged)} rows")

# Validate fix
rv_20d_nulls_after = macro_merged['rv_20d'].isna().sum()
rv_20d_mean = macro_merged['rv_20d'].mean()
rv_20d_std = macro_merged['rv_20d'].std()
print(f"\n  Validation:")
print(f"    rv_20d nulls after fix: {rv_20d_nulls_after} ({rv_20d_nulls_after/len(macro_merged)*100:.1f}%)")
print(f"    rv_20d mean: {rv_20d_mean:.4f}, std: {rv_20d_std:.4f}")

if rv_20d_nulls_after < rv_20d_nulls_before * 0.1:  # < 10% of original nulls
    print(f"  ✅ SUCCESS: Fixed {rv_20d_nulls_before - rv_20d_nulls_after} null values!")
else:
    print(f"  ⚠️  WARNING: Still have {rv_20d_nulls_after} nulls in rv_20d")

# ============================================================================
# Step 4: Regenerate Feature Store (Optional - provide instructions)
# ============================================================================
print("\n[4/4] Next Steps - Regenerate Feature Store:")
print("  Run this command to regenerate the with_macro feature store:")
print()
print("  python3 bin/append_macro_to_feature_store.py \\")
print("      --asset BTC \\")
print("      --start 2022-01-01 \\")
print("      --end 2023-12-31")
print()
print("  Or if you want to rebuild the entire feature store from scratch:")
print()
print("  python3 bin/build_mtf_feature_store.py \\")
print("      --asset BTC \\")
print("      --start 2022-01-01 \\")
print("      --end 2023-12-31")
print()
print("  Then run append_macro_to_feature_store.py to add macro features.")
print()

print("=" * 80)
print("MACRO FEATURES FIX COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  ✅ Computed rv_7d, rv_20d, rv_30d, rv_60d from BTC OHLCV")
print(f"  ✅ Updated macro_history.parquet ({len(macro_merged)} rows)")
print(f"  ✅ Backup saved: {backup_path.name}")
print()
print("Next: Regenerate feature store using instructions above")
print("=" * 80)
