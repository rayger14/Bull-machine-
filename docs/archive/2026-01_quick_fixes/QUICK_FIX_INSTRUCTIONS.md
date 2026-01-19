# Quick Fix Instructions - Macro Feature Pipeline Failure

**Problem:** RV_20, funding, rv_20d are ALL ZEROS/NULLS in 2022-2023 feature store
**Impact:** HMM regime detection cannot train
**Time to fix:** 10-15 minutes

---

## TL;DR - Execute These Commands

```bash
# 1. Fix macro_history.parquet (compute missing RV features)
python3 bin/fix_macro_features_2022_2023.py

# 2. Regenerate feature store with fixed macro data
python3 bin/append_macro_to_feature_store.py \
    --asset BTC \
    --start 2022-01-01 \
    --end 2023-12-31

# 3. Validate the fix
python3 bin/validate_macro_fix.py

# 4. Train HMM (should work now!)
python3 bin/train_hmm_simplified.py
```

---

## What Each Script Does

### 1. `fix_macro_features_2022_2023.py`

**Purpose:** Compute missing realized volatility features from BTC OHLCV data

**What it does:**
- Loads BTC 1H OHLCV data from TradingView
- Computes rv_7d, rv_20d, rv_30d, rv_60d from BTC returns
- Updates `data/macro/macro_history.parquet` with computed features
- Fills missing funding/oi with conservative defaults

**Output:**
- Updated `data/macro/macro_history.parquet`
- Backup saved as `macro_history.parquet.bak_pre_rv_fix`

**Time:** ~2 minutes

### 2. `append_macro_to_feature_store.py`

**Purpose:** Merge macro features into the MTF feature store

**What it does:**
- Loads the base feature store (without macro features)
- Merges macro_history.parquet using time-aligned join
- Saves result as `*_with_macro.parquet`

**Output:**
- `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet`

**Time:** ~1 minute

### 3. `validate_macro_fix.py`

**Purpose:** Verify that all critical features are properly populated

**What it checks:**
- RV_20, rv_20d, rv_60d: Non-zero, non-null values
- funding, funding_Z, VIX_Z: Proper coverage
- Data quality: mean, std, null counts

**Output:**
- ✅ or ❌ for each critical feature
- Summary report

**Time:** ~10 seconds

---

## Expected Results After Fix

### Before Fix (BROKEN)

```
RV_20:      ALL ZEROS (17,475 zeros)  ❌
rv_20d:     ALL NULLS (0 non-null)    ❌
funding:    ALL NULLS (0 non-null)    ❌
```

### After Fix (WORKING)

```
RV_20:      15,000+ non-zero values   ✅
rv_20d:     15,000+ non-null values   ✅
funding:    15,000+ non-null values   ✅ (defaults filled)

Stats:
  rv_20d: mean ≈ 0.55, std ≈ 0.25 (annualized volatility ~55%)
  funding: mean ≈ 0.0001 (default 0.01% per 8h)
  funding_Z: mean ≈ 0.0, std ≈ 1.0 (z-scored)
```

---

## Troubleshooting

### Issue: "File not found: BTC_1H"

**Fix:** TradingView data loader may need configuration
```bash
# Check if TV data exists
ls -la data/tradingview/

# If missing, you may need to fetch it
# (contact team for data loading instructions)
```

### Issue: "macro_history.parquet not found"

**Fix:** File is missing, need to create it first
```bash
# Check if file exists
ls -la data/macro/macro_history.parquet

# If missing, you may need to run the initial macro builder
python3 bin/populate_macro_data.py
```

### Issue: Still getting zeros after fix

**Check:**
1. Did `fix_macro_features_2022_2023.py` complete without errors?
2. Did you regenerate the feature store with `append_macro_to_feature_store.py`?
3. Run validation: `python3 bin/validate_macro_fix.py`

### Issue: HMM training still fails

**Debug steps:**
```python
# Check what features HMM is actually using
import pandas as pd

df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet')

# HMM needs these features (from train_hmm_simplified.py):
required = ['VIX', 'DXY', 'MOVE', 'YIELD_2Y', 'YIELD_10Y',
            'USDT.D', 'BTC.D', 'TOTAL', 'TOTAL2',
            'funding', 'oi', 'rv_20d', 'rv_60d']

for feat in required:
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        print(f"{feat:15s}: {non_null:5d} non-null ({non_null/len(df)*100:.1f}%)")
    else:
        print(f"{feat:15s}: MISSING")
```

---

## Understanding the Root Cause

**Why did this happen?**

1. `macro_history.parquet` was created/updated starting in 2024
2. Historical backfill for 2022-2023 was never run
3. `append_macro_to_feature_store.py` merged NULL values from incomplete macro file
4. Result: Training data has all zeros/nulls for critical regime features

**Why RV_20 vs rv_20d confusion?**

- `RV_20` = Computed feature (from TOTAL market cap returns)
- `rv_20d` = Raw macro feature (from BTC price returns)
- They should be similar but from different sources
- Bug: RV_20 was supposed to be renamed from rv_20d, but rv_20d didn't exist!

**The fix:**

- Compute rv_20d from BTC OHLCV (which we already have)
- Update macro_history.parquet with computed values
- Regenerate feature store with proper data

---

## Long-Term Solutions

### Option A: Backfill Real Funding/OI Data (2 hours)

```bash
# Fetch real historical data from OKX API
python3 bin/backfill_missing_macro_features.py \
    --feature-store data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
    --start-date 2022-01-01 \
    --end-date 2023-12-31
```

**Pros:**
- Real historical funding rates
- Real historical open interest
- Better accuracy for regime detection

**Cons:**
- Takes longer (API rate limits)
- May have data gaps
- Requires OKX/Binance API access

### Option B: Use Current Quick Fix (10 minutes) ⭐ RECOMMENDED

Current fix is good enough because:
- RV features computed from BTC OHLC are accurate
- funding_Z and VIX_Z (z-scored) already work
- Default funding rate (0.01%) is reasonable approximation
- Can upgrade to real data later if needed

---

## Validation Checklist

After running the fix, verify:

- [ ] `fix_macro_features_2022_2023.py` completed without errors
- [ ] Backup created: `macro_history.parquet.bak_pre_rv_fix`
- [ ] `append_macro_to_feature_store.py` completed
- [ ] `validate_macro_fix.py` shows all ✅
- [ ] Feature store file updated (check timestamp)
- [ ] HMM training runs without "all zeros" error

---

## Files Modified

**Updated:**
- `data/macro/macro_history.parquet` (added rv_7d, rv_20d, rv_30d, rv_60d)
- `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_with_macro.parquet` (regenerated)

**Backups Created:**
- `data/macro/macro_history.parquet.bak_pre_rv_fix`
- `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet.bak_pre_macro` (if exists)

**New Scripts:**
- `bin/fix_macro_features_2022_2023.py`
- `bin/validate_macro_fix.py`

---

## Next Steps After Fix

1. **Train HMM regime classifier**
   ```bash
   python3 bin/train_hmm_simplified.py
   ```

2. **Validate regime detection**
   ```bash
   python3 bin/quick_hmm_validation.py
   ```

3. **Run backtests with regime routing**
   ```bash
   python3 bin/validate_regime_routing.py
   ```

---

**Questions?** See full diagnosis: `FEATURE_PIPELINE_FAILURE_DIAGNOSIS.md`
