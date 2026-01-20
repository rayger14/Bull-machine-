# CPCV Smoke Test: Critical Findings

**Date:** 2026-01-18 01:28
**Status:** ⚠️ **BLOCKED** - Critical data quality issues discovered
**Session Duration:** ~30 minutes

---

## TL;DR

The smoke test revealed **severe data quality issues** that block optimization:

1. ✅ **Logging fixed** - Suppressed verbose regime monitoring
2. ✅ **Purging correct** - 246-bar purge window working as designed
3. ✅ **Fold generation correct** - No temporal leakage
4. ❌ **Features missing** - Critical features only exist from 2022-01-02 onwards (42.8% of dataset)
5. ❌ **0 trades generated** - H archetype generates 0 trades even with features present
6. ❌ **Data corruption** - Some features contain garbage data ("range" string repeated thousands of times)

**Bottom line:** Cannot proceed with CPCV optimization until data quality issues are resolved.

---

## What Worked ✅

### 1. Logging Suppression
- **Fix:** Changed root logger to WARNING, only optimizer at INFO
- **Result:** Clean output without 6,000+ regime monitor messages
- **File:** `bin/optimize_constrained_cpcv.py:41-43`

### 2. Logger Formatting Bug
- **Issue:** Tried to format 'N/A' strings as floats
- **Fix:** Conditional formatting with None checks
- **File:** `bin/optimize_constrained_cpcv.py:567-582`

### 3. CPCV Purging Configuration
- **Purge window:** 246 bars (~10.2 days)
- **Calculation:** (200-bar EMA lookback + 24h labels) × 1.1
- **Embargo:** 15 days per fold
- **Result:** No temporal leakage detected

### 4. Archetype Isolation
- **Config:** Only one archetype enabled per trial
- **Verified:** H-only config working correctly
- **No regression** from previous bug

---

## What Failed ❌

### 1. Feature Coverage: Only 42.8% of Dataset

**Discovery:**
```
Features (adx_14, rsi_14, tf4h_*) populated: 2022-01-02 to 2024-12-31
Features missing: 2018-01-01 to 2022-01-01 (35,049 bars = 57.2%)
```

**Impact:**
- Original CPCV Fold 1 (2020-05-02 to 2021-09-25): **0% features** → 0 trades
- Original CPCV Fold 2 (2022-09-01 to 2024-01-25): **100% features** → Should work

**Root Cause:**
- Feature engineering pipeline only ran on recent data
- Or features were added to dataset incrementally and older data wasn't backfilled

**Fix Applied:**
- Modified optimizer to filter data to 2022-01-02 onwards
- Lines 139-141, 682-684 in `bin/optimize_constrained_cpcv.py`

---

### 2. Missing Critical Feature: `tf4h_trend_direction`

**Required by H archetype:** (line 209 of `trap_within_trend.py`)
```python
tf4h_trend = row.get('tf4h_trend_direction', 0)
if tf4h_trend <= self.min_htf_trend:
    return 0.0  # No uptrend = no signal
```

**Status in data:** **COMPLETELY MISSING**

**Alternative features exist:**
- `tf4h_external_trend` (42.8% populated)
- `tf4h_squiggle_direction` (42.8% populated)

**Likely cause:** Feature name mismatch between archetype code and feature engineering

---

### 3. H Archetype Generates 0 Trades (Even With Features)

**Test periods tried:**
1. Q4 2023 (2023-10-01 to 2024-01-01): **0 trades**
2. 2022-09-01 to 2024-01-25: **0 trades**

**Configurations tested:**
1. NO regime gating: **0 trades**
2. WITH regime gating: **0 trades**
3. Lower fusion threshold (0.40): **0 trades**

**Features verified present (Q4 2023):**
- `adx_14`: 100% populated, mean=36.48
- `rsi_14`: 100% populated, mean=52.47
- `tf4h_bos_bearish`: 100% populated, mean=0.024
- `tf4h_fusion_score`: 100% populated (but corrupted - see below)

**Hypothesis:** Missing `tf4h_trend_direction` causes immediate veto on line 210-212:
```python
tf4h_trend = row.get('tf4h_trend_direction', 0)  # Returns 0 (default)
if tf4h_trend <= self.min_htf_trend:  # 0 <= 0 is TRUE
    return 0.0  # VETO - no uptrend
```

Since `tf4h_trend_direction` is missing, it defaults to 0, which vetoes ALL signals.

---

### 4. Data Corruption in tf4h Features

**Error when calculating mean of `tf4h_external_trend`:**
```
TypeError: Could not convert string 'rangerangerange...' to numeric
```

**Evidence:**
- Some tf4h features contain string "range" repeated thousands of times
- This corrupts the data and breaks calculations

**Affected features (suspected):**
- `tf4h_external_trend` (confirmed corrupted)
- Possibly others in the tf4h_* family

---

## Impact Assessment

### Smoke Test Results

**All 3 trials:** FAILED (returned -inf)

```
Trial 0: fusion_threshold=0.606, Trades=0
Trial 1: fusion_threshold=0.640, Trades=0
Trial 2: fusion_threshold=0.559, Trades=0

Warning: Only 2 folds succeeded (need >=3), returning -inf
```

### Why Optimization Cannot Proceed

1. **Feature Coverage:** Only 2022-2024 data usable (42.8%)
   - Reduces CPCV folds from 5 to ~2-3 max (not enough statistical power)

2. **Missing Critical Feature:** `tf4h_trend_direction`
   - H archetype cannot generate ANY signals without it
   - Default value (0) immediately vetoes all trades

3. **Data Corruption:**
   - Some features contain garbage data
   - Can't trust optimization results with corrupted features

---

## Required Fixes (Priority Order)

### 🔴 CRITICAL (Blocks everything)

**Fix 1: Add `tf4h_trend_direction` feature**
- Option A: Rename `tf4h_external_trend` → `tf4h_trend_direction` in archetype code
- Option B: Add `tf4h_trend_direction` as alias in feature engineering
- Option C: Backfill `tf4h_trend_direction` in dataset

**Fix 2: Clean corrupted tf4h features**
- Investigate why "range" strings appear in numeric columns
- Re-run feature engineering for affected features
- Validate data types after regeneration

### 🟡 HIGH (Enables proper CPCV)

**Fix 3: Backfill features to 2018-2024**
- Re-run feature engineering on full historical dataset
- Target: 100% feature coverage for all 61,277 bars
- Enables proper 5-fold CPCV with statistical power

### 🟢 MEDIUM (Nice to have)

**Fix 4: Verify all H archetype feature dependencies**
- Cross-reference archetype code with available features
- Document any other name mismatches
- Add data validation tests

---

## Alternative Paths Forward

### Option A: Fix Data (Recommended)

**Pros:**
- Proper fix - enables all archetypes
- Enables walk-forward validation on full history
- Production-quality data

**Cons:**
- Time-consuming (feature engineering on 4+ years)
- May require debugging feature engineering pipeline

**Timeline:** 1-2 days

---

### Option B: Use Different Archetypes

**Try B (Order Block Retest) or S1 (Liquidity Vacuum):**
- Check if they have different feature dependencies
- May work with available features

**Pros:**
- Faster - no data regeneration needed
- Can test CPCV methodology

**Cons:**
- Doesn't fix underlying data issues
- May hit same problems

**Timeline:** 1 hour

---

### Option C: Simplify H Archetype

**Remove dependency on missing features:**
- Comment out `tf4h_trend_direction` check (lines 209-212)
- Use available features only (adx_14, rsi_14, etc.)
- Test if signals generate

**Pros:**
- Quick test to unblock smoke test
- Validates CPCV methodology

**Cons:**
- Degrades H archetype quality
- Not production-ready
- Technical debt

**Timeline:** 30 minutes

---

## Files Changed This Session

### Modified
1. `bin/optimize_constrained_cpcv.py`
   - Lines 41-43: Logging suppression
   - Lines 139-141: Filter to 2022+ date range
   - Lines 567-582: Fix logger formatting bug
   - Lines 682-684: Filter loaded data to feature range

### Created
2. `bin/diagnose_h_zero_trades.py` - Diagnostic script
3. `CPCV_SMOKE_TEST_CRITICAL_FINDINGS.md` - This file

---

## Recommendations

### Immediate Next Steps

1. **Investigate feature engineering:**
   - Why does `tf4h_trend_direction` not exist?
   - Why is data only populated from 2022 onwards?
   - What caused "range" string corruption?

2. **Choose path forward:**
   - **Path A (best):** Fix data quality → Re-run smoke test
   - **Path B (pragmatic):** Try B or S1 archetypes first
   - **Path C (hacky):** Simplify H → Test CPCV methodology

3. **If continuing with H:**
   - Map `tf4h_external_trend` → `tf4h_trend_direction`
   - Or remove trend check temporarily
   - Re-run diagnostic to confirm trades > 0

### User Decision Required

**Question:** Which path do you want to take?

- [ ] **Option A:** Fix data quality properly (1-2 days)
- [ ] **Option B:** Try different archetypes (B or S1) first
- [ ] **Option C:** Patch H archetype to test CPCV methodology

---

## What We Learned

1. ✅ **CPCV implementation is correct** - Purging, embargo, fold generation all working
2. ✅ **Logging is clean** - No more 6,000-line outputs
3. ❌ **Data quality is the blocker** - Not optimization methodology
4. ❌ **Feature coverage is incomplete** - Only 42.8% of dataset usable
5. ❌ **Feature name mismatches exist** - Code expects features that don't exist
6. ❌ **Data corruption present** - String data in numeric columns

**Key insight:** The user was right - "we're having so many moving parts and aren't sure if the backtest engine is doing it properly." This wasn't an engine problem - it was data quality.

---

**Session Status:** ⏸️ PAUSED - Awaiting user decision on path forward
**Next Session:** Resume after data fixes or archetype selection

