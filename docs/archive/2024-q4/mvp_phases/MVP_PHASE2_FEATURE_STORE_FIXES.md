# Phase 2: Feature Store Fixes - Technical Report

**Date**: 2025-10-22
**Status**: IN PROGRESS
**Issue**: 3-archetype entry system producing 0 trades due to missing/zero feature values

---

## Root Cause Summary

The 3-archetype entry system is **correctly implemented** but cannot function because the **feature store builder** has calculation bugs that produce zero/missing values for critical SMC indicators.

This is NOT an entry logic problem - it's a **data pipeline problem**.

---

## Critical Bugs Identified

### 1. BOMS Displacement Always 0.00 ❌ → ✅ **FIXED**

**Location**: `engine/structure/boms_detector.py:258-286`

**Root Cause**:
- Displacement was only calculated when **ALL** BOMS conditions were met:
  1. Price breaks structure ✓
  2. Volume > threshold ✓
  3. **FVG present** ← STRICT requirement
  4. **No reversal (3 bars)** ← STRICT requirement

- If FVG not present OR reversal occurred → displacement = 0.0
- In 2024 BTC data, FVG+no-reversal combo was VERY rare → 100% of bars had displacement=0.0

**Impact on Archetype System**:
- **Archetype A** requires: `boms_disp >= 1.25 × ATR`
- **Archetype C** requires: `boms_disp >= 1.5 × ATR`
- Both conditions **IMPOSSIBLE** when displacement always 0.0

**Additional Bug - Unit Mismatch**:
- Original displacement calculated as **percentage**: `(close - swing_high) / swing_high` (e.g., 0.02 = 2%)
- Archetype system expects **absolute price**: displacement compared to ATR multiples (e.g., 731.4 when ATR=585.12)
- This would have caused false positives even if displacement was calculated

**Fix Applied**:
1. Calculate displacement **ALWAYS** when structure break occurs (regardless of FVG/reversal)
2. Changed formula from percentage to **absolute price difference**:
   - Bullish: `displacement = close - swing_high` (in price units)
   - Bearish: `displacement = swing_low - close` (in price units)
3. Return best displacement even if full BOMS not confirmed

**Code Changes**:
```python
# BEFORE (broken):
if close > swing_high and volume_surge > volume_threshold:
    fvg_present = detect_fvg_trail(df, i - 3, 'bullish')
    if fvg_present:
        if check_no_immediate_reversal(df, i, 'bullish', bars=3):
            displacement = (close - swing_high) / swing_high  # Percentage!
            return BOMSSignal(..., displacement=displacement)
# If FVG or reversal failed → displacement = 0.0

# AFTER (fixed):
best_bullish_displacement = 0.0
...
if close > swing_high and volume_surge > volume_threshold:
    # ALWAYS calculate displacement in absolute price terms
    displacement = close - swing_high  # Absolute price difference
    best_bullish_displacement = max(best_bullish_displacement, displacement)

    # FVG/reversal only required for full BOMS confirmation
    fvg_present = detect_fvg_trail(df, i - 3, 'bullish')
    if fvg_present and check_no_immediate_reversal(...):
        return BOMSSignal(boms_detected=True, ..., displacement=displacement)

# No full BOMS but structure was broken → return displacement anyway
if best_bullish_displacement > 0.0:
    return BOMSSignal(boms_detected=False, ..., displacement=best_bullish_displacement)
```

**Expected Results After Fix**:
- Displacement will be non-zero when price breaks swing highs/lows
- Example: BTC breaks $65,000 swing high, closes at $65,850 → displacement = 850
- Archetype A/C can now check: `if displacement >= 1.25 × ATR` (e.g., >= 731.4 when ATR=585.12)

**Status**: ✅ **FIXED** in commit `feat(phase2): fix BOMS displacement calculation for archetype system`

---

### 2. BOMS Strength Always 0.000 ⏸️ **PENDING**

**Location**: `bin/build_mtf_feature_store.py:205`

**Current Code**:
```python
boms_1d = detect_boms(window_1d, timeframe='1D', config=config)
features['tf1d_boms_detected'] = boms_1d.boms_detected
features['tf1d_boms_strength'] = boms_1d.displacement if boms_1d.boms_detected else 0.0  # ← BUG
features['tf1d_boms_direction'] = boms_1d.direction if boms_1d.boms_detected else 'none'
```

**Root Cause**:
- `tf1d_boms_strength` is set to `boms_1d.displacement` only when `boms_detected=True`
- Since full BOMS rarely confirmed (needs FVG + no reversal), strength is always 0.0
- **Should normalize displacement to 0.0-1.0 range** for use as "strength" metric

**Impact on Archetype System**:
- **Archetype B** requires: `boms_strength >= 0.68`
- Condition is **IMPOSSIBLE** when strength always 0.0

**Proposed Fix**:
```python
# Calculate strength as normalized displacement (0.0-1.0 range)
# Strength = displacement / (2.0 × ATR), capped at 1.0
# Rationale: 2× ATR displacement = very strong, > 2× ATR = maximum strength
atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]
if atr_1d > 0 and boms_1d.displacement > 0:
    features['tf1d_boms_strength'] = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
else:
    features['tf1d_boms_strength'] = 0.0
```

**Status**: ⏸️ **PENDING** - will fix after verifying displacement fix

---

### 3. Liquidity Scores Always 0.000 ⏸️ **INVESTIGATION NEEDED**

**Location**: Fusion calculation in `bin/backtest_knowledge_v2.py` (context derived feature)

**Root Cause**: Unknown - need to investigate fusion score calculation

**Current Understanding**:
- `liquidity_score` is derived from fusion context, not a direct feature store column
- Likely calculated from HOB (Higher Order Block) + BOMS features
- May be broken due to missing BOMS displacement (cascading failure)

**Impact on Archetype System**:
- **Archetype B** requires: `liq_score >= 0.68`
- **Archetype C** requires: `liq_score >= 0.72`
- Both conditions **IMPOSSIBLE** when liquidity always 0.0

**Investigation Plan**:
1. Check if liquidity score calculation depends on BOMS displacement
2. If yes → may auto-fix after displacement fix
3. If no → need to find and fix liquidity calculation logic

**Status**: ⏸️ **INVESTIGATION NEEDED** - will test after displacement fix

---

### 4. PTI Trap Type Never Populated ⏸️ **PENDING**

**Location**: `bin/build_mtf_feature_store.py:448`

**Current Code**:
```python
features['tf1h_pti_trap_type'] = 'bullish_trap' if pti_score > 0.6 else 'none'
```

**Root Cause**:
- Trap type is hardcoded to 'bullish_trap' or 'none'
- No logic to classify trap types: 'bull_trap', 'bear_trap', 'spring', 'utad'
- PTI score IS calculated correctly (0.0-0.531 range observed)
- Just missing trap classification logic

**Impact on Archetype System**:
- **Archetype A** requires: `pti_trap is not None`
- In code, 'none' is treated as None → condition never met

**Proposed Fix**:
```python
# Classify trap type based on PTI components
if pti_score > 0.6:
    # Analyze which PTI component triggered (RSI div, volume exhaustion, wick trap, failed BO)
    if rsi_div.get('strength', 0.0) > 0.6:
        # Check if bearish or bullish divergence
        if rsi_div.get('type') == 'bearish':
            trap_type = 'bull_trap'  # Price made high but RSI diverged down
        else:
            trap_type = 'bear_trap'  # Price made low but RSI diverged up
    elif vol_exh.get('strength', 0.0) > 0.6:
        # Volume exhaustion suggests end of move
        if df_1h['close'].iloc[-1] > df_1h['close'].iloc[-5]:
            trap_type = 'utad'  # Upthrust after distribution
        else:
            trap_type = 'spring'  # Spring after accumulation
    else:
        trap_type = 'none'
else:
    trap_type = 'none'

features['tf1h_pti_trap_type'] = trap_type
```

**Status**: ⏸️ **PENDING** - requires PTI detector enhancement

---

### 5. tf4h_fusion_score Always 0.000 ⏸️ **INVESTIGATION NEEDED**

**Location**: `bin/build_mtf_feature_store.py:968`

**Current Code**:
```python
# tf4h_fusion_score: Simple weighted average of tf4h Wyckoff/structure indicators
wyckoff_score_map = {
    'accumulation': 0.7, 'markup': 0.9, 'distribution': -0.7, 'markdown': -0.9,
    'm1': 0.5, 'm2': 0.6, 'm3': 0.8, 'm4': 0.9, 'm5': 0.85,
    'unknown': 0.0, 'neutral': 0.0
}

if 'tf4h_wyckoff_phase' in features.columns:
    features['tf4h_fusion_score'] = features['tf4h_wyckoff_phase'].map(wyckoff_score_map).fillna(0.0)
else:
    features['tf4h_fusion_score'] = features.get('tf4h_structure_alignment', 0.0)
```

**Root Cause**:
- Code expects `tf4h_wyckoff_phase` column to exist
- This column is NOT created in the 4H feature computation
- Falls back to `tf4h_structure_alignment` which is boolean (True/False → 1.0/0.0)
- Likely all False → all 0.0

**Impact on Archetype System**:
- **Archetype C Plus-One sizing** requires: `tf4h_fusion >= 0.62`
- Prevents 1.25× sizing optimization (all trades would be 1.0× instead)
- Does NOT block entries, only reduces position size

**Proposed Fix**:
```python
# Calculate fusion score from available 4H features
# Use structure alignment + squiggle confidence + CHOCH as proxies
tf4h_fusion = 0.0

if features.get('tf4h_structure_alignment', False):
    tf4h_fusion += 0.30  # Internal/external aligned

if features.get('tf4h_squiggle_entry_window', False):
    tf4h_fusion += 0.20  # Squiggle 1-2-3 entry window
    tf4h_fusion += features.get('tf4h_squiggle_confidence', 0.0) * 0.20

if features.get('tf4h_choch_flag', False):
    tf4h_fusion += 0.30  # CHOCH detected

features['tf4h_fusion_score'] = min(tf4h_fusion, 1.0)
```

**Status**: ⏸️ **INVESTIGATION NEEDED** - requires proper 4H fusion calculation

---

## Implementation Priority

### P0 (Critical - Blocks ALL Archetypes):
1. ✅ **DONE**: BOMS displacement calculation (`tf4h_boms_displacement`)
   - **Unblocks**: Archetype A, Archetype C
   - **Status**: Fixed, awaiting full-year rebuild and test

### P1 (High - Blocks 2/3 Archetypes):
2. ⏸️ **NEXT**: BOMS strength calculation (`tf1d_boms_strength`)
   - **Unblocks**: Archetype B
   - **Dependency**: None (independent fix)

3. ⏸️ **NEXT**: Liquidity score derivation (fusion context)
   - **Unblocks**: Archetype B, Archetype C
   - **Dependency**: May auto-fix after #1 (displacement fix)

### P2 (Medium - Quality Improvements):
4. ⏸️ **LATER**: PTI trap type classification (`tf1h_pti_trap_type`)
   - **Unblocks**: Archetype A fully
   - **Dependency**: Requires PTI detector enhancement

5. ⏸️ **LATER**: tf4h fusion scoring (`tf4h_fusion_score`)
   - **Enables**: Archetype C Plus-One sizing (1.25×)
   - **Dependency**: Requires proper 4H fusion calculation

---

## Testing Plan

### Step 1: Verify Displacement Fix ✅ **IN PROGRESS**

```bash
# Rebuild feature store with displacement fix
python3 bin/build_mtf_feature_store.py --asset BTC --start 2024-01-01 --end 2024-12-31

# Verify displacement is non-zero
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')
print('tf4h_boms_displacement stats:')
print(df['tf4h_boms_displacement'].describe())
print(f'Non-zero count: {(df[\"tf4h_boms_displacement\"] > 0).sum()} / {len(df)}')
"
```

**Expected**:
- displacement > 0 for 10-20% of bars (when structure breaks occur)
- Range: 50-5000 (price units, not percentages)

### Step 2: Test Archetype System

```bash
# Test with fixed displacement
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31
```

**Expected After Displacement Fix**:
- Some archetype matches (not all 0)
- Archetype A + C may start working (depend on displacement only)
- Archetype B still blocked (needs liquidity + BOMS strength)

### Step 3: Investigate Cascading Fixes

If liquidity scores still 0.000 after displacement fix:
- Locate liquidity calculation logic in fusion engine
- Add debugging to understand why it's zero
- Fix calculation or dependencies

### Step 4: Full System Test

After all P0+P1 fixes:
- Expected: 50-80 trades/year across all 3 archetypes
- Expected distribution:
  - Archetype A (Trap Reversal): 5-15 trades (16% of total)
  - Archetype B (OB Retest): 20-40 trades (56% of total)
  - Archetype C (FVG Continuation): 15-30 trades (28% of total)

---

## Files Modified

### Fixed:
1. `engine/structure/boms_detector.py` - BOMS displacement calculation

### Pending:
2. `bin/build_mtf_feature_store.py` - BOMS strength, PTI trap type, tf4h fusion
3. `bin/backtest_knowledge_v2.py` or `engine/fusion/domain_fusion.py` - Liquidity score calculation

---

## Next Actions

1. ✅ Complete BTC 2024 feature store rebuild with displacement fix
2. ⏸️ Test archetype system to verify displacement fix works
3. ⏸️ Investigate liquidity score calculation (may auto-fix)
4. ⏸️ Fix BOMS strength normalization
5. ⏸️ Test again with all P0+P1 fixes applied
6. ⏸️ (Optional) Add PTI trap classification and tf4h fusion (P2)

---

**Status**: Displacement fix complete, awaiting rebuild and test results.
