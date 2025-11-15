# MVP Phase 2 - Detector Wiring Status

## Summary

Goal: Ensure all MTF feature store detectors are **wired correctly** and returning **real values** from the engine's full knowledge, not defaults/placeholders.

**Status as of**: October 18, 2025

---

## ✅ COMPLETED WIRING (2/6 domains)

### 1. Wyckoff M1/M2 Integration ✅

**Files Modified**:
- `bin/build_mtf_feature_store.py` lines 44-49 (M1/M2 import)
- `bin/build_mtf_feature_store.py` lines 640-729 (enhanced precompute with M1/M2)
- `engine/wyckoff/wyckoff_engine.py` lines 151-189 (fixed None returns)

**Validation**:
```
BTC Q3 2024:
- 22 M1 (spring) signals detected ✓
- 28 M2 (markup) signals detected ✓
- 8 unique Wyckoff phases (was only 1) ✓
- Score range: [0.00, 0.79] with std=0.244 (was constant 0.5) ✓
```

**Result**: Wyckoff features are now **VARYING CORRECTLY** using advanced M1/M2 knowledge.

---

### 2. Macro Context (DXY, Yields, Oil, VIX) ✅

**Problem Found**:
- Macro CSV files exist and loaded successfully
- BUT: Feature store builder was passing wrong keys to `analyze_macro_echo()`
- Passed: `snapshot.get('dxy_series', ...)` (wrong key)
- Should pass: Series extracted from `macro_data['DXY']` DataFrame

**Fix Applied** (`bin/build_mtf_feature_store.py` lines 240-272):
```python
# OLD (broken):
macro_echo = analyze_macro_echo({
    'DXY': snapshot.get('dxy_series', pd.Series([100.0])),  # Wrong!
    'YIELDS_10Y': snapshot.get('yields_series', pd.Series([4.0])),  # Wrong!
    ...
})

# NEW (fixed):
def extract_macro_series(symbol: str, lookback_start_ts, end_ts) -> pd.Series:
    """Extract macro series for 7-day lookback window."""
    df = macro_data[symbol]
    lookback_naive = lookback_start_ts.replace(tzinfo=None)  # Handle tz mismatch
    end_naive = end_ts.replace(tzinfo=None)
    window = df[(df['timestamp'] >= lookback_naive) & (df['timestamp'] <= end_naive)]
    return window['value'].reset_index(drop=True)

macro_echo = analyze_macro_echo({
    'DXY': extract_macro_series('DXY', lookback_start, timestamp),
    'YIELDS_10Y': extract_macro_series('US10Y', lookback_start, timestamp),
    'OIL': extract_macro_series('WTI', lookback_start, timestamp),
    'VIX': extract_macro_series('VIX', lookback_start, timestamp)
}, lookback=7, config=config)
```

**Key Changes**:
1. Extract 7-day Series from macro_data instead of using snapshot
2. Handle timezone mismatch (macro CSVs are tz-naive, timestamps are tz-aware)
3. Properly map symbol names (US10Y → YIELDS_10Y, WTI → OIL)

**Expected Result**: Macro features should now vary across Q3 2024 based on real DXY/yields/oil/VIX trends.

**Validation Needed**: Rebuild feature store and check that:
- `macro_regime`: varies (risk_on, risk_off, neutral, crisis)
- `macro_dxy_trend`: varies (up, down, flat)
- `macro_yields_trend`: varies (up, down, flat)
- `macro_correlation_score`: varies (not constant)

---

## ⚠️ VERIFIED BUT RARE (1/6 domains)

### 3. BOMS (Break of Market Structure) ⚠️ RARE

**Status**: Detector is **WIRED CORRECTLY** but returns all False because **BTC Q3 2024 genuinely had 0 BOMS**.

**Diagnostic Results**:
```
BTC 1D Q3 2024 - BOMS Condition Funnel:
- Total bars tested: 92
- Swing breaks (Condition 1): 3 bars (3.3%) ✓
- Volume confirmed (Condition 2): 2 bars (66.7%) ✓
- FVG present (Condition 3): 2 bars (100%) ✓
- No reversal (Condition 4): 0 bars (0%) ❌

BOTTLENECK: Reversal check too strict
→ Both FVG-confirmed breaks reversed within 3 bars
→ This is VALID - BOMS requires sustained follow-through
```

**Why All False is Valid**:
- BOMS is a **rare, high-conviction signal** requiring ALL 4 conditions:
  1. Close beyond swing high/low ✓
  2. Volume > 1.8x mean ✓
  3. FVG left behind ✓
  4. No reversal for 3 bars ❌ (BTC Q3 2024 failed this)
- Expected frequency: 1-5 BOMS per quarter
- BTC Q3 2024 having 0 BOMS is **plausible** for a ranging/choppy period

**Recommendation**: Leave BOMS detector as-is. It's working correctly - just firing rarely (as designed).

**Files**: Already wired in `bin/build_mtf_feature_store.py` lines 200-203, 213-216.

---

## 🔍 PENDING INVESTIGATION (3/6 domains)

### 4. PTI (Psychology Trap Index) - IN PROGRESS

**Feature Store Calls** (`bin/build_mtf_feature_store.py` lines 231-238):
```python
# PTI on 1D (major reversal signals)
rsi_div = detect_rsi_divergence(window_1d, lookback=10)
vol_exh = detect_volume_exhaustion(window_1d, lookback=5)
features['tf1d_pti_score'] = (
    rsi_div.get('divergence_strength', 0.0) * 0.5 +
    vol_exh.get('exhaustion_score', 0.0) * 0.5
)
features['tf1d_pti_reversal'] = features['tf1d_pti_score'] > 0.7
```

**Current Status**: PTI functions are being called, but need to verify they return varying values.

**Next Steps**:
1. Test `detect_rsi_divergence()` on BTC Q3 2024 to see if it detects any divergences
2. Test `detect_volume_exhaustion()` similarly
3. If both return all 0.0, investigate detector logic
4. PTI also has wick trap and failed breakout detectors - check if those are wired

---

### 5. Range Outcomes (1D & 4H) - PENDING

**Feature Store Calls** (`bin/build_mtf_feature_store.py` lines 218-222):
```python
# Range outcome classification
range_outcome = classify_range_outcome(window_1d, timeframe='1D', config=config)
features['tf1d_range_outcome'] = range_outcome.outcome
features['tf1d_range_confidence'] = range_outcome.confidence
features['tf1d_range_direction'] = range_outcome.direction
```

**Current Status**: Function is called, need to verify if it's returning varying outcomes or always 'none'.

**Next Steps**:
1. Test `classify_range_outcome()` on BTC Q3 2024
2. Check what conditions are required for non-'none' outcomes
3. Similar to BOMS, this may be legitimately rare (only fires in range markets)

---

### 6. Structure Alignment / CHOCH / FVG (4H) - PENDING

**Feature Store Calls** (`bin/build_mtf_feature_store.py` lines ~350-400):
- `tf4h_structure_alignment`: Boolean
- `tf4h_choch_flag`: Boolean
- `tf4h_fvg_present`: Boolean

**Current Status**: Need to verify these are being called in 4H feature computation section.

**Next Steps**:
1. Locate 4H feature computation in `build_mtf_feature_store.py`
2. Verify structure alignment detector is called
3. Verify CHOCH (Change of Character) detector is called
4. Verify FVG (Fair Value Gap) detector is called

---

## Validation Plan

After completing all wiring:

1. **Rebuild Feature Store**:
   ```bash
   python3 bin/build_mtf_feature_store.py --asset BTC --start 2024-07-01 --end 2024-09-30
   ```

2. **Run Domain Analysis**:
   ```bash
   python3 test_feature_store_scores.py
   ```

3. **Expected Results**:
   - Wyckoff: VARYING ✓ (already validated)
   - Macro: VARYING (pending validation)
   - BOMS: ALL FALSE (valid - rare signal) ✓
   - PTI: VARYING (need to validate)
   - Range: VARYING or legitimately rare (need to validate)
   - Structure/CHOCH/FVG: VARYING (need to validate)

4. **Target**: 50-60/69 features (72-87%) varying, with remaining 10-20% being legitimately rare signals

---

## Impact on Optimizer

**Current State**:
- Optimizer assigns 15-44% weight to HOB/Liquidity
- But BOMS (part of liquidity) returns ALL ZEROS → limits fusion scores to 0.4-0.5

**After Wiring Macro/PTI/Range/Structure**:
- More domains will contribute non-zero values
- Fusion scores should reach 0.6-0.8 range (currently capped at 0.5)
- More trading signals (currently 3-20 trades/trial)
- Higher PNL potential (currently +$195-433 per quarter)

**Next Milestone**: Rebuild feature store with macro fix, validate PTI/Range/Structure, then re-run 200-trial optimizer sweep.
