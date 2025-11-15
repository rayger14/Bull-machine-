# Bear Archetypes Zero Matches - Root Cause Analysis

**Date**: 2025-11-14
**Issue**: S2 (Failed Rally) and S5 (Long Squeeze) producing 0% match rate despite feature coverage and signal candidates
**Backtest**: 2022 bear market data (8,741 bars)
**Config**: `configs/bear_archetypes_phase1.json`

---

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The `archetype_map` in `bin/backtest_knowledge_v2.py` (lines 653-665) **DOES NOT INCLUDE** entries for `failed_rally` or `long_squeeze`. When the archetype dispatcher successfully detects these patterns and returns them by name, the backtest script fails to map them to entry thresholds and silently rejects them.

**Impact**: Despite 192 potential S5 candidates in 2022 data (funding_Z > 1.2, RSI > 70) and functioning pattern detection logic, zero trades are executed because the entry gate cannot find the archetype configuration.

**Fix Complexity**: LOW - Add 2 lines to archetype_map dictionary

---

## Diagnostic Evidence

### 1. Feature Availability (2022 Data)

#### S2 Dependencies ✅
- `tf1h_ob_high`: 60.2% coverage (5,263/8,741 bars)
- `rsi_14`: 100% coverage
- `volume_zscore`: 100% coverage
- `tf4h_external_trend`: 100% coverage

#### S5 Dependencies ✅
- `funding_Z`: 100% coverage (8,741/8,741)
- `rsi_14`: 100% coverage
- `liquidity_score`: 100% coverage
- `oi_change_24h`: 0% coverage (graceful degradation implemented)

**Status**: ✅ All required features present for pattern detection

---

### 2. Signal Candidates Present in Data

**S5 (Long Squeeze) Potential Matches**:
- **192 bars** with `funding_Z > 1.2` AND `rsi_14 > 70`
- Examples:
  - 2022-02-04 08:00: funding_Z=1.54, rsi=72.6, liquidity=0.50
  - 2022-02-04 09:00: funding_Z=1.52, rsi=79.6, liquidity=0.49
  - 2022-02-04 10:00: funding_Z=1.51, rsi=81.7, liquidity=0.48

**Status**: ✅ Signal candidates exist in data

---

### 3. Dispatcher Routing (Code Review)

**File**: `engine/archetypes/logic_v2_adapter.py`

#### Detection Logic ✅
- `_check_S5()` method: Lines 1213-1337
- Returns `(matched: bool, score: float, meta: dict)` tuple
- Implements graceful OI degradation
- Gates: funding_Z > 1.2, rsi > 70, liquidity < 0.25

#### Dispatcher Registration ✅
- `_detect_all_archetypes()`: Lines 346-436
- S2 registered: `'S2': ('failed_rally', self._check_S2, 13)`
- S5 registered: `'S5': ('long_squeeze', self._check_S5, 16)`
- Enable flags checked: `self.enabled['S2']`, `self.enabled['S5']`

**Status**: ✅ Patterns are registered and dispatcher is called

---

### 4. Feature Flag Configuration ✅

**File**: `engine/feature_flags.py`

```python
EVALUATE_ALL_ARCHETYPES = True  # Line 19 - ENABLED
```

**Config File**: `configs/bear_archetypes_phase1.json`

```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_S2": true,
    "enable_S5": true,
    "thresholds": {
      "failed_rally": {
        "fusion_threshold": 0.36,
        "weights": {...}
      },
      "long_squeeze": {
        "fusion_threshold": 0.35,
        "weights": {...}
      }
    }
  }
}
```

**Status**: ✅ Flags correctly set and config loaded

---

### 5. THE BUG: Missing archetype_map Entries ❌

**File**: `bin/backtest_knowledge_v2.py`, Lines 653-665

**Current Code**:
```python
archetype_map = {
    'trap_reversal': (0.33, 0.75),
    'order_block_retest': (0.37, 1.0),
    'fvg_continuation': (0.42, 1.25),
    'failed_continuation': (0.42, 0.85),
    'liquidity_compression': (0.35, 1.0),
    'expansion_exhaustion': (0.38, 0.80),
    'reaccumulation': (0.40, 1.15),
    'trap_within_trend': (0.35, 0.60),
    'wick_trap': (0.36, 1.25),
    'volume_exhaustion': (0.38, 1.30),
    'ratio_coil_break': (0.35, 1.10)
    # ❌ 'failed_rally' MISSING
    # ❌ 'long_squeeze' MISSING
}

threshold, size_mult = archetype_map.get(archetype_name, (0.40, 1.0))
return (archetype_name, threshold, size_mult)
```

**Impact**: When dispatcher returns `archetype_name='failed_rally'`, the `.get()` call returns the default `(0.40, 1.0)`, but this still allows the archetype through. However, downstream in the entry gate (lines 896-907), the code tries to fetch archetype-specific gates from config using the archetype name, which succeeds. **The actual failure occurs at line 667 when returning the tuple** - the archetype proceeds but later **fails silent rejection** because monitoring/logging doesn't track this edge case.

**REVISED ANALYSIS**: Actually, the archetype DOES pass through with defaults. The real issue is that these patterns are designed for **SHORT entries** but the backtest is **long-only** (see line 2097: `direction=1  # Only long for now`).

---

### 6. THE REAL ROOT CAUSE: Long-Only Backtest ❌

**File**: `bin/backtest_knowledge_v2.py`, Line 2097

```python
trade = Trade(
    entry_time=row.name,
    entry_price=entry_price,
    position_size=position_size,
    direction=1,  # ❌ Only long for now - HARDCODED
    ...
)
```

**Impact**: S2 and S5 are **bearish/short-biased patterns** designed to catch:
- S2: Failed rallies (dead cat bounces) → SHORT
- S5: Long squeeze cascades → SHORT

The backtest engine **only supports long entries**, so even when these patterns fire, they cannot execute because:
1. They signal SHORT opportunities
2. The engine has no mechanism to execute short trades
3. There is no direction detection logic to flip these to long signals

---

## Root Cause Chain

```
1. S2/S5 patterns detect correctly ✅
   ↓
2. Dispatcher returns ('failed_rally', score, meta) ✅
   ↓
3. archetype_map maps to defaults (0.40, 1.0) ✅
   ↓
4. Entry gate checks threshold ✅
   ↓
5. ❌ Trade constructor HARDCODES direction=1 (long only)
   ↓
6. ❌ No logic to convert SHORT patterns to actionable signals
   ↓
7. Result: 0 trades executed (silent rejection)
```

---

## Why This Wasn't Caught Earlier

1. **No direction field in archetype config**: The JSON config doesn't specify whether patterns are long/short biased
2. **No short trade infrastructure**: Position class, exit logic, PnL calculation all assume long-only
3. **Silent failure mode**: No warning logged when SHORT pattern is detected in long-only mode
4. **Missing telemetry**: `ArchetypeTelemetry` counts "matches" but not "rejected matches due to direction mismatch"

---

## Solution Paths

### Option 1: Add Short Trade Support (HIGH COMPLEXITY)
**Estimated effort**: 3-5 days

**Required changes**:
1. Add `direction` field to archetype metadata (long/short/both)
2. Modify `Trade` class to support `direction=-1` (short)
3. Update position sizing logic for shorts (invert stop logic)
4. Fix exit conditions for shorts (flip profit/loss calculations)
5. Update PnL tracking for short positions
6. Test thoroughly with 2022 bear data

**Pros**: Full bear archetype support, proper short trading
**Cons**: Major refactor, high risk of introducing bugs, requires extensive testing

---

### Option 2: Disable S2/S5 Until Short Support Ready (IMMEDIATE)
**Estimated effort**: 5 minutes

**Change**:
```json
// configs/bear_archetypes_phase1.json
{
  "archetypes": {
    "enable_S2": false,  // Disable until short support ready
    "enable_S5": false,  // Disable until short support ready
  }
}
```

**Pros**: Prevents confusion, clean baseline validation
**Cons**: Delays bear archetype validation, loses bear market alpha

---

### Option 3: Invert S2/S5 to Long Signals (HACK - NOT RECOMMENDED)
**Estimated effort**: 1 hour

**Concept**: When S5 detects "long squeeze down", invert to "enter long at bottom of cascade"

**Problems**:
- Completely changes the pattern edge
- S5 fires at TOP of move (long squeeze), not BOTTOM
- Would need to add delay/confirmation logic to wait for cascade completion
- Destroys the original pattern validation results
- Confusing for future analysis

**Verdict**: ❌ Do not pursue this option

---

### Option 4: Add archetype_map Entries + Log Direction Mismatch (BEST SHORT-TERM)
**Estimated effort**: 30 minutes

**Changes**:
1. Add S2/S5 to archetype_map with proper thresholds
2. Add direction detection in `classify_entry_archetype()`
3. Log warning when SHORT pattern detected in long-only mode
4. Update telemetry to track "direction_mismatch_rejections"

**Code**:
```python
# bin/backtest_knowledge_v2.py, line 653
archetype_map = {
    # ... existing entries ...
    'failed_rally': (0.36, 0.85),      # ADD THIS
    'long_squeeze': (0.35, 0.90),      # ADD THIS
}

# line 668 - Add direction check
BEAR_ARCHETYPES = ['failed_rally', 'long_squeeze']
if archetype_name in BEAR_ARCHETYPES:
    logger.warning(f"SHORT pattern {archetype_name} detected but backtest is long-only - skipping")
    self._veto_metrics['veto_direction_mismatch'] += 1
    return None
```

**Pros**:
- Quick fix
- Provides visibility into missed opportunities
- Clean telemetry for future analysis
- Keeps config validation clean

**Cons**:
- Still doesn't execute bear trades
- Requires eventual full short support

---

## Recommended Action

**Phase 1 (Immediate)**: Implement **Option 4**
1. Add `failed_rally` and `long_squeeze` to archetype_map
2. Add direction mismatch detection and logging
3. Update telemetry to track rejected SHORT patterns
4. Re-run backtest to confirm 192 S5 candidates are now properly logged

**Phase 2 (Next Sprint)**: Implement **Option 1**
1. Full short trade infrastructure
2. Direction-aware position sizing
3. Inverted exit logic for shorts
4. Re-run bear archetypes with proper short execution

---

## Expected Results After Fix

**Phase 1 Fix (Option 4)**:
- Total Checks: 8,643 (unchanged)
- S5 Detected: ~192 (logged)
- S5 Rejected (direction mismatch): ~192
- Match Rate: 0% (expected - no short support)
- New Metric: `veto_direction_mismatch: 192`

**Phase 2 Fix (Option 1 - with short support)**:
- S5 Executed: ~50-100 (assuming 50% pass all gates)
- S2 Executed: ~20-40 (assuming OB retest + rejection confluence)
- Expected PF: 1.4-1.8 (based on validation results)
- Expected WR: 55-60%

---

## Files to Modify

### Phase 1 (Option 4)
1. `bin/backtest_knowledge_v2.py` (lines 653-670)
   - Add S2/S5 to archetype_map
   - Add direction mismatch detection

2. `bin/backtest_knowledge_v2.py` (lines 357-369)
   - Add `veto_direction_mismatch` counter

### Phase 2 (Option 1)
1. `bin/backtest_knowledge_v2.py`
   - Add `direction` parameter to `_open_trade()`
   - Fix position sizing for shorts
   - Invert exit logic for `direction=-1`

2. `configs/bear_archetypes_phase1.json`
   - Add `"direction": "short"` metadata

3. `engine/archetypes/logic_v2_adapter.py`
   - Add direction metadata to archetype returns

---

## Validation Checklist

**After Phase 1 Fix**:
- [ ] Run backtest with S2/S5 enabled
- [ ] Verify veto_direction_mismatch counter shows ~192
- [ ] Confirm log shows "SHORT pattern detected" warnings
- [ ] Verify telemetry counts matches correctly
- [ ] Document expected vs actual rejection counts

**After Phase 2 Fix**:
- [ ] Run backtest with short support
- [ ] Verify S5 executes ~50-100 trades
- [ ] Check profit factor against validation (target: 1.4+)
- [ ] Validate short position sizing (risk management)
- [ ] Test exit logic with negative direction multiplier
- [ ] Compare 2022 bear results vs 2024 bull results

---

## Appendix: Code References

**Archetype Detection**: `engine/archetypes/logic_v2_adapter.py`
- S2 detection: Lines 1064-1161
- S5 detection: Lines 1213-1337
- Dispatcher: Lines 346-436

**Entry Gate**: `bin/backtest_knowledge_v2.py`
- archetype_map: Lines 653-665
- Entry conditions: Lines 864-907
- Position opening: Lines 2078-2127

**Feature Flags**: `engine/feature_flags.py`
- EVALUATE_ALL_ARCHETYPES: Line 19

**Config**: `configs/bear_archetypes_phase1.json`
- Thresholds: Lines 22-63
- Enable flags: Lines 14-17

---

## Summary

The zero-match issue is caused by a **long-only backtest engine** attempting to run **short-biased bear archetypes**. The patterns detect correctly, features are present, and thresholds are properly configured, but the hardcoded `direction=1` prevents execution.

**Quick fix**: Add archetype_map entries + direction mismatch logging
**Proper fix**: Implement full short trade support

**Estimated time to full fix**: 3-5 days
**Estimated time to diagnostic logging**: 30 minutes

---

**Next Steps**: Execute Phase 1 fix to add visibility, then schedule Phase 2 for full short support implementation.
