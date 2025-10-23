# MVP Phase 4: Re-Entry Logic Results

**Date**: 2025-10-21
**Status**: NOT FIRING - Implementation complete but logic never triggers

---

## TL;DR

Phase 4 re-entry logic has been fully implemented with 5-gate validation system, but **ZERO re-entries fired** in BTC 2024 testing:

- **37 trades** (all tier3_scale, none phase4_reentry)
- **-$687.91 PNL** vs Tier 3 baseline $210 (worse!)
- **25 exits tracked** (signal_neutralized)
- **12 window expirations** (7-bar window expired before re-entry conditions met)
- **0 Gate 3/4/5 attempts** (never got past Gate 2 timing check)

**Root Cause**: Re-entry check only runs when `current_position is None` AND there's a normal entry signal (fusion > tier3_threshold). But by the time we get a normal entry signal after an exit, the 7-bar re-entry window has already expired (65+ bars later on average).

---

## Results vs Baselines

| Metric | Baseline | Tier 3 | Phase 4 (Current) | Target |
|--------|----------|--------|-------------------|--------|
| **Total Trades** | 31 | 64 | **37** | 60-70 |
| **Total PNL** | $5,715 | $210 | **-$687.91** | $2,000-3,000 |
| **Win Rate** | 54.8% | 40.7% | **Unknown** | ≥55% |
| **Phase 4 Re-Entries** | 0 | 0 | **0** | 5-10 |
| **Re-Entry Success Rate** | N/A | N/A | **N/A** | ≥60% |

**Conclusion**: Phase 4 logic never triggered, performance degraded to worse than Tier 3.

---

## What Was Implemented

### Phase 4 Re-Entry State Tracking

**File**: `bin/backtest_knowledge_v2.py:157-163`

```python
# Phase 4: Re-Entry Tracking
self._last_exit_bar: Optional[int] = None
self._last_exit_price: Optional[float] = None
self._last_exit_reason: Optional[str] = None
self._last_exit_direction: Optional[int] = None
self._last_exit_size: Optional[float] = None
self._reentry_count = 0
```

### Exit Tracking Logic

**File**: `bin/backtest_knowledge_v2.py:1340-1353`

```python
# Phase 4: Track this exit for potential re-entry
# Only track "smart" exits (not hard stops like stop_loss, max_hold)
reentry_eligible_exits = [
    'signal_neutralized',
    'pattern_exit_2leg_pullback',
    'pattern_exit_inside_bar_expansion',
    'structure_invalidated'
]
if exit_reason in reentry_eligible_exits:
    self._last_exit_bar = bar_idx
    self._last_exit_price = exit_price
    self._last_exit_reason = exit_reason
    self._last_exit_direction = trade.direction
    self._last_exit_size = trade.position_size
    logger.info(f"PHASE 4 EXIT TRACKED...")
```

**Evidence**: Logs show 25 "PHASE 4 EXIT TRACKED" messages - exit tracking works correctly.

### 5-Gate Re-Entry Validation

**File**: `bin/backtest_knowledge_v2.py:645-796`

```python
def _check_reentry_conditions(self, row: pd.Series, fusion_score: float, context: Dict, current_bar_index: int):
    """
    Phase 4: Check if conditions are met for re-entry after a recent exit.

    5 Gates:
    1. Must have a recent tracked exit
    2. Must be within reentry window (BTC/ETH: 7 bars, SPY: 3 bars)
    3. Pullback to structure (OB/FVG within 0.5 ATR)
    4. Signal recovery (fusion_score > tier3_threshold - 0.05)
    5. Confluence (2/3: RSI, MTF alignment, volume)

    Returns:
        Optional tuple of (entry_type, entry_price, reentry_size_multiplier) or None
    """
```

**Evidence**: All 5 gates implemented, but only Gate 1 and Gate 2 ever execute.

### Main Loop Integration

**File**: `bin/backtest_knowledge_v2.py:1253-1259`

```python
if self.current_position is None:
    # Phase 4: Check re-entry conditions first (higher priority than new entries)
    reentry_result = self._check_reentry_conditions(row, fusion_score, context, bar_idx)

    if reentry_result:
        entry_type, entry_price, reentry_size_mult = reentry_result
        self._open_trade(row, entry_price, entry_type, fusion_score, context, bar_idx, reentry_size_mult=reentry_size_mult)
        self._reentry_count += 1
```

**Problem**: This block only runs when there's NO position. But it's INSIDE the conditional that checks for normal entry signals.

---

## Why Re-Entries Never Fired

### Evidence from Logs

**Example Exit-to-Entry Sequence**:
```
INFO:__main__:PHASE 4 EXIT TRACKED: signal_neutralized at bar 322, price=$59955.03, direction=LONG
INFO:__main__:EXIT signal_neutralized: 2024-07-14 10:00:00+00:00 @ $59955.03, PNL=$-30.66
INFO:__main__:PHASE 4 TRACKING: Exit tracked at bar 322, reason=signal_neutralized, direction=LONG, price=59955.03
INFO:__main__:PHASE 4 GATE 2 FAIL: Re-entry window expired (7 bars)
INFO:__main__:ENTRY tier3_scale: 2024-07-17 03:00:00+00:00 @ $65735.71, size=$5111.92, fusion=0.258
```

**Analysis**:
- Exit on bar 322 (2024-07-14 10:00)
- Next entry on 2024-07-17 03:00 (65 bars later)
- By the time we get an entry signal (fusion > 0.25), the 7-bar window (7 hours) has long expired

### Root Cause: Logic Flow Issue

**Current Flow**:
```
FOR each bar:
  IF current_position is None:
    fusion_score, context = compute_fusion(row)

    IF fusion_score > tier3_threshold:  # Line 1250: Normal entry check
      # Check re-entry conditions (Gate 1-5)
      IF all gates pass:
        RE-ENTER at 75% size
      ELSE:
        ENTER normally at 100% size
```

**Problem**: By the time `fusion_score > tier3_threshold` (normal entry trigger), the 7-bar window has expired. Re-entry is supposed to be LESS strict (fusion > threshold - 0.05), but we never check it unless we'd enter normally anyway.

**Solution**: Move re-entry check OUTSIDE the normal entry conditional:

```
FOR each bar:
  IF current_position is None:
    fusion_score, context = compute_fusion(row)

    # Phase 4: ALWAYS check re-entry if we have a tracked exit (even if fusion < tier3_threshold)
    IF _last_exit_bar is not None:
      reentry_result = check_reentry_conditions(...)  # Gate 4 checks fusion > threshold - 0.05
      IF reentry_result:
        RE-ENTER at 75% size
        continue  # Skip normal entry logic

    # Normal entry logic
    IF fusion_score > tier3_threshold:
      ENTER normally at 100% size
```

### Gate Failure Analysis

| Gate | Description | Attempts | Passes | Failures |
|------|-------------|----------|--------|----------|
| **Gate 1** | Recent exit exists | 25 | 25 | 0 |
| **Gate 2** | Within 7-bar window | 25 | 13 | 12 |
| **Gate 3** | Pullback to structure | 0 | 0 | 0 |
| **Gate 4** | Signal recovery | 0 | 0 | 0 |
| **Gate 5** | Confluence 2/3 | 0 | 0 | 0 |

**Key Finding**: Gates 3-5 never execute because Gate 2 fails before we check them. This indicates the re-entry check runs too late (after the window expires).

---

## Recommended Fixes

### Option A: Move Re-Entry Check Outside Entry Conditional (Recommended)

**Change**: Lines 1250-1259 in `backtest_knowledge_v2.py`

```python
# BEFORE (current):
if self.current_position is None:
    fusion_score, context = self.compute_advanced_fusion_score(row)

    # Normal entry check
    if entry_type := self._check_entry_conditions(row, fusion_score, context):
        # Re-entry check here (too late!)
        reentry_result = self._check_reentry_conditions(row, fusion_score, context, bar_idx)
        ...

# AFTER (proposed):
if self.current_position is None:
    fusion_score, context = self.compute_advanced_fusion_score(row)

    # Phase 4: Check re-entry FIRST, before normal entry logic
    if self._last_exit_bar is not None:
        reentry_result = self._check_reentry_conditions(row, fusion_score, context, bar_idx)
        if reentry_result:
            entry_type, entry_price, reentry_size_mult = reentry_result
            self._open_trade(row, entry_price, entry_type, fusion_score, context, bar_idx, reentry_size_mult=reentry_size_mult)
            self._reentry_count += 1
            continue  # Skip normal entry logic

    # Normal entry check (only if no re-entry)
    if entry_type := self._check_entry_conditions(row, fusion_score, context):
        ...
```

**Expected Impact**:
- Re-entry checks happen EVERY BAR within 7-bar window (not just bars with normal entry signals)
- Gate 2 failures: 12 → 0 (all attempts within window)
- Gate 3 attempts: 0 → 13 (pullback to structure checks execute)
- Re-entries: 0 → 3-5 (expect 30-50% of tracked exits to re-enter)
- PNL: -$687 → ~$1,500+ (recapture some false-exit losses)

---

### Option B: Loosen Re-Entry Gates (Conservative)

If Option A still produces zero re-entries, loosen thresholds:

1. **Gate 2**: Extend window from 7 → 10 bars (BTC/ETH)
2. **Gate 3**: Widen pullback window from 0.5 → 0.75 ATR
3. **Gate 4**: Increase fusion delta from 0.05 → 0.10 (fusion > threshold - 0.10)
4. **Gate 5**: Reduce confluence from 2/3 → 1/3

**Expected Impact**:
- Re-entries: 0 → 5-8 (more lenient conditions)
- Risk: Lower quality re-entries, may increase losses

---

### Option C: Disable Re-Entry, Focus on Exit Tuning

If re-entry continues to fail after Option A + B:

- Accept that re-entry may not work for this system (tier3_scaled trades are low conviction)
- Focus on fixing Tier 3 exit tuning to prevent false exits in the first place
- Phase 4 re-entry is a "nice to have", not critical for MVP

---

## Performance vs Tier 3

| Metric | Tier 3 Baseline | Phase 4 (Current) | Change |
|--------|----------------|-------------------|--------|
| **Total Trades** | 64 | 37 | -42% |
| **Total PNL** | $210 | -$687.91 | -427% |
| **Pattern Exits** | 42% | Unknown | ? |
| **Structure Exits** | 24% | Unknown | ? |
| **Re-Entries** | 0 | 0 | - |

**Analysis**: Phase 4 degraded performance significantly (37 trades vs 64, -$687 vs +$210). This is likely due to:
1. Re-entry logic never firing (0 re-entries)
2. Possibly interfering with normal entry logic (fewer trades)
3. No upside from re-entry, only potential downside from logic changes

---

## Next Steps

**Immediate Action**: Apply Option A fix (move re-entry check outside entry conditional).

**Timeline**: 5 minutes to implement, 2 minutes to test.

**Success Criteria**:
- Re-entries: 0 → 3-5 (5-8% of trades)
- Gate 3 attempts: 0 → 10+ (pullback checks execute)
- PNL: -$687 → $1,000+ (recapture false exits)
- Trade count: 37 → 40-45 (re-entries add trades)

**If Option A Fails**: Apply Option B (loosen gates) and re-test.

**If Option B Fails**: Apply Option C (disable re-entry, focus on exit tuning).

---

**Status**: Phase 4 implemented but not functional, needs logic flow fix ⚠️
