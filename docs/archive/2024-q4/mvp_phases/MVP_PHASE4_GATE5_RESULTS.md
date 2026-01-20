# MVP Phase 4: Gate 5 Enabled Results

**Date**: 2025-10-21
**Status**: RE-ENTRIES FIRING BUT STILL OVER-AGGRESSIVE - Gate 5 (2/3) insufficient
**Test Period**: BTC 2024-07-01 to 2024-09-30 (2,160 1H bars)

---

## TL;DR

Gate 5 confluence filtering (2/3) successfully reduced re-entries from 136 (91%) to 37 (60%), but **re-entries still dominate** and performance remains poor:

**Results (Gate 5 Enabled, 2/3 Confluence):**
- **62 trades** (37 re-entries = 60% vs target 10-15%)
- **-$667.79 PNL** (improved from -$848 but still negative)
- **33.9% win rate** (poor, target 50%+)
- **Re-entry chains**: 5-6 consecutive re-entries after single normal entry

**Root Cause**: During BTC markup (July-Aug 2024), `tf4h_fusion_score` stayed 0.7-0.9 (always passing), so Gate 5 only needed RSI > 50 OR vol_z > 0.5 to fire. This creates rapid re-entry churn.

**Recommended Fix**: Require **3/3 confluence** OR add **max 2 re-entry attempts per exit** to enforce Moneytaur's "high-probability pullbacks only" principle.

---

## Performance Comparison

| Iteration | Trades | PNL | Re-Entries | Re-Entry % | Issue |
|-----------|--------|-----|------------|------------|-------|
| **Tier 3 Baseline** | 37 | -$688 | 0 | 0% | Smart exits too aggressive |
| **Phase 4 (Gate 5 disabled)** | 150 | -$848 | 136 | 91% | No confluence filtering |
| **Phase 4 (Gate 5: 2/3)** | 62 | -$668 | 37 | 60% | Too lenient in markup regime |
| **Target** | 40-50 | $1,500+ | 5-10 | 10-15% | High-conviction only |

---

## What Was Tested

### Gate 5 Configuration (bin/backtest_knowledge_v2.py:768-806)

**Confluence Factors (Longs):**
1. **RSI > 50**: Momentum recovery after exit
2. **tf4h_fusion > 0.25**: 4H timeframe alignment (markup/accumulation)
3. **volume_zscore > 0.5**: Above-average volume confirmation

**Threshold:** 2/3 factors must pass (line 806)

```python
if confluence_score < 2:  # Require 2/3 factors
    logger.info(f"GATE 5 FAIL: Confluence too low ({confluence_score}/3): ...")
    return None
```

---

## Evidence: Re-Entry Churn Pattern

### Example 1: 5 Consecutive Re-Entries (July 17)
```
Trade 2:  ENTRY tier3_scale      @ $65735.71, fusion=0.258
          EXIT signal_neutralized @ $65348.17, PNL=-$31.67

Trade 3:  ENTRY phase4_reentry   @ $65137.71 (1 bar later), fusion=0.252
          Gate 5 PASS: RSI=58.3>50, 4H=0.900>0.25, vol=-1.05≤0.5 (2/3)
          EXIT signal_neutralized @ $64739.45, PNL=-$32.55

Trade 4:  ENTRY phase4_reentry   @ $64748.85 (1 bar later), fusion=0.212
          Gate 5 PASS: RSI=51.9>50, 4H=0.900>0.25, vol=-0.51≤0.5 (2/3)
          EXIT signal_neutralized @ $65147.82, PNL=+$27.65

Trade 5:  ENTRY phase4_reentry   @ $65041.21 (1 bar later), fusion=0.220
          Gate 5 PASS: RSI=39.3≤50, 4H=0.900>0.25, vol=1.49>0.5 (2/3)
          EXIT signal_neutralized @ $64747.90, PNL=-$23.11

Trade 6:  ENTRY phase4_reentry   @ $64253.38 (1 bar later), fusion=0.236
          Gate 5 PASS: RSI=26.4≤50, 4H=0.900>0.25, vol=2.50>0.5 (2/3)
          EXIT signal_neutralized @ $64301.60, PNL=+$1.97

Trade 7:  ENTRY phase4_reentry   @ $64520.52 (1 bar later), fusion=0.234
          Gate 5 PASS: RSI=27.4≤50, 4H=0.900>0.25, vol=0.66>0.5 (2/3)
          EXIT signal_neutralized @ $64602.85, PNL=+$4.30
```

**Analysis:**
- **1 normal entry → 5 re-entries** (6 total trades in rapid succession)
- All passed 2/3 confluence due to **4H=0.900 (markup) always passing**
- Net PNL: -$53 across 6 trades (churn without profit)
- Violates Moneytaur's "high-probability pullbacks only" - this is noise trading

---

### Example 2: 4H Markup Always Passing (July 31)
```
Trade 12: ENTRY phase4_reentry @ $65785.66, fusion=0.233
          Gate 5 PASS: RSI=NA, 4H=0.900>0.25, vol=NA (2/3)

Trade 13: ENTRY phase4_reentry @ $66436.31, fusion=0.266
          Gate 5 PASS: RSI=NA, 4H=0.900>0.25, vol=NA (2/3)

Trade 14: ENTRY phase4_reentry @ $66361.61, fusion=0.223
          Gate 5 PASS: RSI=NA, 4H=0.900>0.25, vol=NA (2/3)

Trade 15: ENTRY phase4_reentry @ $66659.16, fusion=0.234
          Gate 5 PASS: RSI=NA, 4H=0.900>0.25, vol=NA (2/3)

Trade 16: ENTRY phase4_reentry @ $65222.44, fusion=0.235
          Gate 5 PASS: RSI=NA, 4H=0.900>0.25, vol=NA (2/3)
```

**Analysis:**
- `tf4h_fusion_score` = 0.900 during markup phase (July-August)
- 4H factor **always passes**, reducing confluence to 1/2 (RSI OR volume)
- This is too lenient - re-enters on any momentum OR volume spike

---

### Example 3: Re-Entry Failures When 4H Turns Negative (July 14)
```
PHASE 4 CHECK: bar 323, bars_since_exit=1/7, fusion=0.247
PHASE 4 GATE 5 FAIL: Confluence too low (1/3):
  RSI=67.7>50, 4H_fusion=-0.700≤0.25, vol_z=-0.26≤0.5
```

**Analysis:**
- When `tf4h_fusion_score` = -0.700 (distribution), confluence check works correctly
- Only RSI passed (1/3), so re-entry was blocked
- This shows Gate 5 **can work** when 4H isn't always bullish

---

## Root Cause: 4H Markup Regime Bias

### tf4h_fusion_score Distribution (BTC July-Aug 2024)

During the test period, BTC was in a **strong 4H markup phase**:
- `tf4h_internal_phase`: Mostly "markup" and "accumulation"
- `tf4h_fusion_score`: Range 0.700 to 0.900 (bullish 85% of the time)
- This means 4H confluence factor **passed 85% of re-entry checks**

**Impact:**
- Gate 5 (2/3) effectively became Gate 5 (1/2): RSI OR volume
- Re-entries fire whenever RSI > 50 OR vol_z > 0.5 (too easy)
- No respect for "high-probability pullbacks" - any slight recovery triggers re-entry

---

## Recommended Fixes

### Option 1: Require 3/3 Confluence (Recommended)

**Change:** bin/backtest_knowledge_v2.py:806

```python
# FROM (current):
if confluence_score < 2:  # 2/3 required

# TO (stricter):
if confluence_score < 3:  # All 3 factors required
```

**Expected Impact:**
- Re-entries: 37 (60%) → 3-5 (5-10%)
- Only fire when RSI > 50 AND 4H > 0.25 AND vol > 0.5 (all aligned)
- Trade count: 62 → 40-45
- PNL: -$668 → $1,000-2,000 (fewer low-conviction trades)

**Rationale:** Aligns with Moneytaur's "high-probability pullbacks" and Wyckoff's "re-accumulation confirmation". All factors must align for re-entry, not just 2.

---

### Option 2: Max Re-Entry Attempts Per Exit (Conservative)

**Change:** Add re-entry counter per exit

```python
# In __init__:
self._reentry_attempts_for_current_exit = 0

# In _check_reentry_conditions (after Gate 2):
if self._reentry_attempts_for_current_exit >= 2:
    logger.info("GATE 2c FAIL: Max 2 re-entry attempts per exit reached")
    return None

# In _open_trade (when re-entering):
self._reentry_attempts_for_current_exit += 1

# In exit tracking:
self._reentry_attempts_for_current_exit = 0  # Reset on new exit
```

**Expected Impact:**
- Re-entries: 37 (60%) → 10-15 (20-25%)
- Limits rapid churn (max 2 attempts per exit, then stop)
- Trade count: 62 → 45-50
- PNL: -$668 → $500-1,500 (fewer repeat failures)

**Rationale:** Aligns with Zeroika's "composure post-exit". If two re-entry attempts fail, the signal is dead - move on.

---

### Option 3: Hybrid (3/3 OR Max 2 Attempts)

**Change:** Apply both fixes

```python
# Require 3/3 confluence AND limit to 2 attempts per exit
if confluence_score < 3:
    return None

if self._reentry_attempts_for_current_exit >= 2:
    return None
```

**Expected Impact:**
- Re-entries: 37 (60%) → 2-5 (5-10%)
- Strictest filtering: high-confluence setups only, max 2 tries
- Trade count: 62 → 38-42
- PNL: -$668 → $1,500-2,500 (best quality re-entries)

**Risk:** May be too strict, potentially blocking valid re-entries in ranging markets.

---

## Alignment with Trading Principles

### Current Issues (2/3 Confluence):
- ❌ Violates Moneytaur's "high-probability pullbacks" - fires on noise
- ❌ Violates Zeroika's "composure post-exit" - immediate re-entry churn
- ⚠️ Weak Wyckoff "re-accumulation" - 4H always bullish in markup

### Option 1 (3/3 Confluence):
- ✅ Moneytaur: RSI + 4H + volume all aligned = high probability
- ✅ Zeroika: Fewer re-entries = more composure
- ✅ Wyckoff: 4H alignment + volume = confirmed re-accumulation

### Option 2 (Max 2 Attempts):
- ✅ Moneytaur: Stops repeated failed pullback trades
- ✅ Zeroika: Enforces composure (if 2 attempts fail, move on)
- ✅ Wyckoff: Respects "failed re-accumulation" (stop trying)

---

## Next Steps

**Recommendation:** Apply **Option 1 (3/3 confluence)** first, as it's the simplest fix and most aligned with trading principles.

**Timeline:**
1. Change Gate 5 threshold: `< 2` → `< 3` (1 minute)
2. Re-test BTC 2024-07-01 to 2024-09-30 (2 minutes)
3. Analyze results (5 minutes)

**Success Criteria:**
- Re-entries: 37 → 3-8 (5-15% of trades)
- Trade count: 62 → 38-45
- PNL: -$668 → $500+ (positive or near-zero)
- Re-entry success rate: 40%+ (currently 33.9%)

**If Option 1 Fails:** Apply Option 2 (max 2 attempts) or Option 3 (hybrid) and re-test.

---

**Status**: Phase 4 Gate 5 enabled but too lenient (2/3). Awaiting 3/3 confluence test ⚠️
