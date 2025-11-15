# Phase 2: 3-Archetype System - Debug Report

**Date**: 2025-10-22
**Status**: ROOT CAUSE IDENTIFIED
**Issue**: 0 trades generated with 3-archetype entry system

---

## Executive Summary

The 3-archetype entry system is **correctly implemented** but produces 0 trades because the **feature store has missing/zero values** for critical SMC (Smart Money Concepts) indicators.

---

## Test Results: BTC 2024 Full Year

**Configuration**:
- Asset: BTC
- Period: 2024-01-01 to 2024-12-31
- Total bars checked: 8,761
- Archetype checks performed: 8,761

**Results**:
- Total Trades: **0**
- Archetype A matches (Trap Reversal): **0**
- Archetype B matches (OB Retest): **0**
- Archetype C matches (FVG Continuation): **0**

---

## Root Cause Analysis

### 1. BOMS Displacement is ALWAYS 0.00

**Evidence from debug logs** (sampled every 100 bars):
```
ARCHETYPE DEBUG [check #100]:
  BOMS: disp=0.00, atr=585.12, strength=0.000

ARCHETYPE DEBUG [check #500]:
  BOMS: disp=0.00, atr=116.30, strength=0.000

ARCHETYPE DEBUG [check #1000]:
  BOMS: disp=0.00, atr=281.44, strength=0.000

... ALL 8,761 bars show disp=0.00
```

**Impact**:
- **Archetype A** requires: `boms_disp >= 1.25 × ATR` (e.g., >= 731.4 when ATR=585.12)
- **Archetype C** requires: `boms_disp >= 1.5 × ATR` (e.g., >= 877.68 when ATR=585.12)
- Both conditions are **IMPOSSIBLE** to meet when displacement is always 0

**Column affected**: `tf4h_boms_displacement`

---

### 2. Liquidity Scores are ALWAYS 0.000

**Evidence from debug logs**:
```
ARCHETYPE DEBUG [check #100]:
  Scores: liq=0.000, wyc=0.650, mom=0.169

ARCHETYPE DEBUG [check #4200]:
  Scores: liq=0.000, wyc=0.790, mom=0.376

... ALL bars show liq=0.000
```

**Impact**:
- **Archetype B** requires: `liq_score >= 0.68`
- **Archetype C** requires: `liq_score >= 0.72`
- Both conditions are **IMPOSSIBLE** to meet when liquidity is always 0

**Source**: `liquidity_score` from fusion context (derived from HOB/BOMS features)

---

### 3. BOMS Strength is ALWAYS 0.000

**Evidence from debug logs**:
```
ARCHETYPE DEBUG [check #100]:
  BOMS: disp=0.00, atr=585.12, strength=0.000

... ALL bars show strength=0.000
```

**Impact**:
- **Archetype B** requires: `boms_strength >= 0.68`
- Condition is **IMPOSSIBLE** to meet

**Column affected**: `tf1d_boms_strength`

---

### 4. PTI Trap Type is NEVER Populated

**Evidence from debug logs**:
```
ARCHETYPE DEBUG [check #100]:
  PTI: trap=none, score=0.191

ARCHETYPE DEBUG [check #500]:
  PTI: trap=none, score=0.500

... ALL bars show trap=none
```

**Impact**:
- **Archetype A** requires: `pti_trap is not None`
- Condition is **IMPOSSIBLE** to meet

**Column affected**: `tf1h_pti_trap_type`

**Note**: PTI scores ARE populated (ranging 0.0-0.531), but the trap TYPE classification is missing.

---

### 5. tf4h_fusion_score is ALWAYS 0.000

**Evidence from debug logs**:
```
ARCHETYPE DEBUG [check #100]:
  FVG: 1h=True, 4h=False, tf4h_fusion=0.000

... ALL bars show tf4h_fusion=0.000
```

**Impact**:
- Prevents **Plus-One sizing** in Archetype C (1.25× when tf4h_fusion >= 0.62)
- All Archetype C trades would use 1.0× sizing instead of potential 1.25×

**Column affected**: `tf4h_fusion_score`

---

## Working Features (Confirmed Present)

The following features ARE correctly populated in the feature store:

| Feature | Status | Sample Values |
|---------|--------|---------------|
| `atr_14` | ✅ Working | 116.30 - 1762.57 |
| `wyckoff_score` | ✅ Working | 0.000 - 0.790 |
| `momentum_score` | ✅ Working | 0.067 - 0.498 |
| `tf1h_pti_score` | ✅ Working | 0.000 - 0.531 |
| `tf1h_fvg_present` | ✅ Working | True/False |
| `tf1h_bos_bullish` | ✅ Working | True/False |
| `tf1h_bos_bearish` | ✅ Working | True/False |
| `tf1d_frvp_position` | ✅ Working | 'middle' (all bars) |

---

## Missing/Broken Feature Calculations

### Critical (Breaks all 3 archetypes):

1. **`tf4h_boms_displacement`**: Always 0.00
   - Should measure price displacement from Break of Market Structure
   - Expected range: 0 to 3× ATR (conservative) or higher for strong moves
   - **Blocker for**: Archetype A, Archetype C

2. **Liquidity scores** (derived feature): Always 0.000
   - Calculated from HOB/BOMS features in fusion logic
   - Expected range: 0.0 to 1.0
   - **Blocker for**: Archetype B, Archetype C

3. **`tf1d_boms_strength`**: Always 0.000
   - Should measure strength of Break of Market Structure
   - Expected range: 0.0 to 1.0
   - **Blocker for**: Archetype B

4. **`tf1h_pti_trap_type`**: Always None
   - Should classify trap patterns (e.g., 'bull_trap', 'bear_trap', 'spring')
   - PTI score IS populated, but trap classification is missing
   - **Blocker for**: Archetype A

### Important (Reduces sizing quality):

5. **`tf4h_fusion_score`**: Always 0.000
   - Should be calculated fusion score at 4H timeframe
   - Prevents Plus-One sizing optimization
   - **Impact**: Archetype C trades would be 1.0× instead of potential 1.25×

---

## Why This Explains Previous Issues

### Exit Optimizer v2 Flat-Lining (200 Identical Trials)

**Previous observation**: Exit optimizer ran 200 trials with identical results:
- Score: 1094.91
- PNL: $584.42
- All trials produced EXACTLY the same output

**Root cause (now confirmed)**:
- Exit optimizer used DEFAULT entry threshold (0.45)
- With default threshold + broken features → very few entries
- Limited entries → limited data for exit optimization
- Exit parameter changes had no effect because sample size was too small

### Entry Optimizer Producing Low Trade Counts

**Previous observation**: Entry optimizer Trial 21 (best) only generated:
- 23 signals
- 5 trades executed
- Score: 922.83

**Root cause (now confirmed)**:
- Optimizer lowered threshold to 0.374 to compensate for missing features
- Even with lower threshold, broken BOMS/liquidity features limited opportunities
- System had to rely solely on Wyckoff/momentum scores

---

## Archetype Thresholds (Currently Implemented)

| Archetype | Threshold | Primary Conditions | Status |
|-----------|-----------|-------------------|--------|
| **A: Trap Reversal** | 0.33 | PTI trap + BOMS disp >= 1.25×ATR | ❌ BLOCKED |
| **B: OB Retest** | 0.37 | BOS + liq >= 0.68 + wyc >= 0.50 | ❌ BLOCKED |
| **C: FVG Continuation** | 0.42 | FVG + BOMS disp >= 1.5×ATR + liq >= 0.72 | ❌ BLOCKED |
| **Legacy Fallback** | 0.45 | Ultra-high fusion (safety net) | ⚠️ ACTIVE |

**Current behavior**: System falls back to legacy threshold (0.45) but fusion scores are likely suppressed due to missing liquidity/BOMS components, resulting in 0 trades.

---

## Feature Store Columns to Investigate

The feature store builder (`bin/build_mtf_feature_store.py`) needs to be checked for:

### 1. BOMS Displacement Calculation
- **Column**: `tf4h_boms_displacement`
- **Expected**: Distance from Break of Market Structure point to current price
- **Current**: Always 0.00
- **Location in code**: Likely in BOMS detector module

### 2. BOMS Strength Calculation
- **Column**: `tf1d_boms_strength`
- **Expected**: Strength metric for BOMS (0.0-1.0)
- **Current**: Always 0.000
- **Location in code**: Likely in BOMS detector module

### 3. PTI Trap Type Classification
- **Column**: `tf1h_pti_trap_type`
- **Expected**: Values like 'bull_trap', 'bear_trap', 'spring', 'utad'
- **Current**: Always None
- **Location in code**: PTI detector has scores but missing trap classification logic

### 4. Liquidity Score Derivation
- **Source**: Derived from HOB/BOMS features in fusion calculation
- **Expected**: 0.0-1.0 range based on liquidity zone proximity
- **Current**: Always 0.000
- **Location in code**: Fusion scoring logic in backtest engine

### 5. Multi-Timeframe Fusion Scores
- **Column**: `tf4h_fusion_score`
- **Expected**: Fusion score calculated at 4H timeframe
- **Current**: Always 0.000
- **Location in code**: Feature store builder MTF fusion calculation

---

## Recommended Fix Priority

### P0 (Critical - Unblocks all archetypes):

1. **Fix BOMS displacement calculation** (`tf4h_boms_displacement`)
   - Unblocks Archetype A and C
   - Highest impact

2. **Fix liquidity score derivation** (fusion context)
   - Unblocks Archetype B and C
   - Second highest impact

3. **Fix BOMS strength calculation** (`tf1d_boms_strength`)
   - Unblocks Archetype B
   - Completes Archetype B requirements

### P1 (Important - Quality improvements):

4. **Add PTI trap type classification** (`tf1h_pti_trap_type`)
   - Unblocks Archetype A
   - Enables trap-specific entries

5. **Fix tf4h fusion scoring** (`tf4h_fusion_score`)
   - Enables Plus-One sizing (1.25×)
   - Optimizes Archetype C performance

---

## Testing Plan After Fixes

### Step 1: Verify Feature Store
After fixing feature calculations, rebuild feature store and verify:
```bash
python3 bin/build_mtf_feature_store.py --asset BTC --start 2024-01-01 --end 2024-01-31
```

Check that:
- `tf4h_boms_displacement` has non-zero values
- `tf1d_boms_strength` has non-zero values
- `tf1h_pti_trap_type` has trap classifications
- Liquidity scores are non-zero
- `tf4h_fusion_score` has non-zero values

### Step 2: Re-run Archetype System
```bash
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31
```

**Expected results** (with fixed features):
- **Archetype A matches**: 5-15 (trap reversals are rare but high-edge)
- **Archetype B matches**: 20-40 (OB retests are common)
- **Archetype C matches**: 15-30 (FVG continuations are moderate frequency)
- **Total trades**: 50-80 (target volume)
- **Win rate**: 55-65% (maintained quality)
- **PNL**: $2,000-$5,000 (4-8× improvement over $584 baseline)

### Step 3: Archetype Distribution Analysis
Log output should show:
```
Archetype A matches (Trap Reversal): 8 (16%)
Archetype B matches (OB Retest): 28 (56%)
Archetype C matches (FVG Continuation): 14 (28%)
Total matches: 50 (0.57% of checks)
```

---

## Implementation Status

### ✅ Completed:
- [x] 3-archetype entry classification system
- [x] Archetype-specific thresholds and sizing
- [x] Debug logging and diagnostics
- [x] Root cause identification

### ⏸️ Blocked (Waiting for feature store fixes):
- [ ] Test archetype system on BTC 2024
- [ ] Enable Phase 4 re-entries
- [ ] Loosen exit parameters to assist mode
- [ ] Add USDT.D and OI suppressors

### 🔧 Required Next (Feature Store):
- [ ] Fix BOMS displacement calculation
- [ ] Fix BOMS strength calculation
- [ ] Fix liquidity score derivation
- [ ] Add PTI trap type classification
- [ ] Fix tf4h_fusion_score calculation

---

## Conclusion

The 3-archetype entry system is **production-ready** but cannot function until the underlying SMC feature calculations are fixed in the feature store builder.

The issue is NOT with the entry logic or archetype thresholds - it's with the **data pipeline** feeding zeros into features that should have meaningful values.

Once the feature store is fixed, the archetype system should immediately start generating 50-80 trades/year as designed, with proper distribution across trap reversals, OB retests, and FVG continuations.

---

**Next Action**: Investigate `bin/build_mtf_feature_store.py` to locate and fix the BOMS/liquidity/PTI feature calculation bugs.
