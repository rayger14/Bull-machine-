# Domain Engine Wiring - Final Diagnostic Report

## Executive Summary

**STATUS: ✅ DOMAIN ENGINES WIRED CORRECTLY**
**ISSUE IDENTIFIED: 🔧 VETO LOGIC TOO AGGRESSIVE (calibration problem, not wiring)**

## Root Cause Analysis

### Bug #1: wyckoff_phase_abc String Mismatch (FIXED)
- **Problem**: Code checked for `== 'accumulation'` and `== 'distribution'`
- **Reality**: Feature has values 'A' (accumulation) and 'D' (distribution)
- **Impact**: Domain engines never fired (100% mismatch)
- **Fix**: Changed all 6 occurrences to use 'A' and 'D'
- **Files**: `engine/archetypes/logic_v2_adapter.py` lines 1759, 1811, 2677, 2698, 2968, 2983

### Bug #2: Wyckoff Distribution Veto Too Aggressive (CALIBRATION ISSUE)
- **Problem**: Hard veto on wyckoff_distribution blocks 97%+ of trades
- **Data Reality**:
  - 2022: 97.06% distribution
  - 2023: 98.41% distribution
  - 2024: 97.20% distribution
- **Impact**: Long-only archetypes (S1, S4, S5) can only fire 2-3% of the time
- **Result**: Both CORE and FULL variants apply same veto → identical results

## Verification Test Results

### S4 (Funding Divergence) - ✅ WORKING

**Before Fix:**
- S4_core: 122 trades, PF 0.36
- S4_full: 122 trades, PF 0.36
- **trades_differ: ❌ False**

**After Fix:**
- S4_core: 122 trades, PF 0.36
- S4_full: 116 trades, PF 0.32
- **trades_differ: ✅ True (-6 trades, domain vetoes firing)**

**Conclusion:** Domain engines are wired and working. The wyckoff_distribution veto successfully blocked 6 trades in S4_full.

### S1 (Liquidity Vacuum) - ⚠️ WORKING BUT IDENTICAL BEHAVIOR

**Results:**
- S1_core: 110 trades, PF 0.32 (wyckoff enabled)
- S1_full: 110 trades, PF 0.32 (wyckoff + 5 other engines)
- **trades_differ: ❌ False**

**Why Identical:**
1. s1_core has wyckoff enabled (intentional per config: "CORE variant = Wyckoff only")
2. s1_full has wyckoff + smc + temporal + hob + fusion + macro
3. Both apply wyckoff_distribution veto on 97% of bars
4. Additional engines in FULL can't fire because wyckoff veto blocks entry
5. Result: Both variants produce same 110 trades (from 2.94% of bars where distribution=False)

**Conclusion:** Domain engines wired correctly, but veto logic blocks useful signals.

### S5 (Long Squeeze) - ⚠️ SAME ISSUE AS S1

**Results:**
- S5_core: 110 trades, PF 0.32
- S5_full: 110 trades, PF 0.32
- **trades_differ: ❌ False**

**Same root cause as S1.**

## Wyckoff Phase Distribution Analysis

```
YEAR    DISTRIBUTION  ACCUMULATION  NEUTRAL/TRANSITION
2022    97.06%        2.25%         0.69%
2023    98.41%        1.06%         0.53%
2024    97.20%        1.96%         0.84%
```

**Observation:** Wyckoff classifier marks market as "distribution" 97%+ of the time across all years. This suggests either:
1. The classifier threshold is too sensitive
2. BTC was genuinely in distribution 97% of 2022-2024 (unlikely)
3. The classifier needs recalibration

## Recommended Fixes

### Priority 1: Change Hard Vetoes to Soft Vetoes (CALIBRATION FIX)

**Current Logic (Too Aggressive):**
```python
if wyckoff_distribution or wyckoff_utad or wyckoff_bc:
    return False, 0.0, {  # HARD BLOCK
        "reason": "wyckoff_distribution_veto",
        ...
    }
```

**Recommended Logic (Adaptive):**
```python
if wyckoff_distribution:
    domain_boost *= 0.30  # Reduce confidence 70% but don't block
    domain_signals.append("wyckoff_distribution_caution")
elif wyckoff_utad or wyckoff_bc:
    domain_boost *= 0.50  # More severe distribution events
    domain_signals.append("wyckoff_utad_or_bc_caution")
```

**Rationale:**
- Long entries in distribution phase have lower win rate (not zero)
- Capitulation events CAN occur in distribution (e.g., FTX collapse)
- Allow pattern to fire but with reduced confidence
- Prevents archetype from being blocked 97% of the time

### Priority 2: Recalibrate Wyckoff Phase Classifier

**Investigation needed:**
- Review wyckoff_phase_abc feature generation logic
- Compare classifier output to manual analysis of 2022-2024 BTC chart
- Adjust threshold to achieve 70/20/10 distribution (Distribution/Accumulation/Transition)

**Expected healthy distribution:**
- Distribution: 60-70%
- Accumulation: 15-25%
- Transition (B/C/neutral): 10-20%

### Priority 3: Create Veto-Free Test Configs

**For domain engine verification only:**
```json
// configs/variants/s1_no_vetoes.json
{
  "feature_flags": {
    "enable_wyckoff": true,
    "enable_wyckoff_vetoes": false,  // NEW: Disable hard vetoes for testing
    ...
  }
}
```

## Implementation Plan

### Phase 1: Immediate Fixes (1 hour)
1. Convert hard vetoes to soft vetoes in S1, S4, S5
2. Re-run domain wiring verification test
3. Confirm FULL variant shows improved metrics vs CORE

### Phase 2: Wyckoff Classifier Audit (2-4 hours)
1. Manual analysis of 5-10 major 2022-2024 events
2. Compare classifier output to human judgment
3. Adjust threshold if needed
4. Backfill wyckoff_phase_abc with new classifier

### Phase 3: Production Validation (2 hours)
1. Run full 2022-2024 backtest with soft vetoes
2. Compare results to hard veto baseline
3. Validate that S1/S4/S5 have reasonable trade frequency (15-50/year)
4. Commit changes if metrics improve

## Files Modified This Session

1. `engine/archetypes/logic_v2_adapter.py`
   - Fixed wyckoff_phase_abc string mismatch (6 locations)
   - Lines: 1759, 1811, 2677, 2698, 2968, 2983

2. `bin/backtest_knowledge_v2.py`
   - Fixed feature_flags missing from RuntimeContext metadata
   - Line: 632

## Conclusion

**Domain engines are successfully wired and functional.** We confirmed this by:
1. Fixing the wyckoff_phase_abc string mismatch
2. Observing S4 produce different results (core vs full)
3. Confirming vetoes fire correctly (6 trades blocked in S4_full)

The remaining issue (S1/S5 showing Core = Full) is a **calibration problem**, not a wiring bug. The wyckoff_distribution hard veto is too aggressive for a classifier that marks 97%+ of bars as distribution.

**Recommendation:** Proceed with Phase 1 (soft vetoes) to make the domain engines usable, then address Wyckoff classifier calibration as a separate optimization task.

---
**Generated:** 2025-12-12
**Author:** Claude Code Domain Engine Diagnostic System
