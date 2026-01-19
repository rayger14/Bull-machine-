# Domain Engine Wiring Verification Results
## Test Date: 2025-12-10
## Test Period: BTC 2022-01-01 to 2022-12-31 (Bear Market)

---

## Executive Summary

**STATUS: ⚠️ PARTIAL WIRING - S4 & S5 Working, S1 Broken**

Domain engine wiring shows **mixed results**:
- ✅ **S4 (Funding Divergence)**: Domain engines wired and functioning
- ✅ **S5 (Long Squeeze)**: Domain engines wired and functioning
- ❌ **S1 (Liquidity Vacuum)**: Domain engines NOT connected (0 archetype matches)

**Key Finding**: S4 and S5 archetypes ARE firing and using domain engines, but the results show NO PERFORMANCE DIFFERENCE between Core and Full variants. This suggests:
1. Domain engines are wired correctly (archetypes fire)
2. But domain engine FEATURES are not being used in scoring/filtering
3. Or domain features have zero discriminatory power in 2022 bear market

---

## Detailed Results by Archetype

### S1 (Liquidity Vacuum) - ❌ BROKEN

| Metric | Core Variant | Full Variant | Difference |
|--------|--------------|--------------|------------|
| **Archetype Matches** | 0 | 0 | 0 |
| **Total Trades** | 110 | 110 | 0 |
| **Profit Factor** | 0.32 | 0.32 | 0.0% |
| **Win Rate** | 31.8% | 31.8% | 0.0% |
| **Sharpe Ratio** | -0.70 | -0.70 | 0.0 |

**Trade Type Analysis:**
- S1_core: 100% "tier1_market" (legacy fusion system)
- S1_full: 100% "tier1_market" (legacy fusion system)

**Diagnosis:**
- S1 archetype NOT firing at all (0 matches)
- All trades come from legacy "tier1_market" fusion system
- Domain engines have no effect because archetype never triggers

**Root Cause:**
Likely issues:
1. S1 thresholds too strict for 2022 data (0.17 liquidity_max, 1.96 volume_z_min)
2. Regime filter blocking all signals (use_regime_filter: false in core, but still 0 matches)
3. Confluence requirements too high (0.65 threshold, 3 min conditions)
4. V2 logic has bugs preventing pattern detection

**Recommended Fix:**
1. Test S1 on 2024 data where we know it fires (17 trades PF 6.17)
2. Relax thresholds for 2022 bear market
3. Debug V2 liquidity vacuum detection logic

---

### S4 (Funding Divergence) - ✅ WIRED BUT NO IMPACT

| Metric | Core Variant | Full Variant | Difference |
|--------|--------------|--------------|------------|
| **Archetype Matches** | 16 | 16 | 0 |
| **Archetype Trades** | 12 | 12 | 0 |
| **Total Trades** | 122 | 122 | 0 |
| **Profit Factor** | 0.36 | 0.36 | 0.0% |
| **Win Rate** | 34.4% | 34.4% | 0.0% |
| **Sharpe Ratio** | -0.59 | -0.59 | 0.0 |

**Trade Type Analysis:**
- S4_core: 12 "archetype_funding_divergence" + 110 "tier1_market"
- S4_full: 12 "archetype_funding_divergence" + 110 "tier1_market"

**Diagnosis:**
- ✅ S4 archetype IS firing (16 matches, 12 trades taken)
- ✅ Runtime feature enrichment working ("Enriched 8718 bars")
- ❌ Core and Full variants produce IDENTICAL results
- ❌ Domain engines (macro) have zero performance impact

**Findings:**
1. Archetype detection is working
2. Core variant (no macro): 16 matches
3. Full variant (with macro regime routing): 16 matches
4. **No difference** between variants suggests:
   - Macro regime routing not affecting archetype selection
   - Or macro features not discriminating signal quality in 2022

**Recommended Action:**
1. Inspect S4 trade logs to verify macro features are being used
2. Check if regime routing weights are actually being applied
3. Test on different time periods (2023-2024) where regime matters more

---

### S5 (Long Squeeze) - ✅ WIRED, WRONG DIRECTION IMPACT

| Metric | Core Variant | Full Variant | Difference |
|--------|--------------|--------------|------------|
| **Archetype Matches** | 24 | 24 | 0 |
| **Archetype Trades** | 24 | 5 | -19 |
| **Total Trades** | 134 | 115 | -19 |
| **Profit Factor** | 0.34 | 0.32 | **-5.9%** ⚠️ |
| **Win Rate** | 34.3% | 31.3% | -3.0% |
| **Sharpe Ratio** | -0.57 | -0.67 | -0.10 |

**Trade Type Analysis:**
- S5_core: 24 "archetype_long_squeeze" + 110 "tier1_market"
- S5_full: 5 "archetype_long_squeeze" + 110 "tier1_market"

**Diagnosis:**
- ✅ S5 archetype IS firing (24 matches in both variants)
- ⚠️ **Full variant BLOCKS 19 archetype trades** (24 → 5)
- ⚠️ **Performance DEGRADES with domain engines** (PF 0.34 → 0.32)
- ❌ Trade count differs (134 → 115) but performance worse

**Findings:**
1. Domain engines ARE affecting behavior (trade count differs)
2. But effect is NEGATIVE:
   - Full variant filters out 19/24 S5 archetype trades
   - Remaining 5 trades perform worse
   - Overall PF drops 5.9%

**Hypothesis:**
- Wyckoff/SMC/Temporal/HOB/Fusion/Macro engines are BLOCKING good S5 trades
- Domain engine filters are too conservative for 2022 bear market
- Or domain engine scoring is inverted (filtering winners, keeping losers)

**Recommended Action:**
1. Analyze which 19 trades were filtered out by Full variant
2. Check if filtered trades were winners (would explain PF drop)
3. Inspect domain engine feature scores for blocked vs. accepted trades
4. Consider relaxing domain engine thresholds for bear markets

---

## Conclusion: What This Test Proves

### ✅ **Wiring Works** (Partially)
- S4 and S5 archetypes ARE connected to domain engines
- Runtime feature enrichment is working
- Feature flags control behavior (S5 shows different trade counts)

### ❌ **Performance Impact Missing/Negative**
- S1: Not firing at all (0 matches) - broken
- S4: Domain engines have ZERO impact (identical results)
- S5: Domain engines DEGRADE performance (-5.9% PF)

### 🔍 **Next Steps Required**

**Immediate (Before Re-optimization):**
1. **Fix S1**: Debug why liquidity vacuum has 0 matches in 2022
   - Test on 2024 data (known working period)
   - Relax thresholds for bear market
   - Verify V2 multi-bar detection logic

2. **Investigate S4**: Why does macro have zero impact?
   - Check regime routing weights are applied
   - Verify macro features in trade decisions
   - Test on regime-volatile periods (2023-2024)

3. **Debug S5**: Why do domain engines filter winners?
   - Compare blocked vs. accepted trades
   - Check domain feature score distributions
   - Identify which engines block good trades

4. **Test Different Period**: 2022 is extreme bear (may not be representative)
   - Try 2023 mixed regime
   - Try 2024 bull market
   - Verify domain engines add value in normal conditions

**Before Proceeding to Re-optimization:**
- ❌ Do NOT re-optimize yet - domain engines may be broken
- ✅ First verify engines improve performance on ANY dataset
- ✅ Find parameter sets where Full > Core (proof of value)
- ✅ Then optimize those working configurations

---

## Raw Test Data

### S1 Results
```
Core: 0 archetype matches, 110 tier1 trades, PF 0.32
Full: 0 archetype matches, 110 tier1 trades, PF 0.32
Difference: IDENTICAL (archetype not firing)
```

### S4 Results
```
Core: 16 archetype matches, 12 archetype trades, PF 0.36
Full: 16 archetype matches, 12 archetype trades, PF 0.36
Difference: IDENTICAL (macro has no effect)
```

### S5 Results
```
Core: 24 archetype matches, 24 archetype trades, PF 0.34
Full: 24 archetype matches, 5 archetype trades, PF 0.32
Difference: Full blocks 19 trades, PF drops 5.9%
```

---

## Recommendation: PAUSE and DEBUG

**Do NOT proceed with re-optimization until:**
1. S1 archetype fires on some dataset
2. Domain engines show positive impact (Full PF > Core PF)
3. We understand why S5 degrades with domain engines

**Test on known-good data first:**
- S1: Test on 2024 bull market (17 trades, PF 6.17 proven)
- S4/S5: Test on 2023-2024 mixed regime periods
- Find conditions where domain engines ADD value

**Then optimize from working baseline, not broken one.**
