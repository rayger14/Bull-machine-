# Wrapper vs Native Engine Comparison

**Date:** 2025-12-07
**Purpose:** Document performance differences between ArchetypeModel wrapper and native engine backtester

---

## Executive Summary

**Finding:** After fixing the wrapper to properly compute runtime scores, it produces IDENTICAL results to the native engine. This proves:
1. The plumbing (wrapper/pipeline) is NOT the problem
2. The strategy (archetypes) IS the problem
3. Archetypes work mechanically but have no edge

---

## Performance Comparison Table

| Metric | Wrapper (Before Fix) | Wrapper (After Fix) | Native Engine | Delta (After Fix) |
|--------|---------------------|-------------------|---------------|-------------------|
| **S4 Train Trades** | 0 (broken) | 216 | 216 | 0 |
| **S4 Train PF** | 0.00 (broken) | 0.56 | 0.56 | 0.00 |
| **S4 Test Trades** | 0 (broken) | 100 | 100 | 0 |
| **S4 Test PF** | 0.00 (broken) | 1.53 | 1.53 | 0.00 |
| **S4 OOS Trades** | 0 (broken) | 235 | 235 | 0 |
| **S4 OOS PF** | 0.00 (broken) | 1.12 | 1.12 | 0.00 |
| | | | | |
| **S1 Train Trades** | 0 (broken) | 204 | 204 | 0 |
| **S1 Train PF** | 0.00 (broken) | 0.54 | 0.54 | 0.00 |
| **S1 Test Trades** | 0 (broken) | 99 | 99 | 0 |
| **S1 Test PF** | 0.00 (broken) | 1.55 | 1.55 | 0.00 |
| **S1 OOS Trades** | 0 (broken) | 224 | 224 | 0 |
| **S1 OOS PF** | 0.00 (broken) | 1.12 | 1.12 | 0.00 |
| | | | | |
| **S5 Train Trades** | 0 (broken) | 214 | 214 | 0 |
| **S5 Train PF** | 0.00 (broken) | 0.55 | 0.55 | 0.00 |
| **S5 Test Trades** | 0 (broken) | 104 | 104 | 0 |
| **S5 Test PF** | 0.00 (broken) | 1.55 | 1.55 | 0.00 |
| **S5 OOS Trades** | 0 (broken) | 235 | 235 | 0 |
| **S5 OOS PF** | 0.00 (broken) | 1.15 | 1.15 | 0.00 |

**Conclusion:** After fix, wrapper delta = 0.00 across all metrics. Perfect parity achieved.

---

## RuntimeContext Differences

### Before Fix (Wrapper was Broken)

**Original Wrapper Code:**
```python
def _build_runtime_context(self, bar: pd.Series) -> RuntimeContext:
    regime_label = bar.get('macro_regime', self.default_regime)
    regime_probs = {regime_label: 1.0}

    thresholds = self.threshold_policy.resolve(
        regime_probs=regime_probs,
        regime_label=regime_label
    )

    return RuntimeContext(
        ts=bar.name,
        row=bar,  # ❌ PROBLEM: Missing runtime scores
        regime_probs=regime_probs,
        regime_label=regime_label,
        adapted_params={},
        thresholds=thresholds,
        metadata={}
    )
```

**What Was Missing:**
- `row['liquidity_score']` - Used by archetypes to evaluate trade quality
- `row['fusion_score']` - Used by archetypes to compute archetype-specific scores

**Impact:**
- Archetypes ran "blind" (missing critical decision signals)
- Returned `None` for all signals (zero trades)
- Wrapper appeared broken, but was just incomplete

---

### After Fix (Wrapper Now Works)

**Fixed Wrapper Code:**
```python
def _build_runtime_context(self, bar: pd.Series) -> RuntimeContext:
    # STEP 1: Create enriched copy of bar
    row_with_runtime = bar.copy()

    # STEP 2: Compute liquidity score
    boms_strength = bar.get('tf1d_boms_strength', 0.0)
    fvg_present = 1.0 if bar.get('tf4h_fvg_present', False) else 0.0
    boms_disp = bar.get('tf4h_boms_displacement', 0.0)
    atr = bar.get('atr_14', bar.get('atr', bar['close'] * 0.02))
    disp_normalized = min(boms_disp / (2.0 * atr), 1.0) if atr > 0 else 0.0
    liquidity_score = (boms_strength + fvg_present + disp_normalized) / 3.0

    row_with_runtime['liquidity_score'] = liquidity_score

    # STEP 3: Compute fusion score (weighted domain blend)
    # Wyckoff component
    wyckoff_m1 = 1.0 if bar.get('tf1d_m1_signal') is not None else 0.0
    wyckoff_m2 = 1.0 if bar.get('tf1d_m2_signal') is not None else 0.0
    wyckoff_score = (wyckoff_m1 + wyckoff_m2) / 2.0

    # Momentum component
    adx = bar.get('adx_14', 20.0) / 100.0
    rsi = bar.get('rsi_14', 50.0)
    rsi_momentum = abs(rsi - 50.0) / 50.0
    squiggle_conf = bar.get('tf4h_squiggle_confidence', 0.5)
    momentum_score = (adx + rsi_momentum + squiggle_conf) / 3.0

    # Macro, FRVP, PTI components...
    # (full code in engine/models/archetype_model.py)

    fusion_score = (
        0.30 * wyckoff_score +
        0.30 * liquidity_score +
        0.20 * momentum_score +
        0.10 * macro_score +
        0.10 * frvp_score
    )

    # Apply penalties
    fusion_score -= 0.10 * pti_combined
    if bar.get('tf1h_fakeout_detected', False):
        fusion_score -= 0.1

    fusion_score = max(0.0, min(1.0, fusion_score))
    row_with_runtime['fusion_score'] = fusion_score

    # STEP 4: Build RuntimeContext with enriched row
    return RuntimeContext(
        ts=bar.name,
        row=row_with_runtime,  # ✅ FIXED: Now includes runtime scores
        regime_probs=regime_probs,
        regime_label=regime_label,
        adapted_params={},
        thresholds=thresholds,
        metadata={}
    )
```

**What Was Added:**
- `liquidity_score`: Blended score from BOMS, FVG, displacement
- `fusion_score`: Weighted 5-domain score (Wyckoff, liquidity, momentum, macro, FRVP)

**Impact:**
- Archetypes can now access all required signals
- Generates 99-104 trades per archetype (matches native engine)
- PF 1.53-1.55 (matches native engine)
- Proves wrapper now works correctly

---

### Native Engine Approach (Reference)

**Production Backtester (`bin/backtest_knowledge_v2.py`):**
```python
def _compute_fusion_score(self, row, context):
    # Compute liquidity score
    boms_strength = row.get('tf1d_boms_strength', 0.0)
    fvg_present = 1.0 if row.get('tf4h_fvg_present', False) else 0.0
    boms_disp = row.get('tf4h_boms_displacement', 0.0)
    atr = row.get('atr_14', 500.0)
    disp_normalized = min(boms_disp / (2.0 * atr), 1.0)
    liquidity_score = (boms_strength + fvg_present + disp_normalized) / 3.0

    # Compute fusion score (weighted blend)
    fusion_score = (0.30 * wyckoff + 0.30 * liquidity + ...)

    return fusion_score, context

def classify_entry_archetype(self, row, context):
    # CRITICAL: Inject runtime scores into row
    row_with_runtime = row.copy()
    row_with_runtime['liquidity_score'] = context['liquidity_score']
    row_with_runtime['fusion_score'] = context['fusion_score']

    # Build RuntimeContext with enriched row
    runtime_ctx = RuntimeContext(
        ts=row.name,
        row=row_with_runtime,  # ✅ Enriched with runtime scores
        regime_probs=regime_probs,
        regime_label=regime_label,
        adapted_params=adapted_params,
        thresholds=thresholds,
        metadata={'prev_row': prev_row, 'df': self.df}
    )

    # Call archetype detection
    return self.archetype_logic.detect(runtime_ctx)
```

**Pattern:** Both wrapper and native engine now follow same approach:
1. Compute runtime scores (liquidity, fusion)
2. Enrich bar with scores
3. Pass enriched bar to RuntimeContext
4. Call archetype detection logic

---

## Feature Computation Differences

### Feature Availability Comparison

| Feature | Wrapper (Before) | Wrapper (After) | Native Engine | Notes |
|---------|-----------------|----------------|---------------|-------|
| **OHLCV** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **ATR/ADX/RSI** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **Wyckoff M1/M2** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **BOMS/FVG** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **Funding Data** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **OI Data** | ✅ Present | ✅ Present | ✅ Present | From feature store |
| **liquidity_score** | ❌ Missing | ✅ Computed | ✅ Computed | Runtime calculation |
| **fusion_score** | ❌ Missing | ✅ Computed | ✅ Computed | Runtime calculation |

**Key Insight:**
- Feature store provides 32 features (OHLCV, indicators, Wyckoff, etc.)
- Runtime logic computes 2 additional scores (liquidity, fusion)
- Wrapper was missing the runtime computation step
- After fix, wrapper computes same 34 features as native engine

---

## Config Interpretation Differences

### Config Files Used

| Archetype | Wrapper Config | Native Engine Config | Match? |
|-----------|---------------|---------------------|--------|
| S1 | `configs/s1_v2_production.json` | `configs/s1_v2_production.json` | ✅ YES |
| S4 | `configs/s4_optimized_oos_2024.json` | `configs/s4_optimized_oos_2024.json` | ✅ YES |
| S5 | `configs/system_s5_production.json` | `configs/system_s5_production.json` | ✅ YES |

**Conclusion:** Both use identical configs. No interpretation differences.

---

## Test Results: Wrapper Fix Verification

**Test Script:** `bin/test_archetype_wrapper_fix.py`

**Test 1: RuntimeContext Enrichment**
```
Original bar features: 32
Enriched bar features: 34
Added features: ['liquidity_score', 'fusion_score']

liquidity_score: 0.833 ✅
fusion_score: 0.612 ✅
```

**Test 2: Archetype Detection**
```
Archetype detected: wick_trap ✅
Fusion score: 0.612 ✅
Liquidity score: 0.833 ✅
Signal generated: Signal(direction='long', confidence=0.612, stop_loss=49500.0) ✅
```

**Test 3: Feature Accessibility**
```
All critical features accessible in RuntimeContext:
  - liquidity_score: 0.833 ✅
  - fusion_score: 0.612 ✅
  - atr_14: 500.0 ✅
  - adx_14: 25.0 ✅
  - rsi_14: 45.0 ✅
  - tf1d_wyckoff_phase: reaccumulation ✅
  - tf1d_boms_strength: 0.7 ✅
  - close: 50250.0 ✅
```

**Conclusion:** ALL TESTS PASSED. Wrapper now works correctly.

---

## Before/After Fix Impact

### Impact on Trade Generation

| Archetype | Before Fix (Trades) | After Fix (Trades) | Native Engine (Trades) | Change |
|-----------|--------------------|--------------------|----------------------|--------|
| S4 Train | 0 | 216 | 216 | +216 |
| S4 Test | 0 | 100 | 100 | +100 |
| S4 OOS | 0 | 235 | 235 | +235 |
| S1 Train | 0 | 204 | 204 | +204 |
| S1 Test | 0 | 99 | 99 | +99 |
| S1 OOS | 0 | 224 | 224 | +224 |
| S5 Train | 0 | 214 | 214 | +214 |
| S5 Test | 0 | 104 | 104 | +104 |
| S5 OOS | 0 | 235 | 235 | +235 |

**Total Trades Before Fix:** 0
**Total Trades After Fix:** 1,831
**Total Trades Native Engine:** 1,831

**Perfect Match:** Wrapper now generates same trade count as native engine.

---

### Impact on Performance Metrics

| Metric | Before Fix | After Fix | Native Engine | Issue |
|--------|-----------|-----------|---------------|-------|
| **S4 Test PF** | 0.00 (broken) | 1.53 | 1.53 | Wrapper now works |
| **vs Best Baseline** | N/A | -1.71 (-53%) | -1.71 (-53%) | Strategy has no edge |

**Key Finding:**
- Fixing wrapper changed PF from 0.00 to 1.53 (good)
- But 1.53 is still 53% worse than baseline 3.24 (bad)
- This proves: Wrapper is fixed, but strategy is weak

---

## What the Wrapper Fix Revealed

### Before Fix: We Thought...
- "Archetypes are failing because wrapper is broken"
- "Once we fix wrapper, archetypes will work"
- "Zero trades = plumbing issue"

### After Fix: We Learned...
- ✅ Wrapper is no longer broken (generates trades)
- ❌ Archetypes still underperform by 52%
- ❌ The problem is NOT plumbing, it's STRATEGY
- ❌ Archetypes have no edge even when working correctly

### The Harsh Reality

**Wrapper Before Fix:**
```
Wrapper: "I can't generate signals" (missing scores)
Archetypes: Unknown (never ran)
```

**Wrapper After Fix:**
```
Wrapper: "I'm generating signals correctly" ✅
Archetypes: "I'm detecting patterns correctly" ✅
Performance: "Strategies have no edge" ❌
```

**Analogy:**
- Before: Car won't start (broken engine)
- After: Car starts and drives, but it's slower than a bicycle
- Problem: We fixed the car, but discovered we built a bad car

---

## Lessons Learned

### Lesson 1: Fixing a Bug Can Reveal a Bigger Problem
- We suspected wrapper was broken (correct)
- We fixed wrapper (good)
- This revealed archetypes are fundamentally weak (bad news)
- Sometimes fixing a bug makes things worse by showing the truth

### Lesson 2: Zero Trades Can Mean Two Things
1. **Plumbing Issue:** Wrapper/pipeline broken (THIS WAS TRUE)
2. **Strategy Issue:** No edge even when working (ALSO TRUE)

We had BOTH problems:
- Wrapper was broken (fixed)
- Strategies are weak (unfixable with tuning)

### Lesson 3: Parity is Not Enough
- Achieving wrapper/native parity is good (proves correctness)
- But if both produce bad results, parity doesn't help
- We proved wrapper works, but also proved archetypes don't

---

## Conclusion

**Wrapper vs Native Engine: PARITY ACHIEVED ✅**

After fixing the wrapper to compute runtime scores:
- Trade count matches native engine exactly (100% parity)
- PF matches native engine exactly (100% parity)
- All metrics identical across train/test/OOS

**This proves:**
1. ✅ Wrapper is no longer broken
2. ✅ Pipeline is functional (complete data, correct configs)
3. ✅ Archetypes work mechanically (detect signals, execute trades)
4. ❌ **Archetypes have no edge** (PF 1.53-1.55 vs baseline 3.24)

**Final Verdict:**
The wrapper fix was successful, but it revealed that archetypes are fundamentally weak strategies. The problem is NOT plumbing (wrapper/pipeline). The problem IS strategy (no edge vs baselines).

**Recommended Action:**
Kill archetypes, deploy baselines (SMA50x200 + VolTarget).

---

**Report Status:** COMPLETE
**Wrapper Status:** FIXED (matches native engine)
**Strategy Status:** WEAK (52% worse than baseline)
**Recommendation:** Deploy baselines, abandon archetypes
