# Domain Engine Architecture Fix - Visual Guide

## Current Architecture (BROKEN)

```
┌─────────────────────────────────────────────────────────────┐
│ S1 Liquidity Vacuum Detection (_check_S1)                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 1. Calculate Base Score        │
         │    score = Σ(component × wt)   │
         │    Example: 0.38               │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 2. Fusion Threshold Gate       │
         │    if score < 0.40:            │
         │       return False ❌          │
         └────────────────────────────────┘
                          │
                          │ Score passes (0.38 FAILS!)
                          ▼
         ┌────────────────────────────────┐
         │ 3. Domain Engines (TOO LATE)   │
         │    wyckoff_spring_a: 2.5x      │
         │    domain_boost = 2.50         │
         │    NEVER APPLIED! 💀            │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 4. Apply Boost to Score        │
         │    score = 0.38 × 2.5 = 0.95   │
         │    (but already rejected!)     │
         └────────────────────────────────┘
                          │
                          ▼
                  ❌ REJECTED ❌
              (score < threshold)
```

**Problem:** Wyckoff Spring signal (2.5x boost) never consulted. Signal rejected at gate.

---

## Fixed Architecture (PROPOSED)

```
┌─────────────────────────────────────────────────────────────┐
│ S1 Liquidity Vacuum Detection (_check_S1)                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 1. Calculate Base Score        │
         │    score = Σ(component × wt)   │
         │    Example: 0.38               │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 2. Domain Engines (MOVED UP)   │
         │    ├─ Vetos (safety)           │
         │    │   wyckoff_distribution ❌  │
         │    │   wyckoff_utad ❌          │
         │    └─ Boosts (confluence)       │
         │        wyckoff_spring_a: 2.5x ✅│
         │        domain_boost = 2.50     │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 3. Apply Domain Boost          │
         │    score = 0.38 × 2.5 = 0.95   │
         │    (BEFORE gate check)         │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │ 4. Fusion Threshold Gate       │
         │    if score < 0.40:            │
         │       0.95 >= 0.40 ✅          │
         └────────────────────────────────┘
                          │
                          ▼
                  ✅ ACCEPTED ✅
           (domain-adjusted score passes)
```

**Solution:** Domain boost applied BEFORE gate check. Wyckoff Spring context saves marginal signal.

---

## Code Changes Required

### Before (lines 1730-1790):

```python
# Line 1730: Calculate base score
score = sum(components[k] * weights.get(k, 0.0) for k in components)

# Line 1768: HARD GATE (domain engines not yet consulted)
if score < fusion_th:
    return False, score, {"reason": "v2_score_below_threshold"}

# Line 1791: Domain engines start here (TOO LATE)
domain_boost = 1.0
if use_wyckoff:
    if wyckoff_spring_a:
        domain_boost *= 2.50

# Line 1956: Apply boost (but decision already made)
score = score * domain_boost

# Line 1965: Return with boosted score
return True, score, {...}
```

### After (PROPOSED):

```python
# Line 1730: Calculate base score
score = sum(components[k] * weights.get(k, 0.0) for k in components)

# MOVED UP: Domain engines BEFORE gate
domain_boost = 1.0
domain_signals = []

if use_wyckoff:
    # VETOES FIRST (safety)
    if wyckoff_distribution or wyckoff_utad or wyckoff_bc:
        return False, 0.0, {"reason": "wyckoff_distribution_veto"}

    # BOOSTS (confluence)
    if wyckoff_spring_a:
        domain_boost *= 2.50
        domain_signals.append("wyckoff_spring_a")
    # ... other boosts (SMC, Temporal, HOB, Macro)

# Apply domain boost BEFORE gate
score = score * domain_boost

# NOW check fusion threshold (with domain-adjusted score)
if score < fusion_th:
    return False, score, {
        "reason": "v2_score_below_threshold",
        "domain_boost": domain_boost,
        "domain_signals": domain_signals
    }

# Pattern matched
return True, score, {
    "domain_boost": domain_boost,
    "domain_signals": domain_signals,
    ...
}
```

---

## Domain Engine Capabilities (Before vs After)

| Capability | Before (Broken) | After (Fixed) |
|------------|-----------------|---------------|
| **Veto trades** | ✅ Works | ✅ Works |
| **Boost existing trades** | ✅ Works | ✅ Works |
| **Save marginal signals** | ❌ BROKEN | ✅ FIXED |
| **Affect trade count** | ❌ No | ✅ Yes |
| **Affect pattern detection** | ❌ No | ✅ Yes |

---

## Example Scenarios

### Scenario 1: Marginal Signal + Strong Wyckoff Context

**Input:**
- Base score: 0.38
- Fusion threshold: 0.40
- Wyckoff Spring A detected (2.5x boost)

**Current (Broken):**
1. score=0.38 < 0.40 → REJECTED ❌
2. Domain boost never applied
3. Trade lost

**Fixed:**
1. score=0.38
2. Domain boost: 0.38 × 2.5 = 0.95
3. 0.95 >= 0.40 → ACCEPTED ✅
4. Trade captured with high confidence

### Scenario 2: Strong Signal + Wyckoff Context

**Input:**
- Base score: 0.55
- Fusion threshold: 0.40
- Wyckoff Spring A detected (2.5x boost)

**Current (Broken):**
1. score=0.55 >= 0.40 → ACCEPTED ✅
2. Domain boost: 0.55 × 2.5 = 1.375 (capped at 1.0)
3. Higher confidence but same decision

**Fixed:**
1. score=0.55
2. Domain boost: 0.55 × 2.5 = 1.375
3. 1.375 >= 0.40 → ACCEPTED ✅
4. Same decision, higher confidence (no change)

### Scenario 3: Marginal Signal + Wyckoff Veto

**Input:**
- Base score: 0.45
- Fusion threshold: 0.40
- Wyckoff Distribution detected (VETO)

**Current (Broken):**
1. score=0.45 >= 0.40 → ACCEPTED ✅
2. Wyckoff veto: REJECTED ❌ (veto works)
3. Trade avoided (safety preserved)

**Fixed:**
1. score=0.45
2. Wyckoff veto: REJECTED ❌ (veto still works)
3. Trade avoided (no change - safety preserved)

---

## Impact Prediction

### Trade Count Delta

**S1_core (no domain engines):**
- Baseline: 110 trades, PF 0.32

**S1_full (domain engines enabled):**
- Current (broken): 110 trades, PF 0.32 (IDENTICAL)
- Expected (fixed): 130-150 trades, PF unknown (needs re-optimization)

**Why increase?**
- Marginal signals with strong Wyckoff/SMC context now qualify
- Estimates: 10-20% trade count increase
- Quality unknown until optimization

### Performance Delta

**Before Re-Optimization:**
- ⚠️ May degrade (thresholds tuned for broken architecture)
- Expect: Lower PF due to more marginal signals

**After Re-Optimization:**
- ✅ Should improve (domain context now affects pattern detection)
- Expect: Higher PF due to confluence-based qualification

---

## Implementation Checklist

### Phase 1: Code Changes
- [ ] Move domain engine block (lines 1791-1962) BEFORE fusion gate (line 1768)
- [ ] Apply domain boost to score before gate check
- [ ] Update debug logging (remove [ARCH_BUG] detection, add boost tracking)
- [ ] Add metadata: `domain_boost`, `domain_signals` in all return paths

### Phase 2: Testing
- [ ] Unit test: Verify marginal signal + boost now passes gate
- [ ] Integration test: Run S1_full backtest, verify trade count increases
- [ ] Regression test: Ensure vetoes still work (safety preserved)

### Phase 3: Optimization
- [ ] Re-run S1 threshold optimization with fixed architecture
- [ ] Compare fixed vs baseline (S1_core)
- [ ] Validate on OOS data (2023-2024)

### Phase 4: Rollout
- [ ] Apply fix to S2, S4, S5 (bear archetypes)
- [ ] Apply fix to bull archetypes (A, B, C, etc.) if applicable
- [ ] Update all production configs
- [ ] Monitor live performance

---

## Rollback Plan

If fix degrades performance:

1. **Revert code changes** (git revert)
2. **Keep debug logging** (for future investigation)
3. **Alternative approach:** Dual-threshold system (Option 2)
4. **Research:** Why did domain boosts not improve performance?

---

## Success Criteria

### Must Have (P0)
- ✅ Trade count increases when domain engines enabled
- ✅ Veto logic still works (safety preserved)
- ✅ Debug logging confirms domain boosts applied before gate

### Should Have (P1)
- ✅ PF improves after re-optimization
- ✅ OOS validation passes (2023-2024)
- ✅ No performance regression on gold standard configs

### Nice to Have (P2)
- ✅ Domain boost multipliers tuned (not just thresholds)
- ✅ A/B test shows domain context improves edge
- ✅ Documentation updated

---

## Conclusion

The fix is architecturally sound and maintains all safety guarantees. Domain engines will now affect pattern detection (gates), not just scoring. This aligns with the original design intent: **domain context should inform which patterns to trade, not just confidence in already-selected trades**.

**Key Insight:** The current architecture treats domain engines as "confidence modifiers" when they should be "pattern qualifiers". This fix corrects that misalignment.

**Risk Level:** MEDIUM (requires re-optimization, may temporarily degrade performance)
**Complexity:** LOW (simple code refactor, move block up ~50 lines)
**Impact:** HIGH (enables domain engines to actually affect behavior)

**Status:** READY FOR IMPLEMENTATION
