# Domain Engine Gate Ordering Bug - Root Cause Analysis

**Date:** 2025-12-11
**Severity:** HIGH - Architectural flaw preventing domain engines from affecting trade decisions
**Status:** IDENTIFIED, DEBUG LOGGING ADDED, FIX PROPOSED

---

## Executive Summary

Domain engines (Wyckoff, SMC, Temporal, HOB, Macro) are fully implemented with real feature signals but have **zero effect** on trade counts despite 2.5x score boosts. Root cause: **gate ordering bug** where domain boosts are applied AFTER the fusion threshold gate, so they only affect scores of already-qualified trades, not pattern detection.

**Impact:** S1_core and S1_full produce identical results (110 trades, PF 0.32) despite domain engines being enabled in S1_full.

---

## Root Cause Analysis

### 1. Architectural Flow (Current - BROKEN)

File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py`

```python
# Line 1730-1741: Calculate base score
score = sum(components[k] * weights.get(k, 0.0) for k in components)

# HARD GATE: Score must pass threshold BEFORE domain engines are consulted
if score < fusion_th:
    return False, score, {
        "reason": "v2_score_below_threshold",
        ...
    }

# Lines 1744-1962: Domain engines applied AFTER gate passes
domain_boost = 1.0
if use_wyckoff:
    if wyckoff_spring_a:
        domain_boost *= 2.50  # 2.5x boost - TOO LATE!
    # ... other boosts

# Line 1956: Apply boost to score (but decision already made)
score = score * domain_boost

# Line 1965: Return True with boosted score
return True, score, {...}
```

### 2. The Bug

**Problem:** Domain engines can only:
1. **Veto** trades early (line 1772-1778) ✅ Works
2. **Boost** scores of trades that already passed (line 1956) ❌ Useless

They **cannot** help marginal signals cross the fusion threshold gate because:
- Marginal signal: score=0.38, threshold=0.40 → REJECTED at line 1768
- Domain boost: 2.5x → would yield 0.95 → NEVER APPLIED
- Result: Signal lost despite strong Wyckoff Spring context

### 3. Why S1_core == S1_full

```bash
# S1_core (no domain engines)
110 trades, PF 0.32

# S1_full (Wyckoff + SMC + Temporal enabled)
110 trades, PF 0.32  # IDENTICAL!
```

**Explanation:**
- Domain engines can veto (reduce trades) ✅
- Domain engines CANNOT help marginal signals qualify (increase trades) ❌
- Domain engines CANNOT change trade selection (just confidence scores) ❌
- Result: Same trades, slightly different scores (not affecting trade count)

---

## Evidence

### A. Code Analysis

**Lines 1768-1788** (Current - BROKEN):
```python
if score < fusion_th:
    return False, score, {...}  # REJECTED - domain boost never consulted

# Domain engines start here (TOO LATE)
if use_wyckoff:
    if wyckoff_spring_a:
        domain_boost *= 2.50  # Would have saved it!
```

### B. Debug Logging Added

**Lines 1733-1788** (New diagnostic code):
```python
# Pre-calculate what domain boost WOULD be
pre_gate_domain_boost = 1.0
if use_wyckoff:
    if wyckoff_spring_a:
        pre_gate_domain_boost *= 2.50

boosted_score = score * pre_gate_domain_boost

if score < fusion_th:
    would_pass_with_boost = (boosted_score >= fusion_th)
    if would_pass_with_boost and pre_gate_domain_boost > 1.0:
        logger.warning(
            f"[ARCH_BUG] S1 signal REJECTED due to gate ordering! "
            f"score={score:.3f} < threshold={fusion_th:.3f}, "
            f"BUT domain boost={pre_gate_domain_boost:.2f}x would yield {boosted_score:.3f} (PASS). "
            f"Signals: {pre_gate_signals}. "
            f"FIX: Move domain engines BEFORE fusion gate."
        )
```

**Expected Result:** When running backtest with S1_full, [ARCH_BUG] warnings will appear in logs showing missed opportunities.

### C. Feature Store Verification

Domain features exist and have real signals:
- Wyckoff events: `wyckoff_spring_a`, `wyckoff_sc`, `wyckoff_lps`
- SMC structure: `tf4h_bos_bullish`, `smc_demand_zone`
- Temporal: `fib_time_cluster`, `temporal_confluence`
- All features present in feature store with non-zero values ✅

---

## Proposed Architectural Fix

### Option 1: Move Domain Engines BEFORE Gate (RECOMMENDED)

**Rationale:** Domain context should inform pattern detection, not just scoring.

```python
# Calculate base score
score = sum(components[k] * weights.get(k, 0.0) for k in components)

# MOVE DOMAIN ENGINES HERE (BEFORE GATE)
domain_boost = 1.0
domain_signals = []

if use_wyckoff:
    # VETOES first
    if wyckoff_distribution or wyckoff_utad or wyckoff_bc:
        return False, 0.0, {"reason": "wyckoff_distribution_veto"}

    # BOOSTS
    if wyckoff_spring_a:
        domain_boost *= 2.50
        domain_signals.append("wyckoff_spring_a")
    # ... other boosts

# Apply domain boost to score BEFORE gate
score = score * domain_boost

# NOW check fusion threshold (with domain-adjusted score)
if score < fusion_th:
    return False, score, {
        "reason": "v2_score_below_threshold",
        "score": score,
        "domain_boost": domain_boost,
        "domain_signals": domain_signals
    }

# Pattern matched
return True, score, {...}
```

**Benefits:**
1. ✅ Domain engines affect pattern detection (increase trade count for confluent signals)
2. ✅ Maintains veto capability (safety)
3. ✅ Simple refactor (move code block up ~50 lines)
4. ✅ Backward compatible (same logic, different order)

**Risks:**
- ⚠️ May increase trade count (more marginal signals qualify)
- ⚠️ Need to re-optimize thresholds after fix

### Option 2: Dual-Threshold System (COMPLEX)

Create separate thresholds for base vs domain-boosted signals:

```python
base_threshold = 0.40  # Strict for raw signals
domain_threshold = 0.35  # Relaxed if domain boost available

if domain_boost > 1.0:
    effective_threshold = domain_threshold
else:
    effective_threshold = base_threshold

if score * domain_boost < effective_threshold:
    return False, ...
```

**Benefits:**
- Explicit control over domain engine influence
- Can tune domain threshold independently

**Drawbacks:**
- More complex (2 parameters per archetype)
- Harder to reason about
- Requires re-optimization of all configs

### Option 3: Probabilistic Blending (OVER-ENGINEERED)

Blend base and domain scores with learnable weights - not recommended for rule-based system.

---

## Implementation Plan

### Phase 1: Validation (COMPLETE ✅)
- [x] Add debug logging to detect missed signals
- [x] Create test script demonstrating the bug
- [x] Document root cause and architecture

### Phase 2: Fix Implementation (NEXT)
1. Move domain engine block BEFORE fusion gate (lines 1791-1962 → lines 1733-1790)
2. Apply domain boost to score before gate check
3. Update debug logging to confirm fix
4. Run backtest comparison:
   - S1_core: baseline (no domain engines)
   - S1_fixed: with gate ordering fix
   - Expected: Trade count increase in S1_fixed

### Phase 3: Re-Optimization (REQUIRED)
1. Fusion thresholds may need adjustment (domain boosts now affect qualification)
2. Re-run S1 optimization with fixed architecture
3. Compare against gold standard (ensure improvement, not degradation)

### Phase 4: System-Wide Rollout
1. Apply same fix to all bear archetypes (S2, S4, S5)
2. Apply same fix to bull archetypes (A, B, C, etc.) if they have domain engines
3. Update all configs with re-optimized thresholds
4. Full system validation

---

## Testing Strategy

### Unit Test (Proof of Concept)

File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/test_domain_engine_gate_bug.py`

**Test Cases:**
1. Marginal signal (score=0.38, threshold=0.40) + Wyckoff Spring (2.5x)
   - Current: REJECTED (score < threshold)
   - Expected with fix: ACCEPTED (0.38 * 2.5 = 0.95 > 0.40)

2. Strong signal (score=0.55) + Wyckoff Spring
   - Current: ACCEPTED, domain boost just increases confidence
   - Expected with fix: Still ACCEPTED, higher confidence

### Integration Test (Backtest)

```bash
# Run S1_full backtest and count [ARCH_BUG] warnings
python bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bear_market_v1.json \
    --start 2022-01-01 \
    --end 2022-12-31 \
    2>&1 | grep "\[ARCH_BUG\]" | wc -l
```

**Expected:** Multiple warnings showing missed Wyckoff Spring signals.

### Regression Test (Post-Fix)

```bash
# Compare trade counts before/after fix
# S1_core (baseline): 110 trades
# S1_fixed (with domain engines): Should be > 110 trades if bug fixed
```

---

## Risk Assessment

### High Risk: Threshold Invalidation
- **Issue:** Fusion thresholds optimized assuming current (broken) architecture
- **Impact:** Fixing bug may degrade performance until re-optimization
- **Mitigation:** Re-optimize thresholds after fix, use walk-forward validation

### Medium Risk: Trade Count Explosion
- **Issue:** Domain boosts may qualify too many marginal signals
- **Impact:** Increased trade frequency, potential over-trading
- **Mitigation:** Monitor trade count delta, adjust domain boost multipliers if needed

### Low Risk: Veto Logic Unchanged
- **Issue:** None - veto logic already works correctly
- **Impact:** Safety preserved
- **Mitigation:** N/A

---

## Reliability & Data Integrity Considerations

### Fault Tolerance
- ✅ Veto logic preserved (safety first)
- ✅ Graceful degradation (if domain features missing, boost=1.0)
- ✅ Explicit logging of domain boost application

### Observability
- ✅ Debug logging added to track:
  - Signals rejected that would have passed with domain boost
  - Domain boost values and signal sources
  - Score before/after domain adjustment
- ✅ Metadata includes domain boost details for audit trail

### Data Consistency
- ✅ Feature store already has all domain features
- ✅ No schema changes required
- ✅ Backward compatible (configs without domain engines work as before)

---

## Deliverables

1. **Debug Logging** ✅
   - File: `engine/archetypes/logic_v2_adapter.py` lines 1733-1788
   - Functionality: Logs [ARCH_BUG] warnings when domain engines would have saved signals

2. **Test Script** ✅
   - File: `bin/test_domain_engine_gate_bug.py`
   - Functionality: Demonstrates gate ordering bug with synthetic test cases

3. **Architecture Proposal** ✅
   - Recommendation: Option 1 (Move domain engines before gate)
   - Rationale: Simple, reliable, maintains safety guarantees

4. **This Report** ✅
   - Root cause analysis
   - Fix proposal with implementation plan
   - Risk assessment and testing strategy

---

## Next Actions

### Immediate (DO NOW)
1. Run S1_full backtest and collect [ARCH_BUG] warnings
2. Quantify: How many signals are we missing due to this bug?
3. Decision: Proceed with Option 1 fix or investigate further?

### Short-term (THIS SPRINT)
1. Implement Option 1 fix (move domain engines before gate)
2. Run A/B backtest: S1_core vs S1_fixed
3. Re-optimize S1 thresholds with fixed architecture

### Long-term (NEXT SPRINT)
1. Apply fix to all bear archetypes (S2, S4, S5)
2. Apply fix to bull archetypes if applicable
3. Full system re-optimization and validation

---

## Conclusion

The domain engines are fully implemented and have valid feature signals, but an architectural gate ordering bug prevents them from affecting trade decisions. The fix is straightforward (move domain boost before threshold gate) and maintains all safety guarantees. Debug logging is now in place to quantify the impact of this bug on real backtests.

**Recommendation:** Proceed with Option 1 fix after running diagnostic backtest to quantify missed signals.

**Owner:** Backend Architect
**Reviewers:** Quant Team, System Reliability
**Status:** READY FOR REVIEW
