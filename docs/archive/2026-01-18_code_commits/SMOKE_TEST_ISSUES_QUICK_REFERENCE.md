# Smoke Test Issues - Quick Reference

**Last Updated**: 2025-12-15
**Test**: Q1 2023 (2,157 bars)

---

## Critical Issues (P0)

### 1. Retest Cluster (L) - BROKEN

**Issue**: 1,586 timestamp arithmetic errors
```
Addition/subtraction of integers and integer-arrays with Timestamp is no longer supported.
Instead of adding/subtracting `n`, use `n * obj.freq`
```

**Impact**: Zero signals produced
**Root Cause**: Outdated pandas timestamp arithmetic in `_check_L()`
**Fix**: Replace `ts + n` with `ts + pd.Timedelta(hours=n)` or `ts + n * df.index.freq`
**Priority**: P0 - CRITICAL
**Assignee**: TBD

---

### 2. Seven Archetypes Producing Zero Signals

| Archetype | Name | Issue | Likely Cause |
|-----------|------|-------|--------------|
| S1 | Liquidity Vacuum | 0 signals | Regime filter (needs crisis/risk_off) or thresholds too strict |
| S5 | Long Squeeze | 0 signals | Regime filter or extreme funding conditions rare in Q1 2023 |
| A | Spring | 0 signals | Wyckoff feature dependencies or thresholds |
| C | Wick Trap | 0 signals | Wick feature dependencies or thresholds |
| L | Retest Cluster | 0 signals + errors | Timestamp arithmetic bug (see #1) |
| M | Confluence Breakout | 0 signals | Multiple feature dependencies or thresholds |
| S8 | Volume Fade Chop | 0 signals | Chop detection logic or thresholds |

**Action Items**:
1. Review regime filters for S1, S5 (may block all signals in neutral regime)
2. Create production configs with proper threshold values
3. Validate feature dependencies exist in test data
4. Consider adding regime_override for smoke testing

**Priority**: P0 - CRITICAL
**Assignee**: TBD

---

### 3. Confidence Score Exceeds 5.0 Limit

**Archetype**: H (Momentum Continuation)
**Issue**: Max score = 5.52 (limit is 5.0)
**Impact**: Violates specification, may cause downstream issues
**Root Cause**: Domain boost multiplication without capping
**Fix**: Add `min(5.0, score)` cap in ArchetypeLogic or domain layer
**Priority**: P0 - CRITICAL
**Assignee**: TBD

---

## High Priority Issues (P1)

### 4. Domain Boost Metadata Missing

**Archetypes Affected**: 13/16 (all except H, B, S4)
**Issue**: `domain_boost_pct = 0%` for most archetypes
**Expected**: >50% of signals should show domain boosts

**Possible Causes**:
1. `_apply_domain_engines()` not setting metadata correctly
2. Metadata key mismatch (looking for wrong field)
3. Domain engines not enabled in minimal smoke test config

**Debugging Steps**:
```python
# Check what metadata is actually returned
result = logic._check_E(ctx)
print(f"Metadata: {result[2]}")
# Look for: domain_boost, boost_multiplier, engine_boosts, etc.
```

**Priority**: P1 - HIGH
**Assignee**: TBD

---

### 5. Direction Metadata Missing

**Archetypes Affected**: All 16 archetypes
**Issue**: All report "No direction info"
**Impact**: Cannot validate Bull archetypes are LONG-biased

**Root Causes**:
1. Archetype methods not setting `direction` field in metadata
2. Metadata key name inconsistent (direction vs side vs bias)

**Fix**:
```python
# Standardize metadata format
metadata = {
    'direction': 'LONG',  # or 'SHORT'
    'confidence': score,
    'domain_boost': boost_multiplier,
    # ... other fields
}
return (True, final_score, metadata)
```

**Priority**: P1 - HIGH
**Assignee**: TBD

---

## Medium Priority Issues (P2)

### 6. Low Signal Counts

**Archetypes**:
- S3 (Whipsaw): 1 signal (too low)
- K (Trap Within Trend): 15 signals (acceptable but low)
- D (Failed Continuation): 13 signals (acceptable but low)

**Issue**: May have overly strict thresholds
**Action**: Review and relax thresholds in production configs
**Priority**: P2 - MEDIUM

---

### 7. Metadata Format Inconsistency

**Issue**: Each archetype may return metadata with different keys/structure
**Impact**: Hard to extract standardized fields (direction, domain_boost, etc.)

**Recommendation**: Create standardized metadata schema
```python
@dataclass
class ArchetypeMetadata:
    direction: str  # 'LONG' or 'SHORT'
    confidence: float
    domain_boost: float = 1.0
    regime: str = 'neutral'
    features_used: List[str] = field(default_factory=list)
    # ... other standard fields
```

**Priority**: P2 - MEDIUM

---

## Low Priority Issues (P3)

### 8. Regime Override for Testing

**Issue**: Smoke test uses neutral regime, blocking some archetypes
**Enhancement**: Add test mode that bypasses regime filters
```python
config = {
    'test_mode': True,  # Bypass regime filters
    'regime_override': 'all',  # Allow all regimes
}
```

**Priority**: P3 - LOW

---

### 9. Feature Dependency Validation

**Enhancement**: Add pre-flight check to validate required features exist
```python
def validate_features(df, archetype):
    required = ARCHETYPE_FEATURES[archetype]
    missing = [f for f in required if f not in df.columns]
    if missing:
        raise ValueError(f"{archetype} missing features: {missing}")
```

**Priority**: P3 - LOW

---

## Working Archetypes (No Issues)

**High Confidence** (>50 signals, domain boosts working):
- H (Momentum Continuation): 565 signals, 2.13x avg boost ✅
- B (Order Block Retest): 46 signals, 2.09x avg boost ✅

**Medium Confidence** (>50 signals, no boost metadata):
- E (Volume Exhaustion): 124 signals ✅
- G (Liquidity Sweep): 97 signals ✅
- F (Exhaustion Reversal): 75 signals ✅

**Low-Medium Confidence** (10-50 signals):
- S4 (Funding Divergence): 14 signals, 1.64x avg boost ✅
- K (Trap Within Trend): 15 signals (needs tuning)
- D (Failed Continuation): 13 signals (needs tuning)

---

## Test Results Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Archetypes with signals | 16/16 | 9/16 | ❌ FAIL |
| Signal diversity (overlap) | <20% | 12.8% | ✅ PASS |
| Valid confidence scores | 16/16 | 15/16 | ❌ FAIL |
| Domain boost detection | >50% | 19% | ❌ FAIL |
| Execution time | <30s | 8.9s | ✅ PASS |

**Overall**: 2/5 metrics passed

---

## Quick Fix Checklist

- [ ] Fix Retest Cluster (L) timestamp arithmetic
- [ ] Create production configs for all 16 archetypes
- [ ] Add confidence score capping at 5.0
- [ ] Investigate S1 regime filter (crisis/risk_off requirement)
- [ ] Investigate S5 regime filter
- [ ] Add direction field to all archetype metadata
- [ ] Verify domain boost metadata structure
- [ ] Review thresholds for A, C, M, S8
- [ ] Tune thresholds for S3, K, D
- [ ] Re-run smoke test with production configs

---

## Next Test Iteration

**Changes for v2**:
1. Use production configs (not minimal test configs)
2. Test on multiple periods (bull, bear, chop)
3. Add regime override for testing
4. Validate feature dependencies before running
5. Add detailed logging for zero-signal archetypes

**Expected Improvements**:
- 16/16 archetypes producing signals
- Domain boost detection >70%
- Direction metadata present for all
- All scores in valid range [0.0-5.0]

---

## Contact

**Questions**: See SMOKE_TEST_EXECUTIVE_SUMMARY.md for full details
**Test Script**: bin/smoke_test_all_archetypes.py
**Raw Results**: smoke_test_results.json
