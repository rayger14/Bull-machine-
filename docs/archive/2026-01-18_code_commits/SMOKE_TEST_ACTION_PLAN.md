# Smoke Test - Action Plan

**Created**: 2025-12-15
**Status**: ACTIVE
**Owner**: TBD
**Target Completion**: 2025-12-20

---

## Overview

Comprehensive action plan to address the 22 issues identified in the archetype smoke test. Prioritized by impact and effort required.

**Current State**: 9/16 archetypes working (56% pass rate)
**Target State**: 16/16 archetypes working (100% pass rate)
**Timeline**: 5 days

---

## Phase 1: Critical Fixes (Day 1-2)

### Action 1.1: Fix Retest Cluster (L) Timestamp Bug

**Issue**: 1,586 errors blocking all L signals
**Error**: `Addition/subtraction of integers and integer-arrays with Timestamp is no longer supported`

**File**: `/engine/archetypes/logic_v2_adapter.py`
**Method**: `_check_L()`

**Fix**:
```python
# BEFORE (broken):
lookback_ts = ts - 168  # 7 days in hours

# AFTER (fixed):
lookback_ts = ts - pd.Timedelta(hours=168)
```

**Search for all instances**:
```bash
grep -n "ts - [0-9]" engine/archetypes/logic_v2_adapter.py
grep -n "ts + [0-9]" engine/archetypes/logic_v2_adapter.py
```

**Testing**:
```bash
python3 -c "
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
config = {'enable_L': True}
logic = ArchetypeLogic(config)
# Test _check_L on sample data
print('L archetype timestamp fix validated')
"
```

**Effort**: 1 hour
**Impact**: HIGH - Unblocks L archetype completely
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 1.2: Add Confidence Score Capping

**Issue**: Archetype H produces scores up to 5.52 (exceeds 5.0 limit)

**File**: `/engine/archetypes/logic_v2_adapter.py`
**Method**: `_apply_domain_engines()` or individual archetype methods

**Fix Option A** (global cap in domain engines):
```python
def _apply_domain_engines(self, base_score: float, ...) -> tuple:
    # ... existing boost logic ...
    final_score = base_score * total_boost

    # ADD THIS:
    final_score = min(5.0, final_score)  # Cap at 5.0

    return final_score, metadata
```

**Fix Option B** (cap in each archetype):
```python
def _check_H(self, context: RuntimeContext) -> tuple:
    # ... existing logic ...
    final_score = base_score * boost
    final_score = min(5.0, final_score)  # Cap at 5.0
    return (True, final_score, metadata)
```

**Recommendation**: Option A (global) - applies to all archetypes

**Testing**:
```bash
# Run smoke test on H archetype only
python3 bin/smoke_test_all_archetypes.py --archetypes H
# Verify max score <= 5.0
```

**Effort**: 30 minutes
**Impact**: MEDIUM - Fixes spec violation
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 1.3: Create Production Configs for All Archetypes

**Issue**: Smoke test used minimal configs with thresholds=0, blocking many signals

**Task**: Create 16 production-ready JSON configs based on existing S1, S4, S5 configs

**Template**:
```json
{
  "version": "archetype_X_production",
  "profile": "Archetype X - Production Config",
  "archetypes": {
    "use_archetypes": true,
    "enable_X": true,
    "thresholds": {
      "archetype_name": {
        "direction": "long",  // or "short"
        "archetype_weight": 1.0,
        "fusion_threshold": 0.3,
        "min_confidence": 0.4,
        // ... archetype-specific thresholds
      }
    }
  },
  "feature_flags": {
    "enable_wyckoff": true,
    "enable_smc": true,
    "enable_temporal": true,
    "enable_hob": true,
    "enable_fusion": true,
    "enable_macro": true
  }
}
```

**Files to Create**:
```
configs/archetypes/A_spring_production.json
configs/archetypes/B_order_block_production.json
configs/archetypes/C_wick_trap_production.json
configs/archetypes/D_failed_continuation_production.json
configs/archetypes/E_volume_exhaustion_production.json
configs/archetypes/F_exhaustion_reversal_production.json
configs/archetypes/G_liquidity_sweep_production.json
configs/archetypes/H_momentum_continuation_production.json
configs/archetypes/K_trap_within_trend_production.json
configs/archetypes/L_retest_cluster_production.json
configs/archetypes/M_confluence_breakout_production.json
configs/archetypes/S3_whipsaw_production.json
configs/archetypes/S8_volume_fade_chop_production.json
```

**Use Existing Configs as Reference**:
- S1: `configs/variants/s1_full.json`
- S4: `configs/variants/s4_full.json`
- S5: `configs/variants/s5_full.json`

**Effort**: 4 hours (16 configs × 15 min each)
**Impact**: CRITICAL - Unblocks zero-signal archetypes
**Owner**: TBD
**Status**: ⬜ Not Started

---

## Phase 2: High Priority Fixes (Day 3)

### Action 2.1: Add Direction Metadata to All Archetypes

**Issue**: All archetypes report "No direction info"

**Task**: Standardize metadata format to include `direction` field

**Implementation**:
```python
# In each _check_X() method:
metadata = {
    'direction': 'LONG',  # or 'SHORT'
    'archetype': 'spring',  # archetype name
    'confidence': base_score,
    'domain_boost': boost_multiplier,
    'timestamp': str(context.ts),
    # ... other fields
}
return (True, final_score, metadata)
```

**Files to Update** (all 16 archetype methods):
- `_check_A()` through `_check_M()`
- `_check_S1()`, `_check_S3()`, `_check_S4()`, `_check_S5()`, `_check_S8()`

**Expected Directions**:
- Bull archetypes (A-M): `'LONG'`
- S1 (Liquidity Vacuum): `'LONG'` (capitulation reversal)
- S4 (Funding Divergence): `'SHORT'` or `'LONG'` (depends on signal)
- S5 (Long Squeeze): `'LONG'` (long squeeze = buy the dip)
- S3 (Whipsaw): `'LONG'` or `'SHORT'` (depends on direction)
- S8 (Volume Fade): `'NEUTRAL'` (chop detector)

**Testing**:
```python
# Verify direction in metadata
result = logic._check_B(ctx)
assert result[2]['direction'] in ['LONG', 'SHORT', 'NEUTRAL']
```

**Effort**: 2 hours
**Impact**: HIGH - Enables direction validation
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 2.2: Investigate Domain Boost Metadata

**Issue**: 13/16 archetypes show 0% domain boost detection

**Investigation Steps**:

1. **Check what metadata is actually returned**:
```python
# In smoke test, add debug logging
for sig in signals:
    print(f"Metadata keys: {sig['metadata'].keys()}")
    print(f"Full metadata: {sig['metadata']}")
```

2. **Verify `_apply_domain_engines()` implementation**:
```bash
grep -A 20 "def _apply_domain_engines" engine/archetypes/logic_v2_adapter.py
```

3. **Check if domain engines are enabled in minimal config**:
```python
# smoke_test_all_archetypes.py
config = {
    'feature_flags': {
        'enable_wyckoff': True,  # Should be enabled
        'enable_smc': True,
        'enable_temporal': True,
        # ...
    }
}
```

4. **Look for correct metadata key**:
```python
# Possible keys to check:
- domain_boost
- boost_multiplier
- engine_boost
- wyckoff_boost
- smc_boost
- temporal_boost
```

**Expected Outcome**: Identify why 13 archetypes show no boosts

**Next Steps** (based on findings):
- If metadata not set: Fix `_apply_domain_engines()`
- If key mismatch: Update smoke test extraction logic
- If engines disabled: Enable in production configs

**Effort**: 3 hours
**Impact**: HIGH - Critical for production validation
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 2.3: Investigate Zero-Signal Archetypes

**Issue**: 7 archetypes produce 0 signals (S1, S5, A, C, L, M, S8)

**Investigation Per Archetype**:

#### S1 (Liquidity Vacuum)
**Hypothesis**: Regime filter too restrictive
```python
# Check regime requirements
ARCHETYPE_REGIMES = {
    'liquidity_vacuum': ['risk_off', 'crisis'],  # May block all in 'neutral'
}
```
**Test**: Override regime to crisis and re-run
**Expected**: Should produce signals if regime is the issue

#### S5 (Long Squeeze)
**Hypothesis**: Similar to S1 - regime filter issue
**Test**: Check `allowed_regimes` in config
**Expected**: May need to allow 'risk_on' or 'neutral'

#### A (Spring), C (Wick Trap), M (Confluence Breakout)
**Hypothesis**: Wyckoff/feature dependencies missing
**Test**:
```python
# Check required features
required_features = ['wyckoff_score', 'wyckoff_event', 'liquidity_score', ...]
missing = [f for f in required_features if f not in df.columns]
print(f"Missing features: {missing}")
```
**Expected**: May need additional feature engineering

#### L (Retest Cluster)
**Fix**: Already covered in Action 1.1 (timestamp bug)

#### S8 (Volume Fade Chop)
**Hypothesis**: Chop detection threshold too high
**Test**: Lower volume fade threshold temporarily
**Expected**: Should produce signals in low-volume periods

**Effort**: 4 hours (30 min per archetype)
**Impact**: CRITICAL - Unblocks 7 archetypes
**Owner**: TBD
**Status**: ⬜ Not Started

---

## Phase 3: Medium Priority (Day 4)

### Action 3.1: Tune Thresholds for Low-Signal Archetypes

**Archetypes**:
- S3 (Whipsaw): 1 signal → target 10-20
- K (Trap Within Trend): 15 signals → target 30-50
- D (Failed Continuation): 13 signals → target 30-50

**Process**:
1. Review current thresholds in code
2. Identify most restrictive parameters
3. Relax thresholds by 10-20%
4. Re-run smoke test
5. Iterate until target signal count reached

**Example** (S3 Whipsaw):
```json
// Current (hypothetical)
"whipsaw": {
  "min_volatility": 0.05,
  "max_trend_strength": 0.2,
  "volume_threshold": 1.5
}

// Relaxed
"whipsaw": {
  "min_volatility": 0.04,  // -20%
  "max_trend_strength": 0.25,  // +25%
  "volume_threshold": 1.3  // -13%
}
```

**Effort**: 3 hours
**Impact**: MEDIUM - Improves signal coverage
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 3.2: Create Standardized Metadata Schema

**Task**: Define and enforce standard metadata format

**Create**: `/engine/archetypes/metadata.py`
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ArchetypeSignalMetadata:
    """Standardized metadata for all archetype signals."""

    # Core fields (required)
    archetype: str  # 'spring', 'order_block_retest', etc.
    direction: str  # 'LONG', 'SHORT', or 'NEUTRAL'
    confidence: float  # Base confidence before boosts

    # Domain boost fields
    domain_boost: float = 1.0  # Total boost multiplier
    wyckoff_boost: float = 1.0
    smc_boost: float = 1.0
    temporal_boost: float = 1.0
    hob_boost: float = 1.0
    macro_boost: float = 1.0

    # Context fields
    regime: str = 'neutral'
    timestamp: str = ''

    # Feature fields (optional)
    features_used: List[str] = field(default_factory=list)
    feature_values: Dict[str, float] = field(default_factory=dict)

    # Additional metadata
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'archetype': self.archetype,
            'direction': self.direction,
            'confidence': self.confidence,
            'domain_boost': self.domain_boost,
            'regime': self.regime,
            # ... all fields
        }
```

**Update All Archetypes**:
```python
from engine.archetypes.metadata import ArchetypeSignalMetadata

def _check_B(self, context: RuntimeContext) -> tuple:
    # ... existing logic ...

    # Create standardized metadata
    metadata = ArchetypeSignalMetadata(
        archetype='order_block_retest',
        direction='LONG',
        confidence=base_score,
        domain_boost=boost_multiplier,
        regime=context.regime_label,
        timestamp=str(context.ts),
        features_used=['orderblock_score', 'liquidity_score'],
    )

    return (True, final_score, metadata.to_dict())
```

**Effort**: 4 hours
**Impact**: MEDIUM - Improves maintainability
**Owner**: TBD
**Status**: ⬜ Not Started

---

## Phase 4: Testing & Validation (Day 5)

### Action 4.1: Re-run Smoke Test with Production Configs

**Prerequisites**:
- All Phase 1-3 actions completed
- Production configs created for all archetypes

**Command**:
```bash
python3 bin/smoke_test_all_archetypes.py --config-dir configs/archetypes/
```

**Expected Results**:
- 16/16 archetypes produce signals ✅
- Domain boost detection >70% ✅
- Direction metadata present for all ✅
- All scores in [0.0-5.0] ✅
- Diversity maintained <20% overlap ✅

**Success Criteria**:
```
✅ PASS: All archetypes produce >0 signals (16/16)
✅ PASS: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0] (16/16)
✅ PASS: Domain boosts present in >50% of signals (>8/16)

Overall: 4/4 criteria passed
```

**Effort**: 1 hour
**Impact**: HIGH - Validates all fixes
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 4.2: Extended Testing on Multiple Regimes

**Test Periods**:
1. **Bull Market**: Q1 2024 (Jan-Mar 2024)
2. **Bear Market**: Q2 2022 (Apr-Jun 2022)
3. **Chop Market**: Q3 2023 (Jul-Sep 2023)

**Script**:
```bash
# Test on bull period
python3 bin/smoke_test_all_archetypes.py \
  --start 2024-01-01 --end 2024-04-01 \
  --report bull_market_test.md

# Test on bear period
python3 bin/smoke_test_all_archetypes.py \
  --start 2022-04-01 --end 2022-07-01 \
  --report bear_market_test.md

# Test on chop period
python3 bin/smoke_test_all_archetypes.py \
  --start 2023-07-01 --end 2023-10-01 \
  --report chop_market_test.md
```

**Expected**:
- Bull archetypes fire more in bull period
- Bear archetypes fire more in bear period
- Chop archetypes fire more in chop period

**Effort**: 2 hours
**Impact**: MEDIUM - Validates regime routing
**Owner**: TBD
**Status**: ⬜ Not Started

---

### Action 4.3: Create Regression Test Suite

**Task**: Prevent future regressions

**Create**: `/tests/test_archetypes_smoke.py`
```python
import pytest
from bin.smoke_test_all_archetypes import test_archetype, load_data

def test_all_archetypes_produce_signals():
    """Ensure all 16 archetypes produce at least 1 signal in Q1 2023."""
    df, test_df = load_data('data/...', '2023-01-01', '2023-04-01')

    archetypes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
                  'S1', 'S3', 'S4', 'S5', 'S8']

    for arch in archetypes:
        result = test_archetype(arch, df, test_df)
        assert result['stats']['count'] > 0, f"{arch} produced zero signals"

def test_confidence_scores_in_range():
    """Ensure all confidence scores in [0.0-5.0]."""
    # ... test logic ...

def test_diversity_maintained():
    """Ensure overlap remains <20%."""
    # ... test logic ...

# Run with: pytest tests/test_archetypes_smoke.py
```

**Effort**: 2 hours
**Impact**: HIGH - Prevents regressions
**Owner**: TBD
**Status**: ⬜ Not Started

---

## Summary Checklist

### Phase 1: Critical (Day 1-2)
- [ ] 1.1 Fix Retest Cluster timestamp bug (1h)
- [ ] 1.2 Add confidence score capping (30m)
- [ ] 1.3 Create production configs for all archetypes (4h)

### Phase 2: High Priority (Day 3)
- [ ] 2.1 Add direction metadata to all archetypes (2h)
- [ ] 2.2 Investigate domain boost metadata (3h)
- [ ] 2.3 Investigate zero-signal archetypes (4h)

### Phase 3: Medium Priority (Day 4)
- [ ] 3.1 Tune thresholds for low-signal archetypes (3h)
- [ ] 3.2 Create standardized metadata schema (4h)

### Phase 4: Testing (Day 5)
- [ ] 4.1 Re-run smoke test with production configs (1h)
- [ ] 4.2 Extended testing on multiple regimes (2h)
- [ ] 4.3 Create regression test suite (2h)

**Total Estimated Effort**: 28.5 hours (~4 days for 1 developer)

---

## Success Metrics

**Before Fixes**:
- Archetypes working: 9/16 (56%)
- Domain boost detection: 3/16 (19%)
- Direction metadata: 0/16 (0%)
- Success criteria passed: 1/4 (25%)

**After Fixes** (Target):
- Archetypes working: 16/16 (100%) ✅
- Domain boost detection: >12/16 (>75%) ✅
- Direction metadata: 16/16 (100%) ✅
- Success criteria passed: 4/4 (100%) ✅

---

## Risk Mitigation

### Risk 1: Zero-signal archetypes still broken after config changes
**Mitigation**: Add debug logging to identify exact failure point
**Fallback**: Disable problematic archetypes, document for future fix

### Risk 2: Domain boost investigation reveals architectural issue
**Mitigation**: Allocate extra time for deeper debugging
**Fallback**: Accept lower domain boost detection rate (>50% instead of >75%)

### Risk 3: Threshold tuning introduces overfitting
**Mitigation**: Validate on out-of-sample data (2024)
**Fallback**: Use conservative thresholds, accept lower signal counts

---

## Communication Plan

**Daily Standup**: Report progress on checklist items
**Blockers**: Escalate if any item takes >2x estimated time
**Final Report**: Update executive summary with results

---

## Next Iteration (Future)

**Enhancements for v2**:
1. Automated threshold optimization using Optuna
2. Multi-period validation (walk-forward)
3. Signal quality metrics (not just quantity)
4. Feature importance analysis per archetype
5. Real-time smoke test in CI/CD pipeline

---

**Document Owner**: TBD
**Last Updated**: 2025-12-15
**Status**: DRAFT - Pending approval
