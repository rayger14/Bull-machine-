# DOMAIN FEATURE AVAILABILITY REPORT (2022)

**Date:** 2025-12-10
**Analyst:** Backend Architect Agent
**Investigation:** Wyckoff, SMC, and Temporal Features in 2022 Data

---

## EXECUTIVE SUMMARY

**CRITICAL FINDING:** 11 out of 16 domain features (69%) that Agent 2 wired into S1/S4/S5 archetypes **DO NOT EXIST** in the 2022 feature store.

**ROOT CAUSE:** Agent 2 wired features that were either:
1. Never computed in the feature engineering pipeline
2. Defined in `registry.py` but not actually generated
3. Named differently than expected

**IMPACT:** Agent 2's domain feature wiring had **ZERO EFFECT** on archetype behavior in 2022 because the features don't exist in the data.

---

## FEATURE STORE DETAILS

**File Analyzed:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

**Data Range (2022):**
- Start: 2022-01-01 19:00:00 UTC
- End: 2022-12-31 23:00:00 UTC
- Total Rows: 8,741
- Total Columns: 171

---

## WYCKOFF FEATURES (8 features checked)

| Feature | Exists | Status | Non-Null % | Mean | Std | Notes |
|---------|--------|--------|------------|------|-----|-------|
| `wyckoff_spring_a` | ✅ YES | ✅ USABLE | 100.0% | 0.0002 | 0.0151 | 2 events in 2022 |
| `wyckoff_spring_b` | ✅ YES | ❌ ALL ZEROS | 100.0% | 0.0000 | 0.0000 | Never triggered |
| `wyckoff_ps` | ❌ NO | ❌ MISSING | N/A | N/A | N/A | **WIRED BUT MISSING** |
| `wyckoff_utad` | ✅ YES | ❌ ALL ZEROS | 100.0% | 0.0000 | 0.0000 | Never triggered |
| `wyckoff_sow` | ✅ YES | ✅ USABLE | 100.0% | 0.0071 | 0.0839 | 62 events in 2022 |
| `wyckoff_phase_abc` | ✅ YES | ✅ USABLE | 100.0% | Categorical: D | N/A | 5 unique phases |
| `wyckoff_pti_confluence` | ❌ NO | ❌ MISSING | N/A | N/A | N/A | **WIRED BUT MISSING** |
| `wyckoff_pti_score` | ❌ NO | ❌ MISSING | N/A | N/A | N/A | **WIRED BUT MISSING** |

**Wyckoff Summary:**
- ✅ Usable: 3/8 (37.5%)
- ❌ Missing: 2/8 (25%)
- ❌ All Zeros: 2/8 (25%)

---

## SMC FEATURES (6 features checked)

| Feature | Exists | Status | Notes |
|---------|--------|--------|-------|
| `smc_score` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `smc_bos` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `smc_liquidity_sweep` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `smc_supply_zone` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `hob_demand_zone` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `hob_supply_zone` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |

**SMC Summary:**
- ✅ Usable: 0/6 (0%)
- ❌ Missing: 6/6 (100%)

**CRITICAL:** ZERO SMC features exist in the feature store despite being heavily referenced in:
- `engine/archetypes/logic_v2_adapter.py` (5 archetypes check `smc_score`)
- `engine/fusion/domain_fusion.py` (SMC scoring logic exists)
- `engine/features/registry.py` (Feature spec defined)

---

## TEMPORAL FEATURES (3 features checked)

| Feature | Exists | Status | Notes |
|---------|--------|--------|-------|
| `temporal_confluence` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `temporal_support_cluster` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** |
| `wyckoff_pti_confluence` | ❌ NO | ❌ MISSING | **WIRED BUT MISSING** (duplicate from Wyckoff) |

**Temporal Summary:**
- ✅ Usable: 0/3 (0%)
- ❌ Missing: 3/3 (100%)

**CRITICAL:** ZERO temporal features exist despite:
- `engine/temporal/temporal_fusion.py` having full implementation
- `engine/archetypes/logic_v2_adapter.py` checking `temporal_confluence` in 4 archetypes

---

## WHAT WYCKOFF FEATURES ACTUALLY EXIST?

The feature store contains 30 Wyckoff features with data:

**Working Event Features (binary flags):**
- `wyckoff_spring_a` (2 events in 2022) ✅
- `wyckoff_sow` (62 events in 2022) ✅
- `wyckoff_sos` (40 events) ✅
- `wyckoff_lps` (1,611 events) ✅
- `wyckoff_lpsy` (1,453 events) ✅
- `wyckoff_st` (5,234 events) ✅
- `wyckoff_ar` (645 events) ✅
- `wyckoff_as` (608 events) ✅

**Non-functioning Event Features (all zeros):**
- `wyckoff_spring_b` (0 events) ❌
- `wyckoff_bc` (0 events) ❌
- `wyckoff_ut` (0 events) ❌
- `wyckoff_utad` (0 events) ❌
- `wyckoff_sc` (1 event) ❌

**Phase Features:**
- `wyckoff_phase_abc` (Categorical: A, B, C, D, neutral) ✅
- `tf1d_wyckoff_phase` (Categorical: 8 phases including "markdown") ✅
- `wyckoff_sequence_position` (5 positions) ✅

**Missing Features (defined in code but not computed):**
- `wyckoff_ps` (Preliminary Support) ❌
- `wyckoff_pti_confluence` (PTI trap confluence) ❌
- `wyckoff_pti_score` (Composite trap score) ❌

---

## WHERE AGENT 2 WIRED MISSING FEATURES

### S1 (Liquidity Vacuum) - `engine/archetypes/logic_v2_adapter.py:1593-1611`
```python
wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)  # ❌ MISSING
smc_score = self.g(context.row, 'smc_score', 0.0)  # ❌ MISSING
wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)  # ❌ MISSING
```

### S2 (Failed Rally) - `engine/archetypes/logic_v2_adapter.py:1762-1783`
```python
wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)  # ❌ MISSING
smc_score = self.g(context.row, 'smc_score', 0.0)  # ❌ MISSING
wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)  # ❌ MISSING
```

### S4 (Funding Divergence) - `engine/archetypes/logic_v2_adapter.py:1934-1952`
```python
wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)  # ❌ MISSING
smc_score = self.g(context.row, 'smc_score', 0.0)  # ❌ MISSING
wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)  # ❌ MISSING
```

### S5 (Long Squeeze) - `engine/archetypes/logic_v2_adapter.py:2695-2715`
```python
smc_score = self.g(context.row, 'smc_score', 0.0)  # ❌ MISSING
wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)  # ❌ MISSING
wyckoff_pti_score = self.g(context.row, 'wyckoff_pti_score', 0.0)  # ❌ MISSING
```

**Result:** All `self.g()` calls return default values (False, 0.0) because columns don't exist.

---

## WHY THESE FEATURES DON'T EXIST

### 1. Feature Registry vs. Computation Mismatch

**Defined in `engine/features/registry.py`:**
```python
FeatureSpec("smc_score", "float64", 2, False, "SMC confluence score")
FeatureSpec("wyckoff_pti_confluence", "bool", 2, False, "PTI-Wyckoff trap confluence")
FeatureSpec("wyckoff_pti_score", "float64", 2, False, "Composite trap score")
```

**BUT:** These specs don't automatically compute features. They're just schema definitions.

### 2. Computation Code Exists But Not Integrated

**PTI Features (`engine/wyckoff/events.py:1019-1045`):**
```python
def detect_wyckoff_pti_confluence(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Detect confluence between Wyckoff traps and Precision Trade Indicator (PTI).

    Returns:
        - wyckoff_pti_confluence (bool)
        - wyckoff_pti_score (float)
    """
    df['wyckoff_pti_confluence'] = False
    df['wyckoff_pti_score'] = 0.0

    # ... implementation exists ...
```

**Problem:** This function is defined but **NEVER CALLED** in the feature pipeline.

### 3. SMC Features Have Stub Implementation

**In `engine/features/builder.py:323`:**
```python
df['smc_score'] = 0.5  # Hard-coded stub!
```

This stub is defined but never actually added to the final parquet output.

### 4. Temporal Features Computed in Memory Only

**In `engine/temporal/temporal_fusion.py:109`:**
```python
def compute_temporal_confluence(self, context: RuntimeContext) -> float:
    """Compute temporal confluence at runtime."""
```

This computes confluence **at runtime** but doesn't persist to feature store.

---

## IMPACT ANALYSIS

### Why Agent 2's Wiring Had Zero Effect

1. **Default Values Always Returned:**
   - `smc_score` always returns `0.0` (threshold checks never trigger)
   - `wyckoff_ps` always returns `False` (never boosts score)
   - `wyckoff_pti_confluence` always returns `False` (never boosts score)

2. **Boost Logic Never Executes:**
   ```python
   # S1 example (logic_v2_adapter.py:1602-1605)
   smc_score = self.g(context.row, 'smc_score', 0.0)  # Returns 0.0
   if smc_score > 0.5:  # Never true
       fusion_score += 0.1  # Never executes
       domain_signals.append("smc_bullish_structure")  # Never happens
   ```

3. **Domain Signals Never Appear:**
   - No "smc_bullish_structure" signals in logs
   - No "temporal_confluence" signals in logs
   - No "wyckoff_ps" signals in logs

---

## SAMPLE DATA VERIFICATION

**Wyckoff Spring A Events in 2022 (only 2):**
```
2022-05-12 07:00:00 UTC  (Bitcoin bottoming around $26k)
2022-11-11 14:00:00 UTC  (Bitcoin bottoming around $16k - actual bottom)
```

**Wyckoff Sign of Weakness Events in 2022 (62 total):**
```
2022-01-05 19:00:00 UTC
2022-01-05 20:00:00 UTC
2022-01-07 03:00:00 UTC
... (59 more)
```

These are **actual working features** that could have been used instead of the missing ones.

---

## RECOMMENDATIONS

### Immediate Actions (Fix Missing Features)

**Option 1: Use Existing Wyckoff Features**
Instead of wiring missing features, use features that actually exist:

| Missing Feature | Replacement | Notes |
|----------------|-------------|-------|
| `wyckoff_ps` | `wyckoff_lps` or `wyckoff_spring_a` | LPS fired 1,611 times vs Spring A only 2 times |
| `wyckoff_pti_confluence` | `wyckoff_spring_a` + `wyckoff_sow` | Combine existing events |
| `smc_score` | Remove or stub to 0.5 | No SMC implementation exists |
| `temporal_confluence` | Remove or compute at runtime | Already computed but not stored |

**Option 2: Compute Missing Features Retroactively**
1. Run `engine/wyckoff/events.py:detect_wyckoff_pti_confluence()` on historical data
2. Implement SMC feature computation in `engine/features/builder.py`
3. Persist temporal features from `engine/temporal/temporal_fusion.py`
4. Re-save feature store parquet files

**Option 3: Remove Dead Code**
If features won't be computed, remove all references:
- Delete boost logic in `engine/archetypes/logic_v2_adapter.py`
- Remove specs from `engine/features/registry.py`
- Clean up fusion code in `engine/fusion/domain_fusion.py`

### Long-Term Actions

1. **Feature Pipeline Audit:**
   - Document which features in `registry.py` are actually computed
   - Create CI test: "All registry features must exist in test parquet file"

2. **Feature Coverage Tests:**
   - Unit test: Load feature store, assert required columns exist
   - Integration test: Mock archetype, verify domain features are non-zero

3. **Documentation:**
   - Create `FEATURE_STORE_SCHEMA.md` listing all computed features
   - Add comments in code: `# Feature 'X' must exist in feature store`

---

## CONCLUSION

**Agent 2's domain feature wiring was a no-op** because:
1. 69% of wired features don't exist in the feature store
2. Feature definitions exist in code but aren't called during feature computation
3. No tests verify feature store completeness

**Next Steps:**
1. Decide: Fix features retroactively OR remove dead wiring code
2. If fixing: Backfill 2022-2024 data with missing features
3. If removing: Clean up logic_v2_adapter.py domain boost code
4. Add tests to prevent future registry/computation mismatches

**Files to Modify:**
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/logic_v2_adapter.py` (remove or fix domain boosts)
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/features/builder.py` (add missing feature computation)
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/add_derived_features.py` (call PTI detection)

---

**Report Generated:** 2025-12-10
**Script Used:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/check_domain_features_2022.py`
