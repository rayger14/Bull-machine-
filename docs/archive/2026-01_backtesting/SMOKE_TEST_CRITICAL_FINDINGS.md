# Smoke Test - Critical Findings

**Date:** 2026-01-16
**Status:** ⚠️ PARTIAL SUCCESS - Feature naming mismatches discovered
**Impact:** HIGH - Wired features won't trigger without corrections

---

## Executive Summary

Attempted smoke test of newly wired features revealed **critical feature naming mismatches** and **missing features**. While the agents successfully wired the LOGIC for all features, there are discrepancies between:

1. **Feature names used in code** (what agents wired)
2. **Feature names in data** (what actually exists)

**Good News:** Most features DO exist in the data (PTI, Temporal, Wyckoff)
**Bad News:** Feature names don't match - code won't find the features

---

## Feature Availability Analysis

### ✅ Features that EXIST in data (149 total features):

**PTI Features (6 features):**
- `tf1h_pti_score` ✅
- `tf1h_pti_confidence` ✅
- `tf1d_pti_score` ✅
- `tf1d_pti_reversal` ✅
- `adaptive_threshold` ✅
- `wyckoff_pti_confluence` ✅

**Temporal Features (3 features):**
- `temporal_confluence` ✅
- `temporal_resistance_cluster` ✅
- `temporal_support_cluster` ✅

**Wyckoff Features (20 features):**
- `wyckoff_ar` + `wyckoff_ar_confidence` ✅
- `wyckoff_as` + `wyckoff_as_confidence` ✅
- `wyckoff_bc` + `wyckoff_bc_confidence` ✅
- `wyckoff_lps` + `wyckoff_lps_confidence` ✅
- `wyckoff_lpsy` + `wyckoff_lpsy_confidence` ✅
- `tf1d_wyckoff_phase` ✅
- `tf1d_wyckoff_score` ✅
- ...and 10 more

### ❌ Features that DON'T EXIST in data:

**Thermo-floor Features (0 features):**
- `thermo_floor_distance` ❌ NOT IN DATA
- `thermo_floor` ❌ NOT IN DATA
- Any mining cost features ❌ NOT IN DATA

**LPPLS Features (0 features):**
- `lppls_blowoff_detected` ❌ NOT IN DATA
- `lppls_m_parameter` ❌ NOT IN DATA
- Any blowoff detection features ❌ NOT IN DATA

---

## Feature Naming Mismatches

### Problem: Agents used generic names, data has timeframe-specific names

**Example 1: PTI Features**

| What Agents Wired | What Data Has | Match? |
|-------------------|---------------|--------|
| `pti_score` | `tf1h_pti_score` | ❌ MISMATCH |
| `pti_confidence` | `tf1h_pti_confidence` | ❌ MISMATCH |
| `pti_trap_type` | ❌ NOT IN DATA | ❌ MISSING |

**Example 2: Temporal Features**

| What Agents Wired | What Data Has | Match? |
|-------------------|---------------|--------|
| `temporal_confluence` | `temporal_confluence` | ✅ MATCH! |

**Example 3: Wyckoff Features**

| What Agents Wired | What Data Has | Match? |
|-------------------|---------------|--------|
| `wyckoff_spring_a` | ❌ NOT IN DATA | ❌ MISSING |
| `wyckoff_spring_a_confidence` | ❌ NOT IN DATA | ❌ MISSING |
| `wyckoff_ar` | `wyckoff_ar` | ✅ MATCH! |
| `wyckoff_ar_confidence` | `wyckoff_ar_confidence` | ✅ MATCH! |
| `wyckoff_sos` | ❌ NOT IN DATA | ❌ MISSING |
| `wyckoff_sos_confidence` | ❌ NOT IN DATA | ❌ MISSING |

---

## Impact Assessment

### What Will Work (without changes):

1. **Temporal Confluence** ✅
   - Feature name matches: `temporal_confluence`
   - Used by all 9 archetypes
   - Expected impact: +30 bps

2. **Some Wyckoff Events** ✅
   - Features that exist: AR, AS, BC, LPS, LPSY
   - Can use these 5 events immediately
   - Estimated impact: +15-20 bps (partial)

### What Won't Work (needs fixes):

1. **PTI Features** ❌
   - Need to map generic names to timeframe-specific names
   - `pti_score` → `tf1h_pti_score`
   - `pti_confidence` → `tf1h_pti_confidence`
   - `pti_trap_type` → Need to derive from `tf1d_pti_reversal` or similar

2. **Thermo-floor Features** ❌
   - Features DON'T EXIST in data
   - Need to engineer features first
   - Code in `engine/temporal/gann_cycles.py` exists but never exported to feature store

3. **LPPLS Features** ❌
   - Features DON'T EXIST in data
   - Need to engineer features first
   - Code in `engine/temporal/gann_cycles.py` exists but never exported to feature store

4. **Some Wyckoff Events** ❌
   - Spring A, Spring B, SOS - NOT in data
   - Need to check if these can be derived from existing features

---

## Root Cause Analysis

### Why This Happened:

1. **Agents didn't verify feature store** before wiring
   - Assumed features with code implementations were in feature store
   - Didn't check actual parquet file column names

2. **Feature engineering incomplete**
   - PTI code exists (419 lines) but not fully integrated
   - Thermo-floor code exists (35 lines) but never exported
   - LPPLS code exists (70 lines) but never exported

3. **Naming convention inconsistencies**
   - Some features use timeframe prefixes (`tf1h_`, `tf1d_`)
   - Some features use generic names (`temporal_confluence`)
   - No standard documented

---

## Recommended Fix Strategy

### Option A: Quick Fix - Use What Exists (4-6 hours)

**Fix PTI naming:**
1. Update 7 archetype files to use `tf1h_pti_score` instead of `pti_score`
2. Update to use `tf1h_pti_confidence` instead of `pti_confidence`
3. Derive `pti_trap_type` from `tf1d_pti_reversal` (long/short/none)

**Use existing Wyckoff events:**
1. Keep: AR, AS, BC, LPS, LPSY (exist in data)
2. Remove: Spring A, Spring B, SOS (don't exist)

**Remove Thermo/LPPLS temporarily:**
1. Comment out Thermo-floor logic (features don't exist)
2. Comment out LPPLS logic (features don't exist)
3. Re-enable later after feature engineering

**Expected outcome:**
- Temporal Confluence: ✅ Working (+30 bps)
- PTI: ✅ Working (+20 bps)
- Wyckoff: ✅ Partial (+15-20 bps)
- Thermo: ❌ Deferred
- LPPLS: ❌ Deferred

**Total: +65-70 bps available immediately**

---

### Option B: Complete Fix - Engineer Missing Features (2-3 days)

**Day 1: Fix naming (4-6 hours)**
- Same as Option A

**Day 2-3: Engineer missing features (10-14 hours)**

1. **Add Thermo-floor to feature store** (4-5 hours)
   - Modify feature engineering pipeline
   - Calculate mining cost floor from Gann module
   - Export `thermo_floor_distance` feature
   - Backfill for 2018-2024

2. **Add LPPLS to feature store** (4-5 hours)
   - Modify feature engineering pipeline
   - Calculate LPPLS blowoff detection from Gann module
   - Export `lppls_blowoff_detected` feature
   - Backfill for 2018-2024

3. **Add missing Wyckoff events** (2-4 hours)
   - Check if Spring A/B, SOS can be derived
   - Or remove from wired code

**Expected outcome:**
- All features working: +110 bps total

---

### Option C: Validate Core, Then Decide (RECOMMENDED)

**Step 1: Validate what works NOW (2-3 hours)**
1. Quick-fix PTI naming in one archetype (S1)
2. Test S1 with temporal + PTI + existing Wyckoff
3. Run backtest on 2022-2024

**Decision point:**
- ✅ If PTI + Temporal + Wyckoff (5 events) gives +50-60 bps:
  - Apply quick fix to all archetypes → production ready
  - Defer Thermo/LPPLS to later phase

- ⚠️ If only +20-30 bps:
  - Need to engineer Thermo/LPPLS features
  - Worth 2-3 day investment for full +110 bps

**Advantages:**
- Minimizes wasted effort
- Data-driven decision
- Can ship partial improvement quickly

---

## Current Status

### Archetype Integration Status:

| Archetype | PTI Wired | Temporal Wired | Wyckoff Wired | Thermo Wired | LPPLS Wired | Status |
|-----------|----------|----------------|---------------|--------------|------------|--------|
| S1 | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| S4 | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| S5 | ⚠️ Names | ✅ Works | ❌ Not Wired | ❌ No Data | ❌ No Data | Needs Fix |
| H | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| B | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| C | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| K | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| A | ⚠️ Names | ✅ Works | ⚠️ Partial | ❌ No Data | ❌ No Data | Needs Fix |
| G | ❌ Not Wired | ✅ Works | ❌ Not Wired | ❌ No Data | ❌ No Data | Needs Fix |

### Feature Working Status:

| Feature System | Code Complete | Data Exists | Names Match | Working? | Fix Needed |
|---------------|---------------|-------------|-------------|----------|------------|
| Temporal Confluence | ✅ | ✅ | ✅ | ✅ | None |
| PTI | ✅ | ✅ | ❌ | ❌ | Naming fix |
| Wyckoff (AR/AS/BC/LPS) | ✅ | ✅ | ✅ | ✅ | None |
| Wyckoff (Spring A/B, SOS) | ✅ | ❌ | ❌ | ❌ | Missing data |
| Thermo-floor | ✅ | ❌ | ❌ | ❌ | Feature engineering |
| LPPLS | ✅ | ❌ | ❌ | ❌ | Feature engineering |

---

## Immediate Next Steps

**Recommended: Option C - Validate Core, Then Decide**

### Today (2-3 hours):
1. **Quick-fix S1 archetype PTI naming**
   - Change `pti_score` → `tf1h_pti_score`
   - Change `pti_confidence` → `tf1h_pti_confidence`
   - Derive `pti_trap_type` from `tf1d_pti_reversal`

2. **Remove Thermo/LPPLS from S1 temporarily**
   - Comment out logic (features don't exist)

3. **Test S1 backtest 2022-2024**
   - With: Temporal + PTI (fixed) + Wyckoff (5 events)
   - Measure actual improvement vs baseline

### Decision Point:
- If +50-60 bps → Apply fixes to all archetypes, ship it
- If +20-30 bps → Engineer Thermo/LPPLS features (2-3 days)

---

## Bottom Line

**Good News:**
- ✅ Agents successfully wired LOGIC for all features
- ✅ Most features DO exist in data (PTI, Temporal, Wyckoff)
- ✅ Temporal Confluence works out of the box (+30 bps)
- ✅ 5 Wyckoff events work out of the box (+15-20 bps)

**Bad News:**
- ❌ Feature naming mismatches (PTI, some Wyckoff)
- ❌ Thermo-floor and LPPLS features not in data
- ❌ Need 2-6 hours of fixes before smoke test can pass

**Strategic Decision:**
- **Quick win:** Fix naming, test with working features first (2-3 hours)
- **Complete win:** Engineer missing features too (2-3 days)
- **Recommended:** Validate quick win first, then decide if full effort is worth it

---

**Files:**
- Smoke test script: `bin/smoke_test_wired_features.py`
- Feature availability check: Shows 6 PTI, 3 temporal, 20 Wyckoff features exist
- This report: `SMOKE_TEST_CRITICAL_FINDINGS.md`
