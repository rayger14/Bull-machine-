# Blocking Issues for Production Deployment

**Generated:** 2025-12-11
**Status:** 1 CRITICAL, 2 HIGH, 3 MEDIUM

---

## CRITICAL (BLOCKING)

### 1. Missing V2 OI Change Spike Features ❌

**Issue:** All 4 V2 OI change spike features are completely missing from the feature store:
- `oi_change_spike_3h`
- `oi_change_spike_6h`
- `oi_change_spike_12h`
- `oi_change_spike_24h`

**Impact:**
- Bear market archetypes (S2, S4, S5) list these as named dependencies
- These features detect capitulation and panic moments
- Critical for liquidity vacuum and funding divergence strategies

**Root Cause:**
- OI data coverage is only 33% (8,639 / 26,236 rows)
- Insufficient data to generate reliable spike features
- Data availability constraint from 2024 onwards

**Resolution Options:**

**Option A: Accept Partial Coverage (FAST)**
```bash
# Generate features with available OI data
python3 bin/add_oi_spike_features.py --allow-partial-coverage
# Result: Features will have 33% coverage, rest will be NaN
```

**Option B: Backfill OI Data (SLOW)**
```bash
# Backfill historical OI data from exchange APIs
python3 bin/backfill_oi_data.py --start=2022-01-01 --end=2024-12-31
# Then regenerate spike features
python3 bin/add_oi_spike_features.py
```

**Option C: Update Archetypes (STRATEGIC)**
```python
# Modify bear archetypes to handle missing V2 features gracefully
# In S2/S4/S5 archetype logic:
if 'oi_change_spike_24h' in features and not pd.isna(features['oi_change_spike_24h']):
    # Use OI spike detection
    spike_signal = features['oi_change_spike_24h']
else:
    # Fallback to alternative panic detection
    spike_signal = features['volume_panic'] or features['capitulation_depth'] > 0.7
```

**RECOMMENDATION:** Implement Option C immediately, then pursue Option B for data quality

**Timeline:**
- Option C: 2-4 hours (code changes)
- Option A: 1 hour (feature generation)
- Option B: 1-2 days (data backfill + verification)

---

## HIGH PRIORITY

### 2. Constant Value Features (16 features stuck) ⚠

**Issue:** 16 features have constant values and never vary:

**Wyckoff Domain:**
- `wyckoff_spring_b` - Always False (should detect Spring Type B)
- `wyckoff_spring_b_confidence` - Constant 0
- `wyckoff_pti_confluence` - Always False
- `tf4h_structure_alignment` - Constant
- `tf4h_range_breakout_strength` - Constant

**Temporal Domain:**
- `temporal_confluence` - Always False (multi-timeframe confluence)

**SMC Domain:**
- `tf4h_choch_flag` - Constant
- `tf4h_fvg_present` - Constant

**MTF Domain:**
- `tf1d_boms_detected` - Constant
- `tf1d_boms_direction` - Constant
- `tf4h_boms_direction` - Constant
- `mtf_alignment_ok` - Constant

**Other:**
- `k2_threshold_delta` - Constant
- `tf1h_kelly_hint` - Constant
- `tf1h_pti_trap_type` - Constant
- `macro_oil_trend` - Constant

**Impact:**
- Broken feature engineering logic
- These features should vary based on market conditions
- May indicate bugs in calculation or filtering logic

**Resolution:**
```bash
# Debug each constant feature
for feature in [wyckoff_spring_b, temporal_confluence, tf4h_choch_flag]:
    1. Review feature engineering code
    2. Check if thresholds are too strict
    3. Verify input data is varying
    4. Test on sample data
    5. Regenerate feature store
```

**Timeline:** 1-2 days (debugging + regeneration)

---

### 3. Low Coverage OI/Funding Features (33% data) ⚠

**Issue:** 10 OI and funding-related features have only 33% coverage:
- `oi` - 33.02% coverage
- `oi_z` - 32.93% coverage
- `oi_change_24h` - 32.92% coverage
- `oi_change_pct_24h` - 32.92% coverage
- `funding` - 33.39% coverage
- `funding_rate` - 100% coverage (but marked as ALWAYS_FIRES)
- `rv_20d`, `rv_60d` - 33.39% coverage

**Impact:**
- Incomplete context for bear market strategies
- OI-based features unreliable for 2/3 of dataset
- Affects risk management during high volatility

**Root Cause:**
- Historical OI data not available pre-2024 from current data source
- Exchange API limitations

**Resolution:**
```bash
# Backfill from alternative sources
python3 bin/backfill_oi_data.py --source=coinglass --start=2022-01-01
# Or accept limitation and document clearly
```

**Timeline:** 1-2 days (data acquisition + verification)

---

## MEDIUM PRIORITY

### 4. Broken FVG Boundary Features ⚠

**Issue:** Fair Value Gap boundary features have incomplete coverage:
- `tf1h_fvg_high` - 46.47% coverage
- `tf1h_fvg_low` - 49.04% coverage

**Impact:**
- SMC still works (main FVG flags functional)
- Missing precision boundaries for entry/exit
- Affects order placement optimization

**Resolution:**
```python
# Review FVG detection logic in engine/smc/fvg_detector.py
# Ensure boundaries are always populated when FVG detected
# Current issue: Boundaries only set when gap is "significant"
```

**Timeline:** 4-8 hours (debug + fix + test)

---

### 5. Missing HOB Metadata Fields ⚠

**Issue:** HOB zone strength and quality fields missing:
- `hob_strength` - Not in store
- `hob_quality` - Not in store

**Impact:**
- LOW - Zone detection works without metadata
- Nice-to-have for confidence scoring
- Affects zone filtering in strategies

**Resolution:**
```python
# Add to HOB zone detection in engine/hob/zone_detector.py
def compute_zone_metadata(zone):
    strength = calculate_volume_profile_strength(zone)
    quality = assess_zone_freshness_and_tests(zone)
    return strength, quality
```

**Timeline:** 4 hours (implementation + testing)

---

### 6. Missing SMC FVG Direction Flags ⚠

**Issue:** Directional FVG flags missing:
- `smc_fvg_bear` - Not in store
- `smc_fvg_bull` - Not in store

**Impact:**
- LOW - Generic FVG detection exists via other features
- Directional clarity missing
- Affects strategy selection logic

**Resolution:**
```python
# Add to SMC feature generation
df['smc_fvg_bull'] = (df['tf1h_fvg_low'] > 0) & (df['close'] > df['open'])
df['smc_fvg_bear'] = (df['tf1h_fvg_high'] > 0) & (df['close'] < df['open'])
```

**Timeline:** 2 hours (add + regenerate)

---

## Non-Blocking Issues

### Verification Script False Positives (108 features)

**Issue:** Many continuous features marked as "ALWAYS_FIRES":
- OHLCV data (close, high, low, open, volume)
- Technical indicators (atr_14, rsi_14, adx_14)
- Fusion scores (tf1h_fusion_score, k2_fusion_score)
- Confidence scores (wyckoff_*_confidence)

**Impact:**
- NONE - This is a verification script bug, not a feature issue
- Features are working correctly
- Skews quality metrics (makes 19.8% look worse than reality)

**Resolution:**
```python
# Update verify_feature_store_quality.py
# Distinguish between continuous features and binary flags
# Don't classify continuous features as ALWAYS_FIRES
def categorize_quality(metrics):
    if is_continuous_feature(feature_name):
        if metrics['non_null_pct'] >= 95:
            return 'GOOD'
    # ... rest of logic
```

**Timeline:** 1 hour (script update)

---

## Summary Matrix

| Issue | Priority | Impact | Timeline | Blocking? |
|-------|----------|--------|----------|-----------|
| Missing V2 OI Spike Features | CRITICAL | HIGH | 2-4 hours (Option C) | ❌ YES |
| Constant Value Features | HIGH | MEDIUM | 1-2 days | ⚠ PARTIAL |
| Low Coverage OI/Funding | HIGH | MEDIUM | 1-2 days | ⚠ PARTIAL |
| Broken FVG Boundaries | MEDIUM | LOW | 4-8 hours | ✅ NO |
| Missing HOB Metadata | MEDIUM | LOW | 4 hours | ✅ NO |
| Missing SMC FVG Flags | MEDIUM | LOW | 2 hours | ✅ NO |
| Verification False Positives | LOW | NONE | 1 hour | ✅ NO |

---

## Recommended Action Plan

### Phase 1: Immediate (Before Production) - 4 hours
```bash
# 1. Update bear archetypes to handle missing V2 features
vim engine/strategies/archetypes/bear/*.py
# Add graceful degradation for missing OI spike features

# 2. Add feature existence checks
vim engine/archetypes/base_archetype.py
# Implement safe_get_feature(name, default=None) method

# 3. Document known limitations
vim docs/FEATURE_STORE_KNOWN_LIMITATIONS.md

# 4. Deploy to production with warnings
```

### Phase 2: Short-Term (Week 1) - 2-3 days
```bash
# 1. Fix constant features (Priority 2)
python3 bin/debug_constant_features.py
# Fix logic, regenerate store

# 2. Generate V2 features with partial coverage (Option A)
python3 bin/add_oi_spike_features.py --allow-partial-coverage

# 3. Fix FVG boundaries
# Debug and fix FVG detection logic

# 4. Re-verify feature store
python3 bin/verify_feature_store_quality.py
```

### Phase 3: Medium-Term (Week 2-4) - 1-2 weeks
```bash
# 1. Backfill OI data (Option B)
python3 bin/backfill_oi_data.py --start=2022-01-01

# 2. Regenerate V2 features with full coverage
python3 bin/add_oi_spike_features.py

# 3. Add missing HOB and SMC features
python3 bin/add_missing_smc_features.py

# 4. Final verification and quality report
python3 bin/verify_feature_store_quality.py
```

---

## Deployment Decision

**Can we deploy to production?** ✅ **YES, WITH CONDITIONS**

**Conditions:**
1. Implement Phase 1 (4 hours) - Graceful degradation for missing V2 features
2. Add monitoring for feature quality degradation
3. Document known limitations clearly
4. Commit to Phase 2 completion within 1 week

**Confidence:** 75%

**Risk Level:** MEDIUM (mitigated by conditions)

---

**Next Action:** Begin Phase 1 - Update bear archetypes with graceful degradation logic

**Verification Required After Fixes:**
```bash
python3 bin/verify_feature_store_quality.py
python3 bin/verify_feature_store_reality.py
python3 tests/test_feature_completeness.py
```
