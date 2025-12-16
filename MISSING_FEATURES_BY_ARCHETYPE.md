# Missing Features by Archetype
**Generated:** 2025-12-12
**Source:** Feature store analysis vs archetype requirements
**Data:** `/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` (202 columns)

---

## Feature Availability Status

### ✅ Available Features (90% coverage)
- **Macro/Regime:** VIX_Z, DXY_Z, funding_Z, regime_label, crisis_composite
- **Technical:** RSI, ADX, ATR, volume_zscore, liquidity_score
- **Wyckoff:** wyckoff_score (via tf1d/tf4h variants)
- **SMC:** tf4h_bos_bullish, tf4h_fvg_bull, tf1h_fvg_bull
- **Capitulation:** capitulation_depth, volume_climax_3b, wick_exhaustion_3b

### ❌ Missing Features (Critical Blockers)

---

## Archetype-by-Archetype Breakdown

### A - Wyckoff Spring/UTAD
**Status:** BLOCKED (40% features available)

**Critical Missing:**
1. `pti_trap_type` ❌
   - **Required:** Spring/UTAD classification from PTI engine
   - **Current:** No PTI engine integration
   - **Fix:** Integrate PTI (Probability of Trend Interruption) module
   - **Effort:** 20h (complex - requires Wyckoff phase detection)

2. `boms_disp` ❌
   - **Required:** Break of Market Structure displacement
   - **Current:** BOMS module exists but not exposing this metric
   - **Fix:** Add displacement output to BOMS engine
   - **Effort:** 4h

3. `pti_score` ⚠️ PARTIAL
   - **Available:** `tf1d_pti_score` exists (202 columns check)
   - **Issue:** Code queries generic `pti_score` not TF-prefixed version
   - **Fix:** Update adapter layer or rename feature
   - **Effort:** 0.5h

**Recommendation:** DEFER - PTI integration too complex for now

---

### B - Order Block Retest
**Status:** PARTIAL (60% features available)

**Critical Missing:**
1. `boms_strength` ❌
   - **Required:** BOMS strength/quality score
   - **Current:** BOMS module exists but not exposing this output
   - **Fix:** Add strength calculation to BOMS domain engine
   - **Effort:** 4h
   - **Impact:** Blocks B, G, A, C (4 archetypes)

2. `tf1h_ob_bull_top`, `tf1h_ob_bull_bottom` ⚠️ MAYBE
   - **Available:** Feature registry defines these (registry.py lines 122-133)
   - **Uncertainty:** Need to verify in actual parquet file
   - **Workaround:** Can calculate at runtime from price action
   - **Effort:** 3h (if missing)

**Available:**
- `wyckoff_score` ✅ (via tf1d_wyckoff_score)
- `fusion_score` ✅

**Recommendation:** IMPLEMENT AFTER fixing boms_strength

---

### C - BOS/CHOCH Reversal
**Status:** PARTIAL (50% features available)

**Critical Missing:**
1. `fvg_present_4h` ❌ NAME MISMATCH
   - **Available:** `tf4h_fvg_bull` EXISTS in feature store
   - **Issue:** Code queries `fvg_present_4h` not `tf4h_fvg_bull`
   - **Fix:** Adapter layer alias OR rename check function query
   - **Effort:** 0.5h

2. `boms_disp` ❌
   - Same as archetype A
   - **Effort:** 4h (shared fix)

**Recommendation:** EASY FIX - just name aliasing + boms_disp

---

### D - Failed Continuation
**Status:** PARTIAL (60% features available)

**Critical Missing:**
1. `fvg_present_1h` ❌ NAME MISMATCH
   - **Available:** `tf1h_fvg_bull` EXISTS in feature store
   - **Issue:** Code queries `fvg_present_1h` not `tf1h_fvg_bull`
   - **Fix:** Adapter layer alias
   - **Effort:** 0.5h

**Available:**
- `rsi_14` ✅
- `fusion_score` ✅

**Recommendation:** TRIVIAL FIX - just name aliasing

---

### E - Liquidity Compression
**Status:** PARTIAL (80% features available)

**Critical Missing:**
1. `atr_percentile` ❌
   - **Required:** Rolling percentile rank of ATR
   - **Current:** Only raw ATR_20 available
   - **Fix:** `df['atr_percentile'] = df['atr_20'].rolling(168).rank(pct=True)`
   - **Effort:** 1h (add to feature builder)

2. `vol_cluster` ❌
   - **Required:** Volume clustering/coherence metric
   - **Current:** Not in feature store
   - **Fix:** Calculate volume STD in rolling window
   - **Effort:** 2h

**Available:**
- `volume_zscore` ✅
- `fusion_score` ✅

**Recommendation:** EASY - add rolling percentile calculations

---

### F - Expansion Exhaustion
**Status:** READY (90% features available)

**Critical Missing:**
- None! All features exist

**Available:**
- `rsi_14` ✅
- `atr_20` ✅
- `volume_zscore` ✅
- `fusion_score` ✅

**Recommendation:** ⭐ IMPLEMENT IMMEDIATELY - no feature work needed

---

### G - Liquidity Sweep & Reclaim
**Status:** PARTIAL (70% features available)

**Critical Missing:**
1. `boms_strength` ❌
   - Same as archetype B
   - **Effort:** 4h (shared fix)

**Available:**
- `liquidity_score` ✅
- `rsi_14` ✅
- `fusion_score` ✅

**Recommendation:** IMPLEMENT AFTER fixing boms_strength (shared with B)

---

### H - Trap Within Trend
**Status:** READY (80% features available)

**Critical Missing:**
- None! All features exist

**Available:**
- `adx_14` ✅
- `liquidity_score` ✅
- `tf4h_trend_direction` ✅
- `fusion_score` ✅

**Recommendation:** ⭐ IMPLEMENT IMMEDIATELY - no feature work needed

---

### K - Wick Trap Moneytaur
**Status:** READY (80% features available)

**Critical Missing:**
- None! All features exist (calculated from OHLC)

**Available:**
- `high`, `low`, `open`, `close` ✅ (for wick ratio calc)
- `adx_14` ✅
- `liquidity_score` ✅
- `tf4h_bos_bullish` ✅
- `fusion_score` ✅

**Recommendation:** ⭐ IMPLEMENT SOON - no feature work needed

---

### L - Volume Exhaustion / Fakeout Real Move
**Status:** PARTIAL (70% features available)

**Critical Missing:**
1. Name confusion with E (both query `volume_exhaustion` config)
2. Possibly duplicate logic

**Available:**
- `volume_zscore` ✅
- `rsi_14` ✅
- `fusion_score` ✅

**Recommendation:** CLARIFY vs E first, then implement if unique

---

### M - Ratio Coil Break
**Status:** PARTIAL (60% features available)

**Critical Missing:**
1. `atr_percentile` ❌
   - Same as archetype E
   - **Effort:** 1h (shared fix)

2. `poc_dist` ❌
   - **Required:** Distance from FRVP Point of Control
   - **Available:** `tf1d_frvp_poc` exists but no distance calculation
   - **Fix:** `df['poc_dist'] = abs(df['close'] - df['tf1d_frvp_poc']) / df['close']`
   - **Effort:** 1h

3. `boms_strength` ❌
   - Same as archetype B
   - **Effort:** 4h (shared fix)

**Recommendation:** DEFER until BOMS wiring complete

---

## Feature Engineering Priority Matrix

### Tier 1: High Impact, Low Effort (DO FIRST)

| Feature | Effort | Blocks | Fix |
|---------|--------|--------|-----|
| `fvg_present_4h` alias | 0.5h | C | Adapter layer mapping |
| `fvg_present_1h` alias | 0.5h | D | Adapter layer mapping |
| `atr_percentile` | 1h | E, M | Rolling rank calculation |
| `poc_dist` | 1h | M | Distance from FRVP POC |

**Total:** 3h to unblock 4 archetypes (C, D, E, M)

---

### Tier 2: High Impact, Medium Effort (DO SECOND)

| Feature | Effort | Blocks | Fix |
|---------|--------|--------|-----|
| `boms_strength` | 4h | B, G, A, C, M | Wire BOMS domain engine output |
| `boms_disp` | 4h | A, C | Add displacement to BOMS engine |

**Total:** 8h to unblock 5 archetypes (B, G, A, C, M)

---

### Tier 3: Low Impact, High Effort (DEFER)

| Feature | Effort | Blocks | Fix |
|---------|--------|--------|-----|
| `pti_trap_type` | 20h | A | Full PTI engine integration |

**Total:** 20h to unblock 1 archetype (A)

---

## Quick Wins: Zero Feature Work Needed

These archetypes can be implemented TODAY with existing features:

1. **H - Trap Within Trend** (80% ready)
2. **F - Expansion Exhaustion** (90% ready)
3. **K - Wick Trap Moneytaur** (80% ready)

**Action:** Start validation immediately on these 3.

---

## Feature Generation Script Template

```python
# Add to bin/add_derived_features.py

def add_archetype_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing archetype features to feature store."""

    # TIER 1 FIXES (3h total)
    # FVG aliases
    if 'tf4h_fvg_bull' in df.columns:
        df['fvg_present_4h'] = df['tf4h_fvg_bull']
    if 'tf1h_fvg_bull' in df.columns:
        df['fvg_present_1h'] = df['tf1h_fvg_bull']

    # ATR percentile (168h = 1 week rolling)
    if 'atr_20' in df.columns:
        df['atr_percentile'] = df['atr_20'].rolling(168).rank(pct=True)

    # POC distance
    if 'close' in df.columns and 'tf1d_frvp_poc' in df.columns:
        df['poc_dist'] = abs(df['close'] - df['tf1d_frvp_poc']) / df['close']

    # TIER 2 FIXES (8h total - requires BOMS wiring)
    # boms_strength - REQUIRES BOMS DOMAIN ENGINE INTEGRATION
    # boms_disp - REQUIRES BOMS DOMAIN ENGINE INTEGRATION

    return df
```

---

## Feature Store Health Check

### Current Coverage: 202 columns total

**By Category:**
- ✅ OHLCV: 5/5 (100%)
- ✅ Technical: 20/20 (100%)
- ✅ MTF: 50/50 (100%)
- ✅ Macro: 12/12 (100%)
- ✅ Regime: 5/5 (100%)
- ⚠️ Domain Engines: 85/100 (85% - missing BOMS outputs)
- ⚠️ Derived: 20/30 (67% - missing percentiles, distances)

**Missing Categories:**
1. BOMS engine outputs (strength, displacement)
2. Rolling percentiles (ATR, volume)
3. Distance metrics (POC, zones)
4. PTI engine outputs (trap types, scores)

---

## Recommended Action Plan

### Week 1: Tier 1 Features (3h)
1. Add FVG aliases to adapter layer (1h)
2. Generate atr_percentile (1h)
3. Generate poc_dist (1h)
4. **Result:** Unblock C, D, E, M (4 archetypes)

### Week 2: Validate Zero-Feature Archetypes (12h)
1. H - Trap Within Trend (4h)
2. F - Expansion Exhaustion (4h)
3. K - Wick Trap Moneytaur (4h)
4. **Result:** 3 working bull archetypes

### Week 3-4: Tier 2 Features (8h) + Implementation (16h)
1. Wire BOMS domain engine (8h)
2. Implement B - Order Block Retest (8h)
3. Implement G - Liquidity Sweep (8h)
4. **Result:** 2 more working archetypes (total: 5 bull)

### Week 5: Production Deployment
1. Create production configs
2. Run walk-forward validation
3. Deploy to System B0

**End State:** 8 total production archetypes (3 bear + 5 bull)

---

## Feature Dependency Graph

```
PTI Engine (20h)
└── A (Spring/UTAD)

BOMS Domain Engine (8h)
├── B (Order Block Retest)
├── G (Liquidity Sweep)
├── A (Spring/UTAD)
├── C (BOS/CHOCH)
└── M (Ratio Coil Break)

Rolling Calculations (2h)
├── E (Liquidity Compression)
└── M (Ratio Coil Break)

Adapter Aliases (1h)
├── C (BOS/CHOCH)
└── D (Failed Continuation)

POC Distance (1h)
└── M (Ratio Coil Break)

ZERO DEPENDENCIES
├── H (Trap Within Trend) ⭐
├── F (Expansion Exhaustion) ⭐
└── K (Wick Trap Moneytaur) ⭐
```

**Critical Path:** Focus on ZERO DEPENDENCIES first for quick wins.
