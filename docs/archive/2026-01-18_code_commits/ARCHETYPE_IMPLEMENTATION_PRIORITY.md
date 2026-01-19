# Archetype Implementation Priority
**Generated:** 2025-12-12
**Context:** 26+ archetype names discussed, only S1/S4/S5 production-ready
**Goal:** Identify which 5 bull archetypes are closest to working and estimate implementation effort

---

## Executive Summary

**REALITY CHECK:**
- **Production Ready (3):** S1, S4, S5 (bear market specialists)
- **Bull Market Coverage:** ZERO working bull archetypes
- **Ghost/Idea Archetypes:** P, Q, N, S3, S6, S7 (6 total - should be removed from docs)
- **Stubbed Bull Archetypes:** A, B, C, D, E, F, G, H, K, L, M (11 total - have code but missing critical pieces)

**CRITICAL FINDING:**
User has a bear market machine, not a bull machine. No bull archetypes are production-ready.

---

## Tier 1: Production Ready (Bear Specialists)

| ID | Name | PF | Trades/Year | Status |
|----|------|----|-----------|---------|
| S4 | Funding Divergence | 2.22-2.32 | 10-15 | ✅ DEPLOYED |
| S5 | Long Squeeze | 1.86 | 8-12 | ✅ DEPLOYED |
| S1 | Liquidity Vacuum V2 | 1.4-1.8 | 40-60 | ✅ DEPLOYED |

**Gap:** All are bear/crisis specialists. Zero coverage in bull markets (by design).

---

## Tier 2: Closest to Working (Bull Archetypes)

### Top 5 Bull Archetypes by Implementation Proximity

#### 1. H - Trap Within Trend ⭐ BEST CANDIDATE
**Completeness:** 80%
**Missing Pieces:**
- None - all features exist in store
- Logic fully implemented
- Config exists (`trap_within_trend`)

**Effort to Complete:** ~4 hours
1. Validate on 2022-2024 data (2h)
2. Optimize thresholds via Optuna (1h)
3. Create production config (0.5h)
4. Document results (0.5h)

**Why First:**
- No feature engineering needed
- Tested logic (just never validated)
- Name mapping correct
- No domain engine dependency

**Expected Performance:** PF 1.2-1.5 (trend continuation pattern)

---

#### 2. F - Expansion Exhaustion ⭐ RUNNER-UP
**Completeness:** 90%
**Missing Pieces:**
- None - all features exist
- Simple RSI + ATR + volume logic
- Config exists (`exhaustion_reversal`)

**Effort to Complete:** ~5 hours
1. Validate threshold ranges (2h)
2. Test on historical data (2h)
3. Optimize if needed (0.5h)
4. Production config (0.5h)

**Why Second:**
- Simplest logic (RSI extreme + volume spike)
- No complex feature dependencies
- Reversal pattern (complements S1/S4/S5)
- Mean-reversion edge (universal)

**Expected Performance:** PF 1.1-1.4 (exhaustion reversal)

---

#### 3. K - Wick Trap Moneytaur
**Completeness:** 80%
**Missing Pieces:**
- None - basic features available
- ADX, liquidity, wick ratio all present
- Config exists (`wick_trap_moneytaur`)

**Effort to Complete:** ~6 hours
1. Validate wick calculation logic (1h)
2. Test ADX thresholds (2h)
3. Optimize parameters (2h)
4. Document pattern (1h)

**Why Third:**
- Wick rejection = proven edge
- ADX filter reduces noise
- Config name matches query key (rare!)
- Pattern studied by Moneytaur (external validation)

**Expected Performance:** PF 1.2-1.6 (stop hunt reversal)

**Risk:** May overlap with S1 (both wick-based). Need to test correlation.

---

#### 4. B - Order Block Retest
**Completeness:** 60%
**Missing Pieces:**
- BOMS domain engine not wired (wyckoff_score exists but not connected)
- boms_strength feature missing (need to calculate from BOMS module)
- Order block zones not in feature store (need runtime calculation)

**Effort to Complete:** ~12 hours
1. Wire BOMS domain engine to runtime (4h)
2. Add boms_strength calculation (3h)
3. Implement OB zone tracking (3h)
4. Validate pattern (2h)

**Why Fourth:**
- Strong edge hypothesis (institutional retest zones)
- BOMS logic exists in codebase (just not connected)
- SMC/ICT pattern (proven in community)
- Config exists and name matches

**Expected Performance:** PF 1.3-1.8 (high conviction retest)

**Risk:** Requires domain engine wiring (more complex than others).

---

#### 5. G - Liquidity Sweep & Reclaim
**Completeness:** 70%
**Missing Pieces:**
- BOMS domain engine not wired
- Liquidity zones not tracked at runtime
- Need liquidity gradient calculation

**Effort to Complete:** ~10 hours
1. Wire BOMS for sweep detection (3h)
2. Add liquidity zone tracking (4h)
3. Implement reclaim logic (2h)
4. Validate pattern (1h)

**Why Fifth:**
- Stop hunt edge (proven concept)
- liquidity_score already exists
- BOMS strength available (after B implementation)
- Config exists

**Expected Performance:** PF 1.2-1.5 (liquidity hunt reversal)

**Dependency:** Should implement after B (shares BOMS wiring).

---

## Tier 3: More Complex (Skip for Now)

### A - Wyckoff Spring/UTAD
**Completeness:** 40%
**Blockers:**
- PTI engine not integrated (pti_trap_type feature missing)
- boms_disp feature missing
- Requires Wyckoff phase detection (complex)

**Effort:** ~20 hours (too high - defer)

**Why Skip:**
- Hard dependency on PTI feature generation
- Wyckoff phase classification needed
- High complexity for unproven edge
- Other patterns offer better ROI

---

### C - BOS/CHOCH Reversal
**Completeness:** 50%
**Blockers:**
- Config name mismatch (queries `wick_trap` but registered as `bos_choch_reversal`)
- fvg_present_4h missing (has tf4h_fvg_bull but name mismatch)
- boms_disp missing

**Effort:** ~8 hours

**Why Skip:**
- Name confusion needs resolution first
- FVG detection needs standardization
- B and G offer similar edges with less confusion

---

## Tier 4: Ghost Archetypes (DELETE FROM DOCS)

**Recommendation:** Remove these from documentation to reduce confusion.

| ID | Name | Reason |
|----|------|--------|
| P | FVG Reclaim | Registry entry only - no implementation |
| Q | Liquidity Cascade | Registry entry only - no implementation |
| N | HTF Trap Reversal | Registry entry only - no implementation |
| S3 | Whipsaw | Stub returns False - never implemented |
| S6 | Alt Rotation Down | Stub returns False - requires TOTAL3 data |
| S7 | Curve Inversion | Stub returns False - requires yield curve data |

**Action Items:**
1. Remove from `engine/archetypes/registry.py`
2. Update all ARCHETYPE_*.md docs to mark as "REMOVED"
3. Add deprecation warnings if configs reference these

---

## Implementation Roadmap

### Phase 1: Quick Wins (2 weeks)
**Goal:** Get 2 bull archetypes working

1. **Week 1:** H (Trap Within Trend)
   - Mon-Tue: Validate on historical data
   - Wed-Thu: Optimize thresholds
   - Fri: Production config + docs

2. **Week 2:** F (Expansion Exhaustion)
   - Mon-Tue: Validate exhaustion thresholds
   - Wed-Thu: Optimize + backtest
   - Fri: Production config + docs

**Deliverable:** 2 working bull archetypes (H, F)

---

### Phase 2: Domain Engine Wiring (3 weeks)
**Goal:** Unlock BOMS-dependent patterns

1. **Week 3-4:** Wire BOMS domain engine
   - Create boms_strength runtime calculator
   - Integrate with RuntimeContext
   - Add tests

2. **Week 5:** Implement B (Order Block Retest)
   - Add OB zone tracking
   - Validate pattern
   - Production config

**Deliverable:** 3 total (H, F, B)

---

### Phase 3: Liquidity Patterns (2 weeks)
**Goal:** Complete liquidity-based suite

1. **Week 6:** K (Wick Trap)
   - Validate wick logic
   - Optimize ADX thresholds
   - Production config

2. **Week 7:** G (Liquidity Sweep)
   - Leverage BOMS from Phase 2
   - Add liquidity zone tracking
   - Validate + deploy

**Deliverable:** 5 total (H, F, B, K, G)

---

## Feature Engineering Priority

### Critical Missing Features (Blocking Multiple Archetypes)

1. **boms_strength** (Blocks: B, G, A, C)
   - Source: BOMS domain engine output
   - Calculation: Need to expose from existing BOMS module
   - Effort: 4h

2. **atr_percentile** (Blocks: E, M)
   - Source: Rolling percentile of ATR_20
   - Calculation: `df['atr_percentile'] = df['atr_20'].rolling(168).rank(pct=True)`
   - Effort: 1h

3. **fvg_present_4h / fvg_present_1h** (Blocks: C, D)
   - Issue: Feature store has `tf4h_fvg_bull` but code queries `fvg_present_4h`
   - Fix: Rename or add alias in adapter layer
   - Effort: 0.5h

4. **pti_trap_type** (Blocks: A)
   - Source: PTI engine (not integrated)
   - Effort: 20h+ (defer)

---

## Success Metrics

After Phase 1-3 completion (7 weeks):
- **Bull Archetypes Working:** 5 (H, F, B, K, G)
- **Total Production Archetypes:** 8 (S1, S4, S5 + 5 bull)
- **Market Coverage:**
  - Bull markets: 5 patterns
  - Bear markets: 3 patterns
  - Portfolio diversity: ✅ Achieved

**Expected Portfolio PF:** 1.5-2.0 (weighted across regimes)

---

## Recommended Order

1. **H** (Trap Within Trend) - 4h - Easiest
2. **F** (Expansion Exhaustion) - 5h - Simple
3. **K** (Wick Trap) - 6h - Proven edge
4. **B** (Order Block Retest) - 12h - Domain wiring needed
5. **G** (Liquidity Sweep) - 10h - Builds on B

**Total Effort:** ~37 hours (~1 week of focused work)

---

## Decision Framework

**Choose H and F first if:**
- Need quick wins to prove bull market capability
- Want to minimize risk
- Limited time budget

**Choose B and G first if:**
- Want higher conviction patterns (PF 1.5+)
- Willing to invest in domain engine infrastructure
- Building for long-term sophistication

**Recommendation:** Start with H + F (quick wins), then invest in BOMS wiring for B + G.
