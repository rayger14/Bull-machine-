# ARCHETYPE INVENTORY & HEALTH AUDIT
**Bull Machine Trading System - Complete Archetype Census**

**Date:** 2025-12-08
**Purpose:** Comprehensive inventory for ML ensemble selection
**Scope:** All archetypes (bull, bear, neutral) across production configs

---

## EXECUTIVE SUMMARY

**Total Archetypes Identified:** 26 unique patterns
**Production Ready:** 3 (S1, S4, S5)
**Configured & Tested:** 7 bull patterns
**Code Exists, Not Wired:** 11 patterns
**Deprecated/Disabled:** 5 patterns

**Critical Finding:** Only 3 bear archetypes have production configs with domain engines enabled. Bull archetypes (A-M) lack standalone production validation.

---

## SECTION 1: ACTIVE PRODUCTION ARCHETYPES
**Status:** Production configs exist, domain engines enabled, validated performance data available

### 1. Liquidity Vacuum (S1)
- **Canonical Name:** `liquidity_vacuum` (alias: `breakdown`, `S1`)
- **Letter Code:** S1
- **Config File:** `/configs/s1_v2_production.json`
- **Direction:** LONG (capitulation reversal)
- **Domain Engines:** 6/6 ACTIVE
  - enable_wyckoff: true
  - enable_smc: true
  - enable_temporal: true
  - enable_hob: true
  - enable_fusion: true
  - enable_macro: true
- **Recent Performance (2024 validation):**
  - Profit Factor: 1.55
  - Win Rate: ~60%
  - Trades/Year: 60.7 (target: 40-60)
  - Events Caught: 4/7 major capitulations (57% recall)
- **Status:** ✅ ACTIVE & FIRING
- **Regime:** risk_off, crisis (bear specialist)
- **Notes:** V2 multi-bar capitulation detection. Confluence logic enabled. Catches LUNA, FTX, Japan Carry events.

---

### 2. Funding Divergence (S4)
- **Canonical Name:** `funding_divergence` (alias: `distribution`, `S4`)
- **Letter Code:** S4
- **Config File:** `/configs/system_s4_production.json`
- **Direction:** LONG (short squeeze reversal)
- **Domain Engines:** 0/6 (feature flags not in config - likely disabled)
- **Recent Performance (2022 validation):**
  - Profit Factor: 2.22
  - Win Rate: 55.7%
  - Trades/Year: 12
  - Improvement: +34% vs baseline
- **Status:** ✅ CONFIGURED & VALIDATED
- **Regime:** risk_off, neutral (bear specialist)
- **Notes:** Detects extreme negative funding (shorts overcrowded). Runtime feature enrichment enabled.

---

### 3. Long Squeeze (S5)
- **Canonical Name:** `long_squeeze` (alias: `S5`)
- **Letter Code:** S5
- **Config File:** `/configs/system_s5_production.json`
- **Direction:** SHORT (long cascade)
- **Domain Engines:** 6/6 ACTIVE
  - enable_wyckoff: true
  - enable_smc: true
  - enable_temporal: true
  - enable_hob: true
  - enable_fusion: true
  - enable_macro: true
- **Recent Performance (2022 validation):**
  - Profit Factor: 1.86
  - Win Rate: 55.6%
  - Trades/Year: 9
  - Total Return: +4.04R
- **Status:** ✅ ACTIVE & FIRING
- **Regime:** risk_off, crisis (bear rally specialist)
- **Notes:** SHORT positions during bear market rallies. Positive funding extremes. Catches LUNA, 3AC, FTX rallies.

---

## SECTION 2: BULL ARCHETYPES (Configured but No Standalone Production Validation)
**Status:** Enabled in mvp_bull_market_v1.json, but no standalone production configs or recent validation data

### 4. Spring / UTAD (A)
- **Canonical Name:** `wyckoff_spring_utad` (alias: `spring`, `trap_reversal`, `A`)
- **Letter Code:** A
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_A: true)
- **Direction:** LONG
- **Domain Engines:** Unknown (bull config lacks feature_flags section)
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** PTI-based spring/UTAD detection. Requires PTI feature (hard gate). Check function: `_check_A`

---

### 5. Order Block Retest (B)
- **Canonical Name:** `order_block_retest` (alias: `OB`, `ob_retest`, `B`)
- **Letter Code:** B
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_B: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** BOMS strength + Wyckoff + near BOS zone. Check function: `_check_B`

---

### 6. BOS/CHOCH Reversal (C)
- **Canonical Name:** `bos_choch_reversal` (alias: `fvg_continuation`, `bos`, `choch`, `C`)
- **Letter Code:** C
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_C: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** Displacement + momentum + recent BOS. Check function: `_check_C`. Config query key: `wick_trap` (NAME MISMATCH!)

---

### 7. Liquidity Sweep & Reclaim (G)
- **Canonical Name:** `liquidity_sweep_reclaim` (alias: `sweep`, `liquidity_sweep`, `G`)
- **Letter Code:** G
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_G: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** BOMS strength + rising liquidity from oversold. Check function: `_check_G`

---

### 8. Trap Within Trend (H)
- **Canonical Name:** `trap_within_trend` (alias: `trap`, `trap_legacy`, `htf_trap`, `H`)
- **Letter Code:** H
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_H: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** HTF trend + LOW liquidity drop + wick against trend. INVERTED LIQUIDITY LOGIC vs other patterns. Check function: `_check_H`

---

### 9. Wick Trap (Moneytaur) (K)
- **Canonical Name:** `wick_trap_moneytaur` (alias: `wick_trap`, `moneytaur`, `K`)
- **Letter Code:** K
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_K: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** Wick anomaly + ADX > 25 + BOS context. Permissive defaults. Check function: `_check_K`

---

### 10. Fakeout Real Move (L)
- **Canonical Name:** `fakeout_real_move` (alias: `frm`, `false_break`, `volume_exhaustion`, `retest_cluster`, `L`)
- **Letter Code:** L
- **Config File:** `/configs/mvp/mvp_bull_market_v1.json` (enable_L: true)
- **Direction:** LONG
- **Domain Engines:** Unknown
- **Recent Performance:** Not validated standalone
- **Status:** ⚠️ CONFIGURED BUT NOT VALIDATED
- **Regime:** risk_on, neutral
- **Notes:** Fakeout followed by genuine structural move. Config query key: `volume_exhaustion` (NAME MISMATCH!). Check function: `_check_L`

---

## SECTION 3: BEAR ARCHETYPES (Code Exists, Not in Production)

### 11. Failed Rally (S2)
- **Canonical Name:** `failed_rally` (alias: `S2`)
- **Letter Code:** S2
- **Config File:** `/engine/strategies/archetypes/bear/failed_rally_runtime.py` (code exists)
- **Direction:** SHORT
- **Domain Engines:** N/A (no production config)
- **Recent Performance:** PF 0.48 after optimization (DEPRECATED)
- **Status:** ❌ DEPRECATED - Pattern fundamentally broken for BTC
- **Regime:** risk_off, neutral
- **Notes:** Failed rally rejection pattern. Runtime enrichment exists but performance too poor for production. Explicitly disabled in mvp configs.

---

### 12. Whipsaw (S3)
- **Canonical Name:** `whipsaw` (alias: `S3`)
- **Letter Code:** S3
- **Config File:** None
- **Direction:** Unknown
- **Domain Engines:** N/A
- **Recent Performance:** Never validated
- **Status:** 🔲 NOT WIRED IN YET
- **Regime:** risk_off, crisis
- **Notes:** Listed in ARCHETYPE_REGIMES but no check function implemented

---

### 13. Alt Rotation Down (S6)
- **Canonical Name:** `alt_rotation_down` (alias: `S6`)
- **Letter Code:** S6
- **Config File:** None
- **Direction:** Unknown
- **Domain Engines:** N/A
- **Recent Performance:** Never validated
- **Status:** 🔲 NOT WIRED IN YET
- **Regime:** risk_off, crisis
- **Notes:** Listed in ARCHETYPE_REGIMES but no check function implemented

---

### 14. Curve Inversion (S7)
- **Canonical Name:** `curve_inversion` (alias: `S7`)
- **Letter Code:** S7
- **Config File:** None
- **Direction:** Unknown
- **Domain Engines:** N/A
- **Recent Performance:** Never validated
- **Status:** 🔲 NOT WIRED IN YET
- **Regime:** risk_off, crisis
- **Notes:** Listed in ARCHETYPE_REGIMES but no check function implemented

---

### 15. Volume Fade Chop (S8)
- **Canonical Name:** `volume_fade_chop` (alias: `S8`)
- **Letter Code:** S8
- **Config File:** None
- **Direction:** Unknown
- **Domain Engines:** N/A
- **Recent Performance:** Never validated
- **Status:** 🔲 NOT WIRED IN YET
- **Regime:** neutral
- **Notes:** Listed in ARCHETYPE_REGIMES but no check function implemented

---

## SECTION 4: EXPERIMENTAL / INCOMPLETE ARCHETYPES
**Status:** Listed in registry but no implementation or minimal code

### 16. Failed Continuation (D)
- **Canonical Name:** `failed_continuation` (alias: `failed_fvg`, `D`)
- **Letter Code:** D
- **Status:** 🔲 CHECK FUNCTION EXISTS (`_check_D`) - Disabled in configs
- **Regime:** risk_on, neutral
- **Notes:** FVG present + weak RSI + falling ADX

---

### 17. Liquidity Compression (E)
- **Canonical Name:** `liquidity_compression` (alias: `compression`, `E`)
- **Letter Code:** E
- **Status:** 🔲 CHECK FUNCTION EXISTS (`_check_E`) - Disabled in configs
- **Regime:** risk_on, neutral
- **Notes:** Low ATR + narrow range + stable book depth. Config query key: `volume_exhaustion` (NAME MISMATCH!)

---

### 18. Expansion Exhaustion (F)
- **Canonical Name:** `expansion_exhaustion` (alias: `exhaustion_reversal`, `F`)
- **Letter Code:** F
- **Status:** 🔲 CHECK FUNCTION EXISTS (`_check_F`) - Disabled in configs
- **Regime:** risk_on, neutral
- **Notes:** Extreme RSI + high ATR + volume spike

---

### 19. Ratio Coil Break (M)
- **Canonical Name:** `ratio_coil_break` (alias: `coil`, `wyckoff_insider`, `confluence_breakout`, `M`)
- **Letter Code:** M
- **Status:** 🔲 CHECK FUNCTION EXISTS (`_check_M`) - Disabled in configs
- **Regime:** risk_on, neutral
- **Notes:** Low ATR + near POC + BOMS strength

---

### 20. FVG Reclaim (P)
- **Canonical Name:** `fvg_reclaim` (alias: `fvg`, `P`)
- **Letter Code:** P
- **Status:** 🔲 REGISTRY ONLY - No check function
- **Regime:** Unknown
- **Notes:** Fair value gap reclaim with volume confirmation (experimental)

---

### 21. Liquidity Cascade (Q)
- **Canonical Name:** `liquidity_cascade` (alias: `cascade`, `Q`)
- **Letter Code:** Q
- **Status:** 🔲 REGISTRY ONLY - No check function
- **Regime:** Unknown
- **Notes:** Multi-level liquidity run with acceleration (experimental)

---

### 22. Range Expansion/Compression Flip (G_alt)
- **Canonical Name:** `range_expansion_compression_flip` (alias: `range_flip`, `G_alt`)
- **Letter Code:** None
- **Status:** 🔲 REGISTRY ONLY - No check function
- **Regime:** Unknown
- **Notes:** Volatility regime change with structure break

---

### 23. BOMS Phase Shift (F_alt)
- **Canonical Name:** `boms_phase_shift` (alias: `boms`, `reaccumulation`, `F_alt`)
- **Letter Code:** None
- **Status:** 🔲 REGISTRY ONLY - No check function
- **Regime:** Unknown
- **Notes:** BOMS phase transition with book depth shift

---

### 24. HTF Trap Reversal (N)
- **Canonical Name:** `htf_trap_reversal` (alias: `N`)
- **Letter Code:** N
- **Status:** 🔲 REGISTRY ONLY - No check function
- **Regime:** Unknown
- **Notes:** Multi-timeframe trap with HTF confirmation (experimental)

---

### 25. Momentum Continuation (H_legacy)
- **Canonical Name:** `momentum_continuation` (alias: `H_legacy`)
- **Letter Code:** None (H now maps to trap_within_trend)
- **Status:** 🔲 REGISTRY ONLY - Legacy archetype
- **Regime:** Unknown
- **Notes:** Strong momentum continuation (legacy archetype, replaced by trap_within_trend)

---

### 26. Breakdown (S1_OLD)
- **Canonical Name:** `breakdown` (alias: `S1` - DEPRECATED)
- **Letter Code:** S1 (OLD)
- **Status:** ❌ DEPRECATED - Replaced by liquidity_vacuum
- **Regime:** risk_off, crisis
- **Notes:** Old S1 logic, replaced by S1 V2 (liquidity_vacuum). Kept as alias for backward compatibility.

---

## SECTION 5: SYSTEM B0 - BASELINE STRATEGY (Non-Archetype)

### System B0 - Drawdown Baseline
- **Name:** Baseline Conservative
- **Config File:** `/configs/system_b0_production.json`
- **Type:** BASELINE (not an archetype)
- **Direction:** LONG
- **Logic:** Simple drawdown entry (-15% from 30d high) + profit target (+8%)
- **Recent Performance (2023 test):**
  - Profit Factor: 3.17
  - Win Rate: 42.9%
  - Sharpe: 1.437
- **Status:** ✅ PRODUCTION DEPLOYED
- **Notes:** Outperforms all archetypes in validation. Simple, proven, no domain engines.

---

## PERFORMANCE COMPARISON MATRIX

| System | Type | PF | WR | Trades/Yr | Status |
|--------|------|----|----|-----------|--------|
| **System B0 Baseline** | Baseline | 3.17 | 42.9% | 76 | 🟢 BEST |
| S1 Liquidity Vacuum | Archetype | 1.55 | 60% | 60.7 | 🟡 Active |
| S4 Funding Divergence | Archetype | 2.22 | 55.7% | 12 | 🟡 Validated |
| S5 Long Squeeze | Archetype | 1.86 | 55.6% | 9 | 🟡 Active |
| S2 Failed Rally | Archetype | 0.48 | - | - | 🔴 Broken |
| Bull Archetypes (A-M) | Archetype | Unknown | - | - | ⚠️ Not Validated |

---

## CRITICAL ISSUES IDENTIFIED

### 1. Bull Archetypes Lack Production Validation
- **Problem:** A, B, C, G, H, K, L enabled in mvp_bull_market_v1.json but no standalone production configs
- **Impact:** No recent performance data, unknown PF/WR, domain engine status unclear
- **Recommendation:** Validate each bull archetype individually OR use ensemble approach

### 2. Name Mapping Inconsistencies
- **Problem:** Config query keys don't match canonical names
  - C (BOS/CHOCH) queries `wick_trap` instead of `bos_choch_reversal`
  - E (Compression) queries `volume_exhaustion` instead of `liquidity_compression`
  - L (Fakeout) queries `volume_exhaustion` (conflict with E!)
- **Impact:** Optimizer writes to wrong config sections, parameters ignored
- **Recommendation:** Use ARCHETYPE_NAME_MAPPING_REFERENCE.md for correct query keys

### 3. Domain Engine Coverage Unknown for Bull Archetypes
- **Problem:** mvp_bull_market_v1.json lacks `feature_flags` section
- **Impact:** Unknown if Wyckoff, SMC, Temporal, HOB, Fusion, Macro are enabled
- **Recommendation:** Add feature_flags to bull config, validate feature coverage

### 4. Deprecated Archetypes Still in Codebase
- **Problem:** S2 (Failed Rally) has runtime code but PF 0.48 (broken)
- **Impact:** Confusion, maintenance burden, risk of accidental enablement
- **Recommendation:** Archive deprecated archetypes, document removal

### 5. Validation Shows Baselines Outperform Archetypes
- **Problem:** System B0 baseline (PF 3.17) >> S1 (PF 1.55)
- **Impact:** Complexity not justified by returns
- **Recommendation:** Use baselines for deployment, archetypes as ensemble features for ML

---

## ML ENSEMBLE RECOMMENDATIONS

### Tier 1: Production-Ready for ML Training
1. **S1 Liquidity Vacuum** - 6/6 engines, validated, bear specialist
2. **S4 Funding Divergence** - Validated, PF 2.22, bear specialist
3. **S5 Long Squeeze** - 6/6 engines, validated, bear short specialist

### Tier 2: Needs Validation Before ML Use
4. **H Trap Within Trend** - Code exists, needs standalone validation
5. **K Wick Trap** - Code exists, needs standalone validation
6. **B Order Block Retest** - Code exists, needs standalone validation

### Tier 3: Experimental (Do Not Use for ML Yet)
- All D, E, F, M, P, Q archetypes (incomplete implementation)
- All S3, S6, S7, S8 (no code)

### Baseline Strategy (Non-Archetype)
- **System B0** - Use as benchmark, not as ensemble feature

---

## NEXT ACTIONS

### Immediate (This Week)
1. ✅ Validate bull archetypes A, B, C, G, H, K, L on 2024 data
2. ✅ Add feature_flags to mvp_bull_market_v1.json
3. ✅ Document which archetypes have complete domain engine coverage

### Short-Term (Next 2 Weeks)
4. ✅ Create standalone production configs for top 3 bull archetypes
5. ✅ Archive S2 (Failed Rally) code to /deprecated/
6. ✅ Fix name mapping inconsistencies (C, E, L query keys)

### Long-Term (Next Month)
7. ✅ Implement ML ensemble using Tier 1 + validated Tier 2 archetypes
8. ✅ Compare ensemble performance vs System B0 baseline
9. ✅ Decide: Deploy ensemble OR stay with baseline

---

## REFERENCES

- **Registry:** `/engine/archetypes/registry.py` (canonical names, aliases)
- **Detection Logic:** `/engine/archetypes/logic_v2_adapter.py` (check functions)
- **Production Configs:** `/configs/s1_v2_production.json`, `/configs/system_s4_production.json`, `/configs/system_s5_production.json`
- **Bull Config:** `/configs/mvp/mvp_bull_market_v1.json`
- **Name Mapping:** `/ARCHETYPE_NAME_MAPPING_REFERENCE.md`
- **Validation Results:** `/results/validation/FINAL_DEPLOYMENT_DECISION.md`

---

**Report Generated:** 2025-12-08
**Author:** Claude Code (Backend Architect)
**Purpose:** Inform ML ensemble archetype selection strategy
