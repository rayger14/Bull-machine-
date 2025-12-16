# Deprecated and Ghost Archetypes

## Executive Summary

This document identifies archetypes that are **documented but not implemented** (ghosts) or **implemented but disabled** (stubs). These should not be enabled in production configurations.

**Date**: 2025-12-12
**Context**: Ghost module cleanup to align documentation with implementation reality

---

## Status Categories

- **EXISTS**: Has working implementation + domain engine wiring + can backtest
- **STUB**: Has placeholder method that returns False (disabled)
- **GHOST**: Documented in registry but no implementation method
- **DEPRECATED**: Old implementation superseded by better version

---

## Ghost Archetypes (No Implementation)

These archetypes are defined in `/engine/archetypes/registry.py` but have NO detection method in `logic_v2_adapter.py`.

### P - FVG Reclaim (GHOST)
- **Registry slug**: `fvg_reclaim`
- **Aliases**: P, fvg
- **Status**: GHOST - No `_check_P()` method exists
- **Description**: Fair value gap reclaim with volume confirmation (experimental)
- **Action**: Remove from registry or implement if needed
- **Risk**: Configs enabling P will silently fail

### Q - Liquidity Cascade (GHOST)
- **Registry slug**: `liquidity_cascade`
- **Aliases**: Q, cascade
- **Status**: GHOST - No `_check_Q()` method exists
- **Description**: Multi-level liquidity run with acceleration (experimental)
- **Action**: Remove from registry or implement if needed
- **Risk**: Configs enabling Q will silently fail

### N - HTF Trap Reversal (GHOST)
- **Registry slug**: `htf_trap_reversal`
- **Aliases**: N
- **Status**: GHOST - No `_check_N()` method exists
- **Description**: Multi-timeframe trap with HTF confirmation (experimental)
- **Action**: Remove from registry or implement if needed
- **Risk**: Configs enabling N will silently fail

---

## Stub Archetypes (Disabled)

These archetypes have implementation methods but are **explicitly disabled** (return False).

### S6 - Alt Rotation Down (STUB)
- **Method**: `_check_S6()`
- **Status**: STUB - Returns False
- **Reason**: "DISABLED: Requires altcoin dominance data not in feature store"
- **Data Dependency**: TOTAL3 (altcoin dominance) vs BTC
- **Action**:
  - Either implement with available data
  - OR remove from archetype map to prevent confusion
- **Risk**: LOW - Method exists but always fails, no silent errors

### S7 - Curve Inversion (STUB)
- **Method**: `_check_S7()`
- **Status**: STUB - Returns False
- **Reason**: "DISABLED: Requires yield curve data not in feature store"
- **Data Dependency**: US Treasury yield curve data
- **Action**:
  - Either implement with macro data pipeline
  - OR remove from archetype map to prevent confusion
- **Risk**: LOW - Method exists but always fails, no silent errors

---

## Deprecated Archetypes (Superseded)

These archetypes have been replaced by better implementations.

### S1 OLD - Breakdown (DEPRECATED)
- **Registry slug**: `breakdown`
- **Status**: DEPRECATED
- **Replaced by**: S1 - Liquidity Vacuum Reversal
- **Reason**: Generic "breakdown" replaced by specific "liquidity vacuum" pattern
- **Action**: Update configs using "breakdown" to use "liquidity_vacuum"
- **Notes**: Both may share same slot in archetype map

### S2 - Failed Rally (DEPRECATED for BTC)
- **Method**: `_check_S2()`
- **Status**: EXISTS but DEPRECATED for BTC
- **Reason**: Poor performance on BTC (works better on altcoins)
- **Production status**: Disabled in production configs
- **Action**: Keep implementation for testing but don't enable for BTC trading
- **Risk**: MEDIUM - Method works but has negative PnL on BTC

### S4 OLD - Distribution (DEPRECATED)
- **Registry slug**: `distribution`
- **Status**: DEPRECATED
- **Replaced by**: S4 - Funding Divergence (Short Squeeze)
- **Reason**: Generic "distribution" replaced by specific "funding divergence" pattern
- **Action**: Update configs using "distribution" to use "funding_divergence"

---

## Working Archetypes (Verified)

These archetypes are **fully implemented and working**:

### Bull Market (Long-biased)
- **A** - Trap Reversal (Spring/UTAD) - EXISTS + domain engines
- **B** - Order Block Retest - EXISTS + domain engines
- **C** - FVG Continuation (BOS/CHOCH) - EXISTS
- **D** - Failed Continuation - EXISTS
- **E** - Liquidity Compression - EXISTS
- **F** - Expansion Exhaustion - EXISTS
- **G** - Re-accumulate (Liquidity Sweep) - EXISTS
- **H** - Trap Within Trend - EXISTS + domain engines
- **K** - Wick Trap (Moneytaur) - EXISTS
- **L** - Volume Exhaustion - EXISTS
- **M** - Ratio Coil Break - EXISTS

### Bear Market (Short-biased)
- **S1** - Liquidity Vacuum Reversal - EXISTS + FULL domain wiring
- **S3** - Whipsaw - EXISTS
- **S4** - Funding Divergence (Short Squeeze) - EXISTS + FULL domain wiring
- **S5** - Long Squeeze Cascade - EXISTS + FULL domain wiring
- **S8** - Volume Fade Chop - EXISTS

---

## New BaseArchetype Implementations

As of 2025-12-12, the following archetypes have **NEW class-based implementations** in `/engine/strategies/archetypes/`:

### Bull Archetypes (5 implemented)
Located in `engine/strategies/archetypes/bull/`:
1. **SpringUTADArchetype** - Spring/UTAD detection with domain engines
2. **OrderBlockRetestArchetype** - SMC order block retest
3. **BOSCHOCHReversalArchetype** - Break of structure / change of character
4. **LiquiditySweepArchetype** - Liquidity sweep reversal
5. **TrapWithinTrendArchetype** - False breakdown continuation

**Note**: These are NEW implementations that may not yet be wired into the main detection loop. The old `_check_X()` methods in `logic_v2_adapter.py` are still the active detection logic.

### Bear Archetypes (3 runtime helpers)
Located in `engine/strategies/archetypes/bear/`:
1. **S2RuntimeFeatures** - Failed rally runtime enrichment
2. **S4RuntimeFeatures** - Funding divergence runtime enrichment
3. **S5RuntimeFeatures** - Long squeeze runtime enrichment

These provide on-demand feature calculation for bear archetypes.

---

## Domain Engine Wiring Status

### Fully Wired (S1, S4, S5)
These archetypes have **complete domain engine integration**:

**S1 - Liquidity Vacuum**:
- Wyckoff: Spring events, SC, BC, AR, ST (2.0x - 2.5x boost)
- SMC: BOS detection, demand zones, liquidity sweeps (1.4x - 2.0x boost)
- Temporal: Fibonacci time clusters (1.8x boost)
- HOB: Demand walls, bid imbalance (1.3x - 1.5x boost)
- PTI: Spring/UTAD detection (1.5x boost)
- **Total possible boost**: ~95x (realistic 8x-12x)

**S4 - Funding Divergence**:
- Similar domain wiring as S1
- SMC veto gates (4H BOS bearish)
- Runtime feature enrichment

**S5 - Long Squeeze**:
- Similar domain wiring as S1
- Wyckoff distribution events (UTAD, LPSY)
- Supply zone detection

### Partially Wired (A, B, H)
- **A (Trap Reversal)**: Has PTI + displacement logic, needs full domain integration
- **B (Order Block)**: Has BOS + BOMS logic, needs full domain integration
- **H (Trap Within Trend)**: Has HTF + liquidity logic, needs full domain integration

### Not Wired (C, D, E, F, G, K, L, M, S2, S3, S8)
These archetypes use **basic feature checks** without domain engine amplification:
- Simple fusion score thresholds
- RSI, ADX, volume checks
- No Wyckoff/SMC/Temporal boosting
- No hard/soft vetoes

---

## Recommended Actions

### Immediate (Before Production)
1. **Remove ghost archetypes from registry** (P, Q, N) OR implement them
2. **Add clear deprecation warnings** in configs for S2 (BTC), old S1, old S4
3. **Disable stub archetypes** (S6, S7) in all configs
4. **Document wiring inconsistency** - only 3/19 archetypes fully wired

### Short-term (Next Sprint)
1. **Wire domain engines** for high-priority archetypes (A, B, H)
2. **Standardize veto patterns** across all archetypes
3. **Add domain boost BEFORE fusion gate** for consistency
4. **Test new BaseArchetype implementations** and migrate if successful

### Long-term (Future)
1. **Migrate to BaseArchetype pattern** for all archetypes
2. **Remove logic_v2_adapter.py** once migration complete
3. **Implement or remove** stub archetypes (S6, S7)
4. **Create archetype performance dashboard** to identify weak performers

---

## Config Cleanup Checklist

Before enabling any archetype in production:

- [ ] Verify it's not in GHOST list (P, Q, N)
- [ ] Verify it's not in STUB list (S6, S7)
- [ ] Check if it's DEPRECATED (old S1, S2, old S4)
- [ ] Confirm it has working `_check_X()` method
- [ ] Check if domain engines are wired (only S1, S4, S5 fully wired)
- [ ] Validate on backtest before enabling
- [ ] Monitor trades in paper trading before live

---

## References

- **Archetype Registry**: `/engine/archetypes/registry.py`
- **Detection Logic**: `/engine/archetypes/logic_v2_adapter.py`
- **New Implementations**: `/engine/strategies/archetypes/bull/` and `/bear/`
- **Domain Wiring**: Check for `domain_boost` pattern in detection methods
- **Feature Flags**: `/engine/feature_flags.py`

---

## Glossary

- **Ghost**: Documented but not implemented (no method exists)
- **Stub**: Implemented but explicitly disabled (returns False)
- **Deprecated**: Superseded by better version
- **Wired**: Has domain engine integration (Wyckoff, SMC, Temporal, etc.)
- **Domain Boost**: Multiplicative amplification from domain engines (1.0x - 95x theoretical)
- **Fusion Score**: Weighted combination of multiple signals
- **Veto**: Hard stop that prevents entry regardless of score
