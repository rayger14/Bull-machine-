# S2 (Failed Rally) - Deprecation & Archive

**Status**: PERMANENTLY DEPRECATED for BTC
**Reason**: Pattern models equity market behavior, not crypto microstructure
**Date**: 2025-11-20

---

## Why S2 Doesn't Work for BTC

### Pattern Expectation vs BTC Reality

| S2 Expects | BTC Actually Does |
|------------|-------------------|
| Weak, low-volume rallies | Violent short-squeeze bounces |
| Obvious wick rejections | High-volatility face-rippers |
| Gradual exhaustion | Explosive moves followed by slow bleeds |
| Textbook "failed push" | Messy, deceptive price action |

### Feature Frequency Problem

S2 features trigger far too often in BTC:
- **Wick rejections**: 40% of bars (noise, not signal)
- **Volume fades**: 26% of bars (common in BTC)
- **RSI < 45**: 50% of bars in bear markets
- **Liquidity < 0.2**: Common in flat markets

**Result**: No threshold combination isolates quality trades. Tightening parameters doesn't find "real S2s" - **because real S2s don't exist in BTC**.

### Structural Over-Firing

50 optimization trials with aggressive filtering:
- fusion_threshold: 0.70-0.85 (vs baseline 0.36)
- cooldown_bars: 12-24 (vs baseline 8)
- rsi_min: 65-80 (vs baseline 70)

**All trials**: 207-284 trades/year (7-10x over target)
**All trials**: PF 0.33-0.54 in 2022 bear market (loses money)

### Quant Verdict

> S2 exhibits structural over-firing due to high feature frequency and poor discriminative geometry relative to actual BTC bear-cycle microstructure. No choice of thresholds produces a subset with positive expectancy. The pattern lacks causal grounding in crypto market behavior and is statistically invalid as a trading signal.

**Recommendation**: Permanently deprecate for BTC.

---

## Archive for Future Equities Testing

S2 may work for traditional equities where:
- Rally failures are gradual, not violent
- Volume patterns are more predictable
- Microstructure matches textbook technical analysis

### Preserved Artifacts

**Optimization Results**:
- `results/s2_calibration/optuna_s2_calibration.db`
- `results/s2_optimization_FINAL.log`
- `S2_FAILED_RALLY_OPTIMIZATION_FINAL_REPORT.md`

**Runtime Enrichment Logic** (working):
- `engine/strategies/archetypes/bear/failed_rally_runtime.py`
- Features: wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag

**Test Configs**:
- `configs/test_s2_manual.json`
- `configs/test_s2_ultra_strict.json`
- `configs/test_s2_ultra_relaxed.json`

**Bug Fixes** (applicable to other archetypes):
1. Runtime enrichment hook in backtest
2. Config nesting structure
3. Parameter unit validation
4. Baseline trade isolation

---

## Real BTC Bear-Market Archetypes

### Working Patterns (BTC-Validated)

**S5 (Long Squeeze)**: ✅ WORKING
- PF: 1.86
- Trades/year: 9
- Win Rate: 55.6%
- **Why it works**: Matches BTC euphoria → capitulation cycles

### Promising Patterns (Not Yet Implemented)

**S1 (Liquidity Vacuum Reversal)**:
- Requires: liquidity_score (now backfilled ✓)
- Pattern: Deep liquidity drain → violent reversal
- BTC behavior: 2022-06-18 (Luna crash), 2022-11-09 (FTX)

**S3 (Distribution Climax Short)**:
- Requires: Volume spike + rejection + unstable structure
- Pattern: Euphoric top → distribution → dump
- BTC behavior: 2021-04, 2021-11, 2024-03 tops

**S4 (Funding Divergence)**:
- Requires: funding_z + liquidity_score (both fixed ✓)
- Pattern: Overcrowded shorts → violent counter-move
- BTC behavior: 2022-08 short squeeze, 2023-01 rally

**S6 (Capitulation Fade)**:
- Requires: Huge wick + massive volume + liquidity vacuum
- Pattern: Panic selling → exhaustion → reversal
- BTC behavior: 2022-05-12 (UST), 2022-11-09 (FTX)

**S7 (Reaccumulation Spring)**:
- Requires: Wyckoff events + PTI (structural)
- Pattern: Deep undercut during downtrend → spring
- BTC behavior: 2022-06 bottom, 2023-01 accumulation

---

## Next Steps

### Phase 1: Archive S2
- [x] Document deprecation rationale
- [x] Preserve optimization artifacts
- [ ] Add "DEPRECATED_FOR_BTC" flag to S2 code
- [ ] Update all configs with deprecation comment

### Phase 2: Implement Real Bear Archetypes
- [ ] S1 - Liquidity Vacuum Reversal
- [ ] S3 - Distribution Climax Short
- [ ] S4 - Funding Divergence
- [ ] S6 - Capitulation Fade
- [ ] S7 - Reaccumulation Spring

### Phase 3: Enable Deep Structure
- [ ] Wyckoff event detection (for S7)
- [ ] Temporal fusion (for S6, S7)
- [ ] Fib time clusters (for S1, S7)
- [ ] PTI integration (for S7)

### Phase 4: Multi-Archetype Optimization
- [ ] Optimize ALL bear archetypes together
- [ ] Balance portfolio with bull archetypes
- [ ] Validate on 2022 bear market
- [ ] Test on 2023 bull market

---

**Conclusion**: S2 is **not broken due to bugs** - it's **broken due to pattern-market mismatch**. Archive for equities, focus on BTC-native patterns.

---

**Generated**: 2025-11-20
**Status**: DEPRECATED_FOR_BTC
**Future**: Archive for equities testing
