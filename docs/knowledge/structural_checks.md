# Structural Check Architecture (2026-03-09)

## Overview
Wired `logic.py` pattern-specific structural checks into the `ArchetypeInstance.detect()` pipeline as PRE-FILTERS before YAML gates + fusion scoring.

Pipeline: `structural_check(logic.py) → cooling → YAML gates → fusion_score → threshold → Signal`

## Architecture: Identity-Only Gates
Each `_check_X` in logic.py checks ONLY the 1-2 DEFINING structural conditions of the pattern. Quality filtering (fusion, liquidity, volume thresholds) is handled by existing YAML gates + adaptive fusion system.

### Files Modified
- `engine/archetypes/logic.py` — Rewrote all 16 `_check_X` methods to identity-only gates
- `engine/archetypes/structural_check.py` — Bridge module (created earlier, FROZEN_ARCHETYPES=empty set)
- `engine/archetypes/archetype_instance.py` — FROZEN_FEATURES=empty set, structural_checker wired
- `engine/integrations/isolated_archetype_engine.py` — Creates StructuralChecker, threads data
- `bin/backtest_v11_standalone.py` — Passes prev_row + lookback_df
- `bin/live/v11_shadow_runner.py` — Bar buffer for lookback data
- `bin/patch_frozen_features.py` — Computed real values for 5 frozen features

### Identity Gates Per Archetype

| Letter | YAML Name | Identity Gate | Notes |
|--------|-----------|---------------|-------|
| K | wick_trap | wick > 35% of candle range | Aligned with YAML derived:wick_anomaly |
| H | trap_within_trend | wick_anomaly + trend context (EMA or HTF fusion) | Differentiates from K |
| A | spring | PTI trap type detected | Very specific structural detection |
| B | order_block_retest | Price within 1.5 ATR of recent BOS | Lookback-dependent |
| C | fvg_continuation | FVG present + recent BOS | Continuation needs direction |
| D | failed_continuation | FVG present + ADX falling | Defining: momentum dying |
| E | liquidity_compression | ATR < 25th percentile | Low volatility |
| F | exhaustion_reversal | RSI > 78 or RSI < 22 | Extreme exhaustion |
| G | liquidity_sweep | Lower wick > 35% of range | Directional wick (longs only) |
| L | retest_cluster | vol_z > 1.0 + RSI extreme | Volume at RSI edge |
| M | confluence_breakout | ATR < 30th pctile + POC dist < 5% | Coil near value area |
| S1 | liquidity_vacuum | Bearish BOS detected | Structure break |
| S3 | whipsaw | Upper wick > 2x body | False breakout above |
| S4 | funding_divergence | Negative funding rate | Fixed: was "Distribution" (wrong) |
| S5 | long_squeeze | Positive funding extreme | Fixed: direction was inverted |
| S8 | volume_fade_chop | vol_z < 0.5 + ADX < 25 | Chop conditions |

## Results (2020-2024, $100K, 2bps+3bps)

| Metric | No Struct | Old AND gates | Identity v3 |
|--------|-----------|---------------|-------------|
| Trades | 837 | 269 | **972** |
| PF | 1.78 | 1.37 | **1.55** |
| PnL | $158K | $30K | **$148K** |
| WR | 80.4% | ? | **77.8%** |
| MaxDD | -7.1% | ? | **-13.6%** |
| Sharpe | 1.45 | ? | **1.14** |
| Archetypes | ~7 | ~5 | **11** |

## Key Fixes That Worked
1. **Wick threshold alignment**: logic.py used `wick > 2x body`, YAML used `wick > 35% range`. Aligned to 35% — wick_trap PnL recovered ($21K→$48.6K).
2. **Liquidity sweep identity**: Was RSI recovery (wrong pattern). Changed to lower wick > 35% — PnL went from $1.3K to $35K.
3. **S4/S5 direction fix**: S4 was "Distribution" (wrong), now checks negative funding. S5 direction was inverted. long_squeeze flipped from -$6.8K to +$1.5K.
4. **FVG continuation fix**: Added actual FVG check (was missing!) + BOS context.

## Key Lessons
60. **Structural check thresholds MUST match YAML gate thresholds** — wick_anomaly in logic.py used 2x body, YAML used 35% range. Different thresholds = structural check blocks signals that YAML would accept. Always align.
61. **Identity-only gates, not all-AND** — old logic.py required 4-6 simultaneous conditions (all-AND). This killed archetypes (trap_within_trend: 60→0 trades). Identity gates (1-2 conditions) let the YAML system handle quality filtering.
62. **Structural check ↔ YAML mapping mismatches are silent bugs** — S4 mapped to "Distribution" (volume climax) but YAML name is "funding_divergence" (funding rate). S5 checked NEGATIVE funding but YAML "long_squeeze" needs POSITIVE funding. Always verify the mapping matches the actual strategy.
63. **More archetypes + structural detection = more trades, slightly lower PF** — PF dropped 13% (1.78→1.55) but 4 archetypes flipped from loser to winner. PnL only down 6% ($158K→$148K). The YAML system needs re-calibration via Optuna for the new signal distribution.
64. **_get_liquidity_score() boms_strength formula was broken** — 0.70×boms (mean=0.013) + 0.30×vol_z made liquidity near-zero. Fixed to vol_z(0.35) + atr_pct(0.25) + fvg(0.20) + oi(0.20).
65. **Frozen features patched in feature store** — 5 cols computed: fusion_smc (14.3% non-zero), tf4h_fvg_present (17.6%), tf4h_squiggle_confidence (17.1%, 2393 unique), tf1h_frvp_distance_to_poc (99.8% non-zero, mean=0.027), tf4h_choch_flag (0.4%). FRVP took 388s.

## What Needs Re-Calibration
- YAML gates were Optuna-optimized for OLD signal distribution (no structural checks)
- MaxDD increased 7.1%→13.6% (structural checks let more bear-regime trades through)
- Re-running Optuna with structural checks enabled would close the PF gap
- oi_divergence still a net loser (2 trades, PF=0.21) — shadow mode OK
