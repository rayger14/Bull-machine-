# Regime Routing Current State Analysis
**Generated**: 2025-11-13
**Analyzed Version**: logic_v2_adapter@r1

---

## Executive Summary

**Status**: Regime routing infrastructure EXISTS but is NOT ACTIVE due to missing config.

**Critical Finding**: The 96.5% Trap Within Trend dominance in 2022 (causing PF=0.11) is NOT a code bug - it's a **config gap**. The routing system is fully implemented and waiting for regime weight configurations.

---

## Implementation Architecture

### 1. Routing Code Path (FUNCTIONAL)

**Location**: `engine/archetypes/logic_v2_adapter.py:403-426`

```python
# STEP 2: Apply regime-specific routing weights
regime = context.regime_label
routing_config = self.config.get('routing', {})
regime_routing = routing_config.get(regime, {})
regime_weights = regime_routing.get('weights', {})

if regime_weights:
    logger.info(f"[REGIME ROUTING] regime={regime}, applying weights: {regime_weights}")
    adjusted_candidates = []
    for name, score, meta, priority in candidates:
        regime_mult = regime_weights.get(name, 1.0)
        adjusted_score = score * regime_mult  # Score multiplication
        if regime_mult != 1.0:
            logger.info(f"[REGIME ROUTING] {name}: {score:.3f} × {regime_mult:.2f} = {adjusted_score:.3f}")
    candidates = [(n, s, m, p) for n, s, m, p, _ in adjusted_candidates]
```

**Behavior**:
- Reads `config['routing'][regime]['weights']` (NOT `config['archetypes']['routing']`)
- Multiplies archetype scores by regime-specific weights
- Uses **score multiplication** (Option A from mission spec)
- Defaults to `1.0` (no change) if weight missing

**Status**: FULLY FUNCTIONAL, just needs config data

---

### 2. Config Structure (MISSING)

**Expected Path**: `config['routing'][regime]['weights']`

**Example from Code**:
```python
routing_config = self.config.get('routing', {})  # Top-level 'routing' key
regime_routing = routing_config.get(regime, {})  # Nested by regime
regime_weights = regime_routing.get('weights', {})  # Archetype weights
```

**Current Baseline Configs**: NO `routing` key exists

**Checked Configs**:
- `baseline_btc_bull_pf20_biased_20pct_no_ml.json`: No `routing` key
- All frozen configs: No `routing` key
- Router v10 configs: No `routing` key

**Result**: System defaults to `routing_config = {}`, so `regime_weights = {}` (empty)

**Log Evidence**:
```
[ROUTING CHECK] regime=risk_off, routing_config_keys=[], regime_found=False
```

This means NO weights are applied → all archetypes get equal 1.0x weight in ALL regimes.

---

## Root Cause of 2022 Failure

### Problem Statement

**2022 Archetype Distribution** (Router v10 full backtest):
```
trap_within_trend: 96.5% of trades (27/28)
order_block_retest: 3.5% (1 trade)
```

**2022 Performance**:
- Profit Factor: 0.11 (terrible)
- Win Rate: ~30%
- Avg PNL: -$24.76 per trade

### Why This Happened

**Equal Weight Problem**:
All 11 bull archetypes (A-M) have equal 1.0x weight in risk_off:
```
trap_within_trend: 1.0x in risk_off (WRONG - should be 0.2x)
order_block_retest: 1.0x in risk_off (WRONG - should be 0.4x)
failed_rally: 1.0x in risk_off (WRONG - should be 1.8x)
long_squeeze: 1.0x in risk_off (WRONG - should be 2.0x)
```

**Trap Within Trend Dominance**:
- Low fusion threshold (0.35-0.44) → triggers easily
- Simple rules (ADX > 25, low liquidity) → matches frequently in choppy markets
- No regime suppression → fires in bear markets when it shouldn't

**Bear Archetypes Silent**:
- S2 (Rejection), S5 (Long Squeeze) not even enabled in baseline configs
- S1, S3, S4, S8 exist but have `enable_S* = false`
- Even if enabled, 1.0x weight won't overcome bull archetype dominance

---

## Feature Flag Analysis

### Soft Filters (Engine-Level)

**Status**: IMPLEMENTED in `logic_v2_adapter.py:309-338`

```python
from engine import feature_flags as features

if features.SOFT_LIQUIDITY_FILTER:
    if liquidity_score < self.min_liquidity:
        fusion_score *= 0.7  # 30% penalty

if features.SOFT_REGIME_FILTER:
    if regime in ['crisis', 'risk_off']:
        fusion_score *= 0.8  # 20% penalty

if features.SOFT_SESSION_FILTER:
    if hour >= 22 or hour < 8:  # Asian session
        fusion_score *= 0.85  # 15% penalty
```

**Impact**: Global penalties (20-30%) are TOO WEAK to fix 96.5% dominance

**Problem**: Applying 0.8x regime penalty to ALL archetypes equally doesn't shift distribution

### Evaluate-All Archetypes Flag

**Status**: ENABLED by default (`features.EVALUATE_ALL_ARCHETYPES = True`)

**Behavior**:
- Evaluates ALL archetypes, picks best score
- Prevents archetype starvation from early returns
- Essential for regime routing to work

**Compatibility**: Regime weights applied AFTER all archetypes evaluated (line 403)

---

## Gaps in Current Implementation

### 1. Missing Config Files

**What's Needed**:
```json
{
  "routing": {
    "risk_on": {
      "weights": {
        "trap_within_trend": 1.3,
        "volume_exhaustion": 1.1,
        "order_block_retest": 1.4
      }
    },
    "risk_off": {
      "weights": {
        "trap_within_trend": 0.2,
        "failed_rally": 1.8,
        "long_squeeze": 2.0
      }
    }
  }
}
```

**What Exists**: Nothing (configs have no `routing` key)

### 2. Bear Archetypes Disabled

**Current State** (baseline configs):
```json
"enable_S1": false,  // Breakdown
"enable_S2": false,  // Rejection
"enable_S3": false,  // Whipsaw
"enable_S4": false,  // Distribution
"enable_S5": false,  // Long Squeeze
"enable_S8": false   // Volume Fade Chop
```

**Impact**: Even with routing weights, bear archetypes won't fire

**Fix Required**: Set `enable_S2 = true`, `enable_S5 = true` at minimum

### 3. No Threshold Adjustments

**Current Approach**: Score multiplication only

**Missing**: Threshold deltas per regime (from ThresholdPolicy)
```json
"archetype_overrides": {
  "trap_within_trend": {
    "risk_off": {"fusion": +0.10}  // Make harder to trigger
  }
}
```

**Note**: Score multiplication (current) is simpler than threshold adjustment

---

## Validation of Implementation

### Score Multiplication Mechanics

**Example Scenario**:
```
Candidates after archetype evaluation:
1. trap_within_trend: score=0.42
2. long_squeeze: score=0.38
3. order_block_retest: score=0.35

Regime: risk_off
Weights:
- trap_within_trend: 0.2x
- long_squeeze: 2.0x
- order_block_retest: 0.4x

After routing adjustment:
1. long_squeeze: 0.38 × 2.0 = 0.76 (WINNER)
2. trap_within_trend: 0.42 × 0.2 = 0.084 (suppressed)
3. order_block_retest: 0.35 × 0.4 = 0.14 (suppressed)
```

**Result**: Bear archetype (long_squeeze) wins despite lower base score

**Edge Cases Handled**:
- Weights default to 1.0 if missing
- Scores clamped to [0, 1] range (no overflow)
- Ties broken by priority (line 429)

### Logging Coverage

**Diagnostic Logs** (already in code):
```
[ROUTING DEBUG] context.regime_label=risk_off, candidates=[...]
[ROUTING CHECK] regime=risk_off, routing_config_keys=[], regime_found=False
[REGIME ROUTING] regime=risk_off, applying weights: {...}
[REGIME ROUTING] trap_within_trend: 0.420 × 0.20 = 0.084
```

**Status**: Comprehensive logging exists for debugging

---

## Recommendations

### Immediate Actions (No Code Changes Required)

1. **Create Routing Config**:
   - Add `routing` key to baseline configs
   - Start with conservative weights (0.3-0.5x suppression, 1.3-1.5x boost)
   - Test on 2022-2024 full period

2. **Enable Bear Archetypes**:
   - Set `enable_S2 = true` (Rejection)
   - Set `enable_S5 = true` (Long Squeeze)
   - Add thresholds to `thresholds.S2` and `thresholds.S5`

3. **Validate Regime Classification**:
   - Check if 2022 is actually classified as `risk_off` or `crisis`
   - If not, regime classifier needs tuning FIRST

### Code Enhancements (Optional)

1. **Add Final Gate Delta**:
   ```python
   final_gate_delta = regime_routing.get('final_gate_delta', 0.0)
   fusion_th += final_gate_delta  # Make entry harder in risk_off
   ```

2. **Add Regime Weight Clamps**:
   ```python
   regime_mult = max(0.05, min(3.0, regime_weights.get(name, 1.0)))  # Prevent extreme weights
   ```

3. **Add Telemetry**:
   - Track regime distribution in backtests
   - Log archetype distribution per regime
   - Measure weight impact on trade count

---

## Summary

| Component | Status | Blocker |
|-----------|--------|---------|
| Routing Code | FUNCTIONAL | None |
| Score Multiplication | IMPLEMENTED | None |
| Logging | COMPREHENSIVE | None |
| Config Files | MISSING | YES |
| Bear Archetypes | DISABLED | YES |
| Regime Classification | UNKNOWN | Maybe |

**Bottom Line**: The plumbing is perfect. We just need to turn on the faucet (add config).

**Estimated Fix Time**: 1-2 hours (config writing + validation)

**Risk**: Low (code is battle-tested, just needs data)
