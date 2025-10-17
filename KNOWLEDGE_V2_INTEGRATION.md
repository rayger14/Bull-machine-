# Knowledge Architecture v2.0 Integration Guide

**Status**: Fusion hooks implemented, ready for feature store integration
**Date**: 2025-10-16

---

## Overview

This document describes how Week 1-4 knowledge modules are integrated into the live fusion pipeline.

### Components Implemented ✅

1. **Fusion Hooks Module** (`engine/fusion/knowledge_hooks.py` - 665 lines)
   - All 8 knowledge hooks implemented
   - Bounded adjustments with safety clamps
   - Conflict guards to prevent double-counting
   - Feature contract assertions

2. **Configuration Template** (`configs/knowledge_v2/BTC_v2_baseline.json`)
   - Knowledge v2.0 enablement flags
   - Per-module configuration
   - Safety bounds (threshold ±0.10, score ±0.30, risk 0.5-1.5)

### Components Pending

1. **Feature Store v2.0 Builder** (extend `bin/build_feature_store.py`)
   - Add Week 1-4 feature computation
   - Schema version 2.0 with all 104 columns
   - Causal computation (past-only data)

2. **Hybrid Runner Integration** (modify `bin/live/hybrid_runner.py`)
   - Call `apply_knowledge_hooks()` after base fusion
   - Log decision reasons in shadow mode
   - Apply adjustments to threshold/score/risk

3. **Unit Tests** (`tests/test_knowledge_hooks.py`)
   - Test each hook individually
   - Test conflict guards
   - Test safety bounds

---

## Fusion Hooks Architecture

### Evaluation Order

To minimize overlap and prevent double-counting:

```
1. Base Fusion
   ├── Wyckoff (30%)
   ├── SMC (25%)
   ├── HOB (25%)
   └── Momentum (20%)
   → fusion_score (0-1)

2. Regime Policy
   → bounded threshold ±0.10

3. Knowledge Hooks (v2.0)
   ├── Structure
   │   ├── Internal vs External (conflict detection)
   │   ├── BOMS (market structure break)
   │   ├── Range Outcomes (breakout/fakeout/rejection)
   │   └── Squiggle (1-2-3 entry window)
   ├── Psychology/Volume
   │   ├── Fakeout Intensity (penalty)
   │   ├── PTI (trap detection, with fakeout guard)
   │   └── FRVP (value area positioning, with HOB guard)
   └── Macro Echo
       └── Correlation score (soft tilt)
   → (threshold_delta, score_delta, risk_multiplier, reasons)

4. Safety Clamps
   ├── threshold_delta ∈ [-0.10, +0.10]
   ├── score_delta ∈ [-0.30, +0.30]
   └── risk_multiplier ∈ [0.5, 1.5]

5. Final Decision
   ├── adjusted_score = fusion_score + score_delta
   ├── adjusted_threshold = base_threshold + threshold_delta + regime_delta
   └── adjusted_risk = base_risk * risk_multiplier
```

### Hook Details

#### 1. Internal vs External Structure
**File**: `engine/structure/internal_external.py`

**Logic**:
- Conflict > 0.75: Threshold +0.05, Score -0.03 (strong reversal warning)
- Conflict > 0.60: Threshold +0.03, Score -0.02 (moderate warning)
- Strong alignment: Score +0.02 (confirmation)

**Reason Codes**: `STRUCT_CONFLICT`, `STRUCT_ALIGN`

---

#### 2. BOMS (Break of Market Structure)
**File**: `engine/structure/boms_detector.py`

**Logic**:
- High-conviction (volume > 2.0x): Score +0.10, Risk x1.2
- With FVG: Score +0.08
- Standard: Score +0.05

**Conflict Guard**: Max one BOMS boost per bar (don't stack with plain BOS)

**Reason Codes**: `BOMS`

---

#### 3. 1-2-3 Squiggle Pattern
**File**: `engine/structure/squiggle_pattern.py`

**Logic**:
- Stage 2 + high quality (>0.8): Threshold -0.05
- Stage 2 + medium quality: Threshold -0.03

**Conflict Guard**: Never stack with Range Breakout boost

**Reason Codes**: `SQUIGGLE`

---

#### 4. Range Outcomes
**File**: `engine/structure/range_classifier.py`

**Logic**:
- Confirmed breakout: Score +0.08, Risk x1.15
- Fakeout: Score -0.10, Threshold +0.08 (penalty)
- Rejection: Score -0.05

**Conflict Guard**: Mutex (only one of breakout/fakeout/rejection per bar)

**Reason Codes**: `RANGE`

---

#### 5. PTI (Psychology Trap Index)
**File**: `engine/psychology/pti.py`

**Logic**:
- High intensity (>0.6): Score -0.15, Threshold +0.05
- Moderate (>0.4): Score -0.08
- Reversal likely: Additional threshold +0.03

**Conflict Guard**: If fakeout already applied, halve PTI penalty

**Reason Codes**: `PTI`

---

#### 6. Fakeout Intensity
**File**: `engine/psychology/fakeout_intensity.py`

**Logic**:
- High intensity (>0.7): Score -0.25, Threshold +0.10, Risk x0.7
- Medium (>0.5): Score -0.15, Threshold +0.05, Risk x0.85
- Fast return (<3 bars): Additional -0.05

**Reason Codes**: `FAKEOUT`

---

#### 7. FRVP (Fixed Range Volume Profile)
**File**: `engine/volume/frvp.py`

**Logic**:
- Long from below VA / Short from above VA: Score +0.05
- Near POC: Score +0.03
- Near LVN: Score -0.05 (gap risk)

**Conflict Guard**: If HOB gave liquidity bonus, halve FRVP's bonus

**Reason Codes**: `FRVP`

---

#### 8. Macro Echo
**File**: `engine/exits/macro_echo.py`

**Logic**:
- Crisis regime: Score -0.20, Risk x0.5
- Risk-off: Score -0.10, Risk x0.7
- Risk-on: Score +0.05, Risk x1.1
- Correlation score: Additional ±0.10 scaled

**Note**: Separate from hard macro veto (which blocks all entries)

**Reason Codes**: `MACRO`

---

## Feature Store v2.0 Schema

### Required Columns (66 new + 38 existing = 104 total)

**Week 1: Structure (29 columns)**
```python
# Internal vs External
'internal_phase', 'external_trend', 'structure_alignment', 'conflict_score',
'internal_strength', 'external_strength',

# BOMS
'boms_detected', 'boms_direction', 'boms_volume_surge', 'boms_fvg_present',
'boms_confirmation', 'boms_break_level', 'boms_displacement',

# Squiggle
'squiggle_stage', 'squiggle_pattern_id', 'squiggle_direction', 'squiggle_entry_window',
'squiggle_confidence', 'squiggle_bos_level', 'squiggle_retest_quality', 'squiggle_bars_since_bos',

# Range Outcomes
'range_outcome', 'range_outcome_direction', 'range_outcome_confidence',
'range_high', 'range_low', 'breakout_strength', 'volume_confirmation', 'bars_in_range',
```

**Week 2: Psychology & Volume (24 columns)**
```python
# PTI
'pti_score', 'pti_trap_type', 'pti_confidence', 'pti_reversal_likely',
'pti_rsi_divergence', 'pti_volume_exhaustion', 'pti_wick_trap', 'pti_failed_breakout',

# FRVP
'frvp_poc', 'frvp_va_high', 'frvp_va_low', 'frvp_hvn_count', 'frvp_lvn_count',
'frvp_current_position', 'frvp_distance_to_poc', 'frvp_distance_to_va',

# Fakeout Intensity
'fakeout_detected', 'fakeout_intensity', 'fakeout_direction', 'fakeout_breakout_level',
'fakeout_return_speed', 'fakeout_volume_weakness', 'fakeout_wick_rejection', 'fakeout_no_followthrough',
```

**Week 4: Macro Echo (7 columns)**
```python
'macro_regime', 'macro_dxy_trend', 'macro_yields_trend', 'macro_oil_trend',
'macro_vix_level', 'macro_correlation_score', 'macro_exit_recommended',
```

**Existing (38 columns)**
```python
# OHLCV
'open', 'high', 'low', 'close', 'volume',

# Technical
'atr_20', 'atr_14', 'adx_14', 'rsi_14', 'sma_20', 'sma_50', 'sma_100',

# Domain Scores
'wyckoff', 'smc', 'hob', 'momentum', 'temporal',

# Macro Flags
'macro_veto', 'macro_exit_flag',

# MTF
'mtf_align',

# (... plus others from existing schema)
```

---

## Integration Steps

### Step 1: Extend Feature Store Builder

**File**: `bin/build_feature_store.py`

Add after line 247 (after domain scores):

```python
# Week 1-4 Features (causal computation)
print("\n🧠 Computing Week 1-4 knowledge features...")

from engine.structure.internal_external import detect_internal_external_structure
from engine.structure.boms_detector import detect_boms
from engine.structure.squiggle_pattern import detect_squiggle_pattern
from engine.structure.range_classifier import classify_range_outcome
from engine.psychology.pti import calculate_pti
from engine.volume.frvp import calculate_frvp
from engine.psychology.fakeout_intensity import detect_fakeout_intensity
from engine.exits.macro_echo import analyze_macro_echo

for i in range(len(df_1h)):
    if i % 100 == 0:
        print(f"   Processing knowledge bar {i}/{len(df_1h)}...")

    # Get causal window
    window_1h = df_1h.iloc[:i+1].tail(200)
    window_4h = df_4h[df_4h.index <= df_1h.index[i]].tail(100)
    window_1d = df_1d[df_1d.index <= df_1h.index[i]].tail(50)

    # Skip if not enough data
    if len(window_1h) < 50:
        continue

    try:
        # Week 1: Structure
        internal_external = detect_internal_external_structure(window_1h, window_4h, window_1d)
        boms = detect_boms(window_1h, window_4h)
        squiggle = detect_squiggle_pattern(window_1h)
        range_outcome = classify_range_outcome(window_1h, window_4h)

        # Week 2: Psychology & Volume
        pti = calculate_pti(window_1h)
        frvp = calculate_frvp(window_1h, lookback=100)
        fakeout = detect_fakeout_intensity(window_1h)

        # Week 4: Macro Echo
        macro_snapshot = fetch_macro_snapshot(macro_data, df_1h.index[i])
        macro_echo = analyze_macro_echo(macro_snapshot)

        # Write to features dataframe
        for col, val in internal_external.to_dict().items():
            features.loc[df_1h.index[i], col] = val
        # ... (repeat for all modules)

    except Exception as e:
        # Fill with neutral values on error
        pass

print("   ✅ Week 1-4 knowledge features computed")

# Update schema version
features.attrs['schema_version'] = '2.0'
```

---

### Step 2: Integrate into Hybrid Runner

**File**: `bin/live/hybrid_runner.py`

Find the fusion decision section and add knowledge hooks:

```python
# After base fusion analysis
fusion_result = analyze_fusion(df_1h, df_4h, df_1d, config)

# NEW: Apply knowledge v2.0 hooks
if config.get('knowledge_v2', {}).get('enabled', False):
    # Get current features from feature store (or compute on-the-fly)
    current_feats = get_current_features(feature_store, current_idx)

    # Apply hooks
    adjusted_score, threshold_delta, risk_mult, reasons = apply_knowledge_hooks(
        fusion_score=fusion_result.fusion_score,
        feats=current_feats,
        current_price=current_price,
        config=config
    )

    # Log in shadow mode
    if config.get('knowledge_v2', {}).get('shadow_mode', False):
        log_decision(
            timestamp=current_time,
            base_score=fusion_result.fusion_score,
            adjusted_score=adjusted_score,
            threshold_delta=threshold_delta,
            risk_mult=risk_mult,
            reasons=reasons
        )

    # Apply adjustments (if not in shadow mode)
    if not config.get('knowledge_v2', {}).get('shadow_mode', False):
        fusion_result.fusion_score = adjusted_score
        entry_threshold += threshold_delta
        risk_pct *= risk_mult
```

---

### Step 3: Testing Strategy

**Phase 1: Shadow Mode** (Log only, don't change decisions)
```bash
# Run with shadow_mode=true
python bin/live/hybrid_runner.py \
    --asset BTC \
    --start 2024-07-01 --end 2024-09-30 \
    --config configs/knowledge_v2/BTC_v2_baseline.json

# Review decision logs
grep "KNOWLEDGE_V2" logs/decisions.log | head -50
```

**Phase 2: Ablation Testing** (Enable one hook at a time)
```bash
# Test with only BOMS enabled
python bin/optimize_v18.py --mode quick \
    --config configs/knowledge_v2/BTC_v2_boms_only.json

# Compare vs baseline
python tools/compare_configs.py \
    --baseline configs/v1.9/BTC_baseline.json \
    --experiment configs/knowledge_v2/BTC_v2_boms_only.json
```

**Phase 3: Full Integration** (All hooks enabled)
```bash
# Sweep configurations to find optimal weights
python sweep_knowledge_v2.py \
    --asset BTC \
    --start 2024-01-01 --end 2024-09-30 \
    --hooks internal_external,boms,squiggle,range,pti,fakeout,frvp,macro_echo
```

---

## Configuration Examples

### Baseline (All Enabled)
```json
{
  "knowledge_v2": {
    "enabled": true,
    "shadow_mode": false
  },
  "structure": {
    "internal_external_enabled": true,
    "boms_enabled": true,
    "squiggle_enabled": true,
    "range_outcomes_enabled": true
  }
}
```

### Conservative (Structure Only)
```json
{
  "knowledge_v2": {
    "enabled": true
  },
  "structure": {
    "internal_external_enabled": true,
    "boms_enabled": true,
    "squiggle_enabled": false,
    "range_outcomes_enabled": true
  },
  "psychology": {
    "pti_enabled": false,
    "fakeout_intensity_enabled": false
  }
}
```

### Aggressive (Maximum Signals)
```json
{
  "knowledge_v2": {
    "enabled": true
  },
  "structure": {
    "internal_external_enabled": true,
    "boms_enabled": true,
    "squiggle_enabled": true,
    "range_outcomes_enabled": true
  },
  "psychology": {
    "pti_enabled": true,
    "fakeout_intensity_enabled": true
  },
  "macro": {
    "echo_enabled": true
  }
}
```

---

## Safety Mechanisms

### 1. Bounded Adjustments
```python
# Global safety clamps
threshold_delta = np.clip(threshold_delta, -0.10, 0.10)
score_delta = np.clip(score_delta, -0.30, 0.30)
risk_multiplier = np.clip(risk_multiplier, 0.5, 1.5)
```

### 2. Conflict Guards
```python
# Example: PTI penalty halved if fakeout already applied
if fakeout_applied:
    pti_penalty *= 0.5
```

### 3. Feature Contract Assertions
```python
# Fail loudly if required columns missing
assert_feature_contract(feature_store, schema_version="2.0")
```

### 4. Shadow Mode
```python
# Log decisions without affecting orders
if config['knowledge_v2']['shadow_mode']:
    log_only(decision)
else:
    execute_decision(decision)
```

---

## Next Steps

1. **Implement Feature Store v2.0 Builder** (extend `build_feature_store.py`)
2. **Integrate into Hybrid Runner** (modify decision pipeline)
3. **Run Shadow Mode Tests** (verify decisions make sense)
4. **Ablation Testing** (isolate best-performing hooks)
5. **Configuration Sweep** (find optimal weights/thresholds)
6. **PyTorch Training** (use features for ML meta-optimization)

---

## References

- Knowledge Modules: `engine/structure/`, `engine/psychology/`, `engine/volume/`, `engine/exits/`
- Fusion Hooks: `engine/fusion/knowledge_hooks.py`
- Feature Inventory: `ML_FEATURE_INVENTORY.md`
- Architecture: `ML_META_OPTIMIZER_ARCHITECTURE.md`
- Complete Knowledge: `COMPLETE_KNOWLEDGE_ARCHITECTURE.md`

---

**Status**: Ready for feature store v2.0 implementation and hybrid runner integration
**Estimated Completion**: Feature store (4-6 hours), Integration (2-4 hours), Testing (8-12 hours)
