# Fusion Pipeline Audit & Cleanup Specification

**Phase**: Phase 3 - Archetypes, Regimes & Fusion Cleanup
**Status**: Requirements Definition
**Objective**: Define ONE canonical fusion pipeline, deprecate zombie modules

---

## Executive Summary

The Bull Machine codebase contains multiple fusion modules with unclear responsibilities and overlapping functionality. This creates:
- **Double-counting risk**: Same signal contributing via multiple paths
- **Debug difficulty**: Unclear which module affects final score
- **Maintenance burden**: Changes required across multiple files

**Goal**: Define a single, linear fusion chain with clear responsibilities per module. Deprecate or no-op unused modules. Provide full transparency via debug trace.

---

## 1. Current State Audit

### 1.1 Discovered Fusion Modules

| Module | Location | Purpose (Documented) | Status (Inferred) |
|--------|----------|---------------------|-------------------|
| **k2_fusion** | `engine/fusion/k2_fusion.py` | Multi-timeframe meta-fusion (1H/4H/1D/macro) | ✅ **LIVE** - Core fusion |
| **domain_fusion** | `engine/fusion/domain_fusion.py` | Integrate Wyckoff/SMC/HOB/Momentum/Temporal | ✅ **LIVE** - Domain scoring |
| **adaptive** | `engine/fusion/adaptive.py` | EMA-smooth regime probabilities, blend weights | ⚠️ **PARTIAL** - Used in v1.8+ |
| **advanced_fusion** | `engine/fusion/advanced_fusion.py` | Delta routing (momentum/macro/HOB deltas) | ❓ **GHOST** - Unclear if active |

### 1.2 Module Responsibilities (As Documented)

#### k2_fusion.py (Lines 1-187)
**Purpose**: Combine multi-timeframe fusion scores with disagreement penalty

**Inputs**:
```python
- tf1h_fusion_score  # Local momentum (1H)
- tf4h_fusion_score  # Structural context (4H)
- tf1d_fusion_score  # Macro trend (1D)
- macro_correlation_score  # Cross-asset regime
```

**Output**:
```python
k2_fusion_score = weighted_mean * (1 - disagreement_penalty)
# Range: [0, 1], penalty ∈ [0.7, 1.0] based on std deviation
```

**Key Functions**:
- `compute_k2_fusion(row, weights)` - Base weighted fusion
- `compute_k2_fusion_with_regime_adaptation(row)` - VIX-based weight shifting
- `validate_k2_fusion_inputs(df)` - Data quality checks

**Status**: ✅ **Core module** - Used by backtest runner as primary fusion score

---

#### domain_fusion.py (Lines 1-671)
**Purpose**: Calculate domain-specific scores (Wyckoff/SMC/HOB/Momentum) and fuse into single score

**Inputs**:
```python
df_1h  # 1H OHLCV
df_4h  # 4H OHLCV
df_1d  # 1D OHLCV
config # Fusion config with weights
```

**Output**:
```python
FusionSignal(
    score: float,              # 0-1 overall fusion
    wyckoff_score: float,      # Wyckoff component
    smc_score: float,          # SMC component
    hob_score: float,          # Liquidity component
    momentum_score: float,     # Momentum component
    mtf_aligned: bool,         # Multi-timeframe agreement
    direction: str,            # 'long' | 'short' | 'neutral'
    ...
)
```

**Key Functions**:
- `analyze_fusion(df_1h, df_4h, df_1d, config)` - Main entry point
- `_wyckoff_to_score()` - Convert Wyckoff phase to [0-1] score
- `_smc_to_score()` - Convert SMC signals to [0-1] score
- `_hob_to_score()` - Convert HOB/liquidity to [0-1] score
- `_momentum_to_score()` - Convert RSI/MACD to [0-1] score
- `_check_mtf_alignment()` - Validate trend alignment across timeframes

**Enhancements** (v1.8.5+):
- Fibonacci confluence bonus (line 463)
- Fourier noise filter multiplier (line 469)
- Event veto (conference/high-leverage) (line 475)
- Narrative trap detection (HODL/distribution) (line 487)

**Enhancements** (v1.8.6+):
- Temporal/Gann signal integration (line 503)
- LPPLS blowoff veto (line 522)
- Macro fusion composite (line 556)

**Status**: ✅ **Domain scoring module** - Computes tf1h/tf4h/tf1d fusion scores consumed by k2_fusion

**Question**: How does this relate to `engine/archetypes/logic.py::calculate_fusion_score()`?

---

#### adaptive.py (Lines 1-295)
**Purpose**: Regime-aware parameter morphing (smooth blending, no discrete switches)

**Inputs**:
```python
regime_info = {
    "regime": "risk_on",
    "proba": {"risk_on": 0.65, "neutral": 0.25, ...}
}
```

**Output**:
```python
{
    "fusion_weights": {"wyckoff": 0.46, "liquidity": 0.26, ...},
    "gates": {"min_liquidity": 0.18, "final_fusion_floor": 0.35},
    "exit_params": {"trail_atr": 1.2, "max_bars": 84},
    "size_mult": 1.15,
    "ml_threshold": 0.32,
    "regime_probs_ema": {...}
}
```

**Key Functions**:
- `ema_smooth(prev_probs, curr_probs, alpha)` - Smooth regime transitions
- `adapt_weights(base_profiles, regime_probs)` - Blend fusion weights across regimes
- `adapt_gates(base_profiles, regime_probs)` - Blend entry thresholds
- `adapt_exit_params(base_profiles, regime_probs)` - Blend exit policies
- `regime_size_mult(sizing_curve, regime_probs)` - Risk sizing adjustment

**Class**: `AdaptiveFusion` - Stateful coordinator with EMA state

**Status**: ⚠️ **PARTIAL** - Used in v1.8+ configs, but unclear if active in current pipeline

**Question**: Does this overlap with `regime_routing` from Phase 3 spec?

---

#### advanced_fusion.py (Lines 1-100+)
**Purpose**: Delta routing to prevent double-counting

**Architecture** (from docstring):
```
Base weights:
  - Wyckoff
  - Liquidity
  - SMC
  - Temporal

Delta channels (boosts):
  - Momentum delta
  - Macro delta
  - HOB volume delta
  - Wyckoff HPS delta

Fusion = base_weighted_score + sum(deltas)
```

**Classes**:
- `FusionTelemetry` - Waterfall trace for debugging
- `AdvancedFusionSignal` - Output with full telemetry
- `AdvancedFusionEngine` - Orchestrator

**Status**: ❓ **GHOST** - No references found in backtest runner or configs

**Question**: Is this module dead code from an abandoned refactor?

---

### 1.3 Pipeline Flow Confusion

**Current Suspected Flow** (needs validation):

```
1. Feature Store Builder (offline)
   ├─ domain_fusion.analyze_fusion() computes tf1h/tf4h/tf1d_fusion_score
   └─ Writes scores to feature store parquet

2. Backtest Runtime (online)
   ├─ Load features from parquet (includes tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion_score)
   ├─ k2_fusion.compute_k2_fusion() combines timeframe scores → k2_fusion_score
   ├─ archetypes/logic.py::calculate_fusion_score() uses k2_fusion_score OR tf1h_fusion_score?
   ├─ adaptive.py adjusts weights/gates based on regime? (if enabled)
   └─ advanced_fusion.py ... does nothing? (not referenced)
```

**Ambiguities**:
1. ❓ Does `domain_fusion` run at feature build OR runtime?
2. ❓ Is `k2_fusion` redundant if domain_fusion already computes tf1h_fusion_score?
3. ❓ Does `archetypes/logic.py` use k2_fusion or recalculate from components?
4. ❓ Is `adaptive.py` actually wired into decision pipeline?
5. ❓ What does `advanced_fusion.py` do (if anything)?

---

## 2. Canonical Fusion Pipeline Definition

### 2.1 Design Principles

**Single Responsibility**:
- Each module has ONE clear purpose
- No overlapping calculations
- No double-counting of signals

**Linear Flow**:
```
Feature Extraction
   ↓
Base Domain Fusion
   ↓
Multi-Timeframe Fusion (K2)
   ↓
Temporal Adjustments
   ↓
Regime Routing
   ↓
Safety Rails
   ↓
Final Fusion Score → Entry Decision
```

**Transparency**:
- Every adjustment logged (debug mode)
- Full waterfall trace available
- No "black box" score jumps

### 2.2 Canonical Fusion Chain

```
┌────────────────────────────────────────────────────────────────┐
│ MODULE 1: Base Domain Fusion (domain_fusion.py)               │
│ Purpose: Compute domain-specific scores from raw OHLCV        │
│ Inputs:  df_1h, df_4h, df_1d (OHLCV DataFrames)              │
│ Outputs: wyckoff_score, smc_score, hob_score, momentum_score │
│ Location: Feature Store Builder (offline)                     │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ MODULE 2: Timeframe Fusion Scoring (domain_fusion.py)         │
│ Purpose: Weight domain scores → timeframe fusion scores       │
│ Inputs:  wyckoff_score, smc_score, hob_score, momentum_score │
│ Outputs: tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion   │
│ Weights: {"wyckoff": 0.30, "smc": 0.15, "liquidity": 0.25,   │
│           "momentum": 0.30}                                    │
│ Location: Feature Store Builder (offline)                     │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ MODULE 3: Multi-Timeframe K2 Fusion (k2_fusion.py)           │
│ Purpose: Combine 1H/4H/1D scores with disagreement penalty    │
│ Inputs:  tf1h_fusion_score, tf4h_fusion_score, tf1d_fusion   │
│ Output:  k2_fusion_score                                       │
│ Weights: {"1h": 0.35, "4h": 0.35, "1d": 0.20, "macro": 0.10} │
│ Penalty: max(0.7, 1.0 - std(scores) * 1.5)                   │
│ Location: Runtime (archetypes/logic.py::calculate_fusion)    │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ MODULE 4: Temporal Adjustments (temporal/gann_cycles.py)      │
│ Purpose: Boost/penalize based on Gann cycles, Fib time       │
│ Inputs:  k2_fusion_score, temporal feature set               │
│ Output:  k2_fusion_score ± temporal_bonus (± 0.15 max)       │
│ Location: Runtime (if temporal.enabled == true)              │
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ MODULE 5: Regime Routing (NEW from Phase 3)                   │
│ Purpose: Apply regime-specific archetype weights & floors     │
│ Inputs:  k2_fusion_score, archetype_slug, regime_info        │
│ Output:  regime_adjusted_fusion_score                         │
│ Weights: Archetype-specific per regime (e.g., H 1.2x in risk_on) │
│ Location: Runtime (archetypes/logic.py::apply_regime_routing)│
└────────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────┐
│ MODULE 6: Safety Rails (threshold_policy.py)                  │
│ Purpose: Enforce minimum thresholds, crisis overrides        │
│ Inputs:  regime_adjusted_fusion_score, liquidity_score       │
│ Checks:  fusion >= min_threshold, liquidity >= min_liq       │
│ Output:  final_fusion_score OR veto (0.0)                    │
│ Location: Runtime (final gate before entry decision)         │
└────────────────────────────────────────────────────────────────┘
                             ↓
                    Final Fusion Score
                    → Entry Decision (if score >= threshold)
```

### 2.3 Module Deprecation Plan

#### ✅ KEEP: domain_fusion.py
**Rationale**: Core domain scoring (Wyckoff/SMC/HOB/Momentum) is fundamental

**Actions**:
- NO CHANGES (module is clean and well-documented)
- Add validation: Ensure tf1h/tf4h/tf1d_fusion_score written to feature store
- Document: Feature store builder dependency

#### ✅ KEEP: k2_fusion.py
**Rationale**: Multi-timeframe combination with disagreement penalty is valuable

**Actions**:
- CLARIFY: Document that this runs at RUNTIME, not feature build
- ADD: Validation function to check inputs exist in feature store
- DOCUMENT: Usage in `archetypes/logic.py::calculate_fusion_score()`

#### ⚠️ EVALUATE: adaptive.py
**Status**: Used in v1.8+ but unclear if active in current pipeline

**Decision Tree**:
```
Is adaptive.py currently wired into backtest runner?
├─ YES
│  └─ Does it overlap with new regime_routing from Phase 3?
│     ├─ YES → DEPRECATE adaptive.py (use regime_routing instead)
│     └─ NO → KEEP adaptive.py (smooth blending complements hard routing)
│
└─ NO
   └─ DEPRECATE adaptive.py (no-op module)
```

**Investigation Required**:
```bash
grep -r "AdaptiveFusion\|adaptive.py" bin/backtest_knowledge_v2.py configs/*.json
```

**If DEPRECATE**:
```python
# engine/fusion/adaptive.py (add warning)
import warnings
warnings.warn(
    "adaptive.py is deprecated - use regime_routing instead. "
    "This module will be removed in v2.0.",
    DeprecationWarning
)
```

**If KEEP**:
- Document relationship with regime_routing
- Clarify use case: "Smooth blending (adaptive) vs hard switches (regime_routing)"

#### ❌ DEPRECATE: advanced_fusion.py
**Rationale**: No references found, likely abandoned experimental code

**Actions**:
```python
# engine/fusion/advanced_fusion.py (add header warning)
"""
DEPRECATED MODULE - DO NOT USE

This module was an experimental delta routing implementation that
was never fully integrated. Use the canonical fusion chain instead:
  1. domain_fusion.py (domain scoring)
  2. k2_fusion.py (multi-timeframe combination)
  3. regime_routing (regime adjustments)

This file is preserved for historical reference only.
Will be removed in v2.0.
"""
raise DeprecationWarning("advanced_fusion.py is deprecated - see fusion pipeline spec")
```

**Move to archive**:
```bash
mkdir -p engine/fusion/deprecated
git mv engine/fusion/advanced_fusion.py engine/fusion/deprecated/
```

---

## 3. Fusion Debug Trace

### 3.1 Purpose

Provide full transparency into fusion calculation for:
- Debugging unexpected scores
- Validating regime routing applied correctly
- Auditing double-counting issues

### 3.2 Trace Format

**Debug Mode Config**:
```json
{
  "fusion": {
    "debug_trace": true,
    "trace_output": "artifacts/fusion_trace.jsonl"
  }
}
```

**Trace Output** (JSON Lines):
```json
{
  "timestamp": "2024-03-15T12:00:00",
  "archetype": "trap_within_trend",

  "fusion_waterfall": {
    "1_domain_scores": {
      "wyckoff_score": 0.65,
      "smc_score": 0.48,
      "hob_score": 0.52,
      "momentum_score": 0.58
    },

    "2_timeframe_fusion": {
      "tf1h_fusion_score": 0.56,
      "tf4h_fusion_score": 0.52,
      "tf1d_fusion_score": 0.60
    },

    "3_k2_fusion": {
      "weighted_mean": 0.556,
      "disagreement_std": 0.04,
      "disagreement_penalty": 0.94,
      "k2_fusion_score": 0.523
    },

    "4_temporal_adjustment": {
      "gann_confluence": 0.72,
      "fib_time_cluster": 0.65,
      "temporal_bonus": 0.05,
      "after_temporal": 0.573
    },

    "5_regime_routing": {
      "regime": "risk_on",
      "regime_confidence": 0.85,
      "archetype_weight": 1.2,
      "global_multiplier": 1.0,
      "after_regime_weight": 0.688,
      "fusion_floor": 0.35,
      "floor_check": "PASS"
    },

    "6_safety_rails": {
      "liquidity_score": 0.42,
      "liquidity_threshold": 0.30,
      "liquidity_check": "PASS",
      "final_fusion_score": 0.688
    }
  },

  "entry_decision": {
    "threshold": 0.40,
    "decision": "ENTER",
    "price": 68500
  }
}
```

### 3.3 Implementation

**Add to `engine/archetypes/logic.py`**:

```python
def calculate_fusion_score_with_trace(self, row: pd.Series) -> Tuple[float, dict]:
    """
    Calculate fusion score with full waterfall trace.

    Returns:
        (final_fusion_score, trace_dict)
    """
    trace = {"fusion_waterfall": {}}

    # 1. Domain scores (from feature store)
    trace["fusion_waterfall"]["1_domain_scores"] = {
        "wyckoff_score": self._get_wyckoff_score(row),
        "smc_score": self._get_smc_score(row),
        "hob_score": self._get_liquidity_score(row),  # HOB/liquidity
        "momentum_score": self._get_momentum_score(row)
    }

    # 2. Timeframe fusion scores (from feature store)
    trace["fusion_waterfall"]["2_timeframe_fusion"] = {
        "tf1h_fusion_score": row.get('tf1h_fusion_score', 0.0),
        "tf4h_fusion_score": row.get('tf4h_fusion_score', 0.0),
        "tf1d_fusion_score": row.get('tf1d_fusion_score', 0.0)
    }

    # 3. K2 fusion (multi-timeframe combination)
    from engine.fusion.k2_fusion import compute_k2_fusion
    k2_score = compute_k2_fusion(row, self.k2_weights)

    trace["fusion_waterfall"]["3_k2_fusion"] = {
        "weighted_mean": k2_score / 0.94,  # Reverse penalty for logging
        "disagreement_penalty": 0.94,  # Would need to extract from k2_fusion
        "k2_fusion_score": k2_score
    }

    # 4. Temporal adjustments (if enabled)
    temporal_bonus = self._calculate_temporal_bonus(row)
    after_temporal = k2_score + temporal_bonus

    trace["fusion_waterfall"]["4_temporal_adjustment"] = {
        "temporal_bonus": temporal_bonus,
        "after_temporal": after_temporal
    }

    # 5. Regime routing (Phase 3)
    # ... (regime adjustment logic) ...

    # 6. Safety rails
    # ... (threshold checks) ...

    final_score = after_temporal  # After all adjustments

    return final_score, trace
```

---

## 4. Validation & Testing

### 4.1 Fusion Module Validation

**Test 1: domain_fusion Smoke Test**
```bash
python -c "
from engine.fusion.domain_fusion import analyze_fusion
import pandas as pd

# Load sample data
df_1h = pd.read_parquet('data/BTC_1H.parquet').tail(100)
df_4h = pd.read_parquet('data/BTC_4H.parquet').tail(50)
df_1d = pd.read_parquet('data/BTC_1D.parquet').tail(20)

config = {'fusion': {'weights': {'wyckoff': 0.3, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.3}}}

signal = analyze_fusion(df_1h, df_4h, df_1d, config)

print(f'✅ Domain Fusion: score={signal.score:.3f}, direction={signal.direction}')
assert 0 <= signal.score <= 1
assert signal.direction in ['long', 'short', 'neutral']
"
```

**Test 2: k2_fusion Validation**
```bash
python -c "
from engine.fusion.k2_fusion import compute_k2_fusion, validate_k2_fusion_inputs
import pandas as pd

df = pd.read_parquet('data/BTC_1H_features.parquet')

# Validate inputs exist
validation = validate_k2_fusion_inputs(df)
assert all(validation.values()), f'Missing K2 inputs: {validation}'

# Test K2 calculation
row = df.iloc[-1]
k2_score = compute_k2_fusion(row)

print(f'✅ K2 Fusion: score={k2_score:.3f}')
assert 0 <= k2_score <= 1
"
```

**Test 3: Fusion Chain Integration**
```bash
python bin/backtest_knowledge_v2.py \
  --config configs/test/fusion_debug.json \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --debug

# Verify fusion trace written
cat artifacts/fusion_trace.jsonl | jq '.fusion_waterfall | keys'
# Expected: ["1_domain_scores", "2_timeframe_fusion", "3_k2_fusion", "4_temporal_adjustment", "5_regime_routing", "6_safety_rails"]
```

### 4.2 Double-Counting Detection

**Potential Double-Count Scenarios**:
1. **Momentum in domain_fusion AND k2_fusion**: ❌ WRONG
   - Fix: Momentum ONLY in domain_fusion (1H score), k2 just combines timeframes
2. **Wyckoff in domain_fusion AND regime_routing**: ✅ CORRECT
   - Domain fusion scores Wyckoff phase
   - Regime routing adjusts weight (no double-count)
3. **Temporal bonus applied twice**: ❌ WRONG
   - Fix: Apply temporal ONCE in canonical chain (step 4)

**Validation**:
```python
# Check fusion waterfall for suspicious jumps
trace = load_fusion_trace()
for i in range(1, 7):
    step_key = f"{i}_..."
    score_before = trace[f"{i-1}_..."]["score"]
    score_after = trace[step_key]["score"]
    delta = score_after - score_before

    # Flag suspicious jumps
    if abs(delta) > 0.20:
        print(f"⚠️ Large jump at {step_key}: {delta:+.3f}")
```

### 4.3 Acceptance Criteria

**Fusion Pipeline**:
- [ ] All modules in canonical chain execute without errors
- [ ] No NaN/inf values in fusion waterfall
- [ ] Final fusion score ∈ [0, 1] (always)
- [ ] Debug trace shows all 6 steps (when enabled)
- [ ] No suspicious jumps > 0.20 between steps

**Module Status**:
- [ ] domain_fusion: ✅ Active, validated
- [ ] k2_fusion: ✅ Active, validated
- [ ] adaptive: Decision made (keep/deprecate)
- [ ] advanced_fusion: ❌ Deprecated, moved to archive

**Documentation**:
- [ ] Canonical fusion chain documented in `docs/FUSION_PIPELINE_AUDIT.md`
- [ ] Module responsibilities clear
- [ ] Feature store dependencies documented
- [ ] Debug trace usage guide created

---

## 5. Implementation Checklist

### 5.1 Code Changes

- [ ] Investigate `adaptive.py` usage:
  ```bash
  grep -r "AdaptiveFusion" bin/*.py configs/*.json
  ```

- [ ] Deprecate `advanced_fusion.py`:
  ```bash
  mkdir -p engine/fusion/deprecated
  git mv engine/fusion/advanced_fusion.py engine/fusion/deprecated/
  ```

- [ ] Add fusion debug trace:
  - [ ] Create `calculate_fusion_score_with_trace()` in `logic.py`
  - [ ] Add config option `fusion.debug_trace`
  - [ ] Write trace to `artifacts/fusion_trace.jsonl`

- [ ] Clarify k2_fusion runtime usage:
  - [ ] Document in `k2_fusion.py` docstring
  - [ ] Add validation helper

### 5.2 Documentation

- [ ] Create `docs/FUSION_PIPELINE_SPEC.md`:
  - [ ] Canonical chain diagram
  - [ ] Module responsibilities
  - [ ] Feature store dependencies
  - [ ] Debug trace guide

- [ ] Update `docs/ARCHITECTURE.md`:
  - [ ] Add fusion pipeline section
  - [ ] Link to module docs

- [ ] Create `docs/decisions/ADAPTIVE_FUSION_DECISION.md`:
  - [ ] Investigate usage
  - [ ] Decide keep/deprecate
  - [ ] Document rationale

### 5.3 Testing

- [ ] Create `tests/unit/test_fusion_pipeline.py`:
  - [ ] Test domain_fusion smoke test
  - [ ] Test k2_fusion validation
  - [ ] Test fusion waterfall integrity
  - [ ] Test double-count detection

- [ ] Integration test:
  ```bash
  python bin/backtest_knowledge_v2.py \
    --config configs/test/fusion_debug.json \
    --start 2024-01-01 --end 2024-01-31
  ```

### 5.4 Validation

- [ ] Run all fusion module validation tests
- [ ] Check fusion trace for sample trades
- [ ] Verify no double-counting
- [ ] Acceptance criteria check

---

## 6. Cleanup Summary

### 6.1 Module Status Table

| Module | Status | Location | Notes |
|--------|--------|----------|-------|
| domain_fusion.py | ✅ KEEP | `engine/fusion/` | Core domain scoring |
| k2_fusion.py | ✅ KEEP | `engine/fusion/` | Multi-timeframe fusion |
| adaptive.py | ⚠️ TBD | `engine/fusion/` | Pending investigation |
| advanced_fusion.py | ❌ DEPRECATED | `engine/fusion/deprecated/` | Moved to archive |

### 6.2 Canonical Fusion Chain

```
domain_fusion → k2_fusion → temporal → regime_routing → safety_rails → final_score
```

### 6.3 Next Steps

1. **Investigate adaptive.py** (priority: HIGH)
   - Check backtest runner integration
   - Compare with regime_routing
   - Decide keep/deprecate

2. **Implement debug trace** (priority: MEDIUM)
   - Add waterfall logging
   - Create trace viewer script
   - Document usage

3. **Validate no double-counting** (priority: HIGH)
   - Review fusion calculation chain
   - Test for suspicious score jumps
   - Document signal flow

---

**End of Audit**
