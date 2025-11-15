# PR#6B: Proper Regime-Aware Architecture

**Status**: Phase 1 infrastructure complete, Phase 2 integration in progress

## Overview

Proper policy-driven architecture for regime-aware trading. All thresholds centralized in ThresholdPolicy, no hardcoded numbers in archetype logic.

## Completed Infrastructure ✅

### 1. RuntimeContext (`engine/runtime/context.py`)
- Immutable context object carrying all decision state
- Clean API: `get_threshold()`, `get_adapted_param()`
- Single source of truth passed through pipeline

```python
runtime_ctx = RuntimeContext(
    ts=row.name,
    row=row,
    regime_probs={'risk_on': 0.65, 'neutral': 0.30, ...},
    regime_label='risk_on',
    adapted_params=adaptive_fusion_output,
    thresholds=threshold_policy.resolve(...)
)
```

### 2. ThresholdPolicy (`engine/archetypes/threshold_policy.py`)
- Centralized threshold management with 5-step resolution:
  1. Base static thresholds from config
  2. Regime probability blending (convex combination)
  3. Regime floor enforcement (from gates_regime_profiles)
  4. Per-archetype overrides (optional deltas by regime)
  5. Global guardrail clamping (min/max sanity bounds)

```python
policy = ThresholdPolicy(
    base_cfg=config,
    regime_profiles=config['gates_regime_profiles'],
    archetype_overrides=config['archetype_overrides'],
    global_clamps=config['global_clamps']
)

thresholds = policy.resolve(regime_probs, regime_label)
# Returns: {'order_block_retest': {'fusion': 0.33, 'liquidity': 0.14}, ...}
```

### 3. Config Updates (`configs/btc_v8_adaptive.json`)
- Tuned regime profiles (0.35-0.52 for fusion floors, up from 0.28-0.50)
- Added `archetype_overrides` for fine-grained regime adjustments
- Added `global_clamps` for sanity bounds

## Remaining Work 🔄

### Phase 2A: Archetype Logic Refactor (2-3 hours)

Refactor all 11 archetype `_check_*` methods to use RuntimeContext.

**Example Pattern** (apply to all 11 methods):

```python
# BEFORE (hardcoded)
def _check_B(self, row: pd.Series, fusion: float) -> bool:
    """Order Block Retest"""
    th = self.thresh_B
    if fusion < 0.374:  # ← HARDCODED!
        return False
    if row.get('liquidity_score', 0) < 0.14:  # ← HARDCODED!
        return False
    # ... structural checks ...
    return True

# AFTER (policy-driven)
def _check_B(self, ctx: RuntimeContext) -> bool:
    """Order Block Retest"""
    # Get thresholds from policy (regime-aware, no hardcoding)
    fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)
    liq_th = ctx.get_threshold('order_block_retest', 'liquidity', 0.14)

    if ctx.row.get('fusion_score', 0) < fusion_th:
        return False
    if ctx.row.get('liquidity_score', 0) < liq_th:
        return False
    # ... structural checks (unchanged) ...
    return True
```

**Files to Update:**
- `engine/archetypes/logic_v2_adapter.py`
  - Update all 11 `_check_*` methods (A-H, K-M)
  - Update `detect()` method signature to accept RuntimeContext
  - Remove all hardcoded threshold constants

**Checklist:**
- [ ] _check_A (spring)
- [ ] _check_B (order_block_retest)
- [ ] _check_C (wick_trap)
- [ ] _check_D (failed_continuation)
- [ ] _check_E (volume_exhaustion)
- [ ] _check_F (exhaustion_reversal)
- [ ] _check_G (liquidity_sweep)
- [ ] _check_H (momentum_continuation)
- [ ] _check_K (trap_within_trend)
- [ ] _check_L (retest_cluster)
- [ ] _check_M (confluence_breakout)
- [ ] detect() method
- [ ] update_adapted_gates() → remove (obsolete with RuntimeContext)

### Phase 2B: Backtest Engine Integration (1 hour)

Wire RuntimeContext into backtest pipeline.

**Location:** `bin/backtest_knowledge_v2.py`

**Changes:**

1. **Initialize ThresholdPolicy** (in `__init__`):
```python
from engine.archetypes.threshold_policy import ThresholdPolicy

self.threshold_policy = ThresholdPolicy(
    base_cfg=self.cfg,
    regime_profiles=self.cfg.get('gates_regime_profiles'),
    archetype_overrides=self.cfg.get('archetype_overrides'),
    global_clamps=self.cfg.get('global_clamps')
)
```

2. **Build RuntimeContext per bar** (in main backtest loop):
```python
from engine.runtime.context import RuntimeContext

# Get regime state
regime_info = self.regime_classifier.classify(macro_row) if self.regime_classifier else None
if regime_info:
    curr_probs = regime_info.get('proba', {})
    self.regime_probs_ema = ema_smooth(self.regime_probs_ema, curr_probs, self.ema_alpha)
    regime_label = max(self.regime_probs_ema, key=self.regime_probs_ema.get)
else:
    regime_label = 'neutral'
    self.regime_probs_ema = {'neutral': 1.0}

# Get adapted parameters
adapted_params = self.adaptive_fusion.update(regime_info) if self.adaptive_fusion else {}

# Resolve thresholds
thresholds = self.threshold_policy.resolve(self.regime_probs_ema, regime_label)

# Build context
runtime_ctx = RuntimeContext(
    ts=row.name,
    row=row,
    regime_probs=self.regime_probs_ema,
    regime_label=regime_label,
    adapted_params=adapted_params,
    thresholds=thresholds
)

# Pass to archetype detection
archetype = self.archetype_logic.detect(runtime_ctx)
```

3. **Remove obsolete code:**
- Delete `update_adapted_gates()` calls
- Delete `context['adapted_params']` wiring
- Delete hardcoded `final_fusion_gate = 0.374` checks

### Phase 2C: Unit Tests (1 hour)

Add comprehensive tests for policy behavior.

**Files to Create:**

1. `tests/test_threshold_policy.py`:
```python
def test_regime_blending():
    """Test convex combination of regime profiles"""

def test_regime_floors():
    """Test floor enforcement across regimes"""

def test_archetype_overrides():
    """Test per-archetype delta adjustments"""

def test_global_clamps():
    """Test sanity bound enforcement"""

def test_determinism():
    """Test reproducibility with same inputs"""
```

2. `tests/test_runtime_context.py`:
```python
def test_immutability():
    """Test context cannot be modified after creation"""

def test_safe_accessors():
    """Test get_threshold and get_adapted_param"""
```

3. `tests/test_archetype_regime_response.py`:
```python
def test_risk_on_expansion():
    """Test archetypes expand gates in risk_on"""

def test_crisis_contraction():
    """Test archetypes tighten gates in crisis"""

def test_no_flicker():
    """Test hysteresis prevents regime flicker"""
```

## Acceptance Criteria

Before marking complete:

- [ ] **No regression**: Static mode (adaptive disabled) matches v8 baseline (±1%)
- [ ] **Bull sanity**: Risk_on (forced) produces 20-40 trades on BTC 2024, PF ≥ baseline ±10%
- [ ] **Bear sanity**: Risk_off produces fewer entries, faster exits, DD ≤ 8% on 2022-2023
- [ ] **Hysteresis**: Synthetic regime flips don't cause entry flicker (<1 flip per N bars)
- [ ] **Policy determinism**: ThresholdPolicy.resolve() is pure & reproducible
- [ ] **Trade count**: BTC 2024 produces 20-35 trades (not 7, not 750)

## Current State Summary

### ✅ Complete
- RuntimeContext infrastructure
- ThresholdPolicy implementation
- Config updates with tuned profiles
- RegimeClassifier probability bug fix

### 🔄 In Progress
- Archetype logic refactor (0/11 methods)
- Backtest engine integration (wiring pending)

### ⏳ Pending
- Unit tests
- Acceptance testing
- Documentation

## Testing Plan

### Quick Validation (15 min)
```bash
# Test static mode (no regression)
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_static.json

# Test adaptive mode (should produce 20-35 trades)
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json
```

### Full Validation (1 hour)
```bash
# Run unit tests
pytest tests/test_threshold_policy.py -v
pytest tests/test_archetype_regime_response.py -v

# Run backtests
python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31 \
  --config configs/btc_v8_adaptive.json

python3 bin/backtest_knowledge_v2.py --asset BTC --start 2022-01-01 --end 2023-12-31 \
  --config configs/btc_v8_adaptive.json
```

## Benefits of This Architecture

1. **Separation of Concerns**: Archetypes = structure only, policy = numbers
2. **Composability**: Regime model evolution doesn't touch archetype logic
3. **Auditability**: Thresholds explainable per bar (log `ctx.thresholds`)
4. **Adaptivity**: Smooth morphing across regimes with guardrails
5. **Predictability**: ML veto uses same regime view as entries
6. **Testability**: Pure functions, deterministic, reproducible
7. **Maintainability**: No scattered magic numbers, single policy layer

## Next Session Action Items

1. **Start with one archetype** as a template (recommend `_check_B`)
2. **Apply pattern to remaining 10** methods
3. **Wire backtest engine** to build RuntimeContext
4. **Run quick validation** on BTC 2024 (expect 20-35 trades)
5. **Add unit tests** for policy behavior
6. **Run full acceptance tests** on 2022-2024 data

Estimated completion time: **3-4 hours** of focused work
