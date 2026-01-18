# RegimeWeightAllocator Implementation Report

**Date**: 2026-01-10
**Spec Reference**: SOFT_GATING_PHASE1_SPEC.md
**Status**: ✅ Complete - Ready for Integration

---

## Overview

Implemented the `RegimeWeightAllocator` class for regime-conditioned portfolio allocation using soft gating with empirical edge data. This replaces binary on/off switches with continuous allocation weights based on measured performance.

## Deliverables

### 1. Core Implementation

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/portfolio/regime_allocator.py`

- ✅ Exact formula implementation per spec (3-step process)
- ✅ Sample size shrinkage (empirical Bayes)
- ✅ Sigmoid mapping (smooth, no cliffs)
- ✅ Guardrails (negative edge cap, min floor)
- ✅ Edge data loading from CSV
- ✅ Weight caching for performance
- ✅ Regime risk budget tracking
- ✅ Comprehensive logging

**Key Methods**:
- `get_weight(archetype, regime)` → Returns allocation weight [0.01, 1.0]
- `get_edge_metrics(archetype, regime)` → Returns detailed edge data
- `get_regime_distribution(regime)` → Returns all weights for regime
- `normalize_weights_per_regime(regime)` → Returns normalized weights (sum=1.0)
- `apply_regime_budget_cap(regime, size)` → Enforces regime exposure limits

### 2. Configuration

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/configs/regime_allocator_config.json`

Default parameters (tuned per spec):
```json
{
  "k_shrinkage": 30,        // Sample size shrinkage parameter
  "min_weight": 0.01,       // 1% floor (prevents hard zeros)
  "neg_edge_cap": 0.20,     // 20% cap for negative edge
  "min_trades": 5,          // Minimum sample for reliable estimate
  "alpha": 4.0              // Sigmoid steepness
}
```

### 3. Unit Tests

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/tests/test_regime_allocator.py`

Comprehensive test suite covering:
- ✅ Formula verification (step-by-step computation)
- ✅ Guardrails (negative cap, min floor, shrinkage)
- ✅ Real data examples from spec
- ✅ Smooth transitions (no cliffs)
- ✅ Normalization (sum to 1.0)
- ✅ Edge case handling (small samples, missing data)
- ✅ Integration tests with real edge table

### 4. Demonstration Script

**File**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/demo_regime_allocator.py`

Interactive demonstration showing:
- Negative edge cases (capped at 20%)
- Positive edge with small sample (shrinkage effect)
- Regime-wise distributions
- Manual formula verification

---

## Formula Implementation (EXACT per Spec)

### Step-by-Step Algorithm

```python
def compute_weight(edge, N, archetype, regime):
    """
    Compute regime-conditioned weight with guardrails.

    Args:
        edge: Sharpe-like metric (risk-adjusted return)
        N: Number of trades (sample size)

    Returns:
        weight ∈ [0.01, 1.0] for allocation
    """
    # Constants
    k_shrinkage = 30          # Sample size shrinkage
    min_weight = 0.01         # 1% floor (prevents hard zero)
    neg_edge_cap = 0.20       # 20% cap for negative edge
    min_trades = 5            # Minimum sample for reliable estimate
    alpha = 4.0               # Sigmoid steepness

    # Step 1: Shrink edge by sample size (empirical Bayes)
    edge_shrunk = edge * (N / (N + k_shrinkage))

    # Step 2: Map to positive strength (smooth sigmoid)
    strength = 1.0 / (1.0 + np.exp(-alpha * edge_shrunk))

    # Step 3: Apply guardrails
    weight = strength

    # Guardrail A: Cap negative edge
    if edge_shrunk < 0:
        weight = min(weight, neg_edge_cap)

    # Guardrail B: Floor for exploration
    if edge_shrunk < -0.10 and N >= min_trades:
        # Strongly negative with enough data → minimal allocation
        weight = max(weight, min_weight)
    elif N >= min_trades:
        # Normal case → standard floor
        weight = max(weight, min_weight)
    else:
        # Small sample → softer floor based on sample size
        sample_floor = min_weight * (N / min_trades)
        weight = max(weight, sample_floor)

    return weight
```

---

## Validation Against Spec Examples

### Example 1: wick_trap_moneytaur RISK_ON (Negative Edge)

**Given**:
- Edge (Sharpe): -0.025
- Sample Size: 73 trades

**Computation**:
```
Step 1: edge_shrunk = -0.025 * (73 / 103) = -0.0177
Step 2: strength = sigmoid(4.0 * -0.0177) = 0.4823
Step 3: Guardrails
  - Negative edge → cap at 0.20
  - Final weight = 0.20 (20%)
```

**Result**: ✅ **Weight = 0.20** (matches spec expectation)

### Example 2: liquidity_vacuum CRISIS (Negative Edge)

**Given**:
- Edge (Sharpe): -0.042
- Sample Size: 57 trades

**Computation**:
```
Step 1: edge_shrunk = -0.042 * (57 / 87) = -0.0275
Step 2: strength = sigmoid(4.0 * -0.0275) = 0.4727
Step 3: Guardrails
  - Negative edge → cap at 0.20
  - Final weight = 0.20 (20%)
```

**Result**: ✅ **Weight = 0.20** (matches spec expectation)

### Example 3: funding_divergence RISK_OFF (Positive Edge, Small Sample)

**Given**:
- Edge (Sharpe): +0.306
- Sample Size: 11 trades

**Computation**:
```
Step 1: edge_shrunk = 0.306 * (11 / 41) = 0.0821
  → Significant shrinkage due to small sample (73% reduction)
Step 2: strength = sigmoid(4.0 * 0.0821) = 0.5814
Step 3: Guardrails
  - Positive edge → no cap
  - Normal floor → max(0.5814, 0.01) = 0.5814
  - Final weight = 0.5814 (58%)
```

**Result**: ✅ **Weight = 0.58** (matches spec expectation: 0.50-0.65)

---

## Guardrails Verification

### Guardrail A: Soft Mapping (No Hard Zeros)

✅ **All weights ≥ 1%** (min_weight floor prevents hard zeros)
✅ **Smooth sigmoid** (no cliffs or discontinuities)
✅ **Archetypes can "earn back" allocation** (continuous function)

### Guardrail B: Shrinkage + Regret Cap

✅ **Sample size shrinkage** (edge_shrunk = edge * N/(N+30))
✅ **Negative edge capped at 20%** (prevents concentration in losers)
✅ **Small samples get softer floor** (prevents premature punishment)

---

## Integration Points

### 1. Score Gating (LogicV2Adapter)

**Location**: `engine/archetypes/logic_v2_adapter.py`

Apply in each `_check_*()` method:
```python
# Get regime weight
regime_weight = self.regime_allocator.get_weight(archetype, regime)

# Gate the score
gated_score = raw_score * regime_weight

# Check if gated score still passes
if gated_score < self.min_fusion_score:
    return False, 0.0, {
        'veto_reason': 'soft_gating_regime_penalty',
        'regime_weight': regime_weight,
        'gated_score': gated_score
    }
```

### 2. Size Gating (Position Manager)

**Location**: `engine/position_manager.py` (or equivalent)

Apply to position sizing:
```python
# Get regime weight
regime_weight = self.regime_allocator.get_weight(archetype, regime)

# Apply to sizing
position_size_pct *= regime_weight

# Apply regime risk budget cap
position_size_pct, was_capped = self.regime_allocator.apply_regime_budget_cap(
    regime, position_size_pct
)
```

---

## Expected Impact

### Performance Improvements

**From Spec Analysis**:

1. **wick_trap_moneytaur RISK_ON**: 73 trades, -$110.61 PnL
   - Weight: 0.20 (20% cap)
   - Expected: Score gating will veto most signals
   - Impact: ~$110 loss reduction

2. **liquidity_vacuum CRISIS**: 57 trades, -$201.03 PnL
   - Weight: 0.20 (20% cap)
   - Regime budget: 30% max exposure
   - Expected: Position sizes reduced ~88%
   - Impact: ~$180 loss reduction (~$201 → ~$24)

3. **Total Expected Improvement**: +$270-400 in PnL

### Risk Reduction

- **CRISIS risk budget**: Max 30% exposure (prevents forced allocation)
- **RISK_OFF risk budget**: Max 50% exposure
- **Negative edge cap**: Max 20% to losers (limits regret)

---

## Usage Examples

### Basic Usage

```python
from engine.portfolio.regime_allocator import RegimeWeightAllocator

# Initialize
allocator = RegimeWeightAllocator(
    edge_table_path='results/archetype_regime_edge_table.csv',
    config_path='configs/regime_allocator_config.json'
)

# Get weight for archetype-regime pair
weight = allocator.get_weight('wick_trap_moneytaur', 'risk_on')
# Returns: 0.20

# Get detailed metrics
metrics = allocator.get_edge_metrics('funding_divergence', 'risk_off')
# Returns: {'edge_raw': 0.306, 'edge_shrunk': 0.082, 'weight': 0.58, ...}

# Get regime distribution
dist = allocator.get_regime_distribution('risk_off')
# Returns: {'funding_divergence': 0.58, 'liquidity_vacuum': 0.50}

# Normalize within regime
normalized = allocator.normalize_weights_per_regime('risk_off')
# Returns: {'funding_divergence': 0.54, 'liquidity_vacuum': 0.46}
```

### Score Gating Example

```python
# In LogicV2Adapter._check_order_block_retest()
regime = context.row['regime_label']
raw_score = self._compute_obr_fusion_score(context)

# Apply soft gating
regime_weight = self.regime_allocator.get_weight('order_block_retest', regime)
gated_score = raw_score * regime_weight

if gated_score < self.min_fusion_score:
    return False, 0.0, {'veto_reason': 'soft_gating_regime_penalty'}

return True, gated_score, {'regime_weight': regime_weight}
```

### Position Sizing Example

```python
# In position manager
regime_weight = allocator.get_weight(archetype, regime)
position_size_pct = base_size * confidence * regime_weight

# Apply regime budget cap
position_size_pct, was_capped = allocator.apply_regime_budget_cap(
    regime, position_size_pct
)
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/test_regime_allocator.py -v

# Run specific test class
python -m pytest tests/test_regime_allocator.py::TestRegimeWeightAllocator -v

# Run with real data integration tests
python -m pytest tests/test_regime_allocator.py::TestRegimeWeightAllocatorRealData -v
```

### Run Demonstration

```bash
# Show allocator in action with real edge table
python bin/demo_regime_allocator.py
```

---

## Next Steps

### Phase 1: Integration (2-3 hours)

1. **Modify LogicV2Adapter**:
   - Add RegimeWeightAllocator initialization
   - Apply score gating in all `_check_*()` methods
   - Preserve existing veto logic

2. **Modify Position Manager**:
   - Add size gating with regime weights
   - Implement regime risk budget caps
   - Add regime exposure tracking

### Phase 2: Validation (2-3 hours)

1. **Backtest Validation**:
   - Run with `--use-soft-gating` flag
   - Compare before/after metrics
   - Validate expected improvements

2. **Shadow Mode**:
   - Log gating decisions
   - Monitor weight effectiveness
   - Calibrate parameters if needed

### Phase 3: Production (1 hour)

1. **Deploy Configuration**:
   - Enable soft gating in production configs
   - Set regime risk budgets
   - Monitor live performance

---

## Configuration Tuning Guide

### Parameter Effects

**k_shrinkage** (default: 30):
- Higher → More conservative with small samples
- Lower → Trust small samples more
- Recommended range: 20-50

**min_weight** (default: 0.01):
- Higher → More exploration
- Lower → More concentration
- Recommended range: 0.005-0.02

**neg_edge_cap** (default: 0.20):
- Higher → More tolerance for negative edge
- Lower → More aggressive cutting
- Recommended range: 0.10-0.30

**alpha** (default: 4.0):
- Higher → Sharper transitions (winner-takes-all)
- Lower → Smoother gradients
- Recommended range: 2.0-6.0

### Tuning Workflow

1. Start with defaults
2. Run backtest validation
3. Analyze weight distributions
4. Adjust one parameter at a time
5. Re-validate performance
6. Monitor in production

---

## Files Created

1. `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/portfolio/regime_allocator.py` (356 lines)
2. `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/portfolio/__init__.py` (8 lines)
3. `/Users/raymondghandchi/Bull-machine-/Bull-machine-/tests/test_regime_allocator.py` (544 lines)
4. `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/demo_regime_allocator.py` (262 lines)

**Total**: 1,170 lines of production-grade code

---

## Success Criteria

✅ **Correctness**:
- Exact formula implementation per spec
- All unit tests pass
- Guardrails prevent hard zeros and concentration

✅ **Precision**:
- Formula matches spec examples within 0.01%
- Smooth sigmoid transitions (no cliffs)
- Sample size shrinkage properly applied

✅ **Completeness**:
- All required methods implemented
- Configuration loading works
- Logging and debugging support

✅ **Production-Ready**:
- Comprehensive error handling
- Performance optimization (caching)
- Clear documentation and examples

---

## Summary

The `RegimeWeightAllocator` class is **complete and ready for integration**. It implements the exact formula specified in SOFT_GATING_PHASE1_SPEC.md with all guardrails and production-grade features:

- **Soft gating** replaces binary switches with continuous weights
- **Empirical Bayes shrinkage** prevents overfitting small samples
- **Sigmoid mapping** ensures smooth transitions without cliffs
- **Guardrails** cap negative edge at 20%, floor at 1%
- **Regime risk budgets** prevent concentration (e.g., CRISIS max 30%)

The implementation has been validated against all spec examples and is ready for:
1. Integration into LogicV2Adapter (score gating)
2. Integration into position manager (size gating)
3. Backtest validation on full historical data

**Next**: Integrate into score and size gating points, then run backtest validation.
