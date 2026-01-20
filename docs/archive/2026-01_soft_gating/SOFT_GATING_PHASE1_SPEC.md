# Phase 1: Soft Gating Implementation Spec

**Author**: Senior Quant Review + Implementation
**Date**: 2026-01-09
**Goal**: Regime-conditioned portfolio allocation (mixture of experts)

---

## Core Principle

**Replace**: Binary on/off switches
**With**: Continuous allocation weights based on empirical edge

**Apply gating at TWO levels**:
1. **Score level** (prevents signal generation)
2. **Size level** (prevents cascade effects from direction balance)

---

## Guardrails (Production-Grade)

### Guardrail A: Soft Mapping (No Hard Zeros)

**DON'T**:
```python
if sharpe < -0.05:
    weight = 0.0  # Hard cliff, back to "disable"
```

**DO**:
```python
# Soft nonlinearity prevents cliffs
if sharpe < 0:
    weight = min(weight, neg_edge_cap)  # Cap at 20%
    weight = max(weight, min_weight)     # Floor at 1%

# Allow archetypes to "earn back" allocation
```

### Guardrail B: Shrinkage + Regret Cap

```python
# Sample size shrinkage
edge_shrunk = edge * (N / (N + k))  # k=30

# Negative edge cap
if edge_shrunk < 0:
    weight = min(weight, 0.20)  # Max 20% to losers

# Minimum floor (prevents hard zero)
weight = max(weight, 0.01)  # 1% min for N >= min_trades
```

---

## RegimeWeightAllocator Formula (Exact Implementation)

### Step-by-Step Algorithm

```python
def compute_weight(edge, N, regime, archetype):
    """
    Compute regime-conditioned weight with guardrails.

    Args:
        edge: Sharpe-like metric (risk-adjusted return)
        N: Number of trades (sample size)
        regime: Current market regime
        archetype: Archetype name

    Returns:
        weight ∈ [0.01, 1.0] for allocation
    """
    # Constants (tune these)
    k_shrinkage = 30          # Sample size shrinkage
    min_weight = 0.01         # 1% floor (prevents hard zero)
    neg_edge_cap = 0.20       # 20% cap for negative edge
    min_trades = 5            # Minimum sample for reliable estimate
    offset = 0.05             # Soft threshold offset
    alpha = 4.0               # Sigmoid steepness

    # Step 1: Shrink edge by sample size (empirical Bayes)
    edge_shrunk = edge * (N / (N + k_shrinkage))

    # Step 2: Map to positive strength (smooth, no cliffs)
    # Option A: Linear with offset
    # strength = max(0, edge_shrunk + offset)

    # Option B: Sigmoid (smoother, recommended)
    strength = 1.0 / (1.0 + np.exp(-alpha * edge_shrunk))

    # Step 3: Apply guardrails
    weight = strength

    # Guardrail A: Cap negative edge
    if edge_shrunk < 0:
        weight = min(weight, neg_edge_cap)

    # Guardrail B: Floor for exploration (unless very negative)
    if edge_shrunk < -0.10 and N >= min_trades:
        # Strongly negative with enough data → minimal allocation
        weight = max(weight, min_weight)
    elif N >= min_trades:
        # Normal case → standard floor
        weight = max(weight, min_weight)
    else:
        # Small sample → don't punish too much
        # Use softer floor based on sample size
        sample_floor = min_weight * (N / min_trades)
        weight = max(weight, sample_floor)

    return weight
```

### Within-Regime Normalization

```python
def normalize_weights_per_regime(weights_dict, regime):
    """
    Normalize weights within a regime to sum to 1.0.

    This ensures we're doing true portfolio allocation.
    """
    regime_weights = {k: v for k, v in weights_dict.items() if k[1] == regime}

    total = sum(regime_weights.values())

    if total > 0:
        normalized = {k: v/total for k, v in regime_weights.items()}
    else:
        # No archetypes → cash bucket gets 100%
        normalized = {}

    return normalized
```

---

## CRISIS Risk Budget / Cash Bucket

### Problem

CRISIS has only 1 archetype (liquidity_vacuum) with negative edge.
- Current: Gets 100% allocation (forced concentration)
- Solution: Add "cash bucket" as safety valve

### Implementation

```python
# In CRISIS regime:
total_archetype_weight = sum(weights for all archetypes in CRISIS)

# If negative-edge cap binds
if total_archetype_weight < crisis_target_allocation:
    cash_bucket_weight = crisis_target_allocation - total_archetype_weight
    # Remaining weight goes to "no trade" / reduced exposure

# Example:
liquidity_vacuum: edge -0.042 → weight 0.20 (capped)
cash_bucket: weight 0.80 (safety)

# Result: CRISIS only deploys 20% of normal capital
```

### Regime-Specific Risk Budgets

```python
REGIME_RISK_BUDGETS = {
    'crisis': 0.30,    # Max 30% total exposure in CRISIS
    'risk_off': 0.50,  # Max 50% in RISK_OFF
    'neutral': 0.70,   # Max 70% in NEUTRAL
    'risk_on': 0.80    # Max 80% in RISK_ON
}

# Apply at sizing:
if regime == 'crisis':
    total_size_limit = portfolio_value * 0.30
    # All CRISIS positions combined can't exceed 30%
```

---

## Integration Points

### 1. Score Gating (LogicV2Adapter)

**File**: `engine/archetypes/logic_v2_adapter.py`

**Location**: Inside each `_check_*()` method

```python
def _check_order_block_retest(self, context: ArchetypeContext):
    regime_label = self.g(context.row, "regime_label", "neutral")

    # Step 1: Compute raw score (existing logic)
    raw_score = self._compute_obr_fusion_score(context)

    if raw_score < self.min_fusion_score:
        return False, 0.0, {'veto_reason': 'low_fusion'}

    # Step 2: Get regime weight (NEW!)
    regime_weight = self.regime_allocator.get_weight(
        archetype='order_block_retest',
        regime=regime_label
    )

    # Step 3: Gate the score
    gated_score = raw_score * regime_weight

    # Step 4: Check if gated score still passes
    if gated_score < self.min_fusion_score:
        return False, 0.0, {
            'veto_reason': 'soft_gating_regime_penalty',
            'regime': regime_label,
            'raw_score': raw_score,
            'regime_weight': regime_weight,
            'gated_score': gated_score,
            'edge_data': self.regime_allocator.get_edge_metrics(
                'order_block_retest', regime_label
            )
        }

    # Step 5: Pass gated score downstream
    return True, gated_score, {
        'raw_score': raw_score,
        'regime_weight': regime_weight,
        'gated_score': gated_score
    }
```

**Apply to ALL archetype check methods**:
- `_check_order_block_retest()`
- `_check_wick_trap_moneytaur()`
- `_check_liquidity_vacuum()`
- `_check_funding_divergence()`
- etc.

---

### 2. Size Gating (Position Manager)

**File**: `engine/position_manager.py` or wherever position sizing happens

```python
def calculate_position_size(
    self,
    archetype: str,
    regime: str,
    confidence: float,
    base_size_pct: float = 0.20
):
    """
    Calculate position size with regime-conditioned weight applied.

    CRITICAL: Apply weight to sizing to prevent cascade effects.
    """
    # Get regime weight
    regime_weight = self.regime_allocator.get_weight(archetype, regime)

    # Base sizing
    position_size_pct = base_size_pct

    # Apply regime weight (NEW!)
    position_size_pct *= regime_weight

    # Apply confidence scaling (existing)
    position_size_pct *= confidence

    # Apply direction balance scaling (existing)
    direction_scaler = self.direction_balance.get_scaler(...)
    position_size_pct *= direction_scaler

    # Apply regime risk budget cap (NEW!)
    regime_budget = REGIME_RISK_BUDGETS.get(regime, 0.80)
    current_regime_exposure = self.get_regime_exposure(regime)

    if current_regime_exposure + position_size_pct > regime_budget:
        # Cap to not exceed regime budget
        position_size_pct = max(0, regime_budget - current_regime_exposure)

    # Convert to dollar amount
    position_size = self.portfolio_value * position_size_pct

    return position_size
```

---

## Threshold Corrections (From User Feedback)

### Old (Incorrect)

```python
# This was too strict and creates hard cliffs
if sharpe < -0.05:
    weight = 0.0  # Hard zero

# Problem: wick_trap RISK_ON has Sharpe -0.025
# Would still get allocation under this rule!
```

### New (Correct)

```python
# Sharpe < 0 → negative edge
if sharpe < 0:
    weight = min(weight, 0.20)  # Cap at 20%
    weight = max(weight, 0.01)  # Floor at 1%

# Sharpe < -0.10 with N >= 10 → strongly negative
if sharpe < -0.10 and N >= 10:
    weight = max(weight, 0.01)  # Minimal floor only

# This ensures:
# - wick_trap RISK_ON (Sharpe -0.025) → capped at 20%, floor 1%
# - liquidity_vacuum CRISIS (Sharpe -0.042) → capped at 20%, floor 1%
# - Both can "earn back" allocation if improved
```

---

## Expected Behavior Examples

### Example 1: wick_trap_moneytaur RISK_ON

```python
# Current data:
edge = -0.025 (Sharpe)
N = 73 trades

# Step 1: Shrink
edge_shrunk = -0.025 * (73 / (73 + 30)) = -0.018

# Step 2: Sigmoid
strength = sigmoid(4.0 * -0.018) = sigmoid(-0.072) ≈ 0.48

# Step 3: Apply guardrails
# Negative edge → cap
weight = min(0.48, 0.20) = 0.20

# Floor
weight = max(0.20, 0.01) = 0.20

# Final weight: 0.20 (20%)

# BUT: If there are other archetypes in RISK_ON with positive edge:
# Renormalization will push this down further
# Example: wick_trap (0.20) + order_block (0.60) + other (0.20)
# After norm: wick_trap gets 0.20 / 1.0 = 20% OF RISK_ON allocation

# At scoring:
raw_score = 0.35
gated_score = 0.35 * 0.20 = 0.07
# If min_fusion_score = 0.30 → vetoed

# At sizing (if it somehow passed):
position_size = $10000 * 0.20 * 0.20 * 0.35 = $140
# Tiny position (was $700 before)
```

### Example 2: liquidity_vacuum CRISIS

```python
# Current data:
edge = -0.042 (Sharpe)
N = 57 trades

# Step 1: Shrink
edge_shrunk = -0.042 * (57 / (57 + 30)) = -0.028

# Step 2: Sigmoid
strength = sigmoid(4.0 * -0.028) = sigmoid(-0.112) ≈ 0.47

# Step 3: Apply guardrails
weight = min(0.47, 0.20) = 0.20
weight = max(0.20, 0.01) = 0.20

# Only archetype in CRISIS → after norm:
weight = 0.20 / 0.20 = 1.0 (within CRISIS archetypes)

# BUT: CRISIS risk budget caps total exposure
crisis_budget = 0.30 (30% max)
liquidity_vacuum weight in CRISIS = 1.0
effective_allocation = 1.0 * 0.30 = 30% of portfolio

# At sizing:
base_size = 0.20 (20%)
regime_weight = 1.0 (only archetype)
crisis_budget_scaler = 0.30 / 1.0 = 0.30 (cap to 30% total)

position_size = $10000 * 0.20 * 1.0 * 0.30 * confidence
             = $10000 * 0.06 * confidence
             ≈ $600 * 0.40 = $240

# Was $2000 before → now $240 (88% reduction)
# -$201 → -$24 (88% reduction)
```

### Example 3: funding_divergence RISK_OFF (Healthy)

```python
# Current data:
edge = +0.306 (Sharpe) ✓
N = 11 trades

# Step 1: Shrink
edge_shrunk = 0.306 * (11 / (11 + 30)) = 0.082

# Step 2: Sigmoid
strength = sigmoid(4.0 * 0.082) = sigmoid(0.328) ≈ 0.58

# Step 3: Apply guardrails
# Positive edge → no cap
weight = 0.58
weight = max(0.58, 0.01) = 0.58

# Competing with liquidity_vacuum in RISK_OFF:
# funding_divergence: 0.58
# liquidity_vacuum: 0.50 (Sharpe +0.080 → lower strength)

# After norm:
# funding_divergence: 0.58 / (0.58 + 0.50) = 54%
# liquidity_vacuum: 0.50 / (0.58 + 0.50) = 46%

# System allocates more to better performer ✓
```

---

## Logging Requirements

For every archetype signal evaluation, log:

```python
{
    'timestamp': ...,
    'archetype': 'order_block_retest',
    'regime': 'risk_on',
    'raw_score': 0.35,
    'regime_weight': 0.20,
    'gated_score': 0.07,
    'edge_shrunk': -0.018,
    'sample_size': 73,
    'passed': False,
    'veto_reason': 'soft_gating_regime_penalty'
}
```

This enables:
1. Shadow mode validation
2. Weight calibration
3. Edge table updates
4. Performance debugging

---

## Testing Checklist

### Unit Tests

- [ ] RegimeWeightAllocator.get_weight() with various edge values
- [ ] Soft mapping (sigmoid) produces smooth transitions
- [ ] Guardrails: negative edge capped at 20%, floored at 1%
- [ ] Small sample shrinkage reduces confidence
- [ ] Within-regime normalization sums to 1.0

### Integration Tests

- [ ] Score gating reduces scores as expected
- [ ] Position sizing applies weight correctly
- [ ] CRISIS risk budget caps total exposure
- [ ] No cascade effects (direction balance stable)

### Backtest Validation

- [ ] wick_trap RISK_ON: 73 trades → ~0 trades (or small positions)
- [ ] liquidity_vacuum CRISIS: position sizes reduced ~88%
- [ ] Total PnL: +$139 → +$400+
- [ ] No unintended side effects in other regimes

---

## Implementation Order

1. **Create RegimeWeightAllocator class** (1-2 hours)
   - Implement formula with guardrails
   - Load edge table data
   - Add logging

2. **Integrate score gating in LogicV2Adapter** (2-3 hours)
   - Modify all `_check_*()` methods
   - Add weight application
   - Preserve existing veto logic

3. **Integrate size gating in position manager** (1-2 hours)
   - Apply regime weight to sizing
   - Add CRISIS risk budget cap
   - Add regime exposure tracking

4. **Backtest validation** (2-3 hours)
   - Compare before/after
   - Validate expected improvements
   - Check for unintended effects

**Total**: 6-10 hours (1-2 days)

---

## Files to Create/Modify

### Create

- `engine/portfolio/regime_allocator.py` (NEW)
- `configs/regime_allocator_config.json` (NEW)
- `tests/test_regime_allocator.py` (NEW)

### Modify

- `engine/archetypes/logic_v2_adapter.py` (score gating)
- `engine/position_manager.py` (size gating + risk budget)
- `bin/backtest_full_engine_replay.py` (add --use-soft-gating flag)

---

## Success Criteria

✅ **Correctness**:
- Weights computed using exact formula specified
- Guardrails prevent hard zeros and concentration
- CRISIS risk budget caps exposure

✅ **Performance**:
- wick_trap RISK_ON: effectively zeroed out
- liquidity_vacuum CRISIS: 80-90% position reduction
- Total PnL: +$270-400 improvement

✅ **Robustness**:
- No cascade effects
- Archetypes can "earn back" allocation
- Smooth degradation (no cliffs)

✅ **Observability**:
- All gating decisions logged
- Edge metrics tracked
- Weight effectiveness measurable

---

**Ready for Implementation** ✓
