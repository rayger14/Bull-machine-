# Soft Gating Analysis - Archetype × Regime Edge Table

**Date**: 2026-01-09
**Approach**: Quant-grade mixture-of-experts with regime-conditioned weights

---

## The Problem with Binary Switches

**Old approach**: `if regime == 'risk_on': return None`

**Issue**: Loses flexibility, throws away potential signal, not how real quant systems work

**New approach**: Soft gating with regime-conditioned weights

---

## Empirical Edge Table (Current State)

### CRISIS Regime

```
Archetype            N    Expectancy   Profit Factor   Win Rate   Sharpe-like   Stop-out %
liquidity_vacuum    57      -$3.53         0.910        35.1%       -0.042         64.9%
```

**Analysis**:
- **Only 1 archetype active in CRISIS** → gets 100% weight by default
- Negative edge (-0.042 Sharpe, <1.0 PF)
- This is the -$201 regression issue
- **Problem**: No other archetypes to compete with for capital

**Implication**: Need to either:
1. Improve liquidity_vacuum for CRISIS
2. Enable other archetypes in CRISIS with better parameters
3. Cap CRISIS allocation regardless of weights

---

### NEUTRAL Regime

```
Archetype               N    Expectancy   Profit Factor   Win Rate   Sharpe-like   Weight
order_block_retest    120      +$0.58         1.024        38.3%       +0.009       26.1%
funding_divergence     10      +$0.50         1.031        30.0%       +0.012       25.8%
wick_trap_moneytaur   138      +$0.11         1.005        40.6%       +0.002       25.8%
trap_within_trend       7     -$11.37         0.364        42.9%       -0.364       22.4%
```

**Analysis**:
- **4 archetypes competing** → weights distributed 22-26%
- All positive Sharpe except trap_within_trend
- trap_within_trend has terrible expectancy but still gets 22% weight!
- **Issue**: Small sample (7 trades) means high uncertainty

**Implication**: Need to:
1. Penalize negative Sharpe more aggressively
2. Apply stricter shrinkage for small samples
3. Set absolute minimum edge threshold (e.g., Sharpe > -0.1)

---

### RISK_OFF Regime

```
Archetype               N    Expectancy   Profit Factor   Win Rate   Sharpe-like   Weight
funding_divergence     11     +$12.96         2.362        36.4%       +0.306       51.3%
liquidity_vacuum       68      +$5.38         1.227        36.8%       +0.080       48.7%
```

**Analysis**:
- **2 archetypes competing** → roughly equal weights
- Both positive edge
- funding_divergence has MUCH better Sharpe (0.306 vs 0.080)
- **Good**: System working as intended, allocating more to better performer

---

### RISK_ON Regime

```
Archetype               N    Expectancy   Profit Factor   Win Rate   Sharpe-like   Weight
trap_within_trend       1     -$66.95         0.000         0.0%        0.000       50.9%
wick_trap_moneytaur    73      -$1.52         0.945        35.6%       -0.025       49.1%
```

**Analysis**:
- **2 archetypes, BOTH NEGATIVE EDGE**
- trap_within_trend: 1 catastrophic trade gets 51% weight
- wick_trap_moneytaur: -$110 total, gets 49% weight
- **CRITICAL PROBLEM**: Softmax can't distinguish "bad" from "terrible"

**Implication**: Need to:
1. Set absolute minimum edge threshold
2. Zero out weights for Sharpe < -0.05 (losing money)
3. Enable order_block_retest or other archetypes in RISK_ON

---

## Key Insights

### 1. Softmax Within Regime Has Blind Spots

When all archetypes in a regime have negative edge, softmax still allocates capital!

**Example**: RISK_ON gets split 51%/49% between two losing strategies.

**Fix**: Add absolute edge threshold:
```python
if sharpe_like < -0.05:
    weight = 0.0  # Don't allocate to negative-edge strategies
```

### 2. Small Sample Bias

trap_within_trend (7 trades in NEUTRAL, 1 in RISK_ON) gets significant weight despite terrible metrics.

**Current shrinkage**: `(N / (N + 30)) * edge`
- 7 trades: shrunk to 19% of raw edge
- But raw edge is -0.364, so still gets allocated

**Fix**: More aggressive shrinkage OR minimum trade count threshold (e.g., N >= 10)

### 3. Regime Coverage Gaps

Some regimes have only 1-2 archetypes active:
- CRISIS: 1 archetype (liquidity_vacuum)
- RISK_OFF: 2 archetypes
- RISK_ON: 2 archetypes (both negative)

**This defeats the purpose of soft gating.**

**Fix**: Enable more archetypes per regime with regime-specific parameters

---

## Improved Weighting Logic

### Current Formula (Naive Softmax)

```python
edge_shrunk = (N / (N + k)) * sharpe_like
weight = softmax(beta * edge_shrunk)
weight = max(weight, min_weight)
```

**Problems**:
- Allocates to negative-edge strategies
- Small sample noise creates false confidence
- No absolute quality bar

### Improved Formula (Quality-Gated Softmax)

```python
# Step 1: Absolute quality filter
if sharpe_like < threshold_edge:  # e.g., -0.05
    weight = 0.0
    return

if n_trades < min_trades:  # e.g., 10
    # Extra shrinkage for small samples
    edge_shrunk = (N / (N + 2*k)) * sharpe_like
else:
    edge_shrunk = (N / (N + k)) * sharpe_like

# Step 2: Penalty for negative edge
if edge_shrunk < 0:
    edge_shrunk = edge_shrunk * penalty_multiplier  # e.g., 0.1

# Step 3: Softmax within regime
weight = softmax(beta * edge_shrunk)

# Step 4: Renormalize (some weights may be zeroed)
weight = weight / sum(all_weights_in_regime)

# Step 5: Apply floor only if edge is positive
if sharpe_like > 0:
    weight = max(weight, min_weight)
```

### Parameters to Tune

```python
threshold_edge = -0.05  # Don't allocate below this Sharpe
min_trades = 10         # Minimum sample size for reliable estimate
k_shrinkage = 30        # Base shrinkage constant
penalty_multiplier = 0.1  # Penalty for negative edge
beta = 2.0              # Softmax temperature
min_weight = 0.02       # Floor for positive-edge archetypes
```

---

## What This Means for Your Two Issues

### Issue #1: liquidity_vacuum CRISIS (-$201)

**Current State**:
- Only archetype in CRISIS → 100% weight
- Negative edge (Sharpe -0.042)
- No competition for capital

**Soft Gating Solution**:
```python
# Option A: Cap allocation to negative-edge archetypes
if sharpe_like < 0:
    weight = min(weight, 0.20)  # Max 20% to losing strategies

# Option B: Enable other archetypes in CRISIS
# - order_block_retest_crisis (strict variant)
# - funding_divergence_crisis (extreme conditions only)
# - wick_trap_crisis (exhaustion signals only)

# Option C: Regime-specific risk budget
if regime == 'crisis':
    total_exposure_cap = 0.30  # Max 30% of capital in CRISIS
    weight *= total_exposure_cap / sum(all_weights_in_crisis)
```

**Expected Impact**:
- Reduces liquidity_vacuum exposure from 100% → 20-30%
- Forces smaller position sizes in CRISIS
- Reduces regression proportionally

### Issue #2: wick_trap_moneytaur RISK_ON (-$110)

**Current State**:
- 2 archetypes in RISK_ON, both negative
- wick_trap gets 49% weight despite Sharpe -0.025

**Soft Gating Solution**:
```python
# Apply quality threshold
if sharpe_like < -0.05:
    weight = 0.0

# Result: wick_trap_moneytaur gets 0% weight in RISK_ON
# Same outcome as "disable," but via allocation logic
```

**Expected Impact**:
- wick_trap gets 0% allocation in RISK_ON
- Capital reallocated to other regimes/archetypes
- Identical to "disable" but preserves framework flexibility

---

## Implementation Roadmap

### Phase 1: Improved Weighting (1 day)

1. Implement quality-gated softmax
2. Add absolute edge thresholds
3. More aggressive small-sample shrinkage
4. Test on current data

**Deliverable**: `engine/portfolio/regime_allocator.py`

### Phase 2: Regime-Specific Parameters (2-3 days)

Create parameter variants:
```
configs/archetypes/
  order_block_retest/
    neutral.json      # Current parameters
    crisis.json       # Stricter: higher conf, tighter SL
  wick_trap_moneytaur/
    neutral.json      # Current parameters
    risk_on.json      # Exhaustion-only: ADX filter, divergence required
  liquidity_vacuum/
    crisis.json       # Improved: better entry conditions
```

**Deliverable**: Config files + variant loading logic

### Phase 3: Walk-Forward Optimization (3-5 days)

For each archetype-regime pair:
1. Define parameter grid (20-100 variants)
2. Run walk-forward folds
3. Optimize multi-objective:
   - Maximize Sharpe
   - Subject to: max drawdown, min trade count, tail risk
4. Select best variant

**Deliverable**: `bin/walk_forward_regime_optimization.py`

### Phase 4: Bandit Allocation (Future - Post Paper Trading)

Once live:
1. Treat parameter variants as "arms"
2. Thompson sampling or UCB
3. Adaptive reweighting based on recent performance
4. Keep 10-20% exploration weight

**Deliverable**: `engine/portfolio/adaptive_allocator.py`

---

## Expected PnL Impact (Conservative)

### Current State
```
Total PnL: +$139.55
CRISIS: -$201.03
RISK_ON: -$110.61 (wick_trap)
```

### After Soft Gating (No Parameter Variants)

```python
# wick_trap_moneytaur RISK_ON: weight 0.49 → 0.0 (negative edge threshold)
# Improvement: +$110.61

# liquidity_vacuum CRISIS: weight 1.0 → 0.20 (cap on negative edge)
# Position sizes reduced by 80%
# Improvement: ~+$160 (80% of -$201)

Total PnL: +$139 + $111 + $160 = +$410
```

### After Parameter Variants (Phase 2)

```python
# Enable other archetypes in CRISIS with strict params
# Diversify allocation across 3-4 archetypes
# Improvement: +$50-100 additional

# wick_trap RISK_ON strict variant (exhaustion-only)
# Small positive edge, 10-20% weight
# Improvement: +$20-40

Total PnL: +$480-550
```

### After Walk-Forward Optimization (Phase 3)

```python
# Optimized parameters per regime
# Better entry/exit conditions
# Improvement: +$100-200 additional

Total PnL: +$580-750
```

---

## Decision Point

You're absolutely right that **"disable" is crude**. Here's what you should choose:

### Option A: Improved Soft Gating Only (Quick)
- Implement quality-gated softmax
- Apply edge thresholds
- Get +$270 improvement (wick_trap zeroed, liquidity_vacuum capped)
- **Effort**: 1 day
- **Risk**: Low

### Option B: Full Quant Implementation (Proper)
- Phase 1: Soft gating
- Phase 2: Regime-specific parameters
- Phase 3: Walk-forward optimization
- Get +$580-750 improvement
- **Effort**: 1-2 weeks
- **Risk**: Medium (optimization overfitting)

### Option C: Hybrid (Pragmatic)
- Implement soft gating now (Phase 1)
- Deploy CRISIS cap + RISK_ON threshold
- Create 2-3 parameter variants manually (Phase 2 lite)
- Defer walk-forward to post-paper-trading
- Get +$400-500 improvement
- **Effort**: 2-3 days
- **Risk**: Low

---

## My Recommendation

**Option C (Hybrid)**:

1. **Today**: Implement quality-gated softmax with edge thresholds
2. **Tomorrow**: Create 3-4 regime-specific parameter variants manually for worst offenders
3. **This week**: Validate with backtest, deploy if positive
4. **Next month**: Walk-forward optimization once paper trading validates approach

This gets you:
- Quant-correct architecture ✓
- No binary "disables" ✓
- Fast iteration ✓
- Measurable improvement ✓
- Framework for future optimization ✓

---

**What do you think? Should I proceed with implementing the quality-gated soft gating allocator?**
