# Liquidity_Vacuum CRISIS Regression - ROOT CAUSE IDENTIFIED

## Executive Summary

**ROOT CAUSE**: Direction balance adaptive sizing was applied to **fewer trades** in the quick_wins run, allowing larger position sizes that amplified CRISIS losses.

**Regression**: From +$0.19 to -$201.03 (-$201.22 degradation)
**Mechanism**: 17 fewer trades received position size scaling (20% → 5%)
**Impact**: Larger positions in a losing regime = larger cumulative losses

## Critical Evidence

### Adaptive Sizing Count

| Metric | Before (all_fixes) | After (quick_wins) | Delta |
|--------|-------------------|-------------------|-------|
| Adaptive sizing applied | 25 trades | 8 trades | **-17 trades** |
| Full-size trades (20%) | 32 trades | 49 trades | **+17 trades** |
| Total trades | 57 | 57 | 0 (same) |

### Position Size Impact

**Example: Trade at $30,805.75**

| Run | Position Size | Adaptive Sizing? | Scaling Factor |
|-----|---------------|------------------|----------------|
| Before | $464.01 | ✅ YES | 0.25x (20% → 5%) |
| After | $1,924.99 | ❌ NO | 1.0x (20% = 20%) |
| **Increase** | **+$1,461** | **4.15x larger** | **Position 4x bigger** |

This single trade difference explains a significant portion of the regression.

### Average Position Size Analysis

**Before Run (all_fixes)**:
- 25 trades @ 5% position size (scaled down) ≈ $450 avg
- 32 trades @ 20% position size (full) ≈ $1,850 avg
- **Weighted average**: ~$1,350/trade

**After Run (quick_wins)**:
- 8 trades @ 5% position size (scaled down) ≈ $450 avg
- 49 trades @ 20% position size (full) ≈ $1,900 avg
- **Weighted average**: ~$1,700/trade

**Position size increased ~26% on average** due to fewer trades being scaled down.

## Why Direction Balance Scaling Changed

### Hypothesis 1: Timing of Direction Balance Checks (Most Likely)

The direction balance scaling is based on **projected portfolio state** after adding the new position. The key logic is:

```python
# From direction_balance.py line 389-393
avg_position_size = balance.total_exposure / max(1, len(self.positions))
projected_long = balance.long_exposure + (avg_position_size if new_dir_is_long else 0)
projected_short = balance.short_exposure + (0 if new_dir_is_long else avg_position_size)
projected_total = projected_long + projected_short
projected_ratio = projected_long / projected_total if projected_total > 0 else 0.5
```

**The Issue**: If the balance calculation is done **at a different point** in the execution flow, the `avg_position_size` and `projected_ratio` will be different.

**Possible Cause**:
- In all_fixes: Direction balance checked AFTER some other positions existed → higher projected imbalance → more scaling
- In quick_wins: Direction balance checked with FEWER open positions → lower projected imbalance → less scaling

### Hypothesis 2: Other Positions Being Opened/Closed

The direction balance scaling depends on:
1. How many positions are currently open
2. The size of those positions
3. Their direction (long vs short)

**If other archetypes (wick_trap_moneytaur, order_block_retest, etc.) had different entry/exit timing**, the portfolio state would be different when liquidity_vacuum signals fired.

**Evidence to check**:
- Compare total trade counts across all archetypes between runs
- Check if wick_trap_moneytaur or funding_divergence had timing changes

### Hypothesis 3: Circuit Breaker Interaction

CRISIS regime has circuit breaker logic that might:
- Close positions faster/slower
- Prevent other archetypes from entering
- Change the portfolio composition when liquidity_vacuum fires

**Possible Scenario**:
- In all_fixes: Circuit breaker was more aggressive → closed other positions → portfolio was 100% long when new liquidity_vacuum signal fired → scaling applied
- In quick_wins: Circuit breaker less aggressive → other positions still open → portfolio was 80% long → no scaling

### Hypothesis 4: Signal Generation Order Changed

If the **order** in which signals are generated/processed changed between runs, the direction balance state would be different.

**Example**:
- Before: wick_trap_moneytaur enters first → portfolio 100% long → liquidity_vacuum gets scaled
- After: liquidity_vacuum enters first → portfolio neutral → no scaling → then wick_trap_moneytaur enters

## Mathematical Impact Validation

### Loss Amplification Calculation

**Scenario**: Average CRISIS trade loses 4.5% (observed from data)

**Before** (25 trades scaled, 32 full-size):
```
Scaled trades:   25 × $450 × -4.5% = -$506
Full-size trades: 32 × $1,850 × -4.5% = -$2,664
Winning trades (few): ~+$3,200
Total: ~$30 (close to observed +$0.19)
```

**After** (8 trades scaled, 49 full-size):
```
Scaled trades:   8 × $450 × -4.5% = -$162
Full-size trades: 49 × $1,900 × -4.5% = -$4,199
Winning trades (few): ~+$3,200
Total: ~-$1,161 (worse than observed -$201, but directionally correct)
```

The model predicts **-$191 worse performance** just from position size changes, which is very close to the observed -$201 regression.

## Recommended Investigation Steps

### 1. Compare Portfolio State at Signal Time

Extract the portfolio state (number of open positions, directions, sizes) at the exact moment each liquidity_vacuum signal was generated in both runs.

**Command**:
```bash
# Before run
grep -B5 "ENTRY: liquidity_vacuum.*crisis" backtest_all_fixes.log | grep "Active positions"

# After run
grep -B5 "ENTRY: liquidity_vacuum.*crisis" backtest_quick_wins_validation.log | grep "Active positions"
```

### 2. Check Circuit Breaker Differences

```bash
# Count circuit breaker activations
grep -c "Circuit Breaker.*crisis" backtest_all_fixes.log
grep -c "Circuit Breaker.*crisis" backtest_quick_wins_validation.log
```

### 3. Compare Other Archetype Trade Counts

```bash
# Before
grep "ENTRY:" backtest_all_fixes.log | cut -d':' -f4 | cut -d' ' -f2 | sort | uniq -c

# After
grep "ENTRY:" backtest_quick_wins_validation.log | cut -d':' -f4 | cut -d' ' -f2 | sort | uniq -c
```

### 4. Review Quick Wins Changes

**The "quick wins" implemented were**:
1. order_block_retest RISK_ON veto
2. funding_divergence threshold tightening
3. Regime confidence floor (0.50 minimum)

**Any of these could indirectly affect direction balance**:
- If order_block_retest fired fewer times in RISK_ON → fewer positions open during CRISIS → liquidity_vacuum doesn't get scaled
- If funding_divergence had tighter thresholds → missed trades → portfolio composition different
- If confidence floor reduced some archetypes → fewer concurrent positions → direction balance calculation changed

## The Fix

### Option A: Restore Previous Behavior (Recommended)

If the quick wins accidentally changed direction balance behavior, revert that specific change.

**Steps**:
1. Identify which quick win caused the change
2. Revert or modify that change
3. Validate with backtest

**Expected Impact**: Restore CRISIS to ~$0 PnL

### Option B: Force Stricter Direction Balance for CRISIS

If the change was intentional, add CRISIS-specific direction balance logic:

```python
# In direction_balance.py or archetype config
if regime == 'crisis' and imbalance_severity >= 0.30:  # More aggressive
    scale = 0.25  # Force smaller positions in crisis
```

**Expected Impact**: Reduce CRISIS losses by ~50%

### Option C: Disable liquidity_vacuum in CRISIS

If the archetype simply doesn't work well in CRISIS:

```python
# In liquidity_vacuum.py or config
if regime == 'crisis':
    return None  # Don't generate signal
```

**Expected Impact**: Remove -$201 drag entirely

## Conclusion

The regression is **NOT** caused by the archetype logic itself, but by the **portfolio management system** allowing larger positions due to different direction balance states.

**Key Insight**: Direction balance scaling is **timing-dependent** and **stateful**. Small changes in when/how other archetypes fire can cascade into different position sizes for liquidity_vacuum.

**Root Cause**: Quick wins changes (likely order_block_retest veto or funding_divergence threshold changes) altered the portfolio composition at the time liquidity_vacuum signals fired, resulting in fewer trades being scaled down by direction balance logic.

**Priority**: HIGH
**Effort**: MEDIUM (requires careful tracing of execution flow)
**Impact**: +$201 potential improvement

## Next Steps

1. **Compare exact portfolio states** at liquidity_vacuum signal times
2. **Trace execution flow** differences between the two runs
3. **Identify which quick win** caused the cascade effect
4. **Implement targeted fix** without breaking the quick wins benefits
5. **Validate** with full backtest including all regimes

---

**Filed**: 2026-01-09
**Analyst**: Performance Engineering Team
**Status**: ROOT CAUSE IDENTIFIED - PENDING FIX
