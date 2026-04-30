# CMI v0 Configuration Tuning Guide

## Config File Location
`configs/bull_machine_isolated_v11_fixed.json` → `adaptive_fusion` section

## Core Parameters

### 1. Base Thresholds
```json
"adaptive_fusion": {
  "enabled": true,
  "base_threshold": 0.18,
  "per_archetype_base_threshold": {
    "trap_within_trend": 0.15,
    "liquidity_vacuum": 0.15,
    "retest_cluster": 0.12,
    "wick_trap": 0.18,
    "failed_continuation": 0.15,
    "liquidity_sweep": 0.15
  }
}
```

**Effect of raising per-arch base threshold**:
- Fewer signals generated
- Higher average fusion score per signal (more selective)
- Fewer losing trades, but also fewer winners
- Better Sharpe, slightly lower total return

**Effect of lowering per-arch base threshold**:
- More signals generated
- Lower average fusion score per signal (more permissive)
- More both winners and losers
- Potentially higher total return, worse max drawdown

### 2. Dynamic Adjustment Ranges

```json
"adaptive_fusion": {
  "temp_range": 0.48,      // Range added/subtracted for risk_temp dimension
  "instab_range": 0.15     // Range added/subtracted for instability dimension
}
```

**temp_range interpretation**:
- Bull market (risk_temp=1.0): threshold = base + 0 = base (permissive)
- Neutral (risk_temp=0.5): threshold = base + 0.24 (selective)
- Bear market (risk_temp=0.0): threshold = base + 0.48 (very selective)

**Tuning temp_range**:
- Increase (0.40 → 0.48): Stronger bear penalty, fewer bear market signals
- Decrease (0.48 → 0.40): Weaker bear penalty, more bear market signals
- Default for current system: 0.48 (Variant A, aggressive bear filtering)

**instab_range interpretation**:
- Stable trending (instability=0.0): no adjustment
- Choppy market (instability=1.0): threshold += 0.15

**Tuning instab_range**:
- Increase (0.15 → 0.20): Penalize choppy markets harder
- Decrease (0.15 → 0.10): More signals in choppy market
- Default: 0.15 (balanced)

### 3. Crisis Control

```json
"adaptive_fusion": {
  "crisis_coefficient": 0.4,              // Multiplier for crisis impact on fusion scores
  "emergency_crisis_threshold": 0.7,      // Above this, apply emergency_size_multiplier
  "emergency_size_multiplier": 0.50       // Size cap when crisis_prob > 0.7
}
```

**crisis_coefficient interpretation**:
- fusion_score *= (1 - crisis_prob * coefficient)
- When crisis_prob=1.0 and coefficient=0.4: fusion *= 0.6 (40% reduction)

**Tuning crisis_coefficient**:
- Increase (0.4 → 0.5): Crisis periods reduce fusion scores more harshly
- Decrease (0.4 → 0.3): Lighter crisis penalty on fusion scores
- Default: 0.4 (balanced, ~40% max penalty)

**Emergency sizing example**:
- crisis_prob=0.75 > emergency_threshold=0.7
- Any signals multiplied by 0.50, reducing position sizing
- Prevents overleveraging in extreme stress

### 4. Risk Temperature Weights

```json
"adaptive_fusion": {
  "cmi_weights": {
    "trend_align": 0.45,
    "trend_strength": 0.25,
    "sentiment_score": 0.15,
    "dd_score": 0.10,
    "derivatives_heat": 0.05
  }
}
```

**Weight interpretation**:
- Increasing trend_align weight: More sensitive to EMA alignment (bearish in downtrends)
- Increasing trend_strength weight: More sensitive to ADX (penalize choppy markets)
- Increasing sentiment_score weight: More contrarian (sell extremes, buy fear)
- Increasing dd_score weight: More conservative (penalize active drawdowns)
- Increasing derivatives_heat weight: More OI/funding/taker focused

**Tuning strategy**:
- For trend-following: increase trend_align and trend_strength, decrease sentiment
- For contrarian: increase sentiment_score and derivatives_heat
- For drawdown-averse: increase dd_score
- Default: Balanced (45% trend + 25% strength + 15% sentiment + 10% dd + 5% deriv)

### 5. Position Management

```json
"adaptive_fusion": {
  "base_max_positions": 3      // Reduced by stress level
}
```

**Stress-scaled position limit**:
```
max_positions = 3 * (1 - 0.5 * stress_level)

where stress_level = max(crisis_prob, instability * 0.5)
```

Examples:
- crisis_prob=0.0, instability=0.0: max_pos = 3 * 1.0 = 3
- crisis_prob=0.4, instability=0.2: max_pos = 3 * (1 - 0.5*0.4) = 2.4 ≈ 2
- crisis_prob=0.8, instability=0.8: max_pos = 3 * (1 - 0.5*0.8) = 1.8 ≈ 1

**Tuning base_max_positions**:
- Increase to 4-5: More concurrent trades (higher returns, higher drawdown)
- Decrease to 2: Fewer concurrent trades (lower returns, lower drawdown)
- Default: 3 (balanced)

## Example Tuning Scenarios

### Scenario 1: System Too Permissive (Too Many Signals)

**Diagnosis**: Backtest generates 2000+ signals over full period, win rate < 75%

**Adjustments**:
```json
{
  "per_archetype_base_threshold": {
    "trap_within_trend": 0.18,    // up from 0.15 (+0.03)
    "liquidity_vacuum": 0.18,     // up from 0.15 (+0.03)
    "retest_cluster": 0.14,       // up from 0.12 (+0.02)
    "wick_trap": 0.20,            // up from 0.18 (+0.02)
    "failed_continuation": 0.17,  // up from 0.15 (+0.02)
    "liquidity_sweep": 0.17       // up from 0.15 (+0.02)
  },
  "temp_range": 0.50,             // up from 0.48 (+0.02)
  "instab_range": 0.17,           // up from 0.15 (+0.02)
  "crisis_coefficient": 0.45      // up from 0.4 (stronger crisis penalty)
}
```

**Expected result**: ~20-30% fewer signals, higher quality per-signal

### Scenario 2: System Too Selective (Too Few Signals)

**Diagnosis**: Backtest generates <500 signals over full period, missing obvious trades

**Adjustments**:
```json
{
  "per_archetype_base_threshold": {
    "trap_within_trend": 0.12,    // down from 0.15 (-0.03)
    "liquidity_vacuum": 0.12,     // down from 0.15 (-0.03)
    "retest_cluster": 0.10,       // down from 0.12 (-0.02)
    "wick_trap": 0.16,            // down from 0.18 (-0.02)
    "failed_continuation": 0.13,  // down from 0.15 (-0.02)
    "liquidity_sweep": 0.13       // down from 0.15 (-0.02)
  },
  "temp_range": 0.42,             // down from 0.48 (-0.06)
  "instab_range": 0.12,           // down from 0.15 (-0.03)
  "crisis_coefficient": 0.30      // down from 0.4 (lighter crisis penalty)
}
```

**Expected result**: ~30-50% more signals, lower average quality

### Scenario 3: Too Many Losses in Bear Market

**Diagnosis**: System loses money in 2022, 2018, other bear periods

**Adjustments**:
```json
{
  "temp_range": 0.55,             // up from 0.48 (stronger bear penalty)
  "crisis_coefficient": 0.50,     // up from 0.4 (stronger crisis penalty)
  "emergency_crisis_threshold": 0.65  // down from 0.7 (trigger emergency sizing sooner)
}
```

**Expected result**: Bear market signals reduced significantly, emergency sizing kicks in earlier

### Scenario 4: Overlevered in Crisis Periods

**Diagnosis**: Max drawdown excessive during 2020-03 or other crisis periods

**Adjustments**:
```json
{
  "emergency_crisis_threshold": 0.60,  // down from 0.7 (more aggressive)
  "emergency_size_multiplier": 0.33,   // down from 0.50 (harder cap)
  "base_max_positions": 2               // down from 3 (fewer concurrent)
}
```

**Expected result**: Smaller positions and fewer entries during crisis, lower max drawdown

## Implementation Steps

1. **Make changes to config file**:
   ```bash
   vim configs/bull_machine_isolated_v11_fixed.json
   ```

2. **Validate JSON syntax**:
   ```bash
   jq . < configs/bull_machine_isolated_v11_fixed.json
   ```

3. **Run quick backtest on small period**:
   ```bash
   python3 bin/backtest_v11_standalone.py \
     --config configs/bull_machine_isolated_v11_fixed.json \
     --data data/btc_1h_2023_Q1.csv \
     --commission-rate 0.0002 \
     --slippage-bps 3 \
     2>&1 | tee backtest.log
   ```

4. **Check key metrics**:
   ```bash
   grep "Total signals:" backtest.log
   grep "Total Trades:" backtest.log
   grep "Win Rate:" backtest.log
   grep "Profit Factor:" backtest.log
   grep "Max Drawdown:" backtest.log
   ```

5. **Full backtest if good**:
   ```bash
   python3 bin/backtest_v11_standalone.py \
     --config configs/bull_machine_isolated_v11_fixed.json \
     --start-date 2020-01-01 \
     --commission-rate 0.0002 \
     --slippage-bps 3 \
     2>&1 | tee backtest_full.log
   ```

6. **Compare against baseline**:
   - Q1 2023 baseline: ~10-15 trades, PF ~1.8, win rate ~82%
   - Full period baseline: ~1500-1600 trades, PF ~1.48, Sharpe ~0.96

## Monitoring Parameters

To inspect live CMI computation during backtest, look for these log lines:

```
[THRESHOLD] bar=... | dynamic_threshold=X.XXX | risk_temp=X.XXX | instability=X.XXX | crisis_prob=X.XXX

[SIGNALS] bar=... | raw=... | post_filter=... | post_threshold=... | passed=...

[FILTER] archetype rejected: fusion=X.XXX*penalty=X.XXX=X.XXX < threshold=X.XXX
```

## Validation Checklist

Before committing config changes:

- [ ] JSON parses without errors
- [ ] No spelling mistakes in archetype names
- [ ] Per-arch thresholds in range [0.10, 0.25]
- [ ] temp_range in range [0.35, 0.55]
- [ ] instab_range in range [0.10, 0.25]
- [ ] crisis_coefficient in range [0.25, 0.55]
- [ ] emergency_crisis_threshold in range [0.60, 0.80]
- [ ] emergency_size_multiplier in range [0.30, 1.0]
- [ ] base_max_positions in range [2, 5]
- [ ] All cmi_weights sum to 1.0
- [ ] Backtest metrics align with expectations

---

**Last updated**: 2026-02-16
**Related docs**: FUSION_QUICK_REFERENCE.md, FUSION_THRESHOLD_ARCHITECTURE.md
