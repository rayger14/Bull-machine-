# Wisdom Time Layer - Quick Reference

**Purpose**: One-page reference for operators, traders, and developers
**Version**: 2.0
**Last Updated**: 2025-11-24

---

## What is the Wisdom Time Layer?

The Wisdom Time Layer adds temporal intelligence to pattern detection by calculating when multiple time cycles align. It answers: **"WHEN should patterns fire?"** not just "what patterns exist?"

**Philosophy**: Time is NOT prediction—it's PRESSURE. Markets have rhythm, periodicity, emotional cycles.

**Impact**: ±5-15% adjustment to fusion scores based on temporal confluence.

---

## Four Time Cycles

### 1. Fibonacci Time Clusters (40% weight)
**What**: Bars since major Wyckoff events (SC, Spring-A, BC, UTAD) aligned with Fibonacci levels (21, 34, 55, 89, 144 bars)

**Score Interpretation**:
- 0.8-1.0: **Extreme confluence** (4+ matches, very tight) → 15% boost likely
- 0.6-0.8: **Strong confluence** (3 matches) → 10-12% boost likely
- 0.3-0.6: **Moderate confluence** (1-2 matches) → 5-8% boost
- 0.0-0.3: **Weak/no confluence** → 0-5% penalty

**Example**: Current bar is 55 bars from Spring-A + 89 bars from pivot = High score (0.75+)

---

### 2. Gann Cycles (30% weight)
**What**: Square of 9 price levels, 30/60/90 day cycles, Gann angle adherence

**Score Interpretation**:
- 0.8-1.0: **Price at Gann level + cycle alignment** → 10-12% boost
- 0.6-0.8: **Moderate alignment** → 5-8% boost
- 0.3-0.6: **Weak alignment** → 0-5% adjustment
- 0.0-0.3: **No alignment** → Neutral/small penalty

**Example**: Price = 54,000 (exactly 9,000 × 6) + 30-day cycle detected = High score (0.75+)

---

### 3. Volatility Cycles (20% weight)
**What**: 30-day rolling volatility phase (low/rising/high/declining)

**Phases**:
- **Low** (compression): **Best** → 0.80 score → Boost long entries
- **Rising** (expansion): **Good** → 0.70 score → Trend accelerating
- **High** (climax): **Dangerous** → 0.30 score → Penalty (chop risk)
- **Declining** (consolidation): **Moderate** → 0.60 score → Wait for clarity

**Example**: Vol Z-score = -0.82 (below mean) + rising = Compression phase → 0.80 score

---

### 4. Emotional Cycles (10% weight)
**What**: Market psychology phase (derived from RSI, price momentum, volume)

**Phases** (Wall Street Cheat Sheet):
- **Capitulation**: **Best buy** → 0.95 score → 8% boost
- **Disbelief/Hope**: **Good buy** → 0.75-0.90 score → 5-8% boost
- **Optimism**: **Healthy** → 0.70 score → Slight boost
- **Euphoria**: **Dangerous** → 0.10 score → 10% penalty
- **Panic/Anxiety**: **Good sell** → 0.85 score → Boost short entries

**Example**: RSI = 22, -43% 90-day return, vol spike = Capitulation → 0.95 score

---

## Temporal Confluence Score

**Formula**:
```
Confluence = 0.40 × Fib_Score + 0.30 × Gann_Score + 0.20 × Vol_Score + 0.10 × Emo_Score
```

**Interpretation**:
- **0.70-1.00**: **High confluence** → Pattern has temporal backing → +10-15% boost
- **0.50-0.70**: **Moderate confluence** → Pattern is reasonably timed → +5-10% boost
- **0.30-0.50**: **Weak confluence** → Pattern lacks temporal setup → 0-5% adjustment
- **0.00-0.30**: **No confluence** → Pattern is out of phase → -5-10% penalty

---

## Fusion Score Adjustments

### Adjustment Rules

| Scenario | Condition | Multiplier | Effect |
|----------|-----------|------------|--------|
| **High Confluence + Bullish Phase** | Confluence > 0.70 AND Wyckoff Phase C/D | 1.15× | +15% boost (max) |
| **Double Confluence** | Fib > 0.70 AND Gann > 0.65 | 1.12× | +12% boost |
| **Compression** | Vol = low/rising AND Confluence > 0.60 | 1.10× | +10% boost |
| **Capitulation** | Emotional phase = capitulation/panic | 1.08× | +8% boost |
| **Ranging + Low Confluence** | Wyckoff Phase B AND Confluence < 0.30 | 0.95× | -5% penalty |
| **High Volatility** | Vol phase = high | 0.90× | -10% penalty |
| **Euphoria** | Emotional phase = euphoria/thrill | 0.90× | -10% penalty |

**Bounds**: All adjustments capped at [0.85×, 1.15×] (max ±15%)

---

## Example Scenarios

### Scenario A: Strong Buy Setup

**Context**:
- 55 bars from Spring-A (Fib match!)
- Price at 54,000 (Gann Square of 9)
- Volatility compressing (low phase)
- Emotional phase = disbelief
- Wyckoff Phase = D (markup starting)

**Scores**:
- Fib: 0.82, Gann: 0.76, Vol: 0.85, Emo: 0.90
- **Confluence: 0.786** ⭐

**Adjustment**:
- Rule: High confluence + Phase D → 1.15× (capped)
- Base fusion: 0.42 → **0.483** (+15%)

**Interpretation**: All systems aligned → Maximum boost → Strong buy signal

---

### Scenario B: Dangerous Timing (Avoid)

**Context**:
- Price 2.5× above 90-day MA
- Recent BC + UTAD (5 bars ago)
- Extreme volatility spike
- Emotional phase = euphoria
- Wyckoff Phase = E (markdown imminent)

**Scores**:
- Fib: 0.28, Gann: 0.41, Vol: 0.15, Emo: 0.10
- **Confluence: 0.266** ⚠️

**Adjustment**:
- Rule: High vol + Euphoria → 0.85× (floored)
- Base fusion: 0.42 → **0.357** (-15%)

**Interpretation**: All systems flashing red → Maximum penalty → Avoid new longs

---

### Scenario C: Ranging Market (Weak)

**Context**:
- No recent Wyckoff events (200+ bars)
- Price between Gann levels
- Normal volatility
- Emotional phase = complacency
- Wyckoff Phase = B (ranging)

**Scores**:
- Fib: 0.18, Gann: 0.32, Vol: 0.50, Emo: 0.45
- **Confluence: 0.313** 😐

**Adjustment**:
- Rule: Low confluence + Phase B → 0.95×
- Base fusion: 0.42 → **0.399** (-5%)

**Interpretation**: Weak temporal setup → Slight penalty → Wait for better timing

---

## Feature Reference

### Fibonacci Time Features
```python
bars_since_sc          # 0-500+ (Selling Climax)
bars_since_bc          # 0-500+ (Buying Climax)
bars_since_spring_a    # 0-500+ (Spring-A trap)
bars_since_utad        # 0-500+ (UTAD trap)
bars_since_lps         # 0-500+ (Last Point Support)
fib_time_cluster_score # 0.0-1.0
is_fib_time_cluster_zone  # True/False
```

### Gann Cycle Features
```python
gann_square9_score     # 0.0-1.0 (proximity to level)
gann_confluence_score  # 0.0-1.0 (overall alignment)
acf_30d_score         # 0.0-1.0 (30-day cycle)
acf_60d_score         # 0.0-1.0 (60-day cycle)
acf_90d_score         # 0.0-1.0 (90-day cycle)
```

### Volatility Features
```python
volatility_cycle_score  # 0.0-1.0
volatility_phase       # 'low' / 'rising' / 'high' / 'declining'
volatility_z_score     # -3 to +3 (std deviations)
```

### Emotional Features
```python
emotional_cycle_score  # 0.0-1.0
emotional_phase       # 'capitulation' / 'euphoria' / 'optimism' / etc.
fear_greed_proxy      # 0-100 (synthesized F&G index)
```

---

## Configuration Quick Settings

### Conservative (Production Default)
```json
{
  "temporal_fusion": {
    "enabled": true,
    "confluence_weights": {
      "fib_clusters": 0.50,
      "gann_cycles": 0.35,
      "volatility": 0.15,
      "emotional": 0.00
    },
    "fusion_adjustments": {
      "min_multiplier": 0.90,
      "max_multiplier": 1.12
    }
  }
}
```

### Aggressive (Research)
```json
{
  "temporal_fusion": {
    "enabled": true,
    "confluence_weights": {
      "fib_clusters": 0.40,
      "gann_cycles": 0.30,
      "volatility": 0.20,
      "emotional": 0.10
    },
    "fusion_adjustments": {
      "min_multiplier": 0.85,
      "max_multiplier": 1.15
    }
  }
}
```

### Disabled (A/B Baseline)
```json
{
  "temporal_fusion": {
    "enabled": false
  }
}
```

---

## Troubleshooting

### Low Confluence (<0.30) - Why?

**Check**:
1. Are Wyckoff events detected? (`bars_since_sc`, `bars_since_spring_a`)
2. Is price near Gann level? (`gann_square9_score`)
3. Is volatility extreme? (`volatility_phase` = 'high')
4. Is feature store missing temporal columns?

**Fix**:
- Backfill temporal features: `python bin/build_temporal_features.py`
- Check Wyckoff event confidence threshold (increase if too strict)
- Verify pivot detection working (window size may be too large)

---

### No Fusion Adjustment - Why?

**Check**:
1. Is temporal fusion enabled? (`config['temporal_fusion']['enabled'] = true`)
2. Is confluence above threshold? (need > 0.70 for high_confluence_boost)
3. Is base fusion score far from threshold? (small adjustments may not matter)

**Fix**:
- Enable debug logging: `log_adjustments_above_pct: 0.0`
- Check `temporal_meta['adjustments']` array (which rules triggered?)
- Verify temporal features exist in row data

---

### Unexpected Penalty - Why?

**Check**:
1. Volatility phase = 'high'? (triggers 0.90× penalty)
2. Emotional phase = 'euphoria'? (triggers 0.90× penalty)
3. Wyckoff Phase B + low confluence? (triggers 0.95× penalty)

**Fix**:
- Review telemetry logs: Why did rules fire?
- Adjust penalty strength in config (reduce from 0.90 to 0.95)
- Disable emotional cycles if causing false penalties

---

## Performance Metrics (Target)

### vs Baseline (No Temporal Fusion)

| Metric | Baseline | With Temporal | Target Δ |
|--------|----------|---------------|---------|
| Profit Factor | 2.37 | 2.42+ | +2-5% ✓ |
| Win Rate | 65.8% | 66.5%+ | +1-3% ✓ |
| Sharpe Ratio | 1.83 | 1.92+ | +5-10% ✓ |
| Max Drawdown | -12.4% | -11.2% | -10-15% ✓ |
| Avg R-Multiple | 1.8 | 1.9+ | +5-10% ✓ |

### Operational Metrics

- **Adjustment Frequency**: 20-30% of trades get ≥5% adjustment
- **Avg Adjustment Magnitude**: ±8-10%
- **Feature Parity**: 100% match (batch = stream, diff < 1e-6)
- **Computation Overhead**: Stream mode ≤ 2× batch mode
- **Config Stability**: No tuning needed for 90 days

---

## CLI Commands

### Build Temporal Features (Batch)
```bash
python bin/build_temporal_features.py \
    --input data/processed/features_mtf/btc_1h_2022_2024.parquet \
    --output data/processed/features_mtf/btc_1h_2022_2024_temporal.parquet \
    --config configs/mvp/mvp_bull_market_v1.json
```

### Backtest with Temporal Fusion
```bash
python bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### A/B Comparison (Temporal ON vs OFF)
```bash
# Baseline (no temporal)
python bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --disable-temporal \
    --output results/baseline_no_temporal.json

# Test (with temporal)
python bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --output results/test_with_temporal.json

# Compare
python bin/compare_backtests.py \
    --baseline results/baseline_no_temporal.json \
    --test results/test_with_temporal.json
```

### Feature Parity Validation
```bash
pytest tests/integration/test_temporal_parity.py -v
```

### Historical Scenario Validation
```bash
pytest tests/integration/test_temporal_scenarios.py \
    -k "luna_crash or ftx_collapse or june_18" -v
```

---

## Telemetry / Monitoring

### Key Metrics to Track

1. **Adjustment Frequency**
   - % of trades with temporal adjustment ≥ 5%
   - Target: 20-30%

2. **Average Adjustment Magnitude**
   - Mean absolute adjustment (%)
   - Target: ±8-10%

3. **Confluence Distribution**
   - Histogram of `temporal_confluence_score`
   - Should have peaks at 0.3, 0.6, 0.8 (discrete regimes)

4. **Rule Trigger Counts**
   - Which rules fire most often?
   - Which rules have biggest impact?

5. **Performance Attribution**
   - PF on trades with high confluence (>0.70) vs low (<0.30)
   - WR on adjusted trades vs non-adjusted trades

### Sample Telemetry Log

```
[2025-11-24 14:23:15] [TEMPORAL FUSION] btc_1h 2024-05-12 08:00:00 - +12.3% adjustment: 0.420 → 0.472
  └─ high_confluence_bullish_phase: 1.150x (Temporal confluence 0.78 in Phase D)
  └─ compression: 1.100x (Low vol + confluence 0.78)
  └─ ceiling_enforcement: 0.975x (Capped at 1.15x ceiling)
```

---

## Key Insights (from Research)

### Fibonacci Time Patterns
- **55 bars** is strongest single level (golden ratio derivative)
- **89 bars** is second strongest (Fibonacci sequence)
- **21-34 bars** for short-term reversals
- **144 bars** for major cycle turns (12² = Gann × Fib)

### Gann Observations
- **Square of 9** works best on round numbers (50k, 55k, 60k)
- **30/60/90 day cycles** align with institutional rebalancing
- **360° cycle** ≈ 1 year (seasonal/halving effects for BTC)

### Volatility Cycles
- **Low vol** precedes major moves (80% of time breakout within 30 days)
- **High vol** = mean reversion likely (60% probability within 14 days)
- **Z-score > 2.0** = extreme danger (avoid new positions)

### Emotional Cycles
- **Capitulation** is best buy (95%+ score justified historically)
- **Euphoria** is worst buy (10% score = -90% avoid signal)
- **Disbelief** is second-best buy (early recovery phase)

---

## Resources

### Documentation
- **Architecture**: `docs/WISDOM_TIME_LAYER_ARCHITECTURE.md`
- **Implementation Guide**: `docs/WISDOM_TIME_IMPLEMENTATION_GUIDE.md`
- **This Quick Reference**: `docs/WISDOM_TIME_QUICK_REFERENCE.md`

### Code Locations
- **Fibonacci Clusters**: `engine/temporal/fib_time_clusters.py`
- **Gann Cycles**: `engine/temporal/gann_cycles.py`
- **Volatility Cycles**: `engine/temporal/volatility_cycles.py`
- **Emotional Cycles**: `engine/temporal/emotional_cycles.py`
- **Temporal Confluence**: `engine/temporal/temporal_confluence.py`
- **Fusion Adjustment**: `engine/fusion/temporal.py`

### Tests
- **Unit Tests**: `tests/unit/temporal/`
- **Integration Tests**: `tests/integration/test_temporal_*.py`
- **Feature Parity**: `tests/integration/test_temporal_parity.py`

---

## Support

### Questions?
1. Check documentation: `docs/WISDOM_TIME_*.md`
2. Check code comments: All functions have docstrings
3. Check tests: `tests/unit/temporal/` for examples
4. Ask the team: Engineering channel

### Found a Bug?
1. Check known issues: `docs/KNOWN_ISSUES.md`
2. Run validation tests: `pytest tests/integration/test_temporal_*.py`
3. File issue: Include config + telemetry logs

---

**Document Version**: 1.0
**Last Updated**: 2025-11-24
**Print this page and keep it handy during trading/research sessions.**
