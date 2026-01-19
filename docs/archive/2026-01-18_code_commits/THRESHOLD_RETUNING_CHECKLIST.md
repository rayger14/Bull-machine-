# Threshold Re-tuning Checklist
## Domain Engine Gate Fix Impact Assessment

**Date**: 2025-12-11
**Context**: After fixing domain engine gate ordering, some threshold parameters may need re-calibration

---

## Priority 1: Fusion Thresholds (RECOMMENDED)

These are the most likely to need adjustment since domain boosts now rescue marginal signals.

### S1: Liquidity Vacuum

**Parameter**: `fusion_threshold`
- **Current**: 0.40 (default)
- **Suggested**: 0.45 (+12.5%)
- **Rationale**: Domain boosts (2.0x - 2.5x) rescue quality signals, can afford higher threshold
- **Testing**: Backtest both values, compare Sharpe/Calmar
- **Config Location**: `configs/mvp/mvp_bear_market_v1.json` → `thresholds.liquidity_vacuum.fusion_threshold`

**Expected Impact**:
- Signal count: -5 to -10% (filters noise)
- Signal quality: +10-15% (keeps only domain-confirmed signals)
- Net effect: Likely improvement in Sharpe/Calmar

---

### S4: Funding Divergence

**Parameter**: `fusion_threshold`
- **Current**: 0.40 (default)
- **Suggested**: 0.45 (+12.5%)
- **Rationale**: Same as S1 - domain boosts (2.0x - 2.5x) rescue quality signals
- **Testing**: Backtest both values
- **Config Location**: `configs/mvp/mvp_bear_market_v1.json` → `thresholds.funding_divergence.fusion_threshold`

**Expected Impact**:
- Signal count: -5 to -10%
- Signal quality: +10-15%

---

### S5: Long Squeeze

**Parameter**: `fusion_threshold`
- **Current**: 0.35 (default)
- **Suggested**: 0.40 (+14%)
- **Rationale**: Same as S1/S4
- **Testing**: Backtest both values
- **Config Location**: `configs/mvp/mvp_bear_market_v1.json` → `thresholds.long_squeeze.fusion_threshold`

**Expected Impact**:
- Signal count: -5 to -10%
- Signal quality: +10-15%

---

## Priority 2: Domain Boost Multipliers (OPTIONAL)

Current boost values may be too aggressive now that they actually affect gate passage.

### Wyckoff Boosts (All Archetypes)

**High Impact Signals**:

| Signal | Current | Suggested | Rationale |
|--------|---------|-----------|-----------|
| `wyckoff_spring_a` | 2.5x | 2.0x | Still strong, but less aggressive |
| `wyckoff_spring_b` | 2.5x | 2.0x | Same as spring_a |
| `wyckoff_accumulation` | 2.0x | 1.8x | Slight reduction |
| `wyckoff_distribution` | 2.0x | 1.8x | Slight reduction |
| `wyckoff_sc` | 2.0x | 1.8x | Selling climax |

**Testing Strategy**:
1. Run backtest with current values (baseline)
2. Run backtest with reduced values (-20%)
3. Compare Sharpe/Calmar/Max Drawdown
4. Choose version with better risk-adjusted returns

**Config Location**: Hard-coded in `logic_v2_adapter.py` (lines ~1825-1870 for S1)
- **Note**: Not configurable, would require code change

---

### SMC Boosts (All Archetypes)

| Signal | Current | Suggested | Rationale |
|--------|---------|-----------|-----------|
| `tf4h_bos_bullish` | 2.0x | 1.8x | 4H institutional shift |
| `smc_liquidity_sweep` | 1.8x | 1.6x | Stop hunt signal |
| `smc_choch` | 1.6x | 1.5x | Change of character |

**Priority**: Low (SMC boosts less aggressive than Wyckoff)

---

### Temporal Boosts (All Archetypes)

| Signal | Current | Suggested | Rationale |
|--------|---------|-----------|-----------|
| `fib_time_cluster` | 1.8x | 1.6x | Fibonacci timing |
| `temporal_confluence` | 1.5x | 1.4x | Multi-timeframe alignment |

**Priority**: Low (Temporal boosts moderate)

---

## Priority 3: Component Weights (LOW PRIORITY)

These are unlikely to need changes (internal scoring, not gate-related).

### S1 V2 Weights

**Current** (default):
```json
{
  "capitulation_depth_score": 0.20,
  "crisis_environment": 0.15,
  "volume_climax_3b": 0.08,
  "wick_exhaustion_3b": 0.07,
  "liquidity_drain_severity": 0.10,
  "liquidity_velocity_score": 0.08,
  "liquidity_persistence_score": 0.07,
  "funding_reversal": 0.12,
  "oversold": 0.08,
  "volatility_spike": 0.05
}
```

**Recommendation**: No change unless post-deployment metrics show component imbalance

**Config Location**: `configs/mvp/mvp_bear_market_v1.json` → `thresholds.liquidity_vacuum.v2_weights`

---

### S4 Weights

**Current** (default):
```json
{
  "funding_negative": 0.40,
  "price_resilience": 0.30,
  "volume_quiet": 0.15,
  "liquidity_thin": 0.15
}
```

**Recommendation**: No change

---

### S5 Weights

**Current** (default with OI data):
```json
{
  "funding_extreme": 0.40,
  "rsi_exhaustion": 0.30,
  "oi_spike": 0.15,
  "liquidity_thin": 0.15
}
```

**Recommendation**: No change

---

## Testing Protocol

### Phase 1: Fusion Threshold Re-tuning (HIGH PRIORITY)

**Objective**: Find optimal fusion_threshold for each archetype

**Method**:
```bash
# Run optimizer for S1 with threshold range
python bin/optimize_s1_regime_aware.py \
  --threshold-range 0.35,0.50 \
  --step 0.05 \
  --metric sharpe

# Repeat for S4, S5
python bin/optimize_s4_calibration.py --threshold-range 0.35,0.50
python bin/optimize_s5_calibration.py --threshold-range 0.30,0.45
```

**Success Criteria**:
- Sharpe ratio improvement vs baseline
- Max drawdown not worse than +10%
- Signal count remains reasonable (>10 matches per year)

**Timeline**: 1-2 days (can run in parallel)

---

### Phase 2: Domain Boost Tuning (OPTIONAL)

**Objective**: Test if reducing boost multipliers improves performance

**Method**:
1. Manually edit `logic_v2_adapter.py` to reduce boosts by 20%
2. Run full backtest suite
3. Compare vs baseline (current values)

**Success Criteria**:
- Sharpe improvement OR
- Max drawdown improvement with minimal Sharpe degradation

**Timeline**: 2-3 days (requires code changes + full backtest)

**Priority**: Low (only if Phase 1 results suggest over-boosting)

---

### Phase 3: Component Weight Tuning (LOW PRIORITY)

**Objective**: Fine-tune internal scoring components

**Method**: Use existing optimizer scripts with component weight search space

**Success Criteria**: Marginal improvement (1-2% Sharpe)

**Timeline**: 3-5 days (full hyperparameter search)

**Priority**: Low (only if Phases 1-2 show promise)

---

## Quick Start: Recommended Actions

### Immediate (Before Production)
- [x] Deploy fix as-is (no threshold changes)
- [ ] Monitor signal counts for 7 days
- [ ] Collect domain_boost telemetry

### Week 1 Post-Deployment
- [ ] Analyze domain_boost distribution
- [ ] Run Phase 1 threshold optimization
- [ ] Update configs with new thresholds (if improvement confirmed)

### Week 2+ (Optional)
- [ ] Run Phase 2 boost multiplier testing (if Phase 1 shows over-boosting)
- [ ] Run Phase 3 component weight tuning (if time permits)

---

## Expected Results by Phase

### Phase 1: Fusion Threshold Increase
- **Signal Count**: -5% to -10%
- **Signal Quality**: +10% to +15%
- **Sharpe**: +5% to +10%
- **Max Drawdown**: -5% to -10% (improvement)

### Phase 2: Boost Reduction (if needed)
- **Signal Count**: -10% to -15%
- **Signal Quality**: +5% to +10%
- **Sharpe**: +2% to +5%
- **Max Drawdown**: -3% to -5%

### Phase 3: Weight Tuning (marginal)
- **Signal Count**: ±2%
- **Signal Quality**: ±3%
- **Sharpe**: +1% to +2%

---

## Monitoring Dashboard

Track these metrics post-deployment:

### Signal Metrics
- [ ] S1 match count (daily/weekly)
- [ ] S4 match count (daily/weekly)
- [ ] S5 match count (daily/weekly)
- [ ] Domain boost distribution (min/mean/median/max)
- [ ] Veto rate (should be stable)

### Performance Metrics
- [ ] Sharpe ratio (rolling 30/60/90 day)
- [ ] Calmar ratio (rolling 90/180 day)
- [ ] Max drawdown (rolling 90 day)
- [ ] Win rate (rolling 30 day)

### Telemetry Analysis
- [ ] Log all `score_before_domain` values
- [ ] Log all `domain_boost` values
- [ ] Log all `domain_signals` arrays
- [ ] Identify most common boost patterns

**Query Example**:
```sql
SELECT
  archetype,
  AVG(domain_boost) as avg_boost,
  COUNT(*) as match_count,
  array_agg(DISTINCT domain_signals) as common_signals
FROM archetype_matches
WHERE ts > NOW() - INTERVAL '7 days'
  AND matched = true
GROUP BY archetype;
```

---

## Summary: Three-Phase Approach

| Phase | Priority | Timeline | Expected Impact |
|-------|----------|----------|-----------------|
| 1. Fusion Threshold | HIGH | 1-2 days | +5-10% Sharpe |
| 2. Boost Multipliers | MEDIUM | 2-3 days | +2-5% Sharpe |
| 3. Component Weights | LOW | 3-5 days | +1-2% Sharpe |

**Recommendation**: Start with Phase 1 only. Proceed to Phase 2/3 only if Phase 1 results warrant further tuning.

---

**Next Steps**:
1. Deploy fix with current thresholds
2. Monitor for 7 days
3. Run Phase 1 optimization
4. Update configs if improvement confirmed

