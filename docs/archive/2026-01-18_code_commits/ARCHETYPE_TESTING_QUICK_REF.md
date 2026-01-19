# Archetype Testing Quick Reference

## TL;DR - Essential Metrics

### Diversity Validation
- **Trade Overlap:** < 30% between unrelated archetypes, < 50% for related ones
- **Return Correlation:** < 0.4 (Pearson), < 0.5 (Spearman)
- **Temporal Diversity:** > 60% of archetype pairs fire in different periods
- **Feature Overlap:** < 50% shared top-5 features

### Realism Checks
- **Signal Count:** 10-20% of bars (10 minimum)
- **Win Rate:** 50-70% realistic, > 80% suspicious
- **Holding Period:** 1-500 bars average
- **Entry/Exit Ratio:** 0.5-1.5
- **Confidence Scores:** Range 0-1, stddev > 0.05

### Smoke Test Config
```python
SMOKE_TEST = {
    'periods': ['2022-bear', '2023-neutral', '2024-bull'],
    'duration': '6 months each',
    'min_signals': 10 per archetype,
    'max_runtime': '60 seconds',
    'asset': 'BTC-USD single asset'
}
```

### Walk-Forward Criteria
- **WFE (Walk Forward Efficiency):** > 50-60%
- **Sharpe Degradation:** < 0.5 drop from IS to OOS
- **Min OOS Periods:** 5+
- **Min Trades per OOS:** 10+

## Quick Test Commands

```bash
# Run smoke test (1 minute)
pytest tests/smoke/test_archetype_smoke.py -v

# Run diversity validation (5 minutes)
pytest tests/smoke/test_diversity_metrics.py -v

# Run full validation (30 minutes)
pytest tests/integration/test_archetype_validation.py -v

# Generate diversity report
python bin/analyze_archetype_diversity.py --output reports/diversity_$(date +%Y%m%d).html
```

## Red Flags to Watch For

### High Priority
- Trade overlap > 60% between different archetypes
- Correlation > 0.7 between strategies
- Win rate > 80% in backtest
- Sharpe drop > 1.0 from train to test
- All signals in one quarter of time period

### Medium Priority
- Confidence scores all identical
- < 10 trades in test period
- Entry/exit ratio > 2.0 or < 0.5
- Feature overlap > 70%

## One-Liner Tests

```python
# Test 1: Trade Overlap
overlap = (entries_A & entries_B).sum() / min(entries_A.sum(), entries_B.sum()) * 100
assert overlap < 30, f"Overlap {overlap}% too high"

# Test 2: Return Correlation
corr = returns_A.corr(returns_B)
assert abs(corr) < 0.5, f"Correlation {corr} too high"

# Test 3: Signal Count
assert 10 <= entries.sum() <= len(df) * 0.2, "Signal count out of range"

# Test 4: Confidence Diversity
assert confidence[entries].std() > 0.05, "Confidence scores not diverse"
```

## Framework Recommendation

**Primary:** VectorBT (fast, vectorized, great for multiple strategies)
**Use For:** Smoke tests, diversity validation, parameter sweeps
**Why:** 10-100x faster, built for strategy comparison

## Critical Thresholds Summary

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Trade Overlap | < 15% | 15-30% | 30-50% | > 50% |
| Correlation | < 0.3 | 0.3-0.4 | 0.4-0.6 | > 0.6 |
| Temporal Diversity | > 70% | 60-70% | 50-60% | < 50% |
| WFE | > 80% | 60-80% | 50-60% | < 50% |
| Signal Count | 15-20% bars | 10-15% | 5-10% | < 5% or > 20% |

## 5-Minute Health Check

```python
# Minimal validation for quick iteration
def quick_health_check(archetype_dict, df):
    for name, arch in archetype_dict.items():
        entries, exits, conf = arch.generate_signals(df)
        n_signals = entries.sum()
        avg_conf = conf[entries].mean()
        conf_std = conf[entries].std()

        print(f"{name:25s} | Signals: {n_signals:3d} | "
              f"Conf: {avg_conf:.2f}±{conf_std:.2f}")

        # Quick assertions
        assert n_signals >= 10, f"{name}: too few signals"
        assert conf_std > 0.05, f"{name}: confidence not diverse"
```

## Full Validation Checklist

- [ ] Individual archetype smoke test passes
- [ ] Pairwise trade overlap < threshold
- [ ] Return correlation matrix acceptable
- [ ] Temporal diversity score > 60%
- [ ] Feature dependency analysis shows diversity
- [ ] Confidence scores meaningful (std > 0.05)
- [ ] No lookahead bias detected
- [ ] Walk-forward efficiency > 50%
- [ ] Performance stable across regimes
- [ ] Runtime < 60s for smoke test

## When to Worry

**Stop and investigate if:**
- 3+ archetypes have > 60% overlap
- Portfolio correlation > 0.7
- Any archetype has < 5 signals in 6 months
- Smoke test takes > 2 minutes
- Test Sharpe < 0.5 of train Sharpe

## Next Steps After Failure

1. **High Overlap:** Review archetype logic, check feature dependencies
2. **High Correlation:** Different exit strategies, different timeframes
3. **Too Few Signals:** Relax thresholds, check data quality
4. **Poor OOS:** Reduce parameters, increase train period
5. **Slow Runtime:** Vectorize code, reduce feature count

---

**For detailed explanations, see:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/TRADING_ARCHETYPE_TESTING_BEST_PRACTICES.md`
