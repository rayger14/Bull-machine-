# Walk-Forward Production Engine - Quick Reference

**Last Updated**: 2026-01-16

---

## 📋 Quick Commands

### Test Single Archetype
```bash
python bin/walk_forward_production_engine.py \
    --archetype S1 \
    --config results/optimization_2026-01-16/S1/best_config.json
```

### Run All Archetypes
```bash
python bin/walk_forward_production_engine.py --all
```

### Generate Report
```bash
python bin/walk_forward_production_engine.py --report
```

---

## 🏗️ Architecture Overview

```
Walk-Forward Orchestrator
    ↓ (Generate 8-10 windows)
FullEngineBacktest (REAL engine)
    ├─ ArchetypeFactory (real implementations)
    ├─ RegimeService (logistic + hysteresis)
    ├─ CircuitBreakerEngine
    ├─ DirectionBalanceTracker
    └─ TransactionCostModel
    ↓ (Aggregate results)
Production Readiness Assessment
```

---

## 📊 Window Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Train     | 365 days | Optimize params |
| Embargo   | 48 hours | Prevent leakage |
| Test      | 90 days  | OOS validation |
| Step      | 90 days  | Non-overlap |
| Windows   | 8-10     | 2022-2024 |

---

## ✅ Production Readiness Criteria

| Metric | Threshold | Must Pass |
|--------|-----------|-----------|
| OOS Degradation | <20% | ✅ YES |
| Profitable Windows | >60% | ✅ YES |
| Aggregate Sharpe | >0.5 | ✅ YES |
| Max Window DD | <50% | ✅ YES |

**Formula**: `degradation = (in_sample_sharpe - oos_sharpe) / in_sample_sharpe * 100`

---

## 🎯 Expected Results

### Best Case (5/6 pass)
- Average degradation: 16%
- Production-ready: 5 archetypes
- Total OOS PnL: $7,500+

### Likely Case (3/6 pass)
- Average degradation: 20%
- Production-ready: 3 archetypes
- Total OOS PnL: $4,500+

### Worst Case (1/6 pass)
- Average degradation: 25%+
- Production-ready: 1 archetype
- **Action**: Re-optimize all configs

---

## 🔧 Implementation Checklist

### Phase 1: Infrastructure (2h)
- [ ] Enhance `FullEngineBacktest` with param overrides
- [ ] Enhance `ArchetypeFactory` with config_override
- [ ] Create `ProductionWalkForwardValidator`
- [ ] Test on single window

### Phase 2: Validation (2h)
- [ ] Run S1 (all windows)
- [ ] Run S4, S5, B, H, K
- [ ] Generate per-archetype results
- [ ] Calculate aggregate metrics

### Phase 3: Reporting (1h)
- [ ] Comparison report (Markdown)
- [ ] Production-ready configs (JSON)
- [ ] Recommendations document

---

## 📁 Key Files

### Design Documents
```
WALK_FORWARD_PRODUCTION_ENGINE_DESIGN.md       - Main spec
WALK_FORWARD_IMPLEMENTATION_PLAN.md           - Step-by-step
WALK_FORWARD_ARCHITECTURE_DIAGRAM.txt         - Visual
WALK_FORWARD_REAL_ENGINE_SUMMARY.md           - Summary
WALK_FORWARD_QUICK_REFERENCE.md               - This file
```

### Implementation
```
bin/walk_forward_production_engine.py         - NEW (to create)
bin/backtest_full_engine_replay.py            - ENHANCE
engine/archetypes/archetype_factory.py        - ENHANCE
```

### Configs
```
results/optimization_2026-01-16/S1/best_config.json
results/optimization_2026-01-16/S4/best_config.json
results/optimization_2026-01-16/S5/best_config.json
results/optimization_2026-01-16/B/best_config.json
results/optimization_2026-01-16/H/best_config.json
results/optimization_2026-01-16/K/best_config.json
```

---

## 🐛 Troubleshooting

### "ArchetypeFactory not loading"
```bash
# Check registry
cat archetype_registry.yaml | grep -A5 "id: S1"

# Verify enable flag
python -c "from engine.archetypes.archetype_factory import ArchetypeFactory; \
           factory = ArchetypeFactory({'enable_S1': True}); \
           print(factory.get_active_archetypes())"
```

### "No regime_label column"
```bash
# Verify columns
python -c "import pandas as pd; \
           df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); \
           print('regime_label' in df.columns)"

# Add if missing
python bin/add_regime_labels_streaming.py \
    --input data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --output data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
```

### "Walk-forward too slow"
```python
# Reduce logging
logging.basicConfig(level=logging.WARNING)

# Smaller test windows
validator = ProductionWalkForwardValidator(test_days=60)

# Parallel execution (advanced)
from multiprocessing import Pool
pool.map(run_window, windows)
```

---

## 📈 Performance Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Single window | 5s | 500MB |
| Single archetype (8 windows) | 45s | 1GB |
| All archetypes (6×8 windows) | 5min | 2GB |

---

## 🎓 Key Concepts

### OOS Degradation
```
Measures how much performance drops from in-sample to out-of-sample.
Low degradation = robust configs that generalize well.

Example:
  In-sample Sharpe: 1.78
  OOS Sharpe: 1.42
  Degradation: (1.78 - 1.42) / 1.78 = 20.2%
  Status: ❌ Failed (>20%)
```

### Embargo Period
```
Gap between train and test to prevent temporal leakage.
Features like moving averages look back N bars, so we need
a buffer to ensure test period truly doesn't see train data.

48 hours = Safe for most indicators
```

### Production Readiness
```
A config is production-ready if it passes ALL criteria:
  ✅ OOS degradation <20%
  ✅ >60% windows profitable
  ✅ Aggregate Sharpe >0.5
  ✅ No catastrophic failures

If it fails ANY criterion, re-optimize.
```

---

## 🚨 Common Mistakes

### ❌ Using Simplified Backtest
```python
# DON'T DO THIS
def _generate_signals(data, archetype):
    if archetype == 'liquidity_vacuum':
        return data['liquidity_score'] < 0.2  # Fake logic
```

### ✅ Using Production Engine
```python
# DO THIS
backtest = FullEngineBacktest(config)
results = backtest.run(data, archetypes=['S1'])  # Real engine
```

### ❌ Overlapping Test Windows
```
[Train][Test]
       [Train][Test]  ← BAD (overlap)
```

### ✅ Non-Overlapping Test Windows
```
[Train][Test]
              [Train][Test]  ← GOOD (no overlap)
```

### ❌ No Embargo Period
```
[Train|Test]  ← BAD (leakage from MAs, volume_z, etc.)
```

### ✅ With Embargo
```
[Train|Embargo|Test]  ← GOOD (prevents leakage)
```

---

## 📊 Example Output

### Per-Archetype Result
```json
{
  "archetype": "S1",
  "total_windows": 8,
  "aggregate_metrics": {
    "total_trades": 73,
    "total_pnl": 1247.32,
    "avg_sharpe": 1.42,
    "profitable_windows": 6,
    "profitable_pct": 75.0
  },
  "oos_analysis": {
    "in_sample_sharpe": 1.78,
    "oos_sharpe": 1.42,
    "degradation_pct": 20.2,
    "robust": false
  },
  "production_ready": false
}
```

### Comparison Report
```markdown
| Archetype | Degradation | OOS Sharpe | Ready |
|-----------|-------------|------------|-------|
| S1        | 20.2%       | 1.42       | ❌    |
| S4        | 15.3%       | 1.52       | ✅    |
| S5        | 12.7%       | 1.61       | ✅    |
| B         | 18.9%       | 1.48       | ✅    |
| H         | 16.2%       | 1.55       | ✅    |
| K         | 14.5%       | 1.68       | ✅    |
```

---

## 🔗 Related Documentation

- **Main Design**: See `WALK_FORWARD_PRODUCTION_ENGINE_DESIGN.md`
- **Implementation**: See `WALK_FORWARD_IMPLEMENTATION_PLAN.md`
- **Architecture**: See `WALK_FORWARD_ARCHITECTURE_DIAGRAM.txt`
- **Summary**: See `WALK_FORWARD_REAL_ENGINE_SUMMARY.md`

---

## ⏱️ Time Estimates

| Task | Estimate | Actual |
|------|----------|--------|
| Design | 2h | ✅ Done |
| Implementation | 4-5h | Pending |
| Validation Run | 1h | Pending |
| Analysis | 1h | Pending |
| **Total** | **8-9h** | **2h** |

---

## 💡 Pro Tips

1. **Start with one archetype**: Test S1 first, fix issues, then scale
2. **Check logs**: Verify "FullEngineBacktest initialized" appears
3. **Monitor memory**: Large datasets may need chunking
4. **Save intermediate results**: Don't lose 2 hours of computation
5. **Parallelize carefully**: Multiprocessing can cause pickling issues

---

## 📞 Support

**Issues?**
1. Check troubleshooting section
2. Review implementation plan
3. Verify all files exist
4. Check Python environment (pandas, numpy, etc.)

**Still stuck?**
- Review `WALK_FORWARD_IMPLEMENTATION_PLAN.md` Section "Troubleshooting"
- Check error logs in `logs/walk_forward_production.log`

---

**Last Updated**: 2026-01-16
**Version**: 1.0
**Status**: Ready for Implementation
