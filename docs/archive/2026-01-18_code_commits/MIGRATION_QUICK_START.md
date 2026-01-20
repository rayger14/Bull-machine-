# Migration Quick Start Guide

**READ THIS FIRST** - Fast reference for executing the migration roadmap

---

## Current Situation (60 seconds)

**Problem:**
- Archetype models exist but unproven vs baseline
- Baseline-Conservative achieves PF 3.17 with simple drawdown strategy
- Need to answer: "Do complex archetypes beat simple baseline?"

**Solution Path:**
Phase 0 (4h) → Phase 1 (2d) → Phase 2 (3d) → Production

---

## Phase 0: TODAY (Next 4 Hours)

### Goal
Answer: "Do archetypes beat PF 3.17?"

### Steps
```bash
# 1. Test S4 archetype on real data (30 min)
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-
python bin/test_archetype_model.py \
  --config configs/s4_optimized_oos_2024.json \
  --data data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
  --period 2023-01-01:2023-12-31

# 2. If 0 trades, create relaxed config (30 min)
# Edit configs/s4_relaxed_test.json
# Lower fusion_threshold from 0.65 to 0.50

# 3. Run full comparison (1 hour)
python examples/baseline_vs_archetype_comparison.py

# 4. Check results (1 hour)
cat results/baseline_vs_archetype_report.txt
# Look for: Test PF for S1, S4
# Decision: Continue if S1 or S4 > 3.17, else pivot
```

### Expected Output
```
Model                   Test_PF   Test_WR  Test_Trades
Baseline-Conservative   3.17      42.9%    7
S1-LiquidityVacuum      X.XX      XX.X%    XXX
S4-FundingDivergence    X.XX      XX.X%    XXX
```

### Decision Point
- **If archetypes WIN:** ✅ Continue to Phase 1
- **If archetypes LOSE:** ❌ Deploy baseline, investigate why
- **If inconclusive:** Run extended test (2022-2024)

---

## Phase 1: Feature Store v2 (1-2 Days)

### Goal
Create enriched feature store with 126+ columns (currently 116)

### Key Tasks
1. **Fix broken columns** (1h)
   - OI derivatives: `oi_change_24h`, `oi_z`
   - No more all-NaN columns

2. **Add derived features** (2h)
   - `fvg_below`, `ob_retest`, `rsi_divergence`
   - Enables bear archetype detection (S1, S2, S4)

3. **Backfill liquidity_score** (2h)
   - Critical for S1 (Liquidity Vacuum)
   - 26,236 rows to calculate

4. **Validate v2** (1h)
   - No NaN values
   - All ranges correct
   - 126+ columns

### Command
```bash
# Build v2
python bin/build_feature_store_v2.py

# Validate
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-2024_v2.parquet \
  --strict

# Test archetypes on v2
python examples/baseline_vs_archetype_comparison.py --data-version v2
```

### Exit Criteria
- ✅ v2 parquet exists (126+ columns)
- ✅ Validation passes (0 NaN)
- ✅ Archetypes generate >0 trades on v2

---

## Phase 2: Optimization (2-3 Days)

### Goal
Optimize archetypes to beat PF 3.17

### Key Tasks
1. **Optimize S1** (6h)
   - Regime-aware optimization
   - 100 trials, 4 parallel processes
   - Target: Test PF > 3.17

2. **Validate S4** (4h)
   - Test on multiple periods (2022, 2023, 2024)
   - Ensure robustness across regimes

3. **Create ensemble** (4h)
   - Combine S1 + S4
   - Target: Test PF > 3.5

4. **Walk-forward validation** (3h)
   - 3 folds (train 12m, test 6m)
   - Average test PF > 3.0

### Command
```bash
# Optimize S1
python bin/optimize_s1_regime_aware.py \
  --data data/features_mtf/BTC_1H_2022-2024_v2.parquet \
  --trials 100 \
  --parallel 4

# Output: configs/s1_optimized_v2.json

# Validate ensemble
python bin/validate_walk_forward.py --model ensemble --folds 3
```

### Exit Criteria
- ✅ S1 or S4 beats baseline (PF > 3.17)
- ✅ Overfit acceptable (<+1.0)
- ✅ Production config created

---

## Decision Tree

```
START
  ├─ Phase 0: Comparison
  │   ├─ Archetypes WIN (PF > 3.17)
  │   │   └─→ Phase 1: Feature Store v2
  │   │       └─→ Phase 2: Optimization
  │   │           ├─ Success (PF > 3.17)
  │   │           │   └─→ Production Deployment
  │   │           └─ Fail (PF < 3.17)
  │   │               └─→ Pivot to ML Baseline
  │   ├─ Archetypes LOSE (PF < 3.17)
  │   │   ├─→ Deploy Baseline-Conservative
  │   │   └─→ Investigate why (features? config? patterns?)
  │   └─ Inconclusive (PF ~= 3.17)
  │       └─→ Extended test (2022-2024)
  │           └─→ Re-evaluate
```

---

## File Locations Reference

### Critical Files
```
examples/baseline_vs_archetype_comparison.py  # Phase 0 comparison script
bin/build_feature_store_v2.py                 # Phase 1 builder (to create)
bin/optimize_s1_regime_aware.py               # Phase 2 optimizer
```

### Data
```
data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet  # v1 (current)
data/features_mtf/BTC_1H_2022-2024_v2.parquet              # v2 (to create)
```

### Configs
```
configs/s1_v2_production.json        # S1 config
configs/s4_optimized_oos_2024.json   # S4 config
configs/s4_relaxed_test.json         # S4 relaxed (to create if needed)
```

### Results
```
results/baseline_vs_archetype_comparison.csv  # Comparison results
results/baseline_vs_archetype_report.txt      # Human-readable report
results/phase2_optimization/                  # Optimization results
```

---

## Common Issues & Solutions

### Issue 1: S4 generates 0 trades
**Solution:** Lower thresholds
```json
// configs/s4_relaxed_test.json
{
  "fusion_threshold": 0.50,  // Was 0.65
  "min_fusion_score": 0.45,  // Was 0.60
  "volume_confirmation": false
}
```

### Issue 2: Feature store validation fails
**Solution:** Check NaN columns
```python
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')
print(df.isnull().sum()[df.isnull().sum() > 0])
# Fix columns with NaN
```

### Issue 3: Optimization runs too slow
**Solution:** Reduce trials or increase parallelization
```bash
# Before: 100 trials, 4 parallel
python bin/optimize_s1_regime_aware.py --trials 100 --parallel 4

# After: 50 trials, 8 parallel
python bin/optimize_s1_regime_aware.py --trials 50 --parallel 8
```

### Issue 4: Import errors
**Solution:** Check Python path
```bash
export PYTHONPATH=/Users/raymondghandchi/Bull-machine-/Bull-machine-:$PYTHONPATH
python examples/baseline_vs_archetype_comparison.py
```

---

## Success Metrics Checklist

### Phase 0 ✓
- [ ] Comparison runs without errors
- [ ] All 4 models generate >0 trades
- [ ] Results documented in `PHASE0_DECISION.md`
- [ ] Decision clear: continue/pivot

### Phase 1 ✓
- [ ] Feature store v2 created (126+ columns)
- [ ] 0 NaN values
- [ ] Validation passes
- [ ] Archetypes work on v2 data

### Phase 2 ✓
- [ ] Optimized model beats baseline (PF > 3.17)
- [ ] Overfit <+1.0
- [ ] Walk-forward validation passes
- [ ] Production config ready

---

## Time Estimates

| Phase | Optimistic | Realistic | Pessimistic |
|-------|------------|-----------|-------------|
| Phase 0 | 2 hours | 4 hours | 8 hours |
| Phase 1 | 1 day | 2 days | 3 days |
| Phase 2 | 2 days | 3 days | 5 days |
| **Total** | **3 days** | **5 days** | **8 days** |

---

## Next Commands to Run

```bash
# RIGHT NOW (Phase 0)
python examples/baseline_vs_archetype_comparison.py

# TOMORROW (Phase 1 - if Phase 0 succeeds)
python bin/build_feature_store_v2.py
python bin/validate_feature_store_schema.py --input data/features_mtf/BTC_1H_2022-2024_v2.parquet

# DAY 3-5 (Phase 2 - if Phase 1 succeeds)
python bin/optimize_s1_regime_aware.py --trials 100 --parallel 4
python bin/validate_walk_forward.py --model ensemble --folds 3
```

---

## Rollback Commands

```bash
# If Phase 1 fails, restore v1
cp data/features_mtf/BTC_1H_2022-2024_backup.parquet \
   data/features_mtf/BTC_1H_2022-2024.parquet

git reset --hard HEAD~1

# If Phase 2 fails, deploy baseline
# Just use Baseline-Conservative config (already proven: PF 3.17)
```

---

## Contact / Escalation

**If Phase 0 fails:**
- Escalate to: System Architect
- Question: "Should we pivot to ML baseline or hybrid approach?"

**If Phase 1 fails:**
- Escalate to: Data Engineering Lead
- Question: "Data quality issues - fix or relax validation?"

**If Phase 2 fails:**
- Escalate to: ML Lead
- Question: "Optimization not improving - try ML or deploy baseline?"

---

## Quick Reference: Baseline Performance

**Baseline-Conservative (Current Champion):**
- Test PF: **3.17**
- Win Rate: **42.9%**
- Trades: **7**
- Overfit: **-1.89** (excellent generalization)
- Strategy: Buy when 30d drawdown < -15%

**This is the bar to beat.**

---

**End of Quick Start Guide**

For full details, see: `MIGRATION_ROADMAP.md`
