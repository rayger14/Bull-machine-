# Engine Weight Optimization - Quick Start Guide

**TL;DR:** Run ML approach in 2 hours to get optimized weights, then validate with Optuna for production rigor.

---

## 1. Test the System (5 minutes)

Verify all components work:

```bash
# Test meta-fusion module
python3 -c "
from engine.archetypes.meta_fusion import MetaFusionEngine
import pandas as pd

config = {'engine_weights': {'structure': 0.35, 'liquidity': 0.25, 'momentum': 0.15, 'wyckoff': 0.15, 'macro': 0.10}}
engine = MetaFusionEngine(config)
print('✓ Meta-fusion working')
"

# Run full validation test (runs minimal Optuna trial)
python3 bin/test_engine_weight_optimizer.py
```

**Expected:** All tests pass, confirms system is ready.

---

## 2. ML Approach - Fast Baseline (1-2 hours)

Train a logistic regression model to learn optimal weights:

```bash
# Basic usage (Logistic Regression)
python3 bin/train_fusion_metalearner.py \
    --input results/enhanced_new_wyckoff_2024.csv \
    --output configs/optimized/engine_weights_ml_logistic.json \
    --model logistic

# Alternative: Gradient Boosted Trees
python3 bin/train_fusion_metalearner.py \
    --input results/enhanced_new_wyckoff_2024.csv \
    --output configs/optimized/engine_weights_gbt.json \
    --model gbt

# Regime-specific (train separate weights per regime)
for regime in risk_on risk_off neutral crisis; do
    python3 bin/train_fusion_metalearner.py \
        --input results/enhanced_new_wyckoff_2024.csv \
        --output configs/optimized/engine_weights_${regime}.json \
        --regime ${regime}
done
```

**Output:**
- `configs/optimized/engine_weights_ml_logistic.json` - Learned weights
- Metrics: AUC, Precision, Recall, CV scores
- Runtime: 1-2 hours

**What to check:**
- Test AUC > 0.55 (better than random)
- Weights sum to 1.0
- Structure + Liquidity weights > 0.50 (expected for crypto markets)

---

## 3. Optuna Approach - Production Rigor (6-8 hours)

Run multi-objective optimization with full backtest validation:

```bash
# Quick test (20 trials, ~1.5 hours)
python3 bin/optimize_engine_weights.py \
    --feature-store data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
    --trials 20 \
    --output-dir results/engine_weights_test

# Full optimization (100 trials, ~8 hours - run overnight)
python3 bin/optimize_engine_weights.py \
    --feature-store data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
    --trials 100 \
    --output-dir results/engine_weights_production
```

**Output:**
- `results/engine_weights_production/optimal_weights.json` - Best weights from Pareto frontier
- `results/engine_weights_production/regime_breakdown.csv` - Performance by regime
- `results/engine_weights_production/weight_sensitivity.png` - Sensitivity analysis
- `results/engine_weights_production/optuna_study.db` - Full trial history

**What to check:**
- Pareto frontier has > 5 non-dominated solutions
- Best PF > 1.2 (meaningful improvement)
- Trade count 25-40/year
- All regimes show PF > 1.0

---

## 4. Compare Results

```bash
# Compare ML vs Optuna weights
python3 -c "
import json

# Load ML weights
with open('configs/optimized/engine_weights_ml_logistic.json') as f:
    ml_w = json.load(f)['engine_weights']

# Load Optuna weights
with open('results/engine_weights_production/optimal_weights.json') as f:
    opt_w = json.load(f)['optimal_weights']

# Compare
print('ML Weights (Logistic Regression):')
for k, v in sorted(ml_w.items(), key=lambda x: -x[1]):
    print(f'  {k:12s}: {v:.3f}')

print('\nOptuna Weights (Multi-Objective):')
for k, v in sorted(opt_w.items(), key=lambda x: -x[1]):
    print(f'  {k:12s}: {v:.3f}')

print('\nDifference:')
for k in ml_w:
    diff = opt_w[k] - ml_w[k]
    print(f'  {k:12s}: {diff:+.3f}')
"
```

**Decision criteria:**
- **Use ML weights if:** Weights are intuitive, AUC > 0.60, fast iteration needed
- **Use Optuna weights if:** Need production rigor, PF > 1.3, multi-objective optimization

---

## 5. Deploy to Production

### Option A: Simple Deployment (No Regime Awareness)

Add to your config (e.g., `configs/mvp/mvp_bull_market_v1.json`):

```json
{
  "archetypes": {
    "use_meta_fusion": true,
    "engine_weights": {
      "structure": 0.35,
      "liquidity": 0.25,
      "momentum": 0.15,
      "wyckoff": 0.15,
      "macro": 0.10
    }
  }
}
```

### Option B: Regime-Aware Deployment

```json
{
  "archetypes": {
    "use_meta_fusion": true,
    "regime_aware": true,
    "engine_weights": {
      "structure": 0.35,
      "liquidity": 0.25,
      "momentum": 0.15,
      "wyckoff": 0.15,
      "macro": 0.10
    },
    "regime_engine_weights": {
      "risk_on": {
        "structure": 0.25,
        "liquidity": 0.35,
        "momentum": 0.25,
        "wyckoff": 0.10,
        "macro": 0.05
      },
      "risk_off": {
        "structure": 0.40,
        "liquidity": 0.20,
        "momentum": 0.10,
        "wyckoff": 0.20,
        "macro": 0.10
      }
    }
  }
}
```

### Test deployment:

```bash
# Run backtest with optimized weights
python3 bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_market_v1.json \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Compare to baseline (without meta-fusion)
# Set "use_meta_fusion": false in config and run again
```

**Expected:**
- PF improvement > 5%
- Trade count stable (within ±10%)
- Max DD similar or improved

---

## 6. Monitor & Retrain

### Monitoring

Track these metrics in production:
- Rolling 6-month PF
- Weight distribution (are domain scores available?)
- Regime distribution (are weights still appropriate?)

### Retraining Schedule

- **Quarterly:** Run ML approach to check for weight drift
- **Semi-annually:** Run Optuna approach if PF degrades > 10%
- **Immediately:** If domain score coverage drops (e.g., liquidity_score becomes 50% NaN)

### Automation

```bash
# Quarterly weight check (fast)
python3 bin/train_fusion_metalearner.py \
    --input results/last_18_months_trades.csv \
    --output configs/optimized/engine_weights_current.json \
    --model logistic

# Compare to production weights
python3 -c "
import json
with open('configs/optimized/engine_weights_ml_logistic.json') as f:
    prod = json.load(f)['engine_weights']
with open('configs/optimized/engine_weights_current.json') as f:
    curr = json.load(f)['engine_weights']

# Check drift
max_drift = max(abs(curr[k] - prod[k]) for k in prod)
print(f'Max weight drift: {max_drift:.3f}')
if max_drift > 0.15:
    print('⚠ ALERT: Weight drift > 15%, consider retraining')
else:
    print('✓ Weights stable')
"
```

---

## 7. Troubleshooting

### Issue: ML approach shows AUC < 0.55

**Cause:** Features not predictive of trade outcomes

**Fix:**
- Check data quality (missing features?)
- Try GBT model (handles nonlinearities better)
- Increase train data size (need >= 100 trades)

### Issue: Optuna weights sum to != 1.0

**Cause:** Constraint not properly enforced

**Fix:**
- Check `optimize_engine_weights.py` line ~270 (normalization)
- Manually normalize: `w_new = {k: v/sum(weights.values()) for k, v in weights.items()}`

### Issue: Meta-fusion not initializing in logic_v2_adapter

**Cause:** Module import failure or config flag missing

**Fix:**
```bash
# Check if module exists
python3 -c "from engine.archetypes.meta_fusion import MetaFusionEngine; print('OK')"

# Check config has flag
grep -r "use_meta_fusion" configs/

# Enable in config
# Add: "use_meta_fusion": true
```

### Issue: Performance worse with optimized weights

**Cause:** Overfitting to test period OR domain score quality degraded

**Fix:**
- Run on OOS data (2024 H2) to check generalization
- Audit domain score coverage (check for NaN spikes)
- Fall back to equal weights or ML weights (more conservative)

---

## 8. File Reference

### Scripts
```
bin/train_fusion_metalearner.py           # ML approach (fast)
bin/optimize_engine_weights.py             # Optuna approach (rigorous)
bin/test_engine_weight_optimizer.py       # Validation test
```

### Modules
```
engine/archetypes/meta_fusion.py           # Core meta-fusion engine
engine/archetypes/logic_v2_adapter.py      # Integration point (lines 131-151, 297-306)
```

### Configs
```
configs/optimized/engine_weights_ml.json           # Placeholder ML weights
configs/optimized/meta_fusion_example.json         # Example config with regime awareness
```

### Documentation
```
ENGINE_WEIGHT_OPTIMIZATION_REPORT.md      # Full implementation report
ENGINE_WEIGHT_QUICK_START.md               # This guide
```

---

## 9. Next Steps

**Week 1:**
1. ✅ Run ML approach (2 hours)
2. ✅ Analyze learned weights
3. ✅ Run quick Optuna test (20 trials, 1.5 hours)

**Week 2:**
1. Run full Optuna optimization (100 trials, overnight)
2. Compare ML vs Optuna results
3. Select production weights

**Week 3:**
1. Test on OOS data (2024 H2)
2. Deploy to staging config
3. Monitor for 1 week

**Week 4:**
1. Deploy to production
2. Set up quarterly retraining pipeline
3. Document weight rationale in config

---

## Questions?

Refer to:
- **Architecture:** `ENGINE_WEIGHT_OPTIMIZATION_REPORT.md`
- **Code examples:** `engine/archetypes/meta_fusion.py`
- **Integration:** `engine/archetypes/logic_v2_adapter.py` (search for "META-FUSION")
- **Validation:** `bin/test_engine_weight_optimizer.py`

**Key Insight:** Engine weight optimization is a meta-optimization step AFTER archetype calibration. It finds the optimal blend of already-calibrated domain engines (structure, liquidity, momentum, etc.) to maximize overall system performance.
