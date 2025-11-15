# Optuna v9 Improved Search - Implementation Summary

## Problem Statement

**Previous Issue (v8):**
- 294 trials → 112 unique configs (62% duplicates)
- Sample/feature ratio: 5.9x (below 10-20x target)
- Meta-optimizer v2 Test R²: -0.289 (insufficient for predictions)

**Root Cause:**
TPE sampler converged too aggressively to high-performing region, creating correlated duplicates rather than diverse exploration.

## Solution: 8-Point Improvement Plan

### 1. Sobol → Multivariate TPE Phasing
**Phase 1 (120 trials):** QMCSampler with Sobol sequences for space-filling exploration
- Quasi-Monte Carlo sampling ensures uniform coverage
- No mode collapse during exploration
- Sets foundation for TPE exploitation

**Phase 2 (240 trials):** Multivariate TPESampler for guided exploitation
- Models parameter interactions (not independent)
- Uses all 120 Sobol trials as startup
- Focuses on high-performing regions discovered in Phase 1

```python
# Phase 1: Exploration
study = optuna.create_study(
    direction='maximize',
    sampler=QMCSampler(qmc_type="sobol", scramble=True, seed=42)
)
study.optimize(objective, n_trials=120)

# Phase 2: Exploitation
tpe_sampler = TPESampler(
    seed=42,
    multivariate=True,
    group=True,
    n_startup_trials=0
)
study.sampler = tpe_sampler
study.optimize(objective, n_trials=240)
```

### 2. Dirichlet Sampling for Fusion Weights
**Problem:** Independent float sampling + normalization violates simplex constraint
**Solution:** Sample gamma variates and normalize (equivalent to Dirichlet)

```python
gamma_wyckoff = trial.suggest_float('w_wyckoff_gamma', 0.1, 3.0, log=True)
gamma_liquidity = trial.suggest_float('w_liquidity_gamma', 0.1, 3.0, log=True)
gamma_momentum = trial.suggest_float('w_momentum_gamma', 0.1, 3.0, log=True)

total = gamma_wyckoff + gamma_liquidity + gamma_momentum
w_wyckoff = gamma_wyckoff / total
w_liquidity = gamma_liquidity / total
w_momentum = gamma_momentum / total
```

**Benefit:** Proper exploration of weight simplex, prevents clustering at simplex vertices

### 3. Expanded Parameter Ranges (2-3x Wider)

| Parameter | v8 Range | v9 Range | Expansion |
|-----------|----------|----------|-----------|
| final_fusion_floor | 0.30-0.38 | 0.18-0.45 | 2.4x |
| min_liquidity | 0.10-0.22 | 0.06-0.25 | 1.6x |
| neutralize_fusion_drop | 0.08-0.18 | 0.05-0.22 | 1.7x |
| neutralize_min_bars | 5-12 | 3-15 | 1.6x |
| size_min | 0.50-0.75 | 0.40-0.85 | 1.8x |
| size_max | 1.05-1.40 | 0.95-1.60 | 1.9x |
| B_fusion | 0.32-0.42 | 0.25-0.50 | 2.5x |
| C_fusion | 0.40-0.50 | 0.35-0.60 | 2.5x |
| trail_atr_mult | 0.85-1.25 | 0.65-1.45 | 2.0x |
| max_bars | 60-96 | 48-120 | 2.0x |

**Benefit:** Allows discovery of genuinely different strategies in unexplored parameter space

### 4. Cross-Regime Objective Function
**Problem:** Single-regime optimization (2024 only) led to overfitting
**Solution:** Robust PF combining both bull and bear regimes

```python
pf_2024 = run_backtest(config, '2024-01-01', '2024-12-31')
pf_22_23 = run_backtest(config, '2022-01-01', '2023-12-31')

# Robust objective: 70% worst-case + 30% average
min_pf = min(pf_2024, pf_22_23)
avg_pf = (pf_2024 + pf_22_23) / 2.0
robust_pf = min_pf * 0.7 + avg_pf * 0.3
```

**Benefit:** Promotes configs that work across different market regimes

### 5. Explicit Config Deduplication
**Implementation:** MD5 hashing with real-time pruning

```python
config_hash = hashlib.md5(
    json.dumps(trial_params, sort_keys=True).encode()
).hexdigest()

if config_hash in SEEN_CONFIGS:
    raise optuna.TrialPruned()

SEEN_CONFIGS.add(config_hash)
```

**Benefit:** Immediate detection and rejection of duplicate configs

### 6. Hard Guardrails on Drawdown
**2024 Regime (Bull):**
- Max DD: 3.0%
- Min trades: 10

**2022-2023 Regime (Bear/Ranging):**
- Max DD: 8.0%
- Min trades: 15

**Benefit:** Prevents exploration of high-risk parameter regions

### 7. Conditional Parameter Sampling
Currently implemented: All parameters always sampled
**Future Enhancement:** Only sample parameters when relevant (e.g., neutralization params only if enabled)

### 8. Real-Time Deduplication Tracking
All trials logged with config hash for post-hoc analysis:
- `BTC_all_trials.csv` includes config_hash column
- Enables deduplication rate tracking
- Facilitates debugging of parameter space coverage

## Expected Outcomes

### Diversity Improvement
- **Target:** >250 unique configs from 360 trials (<31% duplication)
- **Previous:** 112 unique from 294 trials (62% duplication)
- **Improvement:** >2x unique configs with genuine diversity

### Sample/Feature Ratio
- **Target:** 13-16x (250-300 unique / 19 features)
- **Previous:** 5.9x (112 / 19)
- **Improvement:** Sufficient for meta-optimizer training

### Meta-Optimizer v3 Quality
- **Target:** Test R² > 0.3 (positive predictive power)
- **Previous:** Test R² = -0.289 (worse than mean baseline)
- **Improvement:** Reliable config → performance predictions

### Cross-Regime Robustness
- **Target:** Configs with PF > 2.0 on both 2024 and 2022-2023
- **Benefit:** Generalization beyond single regime

## Runtime & Monitoring

**Expected Runtime:** 18-24 hours
- 360 trials × 2 backtests/trial × ~2 min/backtest ≈ 1440 min = 24 hours

**Monitor Progress:**
```bash
# Live log
tail -f reports/optuna_v9_improved/optimization.log

# Trial count
wc -l reports/optuna_v9_improved/BTC_all_trials.csv

# Deduplication rate
cut -d',' -f11 reports/optuna_v9_improved/BTC_all_trials.csv | sort | uniq | wc -l

# Best result so far
cat reports/optuna_v9_improved/BTC_best_score.txt
```

**Phase Transition Checkpoint:**
After 120 trials, check for:
- Best robust PF from Sobol phase
- Deduplication rate (<20% expected)
- Coverage of parameter space edges

## Next Steps (After Completion)

1. **Consolidation:**
   ```bash
   python3 bin/consolidate_trials.py \
     --asset BTC \
     --output reports/ml/config_training_data_v3.csv \
     --min-rows 250
   ```

2. **Meta-Optimizer v3 Training:**
   ```bash
   PYTHONHASHSEED=0 python3 bin/train/train_config_optimizer.py \
     --data reports/ml/config_training_data_v3.csv \
     --target robust_pf \
     --output models/btc_config_optimizer_v3.pkl \
     --test-size 0.25
   ```

3. **Validation:**
   - Target: Test R² > 0.3
   - Generate 12 ML-suggested configs
   - Cross-validate top 5 on both regimes

## Key Improvements Summary

| Aspect | v8 (Previous) | v9 (Improved) |
|--------|---------------|---------------|
| Sampler | TPE only | Sobol (120) → TPE (240) |
| Weight Sampling | Independent floats | Dirichlet (gamma variates) |
| Parameter Ranges | Narrow (baseline) | 2-3x wider |
| Objective | Single regime (2024 PF) | Cross-regime robust PF |
| Deduplication | Post-hoc | Real-time MD5 hashing |
| Guardrails | Single regime | Dual regime (DD limits) |
| Expected Diversity | 38% unique | >69% unique |
| Sample/Feature Ratio | 5.9x | 13-16x |

## Technical References

**Sobol Sequences:**
- Quasi-random low-discrepancy sequences
- Better space coverage than pseudo-random
- Reference: Sobol, I.M. (1967). "On the distribution of points in a cube"

**Dirichlet Distribution:**
- Natural distribution over probability simplexes
- Ensures uniform exploration of weight combinations
- Reference: Wikipedia - Dirichlet distribution

**Multivariate TPE:**
- Models correlations between parameters
- More efficient than independent TPE
- Reference: Optuna documentation - TPESampler

**Cross-Validation Best Practices:**
- Regime diversity prevents overfitting
- Min/avg blending balances robustness and performance
- Reference: Bergstra & Bengio (2012) - Random Search for Hyper-Parameter Optimization
