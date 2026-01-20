# Optimization Tools & Workflows Guide

**Created**: 2025-11-06
**Purpose**: Reference guide for all optimization tools and validation workflows
**Status**: Tools ready, trap v1 optimization in progress (150/200 trials)

---

## 📚 Tools Overview

### 1. **Feature Caching** (`bin/cache_features_with_regime.py`)

**Purpose**: Pre-compute regime labels and features once to save 10-15 seconds per Optuna trial.

**When to use**: Before running any optimization that will execute 100+ trials.

**Usage**:
```bash
# Cache features for full period
python3 bin/cache_features_with_regime.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --output-dir data/cached

# Output: data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet
```

**Benefits**:
- Saves 10-15s per trial
- 200 trials = 33-50 minutes saved
- Consistent regime labels across trials
- Includes event calendar flags

**What it caches**:
- All multi-timeframe features
- Regime labels (RISK_ON, RISK_OFF, etc.)
- Regime probabilities for each regime
- Event calendar flags (FOMC, CPI, NFP, etc.)

---

### 2. **Trap Optimizer v1** (`bin/optuna_trap_v10.py`) - CURRENT

**Status**: 🚧 Running (150/200 trials, ~1.5h remaining)

**Configuration**:
- Train: 2022-2023
- Validation: 2024
- Sizing: Dynamic Kelly + archetype multipliers (⚠️ potential artifact)
- Objective: PF × WR
- Constraints: DD < 10%, trades > 20 (hard reject)

**Issues identified** (see OPTUNA_VALIDATION_PLAN.md):
1. Position sizing artifacts
2. Weak single-period validation
3. Objective can be gamed
4. No regime stratification

**When results complete**:
```bash
# Run 7-phase validation
python3 bin/validate_optuna_results.py \
  --study-dir results/optuna_trap_v10_full \
  --baseline-dir results/router_v10_full_2022_2024_combined \
  --output results/trap_v1_validation
```

---

### 3. **Trap Optimizer v2** (`bin/optuna_trap_v2.py`) - IMPROVED ✨

**Purpose**: Fixed version addressing all v1 issues.

**Key Improvements**:
1. **Fixed sizing**: 0.8% per trade, no Kelly, no archetype multipliers
2. **Rolling OOS**: 4 train/test windows, aggregate median
3. **Better objective**: Expectancy-based with soft penalties
4. **Regime awareness**: Reports breakdown by window
5. **Feature caching**: Loads pre-computed regime labels

**Workflow**:
```bash
# Step 1: Cache features (if not done)
python3 bin/cache_features_with_regime.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-12-31

# Step 2: Run optimization
python3 bin/optuna_trap_v2.py \
  --n-trials 200 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_v2

# Expected runtime: ~6-8 hours (4 windows × 200 trials × 6-8s per backtest)
```

**Objective Function**:
```python
# Base score: Expectancy × sqrt(trades) × stability
expectancy_R = total_pnl / (total_trades * risk_per_trade)
stability = 1.0 / (1.0 + std_R_per_trade)
base_score = expectancy_R × sqrt(total_trades) × stability

# Soft penalties
dd_penalty = 5.0 × max(0, DD - 0.12)
trade_penalty = 2.0 × max(0, 15 - trades)

# Final score (aggregated across windows)
score = median([base_score - dd_penalty - trade_penalty for each window])
```

**Rolling Windows**:
1. Train: 2022-H1 (Jan-Jun) → Test: 2022-H2 (Jul-Dec)
2. Train: 2022 → Test: 2023
3. Train: 2022 + 2023-H1 → Test: 2023-H2
4. Train: 2022-2023 → Test: 2024

**Outputs**:
- `results/optuna_trap_v2/best_params.json` - Best parameters with window breakdown
- `results/optuna_trap_v2/trials.csv` - All trials with window scores
- `results/optuna_trap_v2/trap_v2_optimized_bull.json` - Optimized bull config
- `results/optuna_trap_v2/trap_v2_optimized_bear.json` - Optimized bear config

---

### 4. **Validation Suite** (`bin/validate_optuna_results.py`)

**Purpose**: Automated 7-phase validation of Optuna results.

**Usage**:
```bash
python3 bin/validate_optuna_results.py \
  --study-dir results/optuna_trap_v10_full \
  --baseline-dir results/router_v10_full_2022_2024_combined \
  --output results/trap_validation
```

**7 Validation Phases**:

#### Phase 1: Sanity Checks
- Parameters are sensible (no pathological values)
- Trial distribution analysis
- Rejection rate check
- Duplicate configuration detection

#### Phase 2: Fixed-Size Validation (CRITICAL)
- Re-run optimized params with FIXED sizing
- Re-run baseline with FIXED sizing
- Compare: If optimized doesn't beat baseline → position sizing artifact → REJECT

#### Phase 3: Rolling OOS Validation
- 4 train/test windows
- Aggregate: median PF, min PF, worst DD
- Acceptance: median PF > 1.3, min PF > 1.0, worst DD < 12%

#### Phase 4: Regime Stratification
- Breakdown by RISK_ON, RISK_OFF, NEUTRAL, CRISIS, TRANSITIONAL
- Calculate PF, WR, PNL for each regime
- Acceptance: PF > 1.0 in 4/5 regimes

#### Phase 5: Trade-Level Diagnostics
- Compare optimized vs baseline trade-by-trade
- Identify source of improvements (entries, exits, stops?)
- Flag if improvements come from sizing artifacts

#### Phase 6: Session Analysis
- Breakdown by ASIA, EUROPE, US sessions
- Check for session-specific degradation
- Flag if one session loses heavily

#### Phase 7: Slippage & Cost Sensitivity
- Add 3bp per trade + 1bp stop slippage
- Recalculate metrics
- Acceptance: PF with costs > 1.2

**Decision Criteria**:
```
ACCEPT if:
  ✅ Fixed-size validation shows >10% PF improvement
  ✅ Rolling OOS shows median PF > 1.3, min > 1.0
  ✅ 4/5 regimes have PF > 1.0
  ✅ Trade diagnostics show entry/exit improvements
  ✅ Slippage test maintains PF > 1.2

REJECT if:
  ❌ Fixed-size validation fails
  ❌ One period dominates (e.g., 2024 only)
  ❌ Regime breakdown shows catastrophic losses
  ❌ Improvements come from sizing artifacts

CONDITIONAL (re-run) if:
  ⚠️ Passes most but fails regime robustness
  ⚠️ Suspicious train/val split but otherwise ok
```

**Outputs**:
- `results/trap_validation/VALIDATION_REPORT.md` - Full markdown report
- `results/trap_validation/DECISION.json` - Accept/Reject decision with reasoning

---

## 🔄 Complete Optimization Workflow

### Standard Workflow (v2 Optimizer)

```bash
# 1. Cache features (once per period)
python3 bin/cache_features_with_regime.py \
  --asset BTC --start 2022-01-01 --end 2024-12-31

# 2. Run optimization with v2 (fixed sizing + rolling OOS)
python3 bin/optuna_trap_v2.py \
  --n-trials 200 \
  --cache data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet \
  --output results/optuna_trap_v2

# 3. Review results
cat results/optuna_trap_v2/best_params.json

# 4. Validate on full period with optimized config
python3 bin/backtest_router_v10_full.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --bull-config results/optuna_trap_v2/trap_v2_optimized_bull.json \
  --bear-config results/optuna_trap_v2/trap_v2_optimized_bear.json \
  --output results/trap_v2_validation

# 5. Compare to baseline
# Baseline: +$1,140, 125 trades, WR 50.4%, PF 1.42
# Check if optimized shows improvement
```

### Legacy Workflow (v1 Optimizer - requires manual validation)

```bash
# 1. Wait for v1 optimization to complete
ps aux | grep optuna_trap_v10.py

# 2. Run validation suite
python3 bin/validate_optuna_results.py \
  --study-dir results/optuna_trap_v10_full \
  --output results/trap_v1_validation

# 3. Review decision
cat results/trap_v1_validation/DECISION.json

# 4. If ACCEPT: Use optimized configs
# 5. If REJECT: Re-run with v2
# 6. If CONDITIONAL: Follow recommendations in report
```

---

## 🎯 Optimization Best Practices

### 1. Always Use Fixed Sizing First

**Why**: Isolates entry quality from position sizing effects.

**How**:
- Disable Kelly: `kelly_fraction = 1.0`, no dynamic adjustment
- Disable confidence scaling: `confidence_scaling = False`
- Disable archetype multipliers: `archetype_quality_weight = 0.0`
- Use fixed risk: `base_risk_per_trade_pct = 0.8`

**After locking entry/exit params**: Re-enable and optimize sizing separately.

### 2. Use Rolling OOS Validation

**Why**: Single period can be unrepresentative.

**How**:
- Multiple train/test windows (minimum 3, ideally 4+)
- Aggregate: median or min for robustness
- Report worst-case metrics
- Check consistency across periods

### 3. Prefer Expectancy-Based Objectives

**Why**: PF × WR can be gamed by few fat wins.

**Better options**:
```python
# Option A: Expectancy with stability
score = (expectancy_R × sqrt(trades)) × stability - soft_penalties

# Option B: Calmar-like
score = (Return / MaxDD) - λ × Volatility

# Option C: Multi-objective Pareto
# (more complex, use optuna.multi_objective)
```

### 4. Add Regime Stratification

**Why**: Ensure robustness across all market conditions.

**How**:
- Report PF/WR breakdown by regime
- Minimize across regimes (not just aggregate)
- Reject if any regime shows catastrophic losses

### 5. Cache Features for Speed

**Why**: Saves 10-15 seconds per trial.

**Impact**:
- 200 trials = 33-50 minutes saved
- 1000 trials = 2.8-4.2 hours saved

**How**: Use `bin/cache_features_with_regime.py` before optimization.

---

## 📊 Current Baselines (Reference)

### Trap Archetype (Current - BROKEN)
- Trades: 104 (83% of all trades)
- Win Rate: 46%
- Profit Factor: 0.88
- Avg Win: +$43
- Avg Loss: -$78
- R:R: 0.55:1
- **Total PNL: -$353** ❌

**Target after optimization**:
- Win Rate: 55%+
- Profit Factor: 1.5+
- Avg Loss: < -$50
- R:R: 1.5:1+
- **Total PNL: +$400-600/year** ✅

### Order Block Retest (GOLDMINE)
- Trades: 10 (8% of all trades)
- Win Rate: 90%
- Profit Factor: 15.2
- Avg Win: +$152
- Avg Loss: -$25
- **Total PNL: +$1,518** ✅

**Target after scaling**:
- Trades: 30-40 (10-13/year)
- Win Rate: 70%+ (maintain high precision)
- **Total PNL: +$2,000-3,000/year** ✅

---

## 🔮 Next Steps

### Immediate (When Trap v1 Completes)
1. Run 7-phase validation on trap v1 results
2. Make decision: ACCEPT / REJECT / RE-RUN with v2
3. If re-running needed, execute trap v2 optimization

### Phase 1 Remaining
4. Update bear optimizer to v2 pattern
5. Create OB retest optimizer (requires code changes)
6. Test vacuum_grab baseline, optimize if promising
7. Router/fusion global optimization
8. Exit optimization (TPs, partials, BE, trailing)
9. Position sizing optimization (LAST, after entries/exits locked)

### Phase 2+ (See MASTER_OPTIMIZATION_ROADMAP.md)
- MLP quality multiplier training
- Per-archetype quality classifiers
- Temporal transformer for phase detection
- RL self-learning layer (Specter)

---

**Generated**: 2025-11-06
**Status**: Tools ready, awaiting trap v1 completion
**Estimated time to v1 results**: ~1.5-2 hours
**Next milestone**: Execute validation and decide on v2 re-run
