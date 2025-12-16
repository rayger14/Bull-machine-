# Regime-Aware Optimization Quick Reference

**Companion to:** [REGIME_AWARE_OPTIMIZATION_FRAMEWORK.md](./REGIME_AWARE_OPTIMIZATION_FRAMEWORK.md)

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    REGIME-AWARE SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. REGIME CLASSIFICATION                                       │
│     ┌──────────────┐         ┌──────────────┐                 │
│     │ Macro Data   │────────▶│ GMM Model    │                 │
│     │ (VIX, DXY,   │         │ 4 Clusters   │                 │
│     │  funding...)  │         └──────┬───────┘                 │
│     └──────────────┘                 │                          │
│                                      ▼                          │
│                         ┌────────────────────────┐             │
│                         │ Regime Label Per Bar   │             │
│                         │ - risk_on    (42%)    │             │
│                         │ - neutral    (28%)    │             │
│                         │ - risk_off   (25%)    │             │
│                         │ - crisis     (5%)     │             │
│                         └──────────┬─────────────┘             │
│                                    │                            │
│  2. ARCHETYPE ROUTING               ▼                          │
│     ┌─────────────────────────────────────────────┐           │
│     │ Archetype → Allowed Regimes Mapping         │           │
│     │                                              │           │
│     │  S1: [risk_off, crisis]                     │           │
│     │  S2: [risk_off, neutral]                    │           │
│     │  S4: [risk_off, neutral]                    │           │
│     │  S5: [risk_on, neutral]                     │           │
│     │  A-M: [risk_on, neutral]                    │           │
│     └──────────────┬──────────────────────────────┘           │
│                    │                                            │
│  3. REGIME-FILTERED BACKTEST        ▼                         │
│     ┌────────────────────────────────────────┐               │
│     │ For bar in history:                     │               │
│     │   if bar.regime not in allowed_regimes: │               │
│     │     skip  # Don't evaluate archetype    │               │
│     │   else:                                  │               │
│     │     thresholds = get_regime_thresholds() │               │
│     │     evaluate_archetype(bar, thresholds)  │               │
│     └────────────┬───────────────────────────┘               │
│                  │                                             │
│  4. OPTIMIZATION │                ▼                           │
│     ┌─────────────────────────────────────────┐              │
│     │ Per Archetype-Regime Pair:              │              │
│     │                                          │              │
│     │ Optuna Multi-Objective:                 │              │
│     │   - Maximize Profit Factor              │              │
│     │   - Maximize Event Recall               │              │
│     │   - Minimize Trades/Year                │              │
│     │                                          │              │
│     │ Output: Pareto Frontier                 │              │
│     └─────────────┬───────────────────────────┘              │
│                   │                                            │
│  5. WALK-FORWARD  │              ▼                            │
│     ┌──────────────────────────────────────────┐             │
│     │ Rolling Windows (12mo train, 3mo test):  │             │
│     │                                           │             │
│     │ Train: Q1-Q3 (risk_off bars only)       │             │
│     │ Test:  Q4     (risk_off bars only)       │             │
│     │                                           │             │
│     │ OOS Metrics: Consistency, PF, Sharpe     │             │
│     └─────────────┬─────────────────────────────┘             │
│                   │                                            │
│  6. PORTFOLIO     │             ▼                             │
│     ┌──────────────────────────────────────────┐             │
│     │ Weight(arch, regime) =                   │             │
│     │   regime_freq * PF * risk_adj            │             │
│     │                                           │             │
│     │ Dynamic adjustment every week            │             │
│     └──────────────────────────────────────────┘             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Critical Questions Answered

### Q1: How to handle regime transitions mid-trade?

**Phase 1 Answer: IGNORE**
- Keep original entry regime for entire trade
- Simplest, most consistent approach
- Prevents premature exits from regime noise

**Future: Regime-conditional exits**
- Exit if regime becomes incompatible
- Adjust stops based on new regime favorability

### Q2: Should we allow completely different thresholds per regime?

**Answer: YES (Unconstrained)**
- Crisis fusion_threshold = 0.45
- Risk_off fusion_threshold = 0.65
- Maximum flexibility to capture regime-specific patterns
- Add monotonicity check as validation, not hard constraint

### Q3: What if test window has no regime bars?

**Answer: SKIP WINDOW**
- Accept fewer validation windows for rare regimes (crisis)
- Statistically honest - no contamination
- Alternative: Extend test period to 6 months (Phase 3)

### Q4: How to balance PF vs event recall?

**Answer: Pareto Frontier Analysis**
```python
# Three selection strategies available:

1. Balanced (RECOMMENDED):
   score = 0.50 * normalize(PF) +
           0.30 * normalize(recall) -
           0.20 * normalize(trades/yr)

2. Conservative:
   Sort by (PF desc, trades asc), pick #1

3. Event-Focused:
   Filter PF > 1.5, pick highest recall
```

### Q5: How to adjust weights when regime distribution shifts?

**Answer: Rolling Regime Estimation**
```python
# Weekly monitoring:
recent_dist = calculate_regime_dist(last_90_days)
forecast_dist = forecast_next_30_days(recent_dist)

# Adjust weights if shift > 30%:
adjusted_weight = base_weight * (forecast_freq / historical_freq)
bounded_adjustment = clip(adjusted_weight, 0.5, 2.0)
```

---

## Config Structure Cheat Sheet

### Legacy Config (BEFORE)
```json
{
  "archetypes": {
    "S1": {
      "thresholds": {
        "fusion_threshold": 0.65,
        "crisis_composite_min": 0.35
      }
    }
  }
}
```

### Regime-Aware Config (AFTER)
```json
{
  "regime": {
    "enabled": true,
    "model_path": "models/regime_classifier_gmm.pkl",
    "feature_order": ["VIX", "DXY", "funding", "..."],
    "min_confidence": 0.6
  },

  "archetypes": {
    "S1": {
      "allowed_regimes": ["risk_off", "crisis"],

      "thresholds": {
        "fusion_threshold": 0.65,          // Global default
        "crisis_composite_min": 0.35,

        "regime_thresholds": {
          "risk_off": {
            "fusion_threshold": 0.65,
            "crisis_composite_min": 0.35
          },
          "crisis": {
            "fusion_threshold": 0.55,      // More lenient
            "crisis_composite_min": 0.45   // Higher bar
          }
        }
      }
    }
  }
}
```

---

## Common Operations

### 1. Label Historical Bars with Regime
```python
from engine.context.regime_classifier import RegimeClassifier

rc = RegimeClassifier.load(
    model_path='models/regime_classifier_gmm.pkl',
    feature_order=['VIX', 'DXY', 'funding', ...]
)

labeled_bars = rc.label_historical_bars(
    bars_df=ohlcv_df,
    macro_df=macro_features_df,
    min_confidence=0.6
)

# Result: bars_df with added columns
#   - regime: str
#   - regime_confidence: float
#   - regime_duration: int
#   - regime_transition: bool
```

### 2. Run Regime-Filtered Backtest
```python
from engine.backtest.regime_aware_backtest import RegimeAwareBacktest

backtest = RegimeAwareBacktest(config)

result = backtest.run(
    bars_df=ohlcv_df,
    macro_df=macro_features_df,
    archetype='S1',
    thresholds={
        'fusion_threshold': 0.65,
        'crisis_composite_min': 0.35,
        'regime_thresholds': {
            'crisis': {'fusion_threshold': 0.55}
        }
    },
    regime_filter=['risk_off', 'crisis']  # S1 only trades in these regimes
)

print(f"Total trades: {result.total_trades}")
print(f"Profit factor: {result.profit_factor}")
print(f"Regime metrics: {result.regime_metrics}")
```

### 3. Optimize Archetype-Regime Pair
```python
from engine.optimization.regime_aware_objective import optimize_archetype_regime

study = optimize_archetype_regime(
    archetype='S1',
    regime='crisis',
    bars_df=ohlcv_df,
    macro_df=macro_features_df,
    config=base_config,
    n_trials=200,
    storage='sqlite:///optuna_s1_crisis.db'
)

# Get Pareto frontier
best_trials = study.best_trials

print(f"Pareto frontier: {len(best_trials)} solutions")
for trial in best_trials[:5]:
    pf = -trial.values[0]
    recall = -trial.values[1]
    trades_yr = trial.values[2]
    print(f"  PF={pf:.2f}, Recall={recall:.1%}, Trades/yr={trades_yr:.1f}")
```

### 4. Select Best Threshold from Pareto
```python
from engine.optimization.pareto_selector import ParetoSelector

# Balanced approach (50% PF, 30% recall, 20% trade frequency)
best_trial = ParetoSelector.select_balanced(study.best_trials)

best_thresholds = best_trial.params
print(f"Selected thresholds: {best_thresholds}")

# Conservative approach (maximize PF, minimize trades)
conservative_trial = ParetoSelector.select_conservative(study.best_trials)
```

### 5. Run Walk-Forward Validation
```python
from engine.validation.regime_aware_walk_forward import RegimeAwareWalkForward

wf_validator = RegimeAwareWalkForward(
    train_months=12,
    test_months=3,
    step_months=1,
    min_train_bars_per_regime=200,
    min_test_bars_per_regime=50
)

wf_result = wf_validator.validate_archetype_regime(
    archetype='S1',
    regime='crisis',
    bars_df=ohlcv_df,
    macro_df=macro_features_df,
    config=base_config
)

print(f"OOS Metrics:")
print(f"  Avg PF: {wf_result.avg_oos_pf:.2f}")
print(f"  Avg Sharpe: {wf_result.avg_oos_sharpe:.2f}")
print(f"  Consistency: {wf_result.oos_metrics['consistency_score']:.2%}")
print(f"  Windows validated: {wf_result.n_windows}")
```

### 6. Calculate Regime-Weighted Portfolio
```python
from engine.portfolio.regime_weighted_portfolio import calculate_regime_weighted

regime_distribution = {
    'regime_fractions': {
        'risk_on': 0.42,
        'neutral': 0.28,
        'risk_off': 0.25,
        'crisis': 0.05
    }
}

archetype_performance = {
    'S1': {
        'risk_off': {'pf': 2.3, 'sharpe': 1.2},
        'crisis': {'pf': 3.5, 'sharpe': 1.8}
    },
    'S5': {
        'risk_on': {'pf': 2.0, 'sharpe': 1.1},
        'neutral': {'pf': 1.6, 'sharpe': 0.7}
    }
}

weights = calculate_regime_weighted(
    archetype_performance,
    regime_distribution
)

for (arch, regime), weight in weights.items():
    print(f"{arch} in {regime}: {weight:.1%}")
```

---

## Validation Checklist

### Pre-Optimization
- [ ] Regime classifier labels all bars (no errors)
- [ ] Regime periods align with known events (LUNA, FTX in crisis)
- [ ] Each archetype has >= 200 bars per allowed regime
- [ ] Config has `regime_thresholds` section for all archetypes

### Post-Optimization
- [ ] Pareto frontier has >= 5 non-dominated solutions
- [ ] Selected thresholds pass monotonicity check (crisis < risk_off < neutral)
- [ ] PF variance across regimes < 3x (consistency check)
- [ ] Known event recall >= 80% for crisis archetypes (S1, S2)

### Post-Walk-Forward
- [ ] OOS consistency score > 0.6 (low overfitting)
- [ ] Positive windows >= 60% (majority profitable)
- [ ] Avg OOS PF > 1.5 (profitable after costs)
- [ ] >= 4 walk-forward windows validated per regime

### Pre-Production
- [ ] A/B test shows >= 15% Sharpe improvement vs baseline
- [ ] Paper trading for 2 weeks with no errors
- [ ] Regime classifier updates in < 1 second (real-time)
- [ ] Monitoring dashboard correctly displays regime and weights

---

## Known Pitfalls

### 1. Optimizing on Too Few Bars
**Problem:** Crisis regime only 50 bars → overfitting
**Solution:** Aggregate crisis bars across multiple years, accept non-stationarity

### 2. Ignoring Regime Transitions
**Problem:** Hold S1 position from crisis into risk_on → losses
**Solution (Phase 1):** Accept temporary misalignment, plan Phase 3 exit logic

### 3. Regime Misclassification
**Problem:** GMM labels crisis as neutral (low confidence)
**Solution:** Use manual overrides for known events, fallback to neutral if confidence < 0.6

### 4. Empty Test Windows
**Problem:** Test period has 0 crisis bars → cannot validate
**Solution:** Skip window (statistically honest), extend test period to 6mo if needed

### 5. Regime Distribution Shift
**Problem:** 2022 was 40% crisis, 2024 is 5% crisis → weights obsolete
**Solution:** Weekly monitoring, dynamic adjustment if shift > 30%

---

## Metrics Glossary

| Metric | Definition | Target |
|--------|-----------|--------|
| **Profit Factor** | (Total Wins) / (Total Losses) | > 1.5 |
| **Event Recall** | % of known events captured | > 80% (crisis) |
| **Consistency Score** | 1 - (CV of PF across windows) | > 0.6 |
| **OOS Sharpe** | Out-of-sample Sharpe ratio | > 0.8 |
| **Regime Coverage** | % of regime bars with position | 5-15% |
| **Positive Windows** | % of walk-forward windows with PF > 1 | > 60% |

---

## File Paths Reference

```
Bull-machine-/
├── engine/
│   ├── context/
│   │   ├── regime_classifier.py          # GMM-based regime classification
│   │   └── regime_policy.py              # Regime-based parameter adjustments
│   │
│   ├── backtest/
│   │   └── regime_aware_backtest.py      # NEW: Regime-filtered backtesting
│   │
│   ├── optimization/
│   │   ├── threshold_manager.py          # NEW: Hierarchical threshold loading
│   │   ├── regime_aware_objective.py     # NEW: Multi-objective Optuna
│   │   └── pareto_selector.py            # NEW: Pareto frontier selection
│   │
│   ├── validation/
│   │   └── regime_aware_walk_forward.py  # NEW: Walk-forward with regimes
│   │
│   └── portfolio/
│       └── regime_weighted_portfolio.py  # NEW: Portfolio weighting
│
├── configs/
│   ├── mvp/
│   │   ├── mvp_bull_market_v1.json       # Legacy config
│   │   └── mvp_regime_aware_v1.json      # NEW: Regime-aware config
│   │
│   └── regime/
│       ├── regime_classifier_config.json # Regime model config
│       └── regime_policy.json            # Regime adjustment policy
│
├── bin/
│   ├── optimize_regime_pairs.py          # NEW: CLI for optimization
│   ├── validate_regime_aware.py          # NEW: CLI for validation
│   └── migrate_config_to_regime.py       # NEW: Config migration tool
│
├── models/
│   └── regime_classifier_gmm.pkl         # Trained GMM model
│
└── docs/
    ├── REGIME_AWARE_OPTIMIZATION_FRAMEWORK.md  # THIS DOCUMENT
    └── REGIME_AWARE_QUICK_REFERENCE.md         # YOU ARE HERE
```

---

## Next Steps

1. **Implement Phase 1 (Foundation):**
   ```bash
   # Add label_historical_bars() to RegimeClassifier
   python -c "from engine.context.regime_classifier import RegimeClassifier; \
              rc = RegimeClassifier.load('models/regime_classifier_gmm.pkl', ...); \
              labeled = rc.label_historical_bars(bars_df, macro_df)"
   ```

2. **Validate Regime Labeling:**
   ```bash
   # Check LUNA event is labeled as crisis
   python bin/validate_regime_labels.py --event LUNA --expected crisis
   ```

3. **Run First Regime-Filtered Backtest:**
   ```bash
   # Backtest S1 on crisis bars only
   python bin/backtest_regime_aware.py \
       --archetype S1 \
       --regime crisis \
       --start 2022-01-01 \
       --end 2022-12-31
   ```

4. **Optimize S1 on Crisis Regime:**
   ```bash
   # 200 trials, multi-objective
   python bin/optimize_regime_pairs.py \
       --archetype S1 \
       --regime crisis \
       --trials 200 \
       --output configs/optimized_s1_crisis.json
   ```

5. **Walk-Forward Validation:**
   ```bash
   # Validate S1 crisis thresholds out-of-sample
   python bin/validate_regime_aware.py \
       --archetype S1 \
       --regime crisis \
       --config configs/optimized_s1_crisis.json
   ```

---

## Support

**Questions?** See main framework document: [REGIME_AWARE_OPTIMIZATION_FRAMEWORK.md](./REGIME_AWARE_OPTIMIZATION_FRAMEWORK.md)

**Implementation Issues?** Check:
- Unit tests: `tests/unit/test_regime_aware_backtest.py`
- Integration tests: `tests/integration/test_regime_optimization.py`
- Example notebooks: `examples/regime_aware_optimization.ipynb`
