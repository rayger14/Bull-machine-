# Regime-Aware Optimization Implementation Checklist

**Status:** READY TO IMPLEMENT
**Est. Timeline:** 9 weeks (5 phases)
**Owner:** Optimization Team

---

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Regime Classifier Enhancement
- [ ] Add `label_historical_bars()` method to `RegimeClassifier`
  - Input: `bars_df` (OHLCV), `macro_df` (features)
  - Output: `bars_df` with `regime`, `regime_confidence`, `regime_duration`, `regime_transition` columns
  - Performance: < 5 seconds for 2 years of data

- [ ] Add `get_regime_periods()` method
  - Extract contiguous regime periods
  - Calculate metadata (duration, volatility, drawdown, events)

- [ ] Unit tests for regime labeling
  ```python
  test_label_single_bar()
  test_label_historical_series()
  test_regime_transition_detection()
  test_manual_regime_override()
  ```

**Acceptance Criteria:**
- ✅ Regime classifier labels 2022-2023 BTC data without errors
- ✅ LUNA crash (2022-05-09) labeled as `crisis`
- ✅ FTX collapse (2022-11-08) labeled as `crisis`
- ✅ Performance benchmark: < 5s for 730 days

**Deliverable:** `engine/context/regime_classifier.py` (enhanced)

---

### 1.2 Regime-Aware Backtest Engine
- [ ] Create `engine/backtest/regime_aware_backtest.py`
  - `ARCHETYPE_REGIMES` mapping (S1 → [risk_off, crisis], etc.)
  - `RegimeAwareBacktest.run()` method with regime filtering
  - `_get_regime_thresholds()` for hierarchical threshold loading
  - `_calculate_metrics()` with regime-stratified output

- [ ] Define `BacktestResult` dataclass
  ```python
  @dataclass
  class BacktestResult:
      archetype: str
      total_trades: int
      profit_factor: float
      win_rate: float
      regime_metrics: Dict[str, RegimeMetrics]
      crisis_event_capture: Dict[str, bool]
  ```

- [ ] Integration tests
  ```python
  test_regime_filtered_backtest()  # S1 on crisis bars only
  test_regime_threshold_loading()  # Crisis vs risk_off thresholds
  test_empty_regime_handling()     # Skip if no regime bars
  ```

**Acceptance Criteria:**
- ✅ S1 backtest on crisis bars produces different results than full backtest
- ✅ Regime-specific thresholds load correctly (crisis < risk_off)
- ✅ Empty regime windows are skipped without errors

**Deliverable:** `engine/backtest/regime_aware_backtest.py` (new)

---

### 1.3 Threshold Manager
- [ ] Create `engine/optimization/threshold_manager.py`
  - `ThresholdManager.get_thresholds(archetype, regime)`
  - Hierarchical loading: regime_specific → global → default
  - `validate_thresholds()` for consistency checks

- [ ] Config migration tool
  ```python
  migrate_config_to_regime_aware(
      legacy_config='configs/mvp/mvp_bull_market_v1.json',
      output='configs/mvp/mvp_regime_aware_v1.json'
  )
  ```

**Acceptance Criteria:**
- ✅ Thresholds load with correct precedence (regime > global)
- ✅ Validation detects missing regime parameters
- ✅ Migration tool converts legacy configs without data loss

**Deliverable:** `engine/optimization/threshold_manager.py` (new)

---

## Phase 2: Optimization (Weeks 3-4)

### 2.1 Multi-Objective Function
- [ ] Create `engine/optimization/regime_aware_objective.py`
  - `RegimeAwareObjective.__call__()` returns tuple: (neg_pf, neg_recall, trades_per_year)
  - `_suggest_thresholds()` for S1, S2, S4, S5 parameter spaces
  - `_calculate_event_recall()` for known events (LUNA, FTX, etc.)

- [ ] Known event definitions
  ```python
  KNOWN_CRISIS_EVENTS = {
      'LUNA': {'date': '2022-05-09', 'regime': 'crisis'},
      'June18': {'date': '2022-06-18', 'regime': 'crisis'},
      'FTX': {'date': '2022-11-08', 'regime': 'crisis'},
  }
  ```

**Deliverable:** `engine/optimization/regime_aware_objective.py` (new)

---

### 2.2 Optuna Study Runner
- [ ] Create `bin/optimize_regime_pairs.py`
  ```bash
  python bin/optimize_regime_pairs.py \
      --archetype S1 \
      --regime crisis \
      --trials 200 \
      --storage sqlite:///optuna_s1_crisis.db \
      --output configs/optimized_s1_crisis.json
  ```

- [ ] Implement parallel optimization for all archetype-regime pairs
  ```python
  optimize_all_regime_pairs(
      archetypes=['S1', 'S2', 'S4', 'S5'],
      bars_df=ohlcv_df,
      macro_df=macro_features_df,
      n_trials_per_pair=200,
      parallel=True
  )
  ```

**Acceptance Criteria:**
- ✅ 200 trials per pair complete in < 2 hours
- ✅ Pareto frontier contains >= 5 non-dominated solutions
- ✅ Storage uses separate SQLite DB per pair (avoid concurrency issues)

**Deliverable:** `bin/optimize_regime_pairs.py` (new)

---

### 2.3 Pareto Selector
- [ ] Create `engine/optimization/pareto_selector.py`
  - `select_balanced()`: 50% PF, 30% recall, 20% trades
  - `select_conservative()`: Max PF, min trades
  - `select_event_focused()`: Max recall with PF > 1.5

**Deliverable:** `engine/optimization/pareto_selector.py` (new)

---

### 2.4 Initial Optimization Run
- [ ] Optimize S1 on crisis regime
  - Target: PF > 2.0, Recall > 80%, Trades/yr < 10
  - Validate: Different thresholds than risk_off regime

- [ ] Optimize S1 on risk_off regime
  - Target: PF > 1.8, Trades/yr < 20

- [ ] Optimize S4 on risk_off and neutral
- [ ] Optimize S5 on risk_on and neutral

**Acceptance Criteria:**
- ✅ S1 crisis fusion_threshold < S1 risk_off (more lenient in crisis)
- ✅ Event recall: LUNA captured in >= 80% of trials
- ✅ Selected thresholds achieve PF > 1.5 on validation set

**Deliverable:** `configs/optimized_s1_crisis.json`, `configs/optimized_s1_risk_off.json`

---

## Phase 3: Walk-Forward Validation (Weeks 5-6)

### 3.1 Walk-Forward Engine
- [ ] Create `engine/validation/regime_aware_walk_forward.py`
  - `generate_windows()` with regime stratification
  - `validate_archetype_regime()` runs optimization on each window
  - `_aggregate_oos_metrics()` calculates consistency score

- [ ] Define `WalkForwardResult` dataclass
  ```python
  @dataclass
  class WalkForwardResult:
      archetype: str
      regime: str
      windows: List[Dict]  # Per-window results
      oos_metrics: Dict    # Aggregated OOS
      avg_oos_pf: float
      consistency_score: float
  ```

**Deliverable:** `engine/validation/regime_aware_walk_forward.py` (new)

---

### 3.2 Validation Execution
- [ ] Run walk-forward for S1 (crisis)
  - Target: 4+ windows, consistency > 0.6, avg OOS PF > 1.8

- [ ] Run walk-forward for S1 (risk_off)
- [ ] Run walk-forward for S2 (risk_off, neutral)
- [ ] Run walk-forward for S4 (risk_off, neutral)
- [ ] Run walk-forward for S5 (risk_on, neutral)

**Acceptance Criteria:**
- ✅ >= 4 windows validated per regime
- ✅ OOS consistency score > 0.6 (low overfitting)
- ✅ Positive windows >= 60%
- ✅ Event recall maintained on OOS windows

**Deliverable:** Validation reports for each archetype-regime pair

---

### 3.3 Validation Report Generator
- [ ] Create `generate_validation_report()` function
  - Regime coverage table
  - OOS performance by regime
  - Known event capture summary
  - Consistency checks (threshold monotonicity, PF variance)

- [ ] CLI tool
  ```bash
  python bin/validate_regime_aware.py \
      --archetype S1 \
      --config configs/optimized_s1_crisis.json \
      --output docs/validation_reports/s1_crisis.md
  ```

**Acceptance Criteria:**
- ✅ Report includes all critical sections (coverage, OOS, events, health)
- ✅ Warnings flagged for monotonicity violations or high variance

**Deliverable:** `bin/validate_regime_aware.py` (new)

---

## Phase 4: Portfolio Construction (Weeks 7-8)

### 4.1 Portfolio Weighting
- [ ] Create `engine/portfolio/regime_weighted_portfolio.py`
  - `analyze_regime_distribution()`: Historical regime frequencies
  - `calculate_regime_weighted()`: Weight = freq × PF × risk_adj
  - `calculate_kelly_weights()`: Kelly criterion with regime adjustment

**Deliverable:** `engine/portfolio/regime_weighted_portfolio.py` (new)

---

### 4.2 Dynamic Weight Adjustment
- [ ] Implement `DynamicWeightAdjuster`
  - Rolling 90-day regime estimation
  - Forecast next 30 days using transition matrix
  - Re-weight if regime shift > 30%

- [ ] Monitoring script
  ```bash
  python bin/monitor_regime_shift.py --alert-threshold 0.30
  ```

**Acceptance Criteria:**
- ✅ Dynamic adjustment triggers <= 12 times/year
- ✅ Weights bounded to [0.5x, 2.0x] of base weight
- ✅ Alerts sent when regime shift > 30%

**Deliverable:** `bin/monitor_regime_shift.py` (new)

---

### 4.3 Portfolio Backtest
- [ ] Run portfolio backtest on 2022-2023
  - Use regime-weighted allocation
  - Compare vs equal-weight portfolio
  - Target: 20-30% Sharpe improvement

**Acceptance Criteria:**
- ✅ Portfolio Sharpe > 1.0 on 2022-2023
- ✅ >= 20% improvement vs equal-weight
- ✅ Max drawdown < 25%

**Deliverable:** Portfolio backtest report

---

### 4.4 Production Config Generation
- [ ] Aggregate best thresholds from all archetype-regime pairs
- [ ] Generate unified production config
  ```json
  {
    "regime": {"enabled": true, ...},
    "archetypes": {
      "S1": {
        "thresholds": {
          "regime_thresholds": {
            "risk_off": {...},
            "crisis": {...}
          }
        }
      }
    },
    "portfolio": {
      "weights": {
        "S1_risk_off": 0.18,
        "S1_crisis": 0.12,
        ...
      }
    }
  }
  ```

**Deliverable:** `configs/production_regime_aware_v1.json`

---

## Phase 5: Production Deployment (Week 9)

### 5.1 A/B Testing Framework
- [ ] Implement dual-mode operation
  ```python
  class ArchetypeEngine:
      mode: str  # 'disabled', 'enabled', 'ab_test'
  ```

- [ ] Create `ABTestLogger`
  - Log parallel results (legacy vs regime-aware)
  - Generate comparison report (agreement rate, signal frequency)

**Deliverable:** `engine/ab_test/ab_test_logger.py` (new)

---

### 5.2 Paper Trading
- [ ] Deploy to paper trading with A/B mode
- [ ] Run for 2 weeks, collect metrics
- [ ] Compare:
  - Sharpe ratio (target: +15% improvement)
  - Profit factor
  - Max drawdown
  - Signal agreement rate

**Acceptance Criteria:**
- ✅ Regime-aware system shows >= 15% Sharpe improvement
- ✅ Zero production errors during paper trading
- ✅ Regime classifier updates in < 1 second

**Deliverable:** Paper trading report

---

### 5.3 Monitoring Dashboard
- [ ] Real-time regime display
- [ ] Current portfolio weights
- [ ] Regime transition alerts
- [ ] Performance tracking by regime

**Deliverable:** Monitoring dashboard (Grafana/Streamlit)

---

### 5.4 Production Cutover
- [ ] Switch from `ab_test` to `enabled` mode
- [ ] Monitor for 48 hours
- [ ] Validate:
  - Regime classification working
  - Thresholds loading correctly
  - Trades entering in correct regimes

**Acceptance Criteria:**
- ✅ Zero errors in first 48 hours
- ✅ Regime-aware signals match paper trading behavior
- ✅ Monitoring dashboard operational

**Deliverable:** Production system live

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regime classifier fails in prod | HIGH | Fallback to neutral regime, manual override for events |
| Empty test windows for crisis | MEDIUM | Skip windows, accept fewer validation points |
| Regime distribution shift | MEDIUM | Weekly monitoring, dynamic weight adjustment |
| Mid-trade regime transition losses | LOW | Accept in Phase 1, add conditional exits in Phase 3 |
| Overfitting to sparse crisis data | MEDIUM | Aggregate crisis bars across years, use regularization |

---

## Success Metrics

### Phase 1 (Foundation)
- ✅ Regime labeling completes without errors
- ✅ Filtered backtest produces different results

### Phase 2 (Optimization)
- ✅ Pareto frontier >= 5 solutions
- ✅ Event recall >= 80% for crisis archetypes
- ✅ PF > 1.5 on validation set

### Phase 3 (Walk-Forward)
- ✅ OOS consistency > 0.6
- ✅ >= 4 windows per regime
- ✅ Positive windows >= 60%

### Phase 4 (Portfolio)
- ✅ Portfolio Sharpe > 1.0 (2022-2023)
- ✅ >= 20% improvement vs baseline

### Phase 5 (Production)
- ✅ A/B test shows +15% Sharpe
- ✅ Zero production errors
- ✅ Regime classifier < 1s latency

---

## Dependencies

### Data Requirements
- Historical OHLCV data (2020-2024)
- Macro features (VIX, DXY, funding, OI, etc.)
- Trained GMM regime classifier model

### Software Requirements
- Python 3.9+
- Optuna 3.x
- scikit-learn (GMM)
- pandas, numpy
- SQLite (Optuna storage)

### Personnel
- 1 Senior Quant (design review)
- 1 ML Engineer (regime classifier)
- 1 Backend Engineer (production deployment)
- 1 QA Engineer (validation)

---

## Sign-Off

| Role | Name | Approval | Date |
|------|------|----------|------|
| System Architect | [Name] | ☐ | YYYY-MM-DD |
| Head of Research | [Name] | ☐ | YYYY-MM-DD |
| Lead Quant | [Name] | ☐ | YYYY-MM-DD |
| Engineering Manager | [Name] | ☐ | YYYY-MM-DD |

---

**Document Version:** 1.0
**Last Updated:** 2025-11-24
**Next Review:** After Phase 2 completion
