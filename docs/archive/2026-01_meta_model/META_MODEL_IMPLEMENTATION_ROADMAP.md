# Meta-Model Implementation Roadmap

**Project**: Overlap-as-Feature Meta-Model
**Timeline**: 9-12 weeks
**Owner**: System Architect

---

## Executive Summary

Build and deploy a meta-learning model that treats archetype signal overlap as a feature, improving win rate from 55% → 60-65% and Sharpe ratio by +0.3.

**Expected Impact**:
- Win rate improvement: +5-10% absolute
- Sharpe ratio improvement: +0.3 (e.g., 1.5 → 1.8)
- Trade frequency optimization: Filter out 30-40% of losing signals
- Drawdown reduction: -5% (e.g., 18% → 13%)

---

## Phase 1: Research & Validation (Weeks 1-3)

### Week 1: Data Preparation

**Tasks**:
- [ ] Extract historical archetype signals from backtest logs (2022-2024)
- [ ] Compute forward returns (24h, 48h, 72h horizons)
- [ ] Label dataset (WIN if return >2%, LOSS if return <-1%, else NEUTRAL)
- [ ] Remove neutral samples (keep only clean wins/losses)
- [ ] Exploratory data analysis (EDA)

**Deliverables**:
- `data/signals_historical.parquet` (~40,000 labeled signals)
- `docs/overlap_pattern_analysis.md` (EDA report)
- `data/signals_metadata.json` (dataset statistics)

**Scripts to Create**:
```python
bin/extract_historical_signals.py
  --backtest-logs logs/backtest_2022_2024/
  --price-data data/btc_1h_2022_2024.parquet
  --output data/signals_historical.parquet

bin/analyze_overlap_patterns.py
  --signals data/signals_historical.parquet
  --output docs/overlap_pattern_analysis.md
```

**Success Criteria**:
- Dataset size: 35,000-45,000 labeled signals
- Class balance: 45-55% wins, 45-55% losses
- Regime balance: ≥20% per regime (risk_on, neutral, risk_off, crisis)
- Overlap patterns identified: Top 20 pairs by mutual information

---

### Week 2: Baseline Modeling

**Tasks**:
- [ ] Feature engineering (overlap aggregates, pairwise interactions)
- [ ] Train/val/test split (80/10/10, temporal)
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest strong baseline
- [ ] Evaluate on OOS validation set
- [ ] Feature importance analysis (coefficients for LR, Gini for RF)

**Deliverables**:
- `models/baseline_logistic_v1.pkl` (baseline model)
- `models/baseline_rf_v1.pkl` (strong baseline)
- `results/baseline_metrics.json` (AUC, precision, recall)
- `results/feature_importance_baseline.csv` (top 20 features)

**Scripts to Create**:
```python
bin/train_baseline_models.py
  --data data/signals_historical.parquet
  --output-dir models/baselines/
  --models logistic,random_forest

bin/evaluate_baseline.py
  --model models/baseline_rf_v1.pkl
  --test-data data/signals_val.parquet
  --output results/baseline_metrics.json
```

**Success Criteria**:
- Logistic Regression validation AUC: >0.60
- Random Forest validation AUC: >0.63
- Top feature identified: `A_and_C` or similar overlap interaction

---

### Week 3: Production Model Training

**Tasks**:
- [ ] Train LightGBM production model
- [ ] Hyperparameter tuning with Optuna (100 trials)
- [ ] 5-fold cross-validation
- [ ] Walk-forward validation (6-month rolling windows)
- [ ] SHAP feature importance analysis
- [ ] Threshold calibration (maximize F1)

**Deliverables**:
- `models/meta_model_v1.pkl` (production model)
- `results/walk_forward_validation.json` (OOS performance per month)
- `results/shap_analysis/` (SHAP plots and CSV)
- `results/threshold_analysis.json` (precision-recall tradeoffs)

**Scripts to Create**:
```python
bin/train_meta_model.py
  --data data/signals_historical.parquet
  --model-type lightgbm
  --output models/meta_model_v1.pkl
  --n-trials 100

bin/validate_walk_forward.py
  --model models/meta_model_v1.pkl
  --signals data/signals_historical.parquet
  --window-months 6
  --output results/walk_forward_validation.json

bin/shap_analysis.py
  --model models/meta_model_v1.pkl
  --data data/signals_val.parquet
  --output results/shap_analysis/
```

**Success Criteria**:
- LightGBM validation AUC: **>0.65** ✅
- Walk-forward mean AUC: **>0.63** (consistent performance)
- SHAP top feature: Overlap interaction (e.g., `A_and_C`)
- Optimal threshold identified: ~0.65 (precision >70%, recall >60%)

**Milestone**: If validation AUC <0.65, STOP and debug before proceeding.

---

## Phase 2: Integration (Week 4)

### Week 4: Code Integration

**Tasks**:
- [ ] Build `MetaFilter` class (model wrapper)
- [ ] Integrate into `ArchetypeLogic` (filter layer)
- [ ] Add config management (`use_meta_filter` flag)
- [ ] Unit tests (test individual methods)
- [ ] Integration tests (test full pipeline)
- [ ] Backtest with meta-filter enabled

**Deliverables**:
- `engine/models/meta_filter.py` (production-ready class)
- `tests/unit/models/test_meta_filter.py` (unit tests)
- `tests/integration/test_archetype_meta_filter.py` (integration tests)
- `configs/production_with_meta_filter.json` (example config)
- `results/backtest_with_meta_filter.json` (backtest comparison)

**Code to Create**:

```python
# engine/models/meta_filter.py
class MetaFilter:
    def __init__(self, model_path: str, threshold: float = 0.65):
        """Load meta-model and set decision threshold."""
        ...

    def filter_signal(self, archetype_signals: list, context: RuntimeContext):
        """Filter archetype signals, return (should_take, prob, metadata)."""
        ...

    def _extract_features(self, archetype_signals: list, context: RuntimeContext):
        """Extract 70 meta-features from signals."""
        ...

# engine/archetypes/logic_v2_adapter.py (modified)
class ArchetypeLogic:
    def __init__(self, config: dict):
        ...
        if config.get('use_meta_filter', False):
            self.meta_filter = MetaFilter(
                config['meta_filter_path'],
                config.get('meta_filter_threshold', 0.65)
            )

    def evaluate_all_archetypes(self, context: RuntimeContext):
        raw_signals = [...]  # Existing logic

        if self.use_meta_filter and raw_signals:
            should_take, prob, metadata = self.meta_filter.filter_signal(
                raw_signals, context
            )
            if should_take:
                return raw_signals
            else:
                return []  # Filtered out
        else:
            return raw_signals
```

**Tests**:
```python
# tests/unit/models/test_meta_filter.py
def test_meta_filter_loads_model():
    """Test model loading."""
    filter = MetaFilter('models/meta_model_v1.pkl')
    assert filter.model is not None

def test_meta_filter_extracts_features():
    """Test feature extraction from signals."""
    signals = [
        {'archetype': 'A', 'conf': 0.82, 'direction': 1},
        {'archetype': 'C', 'conf': 0.91, 'direction': 1},
    ]
    features = filter._extract_features(signals, mock_context)
    assert features['A_fired'] == 1
    assert features['C_fired'] == 1
    assert features['A_and_C'] == 1
    assert features['num_fired'] == 2

def test_meta_filter_decision_take():
    """Test TAKE decision (high prob)."""
    should_take, prob, _ = filter.filter_signal(high_conf_signals, mock_context)
    assert should_take == True
    assert prob > 0.65

def test_meta_filter_decision_skip():
    """Test SKIP decision (low prob)."""
    should_take, prob, _ = filter.filter_signal(low_conf_signals, mock_context)
    assert should_take == False
    assert prob < 0.65
```

**Success Criteria**:
- All unit tests pass (100% coverage for `MetaFilter`)
- Integration tests pass (full pipeline with meta-filter)
- Backtest shows Sharpe improvement: **>+0.3** vs baseline ✅

**Milestone**: If backtest Sharpe improvement <+0.3, STOP and debug.

---

## Phase 3: Paper Trading (Weeks 5-8)

### Week 5: Deployment Setup

**Tasks**:
- [ ] Deploy to paper trading environment
- [ ] Set up A/B testing framework (50% with meta-filter, 50% without)
- [ ] Configure monitoring dashboard (Grafana/custom)
- [ ] Set up alerting (Slack notifications)
- [ ] Create rollback script (disable meta-filter)

**Deliverables**:
- `scripts/deploy_paper_trading.sh` (deployment script)
- `configs/paper_trading_ab_test.json` (A/B config)
- `monitoring/meta_filter_dashboard.json` (Grafana dashboard)
- `scripts/rollback_meta_filter.sh` (emergency rollback)

**Monitoring Metrics**:
- Win rate (daily, 7-day rolling)
- Sharpe ratio (7-day rolling)
- Filter rate (% signals rejected)
- Avg meta-prob (signal confidence)
- Feature drift (weekly check)

---

### Weeks 6-8: A/B Testing & Monitoring

**Tasks**:
- [ ] Run A/B test for 3 weeks (minimum 100 signals per arm)
- [ ] Monitor metrics daily (log to dashboard)
- [ ] Weekly reviews (compare A vs B performance)
- [ ] Feature drift checks (compare to training distribution)
- [ ] Acceptance testing (check criteria after Week 8)

**Deliverables**:
- `results/paper_trading_week_6.json` (Week 6 results)
- `results/paper_trading_week_7.json` (Week 7 results)
- `results/paper_trading_week_8.json` (Week 8 results)
- `results/ab_test_final_report.md` (comprehensive comparison)
- `results/acceptance_test_results.json` (pass/fail on criteria)

**Weekly Review Template**:
```markdown
## Week X Paper Trading Review

### Group A (With Meta-Filter)
- Signals: 45
- Win rate: 61.2%
- Sharpe: 1.85
- Filter rate: 35%

### Group B (Without Meta-Filter)
- Signals: 72
- Win rate: 54.8%
- Sharpe: 1.51
- Filter rate: 0%

### Comparison
- Win rate delta: +6.4% ✅
- Sharpe delta: +0.34 ✅
- Filter rate: 35% (within [20%, 60%]) ✅

### Observations
- Meta-filter correctly avoided 3 major losses
- No feature drift detected
- Continue monitoring

### Action Items
- None (on track)
```

**Success Criteria (Acceptance Testing)**:
- [ ] OOS AUC >0.65 (on paper trading data)
- [ ] Sharpe improvement >+0.3 vs baseline
- [ ] Win rate improvement >+5%
- [ ] Max drawdown reduction >-5%
- [ ] Filter rate: 20-60%
- [ ] No significant feature drift detected

**Milestone**: If acceptance criteria fail, ABORT production deployment. Diagnose, retrain, or redesign.

---

## Phase 4: Production Deployment (Week 9+)

### Week 9: Production Rollout (10% Capital)

**Tasks**:
- [ ] Deploy meta-filter to live trading (10% of capital)
- [ ] Monitor real-time performance (hourly checks for first week)
- [ ] Set up auto-retrain schedule (monthly cron job)
- [ ] Configure feature drift alerts (weekly checks)
- [ ] Document rollback procedures

**Deliverables**:
- `scripts/deploy_production.sh` (production deployment)
- `cron/retrain_meta_model.sh` (monthly auto-retrain)
- `monitoring/production_dashboard.json` (live dashboard)
- `docs/ROLLBACK_PROCEDURES.md` (emergency rollback guide)

**Monitoring (Real-Time)**:
- Win rate (24h, 7d, 30d)
- Sharpe ratio (7d, 30d)
- Filter rate (daily)
- Meta-prob distribution (daily)
- Feature drift (weekly)
- Model decay (monthly)

**Alerting Rules**:
- 🚨 Win rate <50% for 7 days → Alert + manual review
- ⚠️ Filter rate <10% or >80% for 3 days → Alert
- ⚠️ Feature drift KL divergence >0.3 → Alert (retrain needed)
- 🚨 Sharpe ratio <1.0 for 30 days → Alert + consider rollback

---

### Weeks 10-12: Monitoring & Optimization

**Tasks**:
- [ ] Monitor live performance daily
- [ ] Weekly feature drift checks
- [ ] Monthly retrain (if drift detected)
- [ ] Gradual capital increase (10% → 25% → 50% → 100%)
- [ ] SHAP analysis on live data (interpretability check)

**Deliverables**:
- `results/production_week_10.json` (Week 10 performance)
- `results/production_week_11.json` (Week 11 performance)
- `results/production_week_12.json` (Week 12 performance)
- `results/production_final_report.md` (comprehensive analysis)

**Capital Ramp Schedule**:
```
Week 9:  10% capital (testing)
Week 10: 25% capital (if metrics hold)
Week 11: 50% capital (if metrics hold)
Week 12: 100% capital (if metrics hold)
```

**Rollback Triggers**:
- Win rate <baseline for 14 days
- Sharpe ratio <baseline for 30 days
- Feature drift unresolved for 2 retrains
- Any catastrophic loss (>10% single trade)

---

## Rollback Plan

### Immediate Action (<1 hour)

```bash
# 1. Disable meta-filter via config
# configs/production.json
{
  "use_meta_filter": false  # ← Change to false
}

# 2. Restart trading engine
./scripts/restart_trading_engine.sh

# 3. Verify fallback to raw archetypes
tail -f logs/trading.log | grep "Meta-filter DISABLED"

# 4. Alert team
./scripts/alert_slack.sh "Meta-filter disabled due to underperformance"
```

### Root Cause Analysis (24 hours)

```python
# bin/diagnose_meta_filter_failure.py

# 1. Compare last 7 days vs baseline
# 2. Check feature drift (KL divergence)
# 3. Analyze SHAP values on failing signals
# 4. Identify failure mode:
#    - Overfitting? (high train AUC, low OOS AUC)
#    - Regime shift? (crisis regime not in training data)
#    - Feature drift? (market structure changed)
#    - Bug? (incorrect feature extraction)

# Output: Diagnosis report (docs/meta_filter_failure_diagnosis.md)
```

### Fix or Retrain (1 week)

**Option 1: Feature Drift → Retrain**
```bash
# Retrain on last 3 months of data
python bin/retrain_meta_model.py \
    --data data/signals_last_3m.parquet \
    --output models/meta_model_v2.pkl \
    --validate

# Re-deploy if validation passes
```

**Option 2: Regime Shift → Add Regime-Specific Model**
```python
# Train separate model for new regime
python bin/train_regime_specific_model.py \
    --regime crisis_v2 \
    --data data/signals_crisis_v2.parquet \
    --output models/meta_model_crisis_v2.pkl
```

**Option 3: Overfitting → Simplify Model**
```python
# Increase regularization
params = {
    'lambda_l1': 1.0,  # Increase from 0.1
    'lambda_l2': 1.0,  # Increase from 0.1
    'max_depth': 6,    # Reduce from 8
}
```

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting | Medium | High | Walk-forward validation, L1/L2 regularization, early stopping |
| Feature drift | High | Medium | Weekly drift checks, monthly auto-retrain, alerts |
| Lookahead bias | Low | Critical | Strict temporal split, no future data in features |
| Model fails in production | Medium | High | Gradual capital ramp, rollback plan, A/B testing |
| Inference latency >10ms | Low | Medium | LightGBM (fast), feature caching, profiling |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient training data | Low | Medium | Require 40k+ signals, augment with synthetic data if needed |
| Regime shift unseen in training | Medium | High | Train regime-specific models, monitor regime transitions |
| Team bandwidth | Medium | Low | Phased rollout, automate monitoring, clear ownership |
| Rollback fails | Low | Critical | Test rollback script monthly, maintain baseline system |

---

## Success Metrics Tracking

### Phase 1 (Research & Validation)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Dataset size | 35k-45k | TBD | ⏳ Pending |
| Validation AUC | >0.65 | TBD | ⏳ Pending |
| Walk-forward AUC | >0.63 | TBD | ⏳ Pending |
| SHAP top feature | Overlap interaction | TBD | ⏳ Pending |

### Phase 2 (Integration)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Unit test coverage | 100% | TBD | ⏳ Pending |
| Integration tests pass | 100% | TBD | ⏳ Pending |
| Backtest Sharpe improvement | >+0.3 | TBD | ⏳ Pending |

### Phase 3 (Paper Trading)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Win rate improvement | >+5% | TBD | ⏳ Pending |
| Sharpe improvement | >+0.3 | TBD | ⏳ Pending |
| Max DD reduction | >-5% | TBD | ⏳ Pending |
| Filter rate | 20-60% | TBD | ⏳ Pending |
| Feature drift detected | No | TBD | ⏳ Pending |

### Phase 4 (Production)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Live Sharpe (30d) | >1.8 | TBD | ⏳ Pending |
| Live win rate (30d) | >60% | TBD | ⏳ Pending |
| Capital deployed | 100% | TBD | ⏳ Pending |
| Rollback events | 0 | TBD | ⏳ Pending |

---

## Go/No-Go Decision Points

### After Week 3 (Research Complete)
**GO criteria**:
- [x] Validation AUC >0.65
- [x] Walk-forward AUC >0.63
- [x] SHAP analysis shows overlap features matter

**NO-GO**: If validation AUC <0.65 → Debug, retrain, or abort.

---

### After Week 4 (Integration Complete)
**GO criteria**:
- [x] All tests pass
- [x] Backtest Sharpe improvement >+0.3

**NO-GO**: If backtest Sharpe improvement <+0.3 → Debug integration or abort.

---

### After Week 8 (Paper Trading Complete)
**GO criteria**:
- [x] OOS AUC >0.65
- [x] Sharpe improvement >+0.3
- [x] Win rate improvement >+5%
- [x] Max DD reduction >-5%
- [x] Filter rate: 20-60%
- [x] No significant feature drift

**NO-GO**: If any criterion fails → Diagnose, retrain, or abort production deployment.

---

### After Week 12 (Production Ramp Complete)
**GO criteria**:
- [x] Live Sharpe sustained >1.8 (30d)
- [x] Live win rate sustained >60% (30d)
- [x] No rollback events
- [x] Capital deployed: 100%

**NO-GO**: If metrics degrade → Rollback to raw archetypes.

---

## Team & Resources

### Ownership
- **System Architect**: Architecture design, code review, final approval
- **ML Engineer**: Model training, hyperparameter tuning, SHAP analysis
- **Backend Engineer**: Integration, testing, deployment
- **DevOps**: Monitoring, alerts, auto-retrain setup
- **Quant Analyst**: EDA, overlap pattern analysis, acceptance testing

### Compute Resources
- **Training**: 1x GPU instance (AWS p3.2xlarge or similar)
  - Estimated cost: $3/hour × 10 hours = $30 (Optuna tuning)
- **Inference**: CPU-only (LightGBM is fast)
  - Estimated latency: <10ms per prediction
- **Storage**: 10 GB (signals dataset + models + logs)

### External Dependencies
- **Python packages**: `lightgbm`, `optuna`, `shap`, `scikit-learn`, `pandas`, `numpy`
- **Monitoring**: Grafana (optional, can use custom dashboard)
- **Alerting**: Slack API (or email)

---

## Appendix: File Structure

```
Bull-machine-/
├── docs/
│   ├── META_MODEL_ARCHITECTURE_OVERLAP_AS_FEATURE.md  ← Full spec
│   ├── META_MODEL_VISUAL_ARCHITECTURE.md  ← Visual diagrams
│   ├── META_MODEL_QUICK_START.md  ← Quick reference
│   ├── overlap_pattern_analysis.md  ← EDA report (TO BE CREATED)
│   └── ROLLBACK_PROCEDURES.md  ← Emergency rollback (TO BE CREATED)
│
├── engine/
│   └── models/
│       ├── meta_filter.py  ← Production filter class (TO BE CREATED)
│       └── meta_ensemble.py  ← Existing (already present)
│
├── bin/
│   ├── extract_historical_signals.py  ← TO BE CREATED
│   ├── analyze_overlap_patterns.py  ← TO BE CREATED
│   ├── train_baseline_models.py  ← TO BE CREATED
│   ├── train_meta_model.py  ← TO BE CREATED
│   ├── validate_walk_forward.py  ← TO BE CREATED
│   ├── shap_analysis.py  ← TO BE CREATED
│   ├── evaluate_meta_model.py  ← TO BE CREATED
│   ├── test_meta_model_acceptance.py  ← TO BE CREATED
│   ├── monitor_meta_filter_health.py  ← TO BE CREATED
│   └── diagnose_meta_filter_failure.py  ← TO BE CREATED
│
├── tests/
│   ├── unit/models/
│   │   └── test_meta_filter.py  ← TO BE CREATED
│   └── integration/
│       └── test_archetype_meta_filter.py  ← TO BE CREATED
│
├── configs/
│   ├── production_with_meta_filter.json  ← TO BE CREATED
│   └── paper_trading_ab_test.json  ← TO BE CREATED
│
├── models/
│   ├── baseline_logistic_v1.pkl  ← TO BE CREATED
│   ├── baseline_rf_v1.pkl  ← TO BE CREATED
│   └── meta_model_v1.pkl  ← TO BE CREATED
│
├── data/
│   ├── signals_historical.parquet  ← TO BE CREATED
│   ├── signals_val.parquet  ← TO BE CREATED
│   └── signals_test.parquet  ← TO BE CREATED
│
├── results/
│   ├── baseline_metrics.json  ← TO BE CREATED
│   ├── walk_forward_validation.json  ← TO BE CREATED
│   ├── shap_analysis/  ← TO BE CREATED
│   ├── ab_test_final_report.md  ← TO BE CREATED
│   └── production_final_report.md  ← TO BE CREATED
│
├── scripts/
│   ├── deploy_paper_trading.sh  ← TO BE CREATED
│   ├── deploy_production.sh  ← TO BE CREATED
│   ├── rollback_meta_filter.sh  ← TO BE CREATED
│   └── restart_trading_engine.sh  ← Existing or TO BE CREATED
│
├── cron/
│   └── retrain_meta_model.sh  ← TO BE CREATED
│
└── monitoring/
    ├── meta_filter_dashboard.json  ← TO BE CREATED (Grafana)
    └── production_dashboard.json  ← TO BE CREATED
```

---

## Next Actions

**Immediate (Week 1)**:
1. Extract historical signals: `bin/extract_historical_signals.py`
2. EDA: Analyze overlap patterns
3. Label dataset (WIN/LOSS)
4. Create train/val/test splits

**Start Date**: TBD
**Owner**: TBD

---

**Questions?** See:
- Full architecture: `docs/META_MODEL_ARCHITECTURE_OVERLAP_AS_FEATURE.md`
- Quick start: `META_MODEL_QUICK_START.md`
- Visual guide: `docs/META_MODEL_VISUAL_ARCHITECTURE.md`
