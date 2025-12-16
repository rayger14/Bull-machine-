# QUANT LAB 72-HOUR CHECKLIST

**Purpose:** Day-by-day task checklist with validation criteria for each milestone.

---

## DAY 1: FRAMEWORK + BASELINES

### Morning Session (4 hours): Framework Setup

**Core Interfaces** (45 min)
- [ ] Create `engine/backtesting/base_model.py` with BaseModel abstract class
- [ ] Create `engine/backtesting/backtest_results.py` with BacktestResults dataclass
- [ ] Test imports work: `python -c "from engine.backtesting.base_model import BaseModel"`

**Backtest Engine** (90 min)
- [ ] Create `engine/backtesting/backtest_engine.py`
- [ ] Implement `_signals_to_trades()` method
- [ ] Implement `_calculate_metrics()` method
- [ ] Implement transaction cost handling (slippage + fees)
- [ ] Test engine initialization: `python -c "from engine.backtesting.backtest_engine import BacktestEngine; e = BacktestEngine()"`

**Experiment Config** (30 min)
- [ ] Create `configs/experiment_btc_1h_2020_2025.json`
- [ ] Define train/test/OOS periods (2020-2022 / 2023 / 2024)
- [ ] Define baseline model configs
- [ ] Validate JSON: `python -c "import json; json.load(open('configs/experiment_btc_1h_2020_2025.json'))"`

**Runner Script** (45 min)
- [ ] Create `bin/run_quant_suite.py`
- [ ] Implement config loading
- [ ] Implement data loading and period splitting
- [ ] Implement results saving (CSV + markdown)
- [ ] Test script runs: `python bin/run_quant_suite.py --help`

**Unit Tests** (30 min)
- [ ] Create `tests/unit/backtesting/test_backtest_engine.py`
- [ ] Write test for engine initialization
- [ ] Write test for engine.run() with dummy model
- [ ] Run tests: `pytest tests/unit/backtesting/ -v`

**VALIDATION:**
```bash
# All imports work
python -c "from engine.backtesting.base_model import BaseModel"
python -c "from engine.backtesting.backtest_engine import BacktestEngine"
python -c "from engine.backtesting.backtest_results import BacktestResults"

# Config is valid
python -c "import json; json.load(open('configs/experiment_btc_1h_2020_2025.json')); print('✓')"

# Tests pass
pytest tests/unit/backtesting/ -v
```

---

### Afternoon Session (4 hours): Baseline Implementation

**Baseline Models** (30 min)
- [ ] Create `engine/backtesting/baseline_models.py`
- [ ] Implement BuyAndHoldModel (B0)
- [ ] Implement SMA200TrendModel (B1)
- [ ] Implement SMACrossoverModel (B2)
- [ ] Implement RSIMeanReversionModel (B3)
- [ ] Implement VolTargetTrendModel (B4)
- [ ] Create MODEL_REGISTRY dictionary

**Unit Tests** (90 min)
- [ ] Create `tests/unit/backtesting/test_baseline_models.py`
- [ ] Test BuyAndHoldModel generates signals
- [ ] Test SMA200TrendModel generates signals
- [ ] Test SMACrossoverModel generates signals
- [ ] Test RSIMeanReversionModel generates signals
- [ ] Test VolTargetTrendModel generates signals
- [ ] Run tests: `pytest tests/unit/backtesting/test_baseline_models.py -v`

**Test Data** (60 min)
- [ ] Create `bin/prepare_test_data.py`
- [ ] Extract OHLCV data from database (2020-2025)
- [ ] Save to `features/btc_ohlcv_2020_2025.csv`
- [ ] Verify data quality (no missing values, correct date range)
- [ ] Run: `python bin/prepare_test_data.py`

**Integration Test** (60 min)
- [ ] Run baseline suite: `python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only`
- [ ] Check results directory created: `ls -la results/quant_suite/`
- [ ] Verify CSV has 15 rows (5 models × 3 periods): `wc -l results/quant_suite/results_LATEST.csv`
- [ ] Review markdown report: `cat results/quant_suite/report_LATEST.md`

**VALIDATION:**
```bash
# All baseline tests pass
pytest tests/unit/backtesting/test_baseline_models.py -v

# Suite runs successfully
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only

# Results exist and are valid
test -f results/quant_suite/results_LATEST.csv && echo "✓ CSV exists"
test -f results/quant_suite/report_LATEST.md && echo "✓ Report exists"

# CSV has correct row count (15 + header = 16)
test $(wc -l < results/quant_suite/results_LATEST.csv) -eq 16 && echo "✓ Correct row count"

# No models have zero profit factor
! grep ",0.0," results/quant_suite/results_LATEST.csv && echo "✓ All models have non-zero PF"
```

**END OF DAY 1 CHECKPOINT:**
- [ ] Can run baseline suite end-to-end
- [ ] Results show 5 baselines across 3 periods
- [ ] All tests pass
- [ ] No errors in execution

---

## DAY 2: BULL MACHINE INTEGRATION

### Morning Session (4 hours): Archetype Backtests

**S4 Funding Divergence** (60 min)
- [ ] Run S4 on train period (2020-2022): `python bin/backtest.py --config configs/s4_funding_divergence.json --start 2020-01-01 --end 2022-12-31 --output results/archetypes/s4_train.json`
- [ ] Run S4 on test period (2023): Similar command with 2023 dates
- [ ] Run S4 on OOS period (2024): Similar command with 2024 dates
- [ ] Verify all 3 result files exist: `ls -la results/archetypes/s4_*.json`

**S1 V2 Liquidity Vacuum** (60 min)
- [ ] Run S1 V2 on train period
- [ ] Run S1 V2 on test period
- [ ] Run S1 V2 on OOS period
- [ ] Verify all 3 result files exist: `ls -la results/archetypes/s1_v2_*.json`

**S5 Long Squeeze** (60 min)
- [ ] Run S5 on train period
- [ ] Run S5 on test period
- [ ] Run S5 on OOS period
- [ ] Verify all 3 result files exist: `ls -la results/archetypes/s5_*.json`

**Extract Metrics** (60 min)
- [ ] Create `bin/extract_archetype_metrics.py`
- [ ] Implement metric extraction from backtest JSON
- [ ] Run extraction: `python bin/extract_archetype_metrics.py`
- [ ] Verify CSV created: `cat results/archetypes/archetype_metrics.csv`
- [ ] Check CSV has 9 rows (3 archetypes × 3 periods): `wc -l results/archetypes/archetype_metrics.csv`

**VALIDATION:**
```bash
# All archetype results exist
test -f results/archetypes/s4_train.json && echo "✓ S4 train"
test -f results/archetypes/s4_test.json && echo "✓ S4 test"
test -f results/archetypes/s4_oos.json && echo "✓ S4 oos"
test -f results/archetypes/s1_v2_train.json && echo "✓ S1 V2 train"
test -f results/archetypes/s1_v2_test.json && echo "✓ S1 V2 test"
test -f results/archetypes/s1_v2_oos.json && echo "✓ S1 V2 oos"
test -f results/archetypes/s5_train.json && echo "✓ S5 train"
test -f results/archetypes/s5_test.json && echo "✓ S5 test"
test -f results/archetypes/s5_oos.json && echo "✓ S5 oos"

# Metrics CSV exists and has correct rows
test -f results/archetypes/archetype_metrics.csv && echo "✓ Metrics CSV"
test $(wc -l < results/archetypes/archetype_metrics.csv) -eq 10 && echo "✓ Correct row count (9 + header)"
```

---

### Afternoon Session (4 hours): Unified Comparison

**Merge Results** (45 min)
- [ ] Create `bin/merge_baseline_archetype_results.py`
- [ ] Load baseline CSV
- [ ] Load archetype CSV
- [ ] Merge into unified format
- [ ] Run: `python bin/merge_baseline_archetype_results.py`
- [ ] Verify: `cat results/unified_comparison.csv`

**Calculate Overfit** (45 min)
- [ ] Create `bin/calculate_overfit_scores.py`
- [ ] Calculate overfit score: (Train_PF - Test_PF) / Train_PF
- [ ] Run: `python bin/calculate_overfit_scores.py`
- [ ] Verify: `cat results/overfit_analysis.csv`

**Create Ranking** (90 min)
- [ ] Create `bin/create_unified_ranking.py`
- [ ] Pivot results to wide format
- [ ] Merge with overfit scores
- [ ] Sort by test PF
- [ ] Generate markdown report with red flags
- [ ] Run: `python bin/create_unified_ranking.py`
- [ ] Review: `cat results/unified_ranking_report.md`

**Analysis** (60 min)
- [ ] Identify best baseline
- [ ] Identify archetypes beating best baseline
- [ ] Identify models with high overfit (>0.5)
- [ ] Identify models with OOS collapse (<1.0 PF)
- [ ] Identify models with low trade count (<20)

**VALIDATION:**
```bash
# All analysis files exist
test -f results/unified_comparison.csv && echo "✓ Unified comparison"
test -f results/overfit_analysis.csv && echo "✓ Overfit analysis"
test -f results/unified_ranking.csv && echo "✓ Unified ranking"
test -f results/unified_ranking_report.md && echo "✓ Ranking report"

# Unified comparison has all models (24 rows + header = 25)
test $(wc -l < results/unified_comparison.csv) -eq 25 && echo "✓ All models included"

# Ranking report contains key sections
grep "Rankings by Test Period Profit Factor" results/unified_ranking_report.md && echo "✓ Rankings section"
grep "Red Flags" results/unified_ranking_report.md && echo "✓ Red flags section"
grep "Key Insights" results/unified_ranking_report.md && echo "✓ Insights section"
```

**END OF DAY 2 CHECKPOINT:**
- [ ] Have metrics for all archetypes
- [ ] Unified comparison table complete
- [ ] Know which archetypes (if any) beat baselines
- [ ] Red flags identified
- [ ] Ready to make decisions

---

## DAY 3: DECISION MAKING

### Morning Session (4 hours): Analysis & Acceptance

**Apply Criteria** (90 min)
- [ ] Create `bin/apply_acceptance_criteria.py`
- [ ] Define acceptance criteria (min PF, max overfit, min trades)
- [ ] Apply criteria to each model
- [ ] Determine keep/improve/kill decisions
- [ ] Run: `python bin/apply_acceptance_criteria.py`
- [ ] Review: `cat results/model_decisions.csv`

**Generate Decision Report** (90 min)
- [ ] Create `bin/generate_decision_report.py`
- [ ] Summarize keep/improve/kill counts
- [ ] Detail each KEEP model with next steps
- [ ] Detail each IMPROVE model with action items
- [ ] List KILL models with reasons
- [ ] Generate key insights and recommendations
- [ ] Run: `python bin/generate_decision_report.py`
- [ ] Review: `cat results/DECISION_REPORT.md`

**Document Findings** (60 min)
- [ ] Review decision report thoroughly
- [ ] Note surprising findings
- [ ] Document key learnings
- [ ] Identify patterns (what worked, what didn't)

**VALIDATION:**
```bash
# Decision files exist
test -f results/model_decisions.csv && echo "✓ Decisions CSV"
test -f results/DECISION_REPORT.md && echo "✓ Decision report"

# Decision report has all sections
grep "Executive Summary" results/DECISION_REPORT.md && echo "✓ Summary"
grep "Models to KEEP" results/DECISION_REPORT.md && echo "✓ Keep section"
grep "Models to IMPROVE" results/DECISION_REPORT.md && echo "✓ Improve section"
grep "Models to KILL" results/DECISION_REPORT.md && echo "✓ Kill section"
grep "Key Insights" results/DECISION_REPORT.md && echo "✓ Insights"
grep "Recommendations" results/DECISION_REPORT.md && echo "✓ Recommendations"

# Decision CSV has decisions for all models
test $(wc -l < results/model_decisions.csv) -eq 9 && echo "✓ All models have decisions (8 + header)"
```

---

### Afternoon Session (4 hours): Next Steps Planning

**Deployment Roadmap** (90 min)
- [ ] Create `bin/create_deployment_roadmap.py`
- [ ] For each KEEP model, define:
  - [ ] Paper trading timeline (14 days)
  - [ ] Acceptance criteria for live deployment
  - [ ] Risk parameters
  - [ ] Monitoring requirements
- [ ] Run: `python bin/create_deployment_roadmap.py`
- [ ] Review: `cat results/DEPLOYMENT_ROADMAP.md`

**Improvement Plan** (90 min)
- [ ] Create `bin/create_improvement_plan.py`
- [ ] For each IMPROVE model, define:
  - [ ] Diagnostic tasks based on failure reason
  - [ ] Success criteria for re-evaluation
  - [ ] Timeline (4 weeks)
- [ ] Run: `python bin/create_improvement_plan.py`
- [ ] Review: `cat results/IMPROVEMENT_PLAN.md`

**Experiment Queue** (60 min)
- [ ] Create `bin/create_experiment_queue.py`
- [ ] Define high-priority experiments:
  - [ ] Temporal layer ablation
  - [ ] Regime slicing
  - [ ] Multi-asset validation
- [ ] Define medium-priority experiments
- [ ] Define low-priority experiments
- [ ] Run: `python bin/create_experiment_queue.py`
- [ ] Review: `cat results/EXPERIMENT_QUEUE.md`

**Final Review** (30 min)
- [ ] Review all deliverables
- [ ] Ensure actionable next steps for each model
- [ ] Verify timelines are realistic
- [ ] Document any open questions

**VALIDATION:**
```bash
# Planning documents exist
test -f results/DEPLOYMENT_ROADMAP.md && echo "✓ Deployment roadmap"
test -f results/IMPROVEMENT_PLAN.md && echo "✓ Improvement plan"
test -f results/EXPERIMENT_QUEUE.md && echo "✓ Experiment queue"

# Each document has content
test $(wc -l < results/DEPLOYMENT_ROADMAP.md) -gt 10 && echo "✓ Deployment roadmap has content"
test $(wc -l < results/IMPROVEMENT_PLAN.md) -gt 10 && echo "✓ Improvement plan has content"
test $(wc -l < results/EXPERIMENT_QUEUE.md) -gt 10 && echo "✓ Experiment queue has content"
```

**END OF DAY 3 CHECKPOINT:**
- [ ] Clear keep/improve/kill decisions made
- [ ] Deployment roadmap created for winners
- [ ] Improvement plan created for candidates
- [ ] Experiment queue prioritized
- [ ] Next 2 weeks planned

---

## FINAL VALIDATION

**Complete Lab Checklist:**

```bash
# Run comprehensive validation
./bin/validate_quant_lab.sh
```

**Manual Checks:**

- [ ] Can explain why each KEEP model is worth deploying
- [ ] Can explain why each IMPROVE model failed and how to fix it
- [ ] Can explain why each KILL model is not worth pursuing
- [ ] Can defend each decision to a skeptical quant
- [ ] Have confidence in the process (not just results)
- [ ] Can repeat this process on new data tomorrow

**Documentation Checklist:**

- [ ] All code has comments explaining logic
- [ ] All scripts have usage examples
- [ ] All decisions have documented rationale
- [ ] All red flags have been identified
- [ ] All next steps have clear owners and timelines

**Repeatable Process:**

- [ ] Can run baseline suite in <5 minutes
- [ ] Can add new baseline in <1 hour
- [ ] Can integrate new archetype in <2 hours
- [ ] Can re-run full analysis weekly
- [ ] Process is documented well enough for another person to run

---

## TROUBLESHOOTING CHECKLIST

**If baseline suite fails:**
- [ ] Check data file exists: `test -f features/btc_ohlcv_2020_2025.csv`
- [ ] Check data has required columns: `head -1 features/btc_ohlcv_2020_2025.csv`
- [ ] Check date ranges match data: `head -5 features/btc_ohlcv_2020_2025.csv`
- [ ] Check for NaN values: `grep -c "nan" features/btc_ohlcv_2020_2025.csv`
- [ ] Run with --verbose flag for debugging

**If archetype backtests fail:**
- [ ] Check config files exist
- [ ] Verify date ranges are valid
- [ ] Check database connection if using DB
- [ ] Review error logs
- [ ] Test on smaller date range first

**If tests fail:**
- [ ] Run pytest with -v flag for details
- [ ] Check imports are correct
- [ ] Verify test data generation works
- [ ] Run tests individually to isolate issues

**If no models meet criteria:**
- [ ] Lower acceptance thresholds temporarily
- [ ] Verify criteria are appropriate for your use case
- [ ] Check if market regime is unusual
- [ ] Consider deploying best baseline
- [ ] Re-evaluate feature engineering

**If results don't make sense:**
- [ ] Verify transaction costs are reasonable
- [ ] Check for data quality issues
- [ ] Review signal generation logic
- [ ] Compare to manual backtest
- [ ] Check for lookahead bias

---

## EMERGENCY ROLLBACK

**If something goes seriously wrong:**

```bash
# Restore from git
git status
git diff
git restore .

# Or restore from backup
cp -r results_backup/ results/

# Start fresh
rm -rf results/
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only
```

**When to rollback:**
- Results are clearly wrong (PF > 10, negative trade counts, etc.)
- Code changes broke working functionality
- Data corruption detected
- Unrecoverable errors

**How to prevent:**
- Commit working code frequently: `git commit -am "Working baseline suite"`
- Back up results before major changes: `cp -r results/ results_backup/`
- Test on small date ranges first
- Validate results against known baselines

---

**Remember: The goal is not perfect results. The goal is a repeatable process that gives you truth.**
