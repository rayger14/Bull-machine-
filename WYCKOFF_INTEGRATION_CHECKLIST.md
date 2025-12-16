# Wyckoff Integration Checklist

**Status**: Integration Required
**Priority**: HIGH (Blocks all Wyckoff optimization work)
**Estimated Time**: 4-6 hours

---

## Phase 0: Integration (REQUIRED - 2-4 hours)

### Task 0.1: Backfill Wyckoff Events to Feature Store

- [ ] **Step 1**: Run dry-run validation (30 sec)
  ```bash
  python3 bin/backfill_wyckoff_events.py \
    --asset BTC --start 2022-01-01 --end 2024-12-31 \
    --config configs/wyckoff_events_config.json --dry-run
  ```
  - Expected: ~17,000 total events detected
  - Expected: BC events: 15-30, LPS events: 1,000-1,500, Spring-A: 2-5

- [ ] **Step 2**: Run actual backfill (5-10 min)
  ```bash
  python3 bin/backfill_wyckoff_events.py \
    --asset BTC --start 2022-01-01 --end 2024-12-31 \
    --config configs/wyckoff_events_config.json
  ```
  - Check for backup created: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet`

- [ ] **Step 3**: Verify columns added (30 sec)
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
  wyckoff_cols = [col for col in df.columns if 'wyckoff' in col.lower()]
  print(f'Wyckoff columns: {len(wyckoff_cols)}')
  for col in sorted(wyckoff_cols):
      if df[col].dtype == bool:
          print(f'{col}: {df[col].sum()} events')
  "
  ```
  - Expected: 28+ Wyckoff columns (18 events × 2 cols + phase/sequence cols)

### Task 0.2: Integrate Boost Logic into Backtest Engine

- [ ] **Step 1**: Open `bin/backtest_knowledge_v2.py`
- [ ] **Step 2**: Find `compute_advanced_fusion_score()` method (around line 450)
- [ ] **Step 3**: Add Wyckoff event boost logic (see docs/WYCKOFF_OPTIMIZATION_STRATEGY.md section 0.2)
  - Add 50 lines of code after `context['wyckoff_phase'] = ...`
  - Includes BC/UTAD avoidance (hard veto)
  - Includes LPS/Spring-A/SOS boost logic
  - Includes PTI confluence detection

- [ ] **Step 4**: Run smoke test (1 min)
  ```bash
  python3 bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_wyckoff_v1.json \
    --asset BTC --start 2024-03-01 --end 2024-03-10 --verbose
  ```
  - Expected log: "Wyckoff event detected: wyckoff_lps"
  - Expected log: "Applying boost: +0.10 to Wyckoff score"

---

## Phase 1: Baseline Validation (30 min - 1 hour)

### Task 1.1: Run Baseline Backtests

- [ ] **Step 1**: Create no-Wyckoff config for comparison
  ```bash
  cp configs/mvp/mvp_bull_wyckoff_v1.json configs/mvp/mvp_bull_no_wyckoff_v1.json
  # Edit configs/mvp/mvp_bull_no_wyckoff_v1.json: set "wyckoff_events.enabled": false
  ```

- [ ] **Step 2**: Run Wyckoff baseline (10-15 min)
  ```bash
  python3 bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_wyckoff_v1.json \
    --asset BTC --start 2024-01-01 --end 2024-09-30 \
    --output results/wyckoff_baseline_2024Q1-Q3.json
  ```

- [ ] **Step 3**: Run no-Wyckoff baseline (10-15 min)
  ```bash
  python3 bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_no_wyckoff_v1.json \
    --asset BTC --start 2024-01-01 --end 2024-09-30 \
    --output results/no_wyckoff_baseline_2024Q1-Q3.json
  ```

### Task 1.2: Compare Results

- [ ] **Step 1**: Run comparison script (see docs/WYCKOFF_OPTIMIZATION_STRATEGY.md section 1.2)
- [ ] **Step 2**: Check success criteria:
  - [ ] Win rate improvement: +8-15%?
  - [ ] Profit factor: ≥1.8?
  - [ ] Sharpe ratio: ≥1.3?
  - [ ] BC avoidance working? (no trades within 24h after BC at $70,850)
  - [ ] LPS boost working? (fusion score +0.10 visible in logs)

### Decision Point

- [ ] **IF 4/5 criteria met**: ✅ Proceed to Phase 2 (optimization)
- [ ] **IF win rate +10-15% AND PF >2.0**: ✅ Deploy as-is (skip Phase 2)
- [ ] **IF <4 criteria met**: ❌ STOP and investigate integration bug

---

## Phase 2: Optimization (CONDITIONAL - 2-3 hours)

**ONLY run if Phase 1 shows +8-12% win rate improvement AND PF >1.8**

### Task 2.1: Create Optimization Script

- [ ] **Step 1**: Create `bin/optimize_wyckoff_boosts.py` (see docs/WYCKOFF_OPTIMIZATION_STRATEGY.md section 2.5)
- [ ] **Step 2**: Verify search space:
  - lps_boost: 0.05-0.20
  - spring_a_boost: 0.08-0.25
  - sos_boost: 0.03-0.15
  - pti_confluence_boost: 0.10-0.30
  - min_confidence: 0.55-0.75

### Task 2.2: Run Optimization

- [ ] **Step 1**: Run Optuna (2-3 hours)
  ```bash
  python3 bin/optimize_wyckoff_boosts.py --trials 40 --workers 4
  ```

- [ ] **Step 2**: Review results
  ```bash
  python3 -c "
  import json
  with open('results/wyckoff_optimization_results.json') as f:
      results = json.load(f)
  print('Best Sharpe:', results['best_value'])
  for k, v in results['best_params'].items():
      print(f'{k}: {v}')
  "
  ```

### Task 2.3: Validate on Test Period

- [ ] **Step 1**: Update config with best params
- [ ] **Step 2**: Run test backtest (Jul-Sep 2024)
  ```bash
  python3 bin/backtest_knowledge_v2.py \
    --config configs/mvp/mvp_bull_wyckoff_optimized_v1.json \
    --asset BTC --start 2024-07-01 --end 2024-09-30 \
    --output results/wyckoff_optimized_test.json
  ```

- [ ] **Step 3**: Check generalization:
  - [ ] Test Sharpe ≥ 85% of train Sharpe? (no overfitting)
  - [ ] Test Sharpe > baseline Sharpe × 1.10? (at least 10% improvement)
  - [ ] Win rate improvement: +3-5% over hand-tuned baseline?

### Decision Point

- [ ] **IF all 3 criteria met**: ✅ Deploy optimized params
- [ ] **IF overfitting detected**: ⚠️ Use hand-tuned params instead
- [ ] **IF <10% improvement**: ⚠️ Not worth deploying optimized params

---

## Success Metrics

### Phase 0 Success (Integration)
- ✅ 18+ event columns in feature store
- ✅ 10,000-20,000 total events detected
- ✅ BC: 15-30, LPS: 1,000-1,500, Spring-A: 2-5
- ✅ Smoke test logs show boost application

### Phase 1 Success (Baseline)
- ✅ Win rate improvement: +8-15%
- ✅ Profit factor: ≥1.8
- ✅ Sharpe ratio: ≥1.3
- ✅ BC avoidance validated
- ✅ LPS boost validated

### Phase 2 Success (Optimization)
- ✅ Train Sharpe > baseline + 0.2
- ✅ Test Sharpe ≥ 85% of train Sharpe
- ✅ Test Sharpe > baseline × 1.10
- ✅ Win rate +3-5% on test period

---

## File Locations

### Created Files (This Session)
- ✅ `bin/backfill_wyckoff_events.py` - Event backfill script
- ✅ `docs/WYCKOFF_OPTIMIZATION_STRATEGY.md` - Full strategic plan
- ✅ `WYCKOFF_INTEGRATION_CHECKLIST.md` - This checklist

### Existing Files (To Modify)
- ⏳ `bin/backtest_knowledge_v2.py` - Add boost logic (line ~450)
- ⏳ `configs/mvp/mvp_bull_no_wyckoff_v1.json` - Create for comparison

### Files to Create (Phase 2)
- ⏳ `bin/optimize_wyckoff_boosts.py` - Optuna optimization script
- ⏳ `configs/mvp/mvp_bull_wyckoff_optimized_v1.json` - Optimized config

### Output Files (Results)
- ⏳ `results/wyckoff_baseline_2024Q1-Q3.json` - Phase 1 baseline
- ⏳ `results/no_wyckoff_baseline_2024Q1-Q3.json` - Phase 1 comparison
- ⏳ `results/wyckoff_optimization_results.json` - Phase 2 optimization
- ⏳ `results/wyckoff_optimized_test.json` - Phase 2 validation

---

## Quick Reference Commands

### Backfill Events
```bash
python3 bin/backfill_wyckoff_events.py --asset BTC --start 2022-01-01 --end 2024-12-31 --config configs/wyckoff_events_config.json
```

### Smoke Test Integration
```bash
python3 bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_wyckoff_v1.json --asset BTC --start 2024-03-01 --end 2024-03-10 --verbose
```

### Run Baseline
```bash
python3 bin/backtest_knowledge_v2.py --config configs/mvp/mvp_bull_wyckoff_v1.json --asset BTC --start 2024-01-01 --end 2024-09-30 --output results/wyckoff_baseline_2024Q1-Q3.json
```

### Optimize (Phase 2 Only)
```bash
python3 bin/optimize_wyckoff_boosts.py --trials 40 --workers 4
```
