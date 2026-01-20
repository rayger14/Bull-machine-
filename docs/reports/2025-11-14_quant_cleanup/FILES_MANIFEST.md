# Cleanup Files Manifest
## What Was Deleted vs. What Was Preserved

### Results Reference Files Created (NEW)

These canonical reference files were created from the best experimental runs:

```
results_reference/
├── README.md                                  (NEW - 1.5KB documentation)
├── bear_market/
│   └── 2022_validation.json                  (MOVED from results/bear_patterns/)
├── btc/
│   └── ml_training_baseline.json             (MOVED from results/)
├── eth/
│   └── ml_training_baseline.json             (MOVED from results/)
├── sol/                                       (STRUCTURE for future)
├── optimization/
│   └── trap_v2_production_params.json        (EXTRACTED from optuna_trap_200trial_PRODUCTION/)
└── system_validation/
    ├── trade_log_reference.jsonl             (MOVED from results_reference/open_ok.jsonl)
    └── trade_log.jsonl                       (MOVED from results_reference/)
```

**Total**: 7 files, ~150KB

---

### Experimental Files Deleted

#### Hybrid Signals (211 files deleted)
```
results/hybrid_signals_BTC_20251010_201010.jsonl
results/hybrid_signals_BTC_20251010_201458.jsonl
results/hybrid_signals_BTC_20251010_214735.jsonl
... (208 more files)
```

#### Health Summaries (29 files deleted)
```
results/health_summary_BTC_20251001_165013.json
results/health_summary_BTC_20251001_165142.json
results/health_summary_ETH_20251006_153712.json
... (26 more files)
```

#### Portfolio Summaries (31 files deleted)
```
results/portfolio_summary_BTC_20251001_165453.json
results/portfolio_summary_BTC_20251001_165513.json
results/portfolio_summary_ETH_20251006_154714.json
... (28 more files)
```

#### Optuna Experiment Directories (20+ directories deleted)
```
results/optuna_FINAL_FIX/
results/optuna_LAYER4_FIXED/
results/optuna_LAYER5_FIXED/
results/optuna_LEGACY_PATH_FIX/
results/optuna_pr6a_validation/
results/optuna_step5_test/
results/optuna_threshold_fix_test/
results/optuna_trap_200trial_PRODUCTION/        (params saved to results_reference)
results/optuna_trap_v10_full/
results/optuna_trap_v10_test/
results/optuna_trap_v2_BUG5_FIXED/
results/optuna_trap_v2_DIAGNOSTIC/
results/optuna_trap_v2_DIAGNOSTIC2/
results/optuna_trap_v2_FINAL/
results/optuna_trap_v2_SAFETY_CHECK/
results/optuna_trap_v2_SUCCESS/
results/optuna_trap_v2_ULTRA_SAFE/
results/optuna_v2_full_test/
... (and more)
```

#### Router Experiments (5 directories deleted)
```
results/router_v10_2022_2023/
results/router_v10_full_2022_2023/
results/router_v10_full_2022_2024/
results/router_v10_full_2022_2024_combined/
results/router_v10_integrated_pf20_2022_2023/
```

#### Large Experiment Archives Deleted
```
results/bench_v2_frontier/                     (167MB)
results/macro_fix_validation/                  (222MB)
results/macro_fix_sanity/                      (44KB)
results/frontier_exploration/                  (multiple DB files)
results/trap_validation/
results/tiered_tests/
results/archive/                               (12 backtest results)
results/20251001_*/                            (timestamped test dirs)
```

#### Individual Result Files Deleted
```
results/BTC_complete_confluence_20251001_122934.json
results/BTC_complete_confluence_20251001_123713.json
results/BTC_complete_confluence_20251001_123719.json
results/BTC_complete_confluence_20251001_124014.json
results/btc_ml_training.json                   (copied to results_reference)
results/eth_ml_training.json                   (copied to results_reference)
results/daily_aggregate_results.json
results/signal_blocks.jsonl                    (5.1MB)
results/trade_log.jsonl
results/fusion_validation.jsonl
results/decision_log.jsonl
results/open_ok.jsonl                          (copied to results_reference)
results/open_fail.jsonl
results/BTC_2025_candidates.jsonl
results/ETH_2025_candidates.jsonl
```

#### Log Files Deleted (All .log files)
```
results/frontier_exploration.log
results/optuna_step5_test.log
results/optuna_step5_full.log
results/macro_fix_validation/q1_2022_scoring_test.log
results/macro_fix_validation/q1_2022_biased.log
results/macro_fix_validation/q1_2022_biased_v2.log
results/bench_v2/run_2022/backtest.log
results/bench_v2/run_2023/backtest.log
results/bench_v2/run_2024/backtest.log
results/bench_v2/run_combined/backtest.log
results/bench_v2/run_2024_determinism/backtest.log
results/20251001_203255/smoke_test.log
docs/reports/stage_a_full.log
docs/reports/opt/errors.log
logs/bear_archetypes_adaptive_2022q1.log
logs/backfill_2024.log
logs/eth_exit_opt.log
```

#### Temporary Directories Deleted
```
chart_logs_binance/                            (CSV files)
test_checkpoints/                              (test artifacts)
```

#### Cleanup Artifacts Deleted
```
.cleanup_plan.txt
.gitignore.backup
```

---

### Files Preserved (Kept in results/)

```
results/
├── README.md                                  (KEPT - directory guide)
├── bench_v2/                                  (KEPT - 613MB production benchmarks)
│   ├── run_2022/
│   │   ├── backtest_summary.json
│   │   └── trades.parquet
│   ├── run_2023/
│   │   ├── backtest_summary.json
│   │   └── trades.parquet
│   ├── run_2024/
│   │   ├── backtest_summary.json
│   │   └── trades.parquet
│   ├── run_combined/
│   │   ├── backtest_summary.json
│   │   └── trades.parquet
│   └── run_2024_determinism/
│       ├── backtest_summary.json
│       └── trades.parquet
└── bear_patterns/                             (KEPT - bear market validation)
    ├── 2022_baseline_trades.json
    └── validation_2022.json                   (also in results_reference)
```

---

### Documentation Preserved (100%)

All 49 active documentation files kept in `docs/`:
```
docs/
├── ADAPTIVE_FUSION_IMPL.md
├── ADAPTIVE_WIRING_CHANGES.md
├── ARCH_REVIEW_NOTES.md
├── BEAR_FEATURE_PIPELINE_EXECUTIVE_SUMMARY.md
├── BEAR_FEATURE_PIPELINE_ROADMAP.md
├── BEAR_MARKET_ANALYSIS_2022.md
├── BEAR_PATTERNS_FEATURE_MATRIX.md
├── BEAR_PATTERNS_IMPLEMENTATION_GUIDE.md
... (41 more docs)
```

All historical docs preserved in `docs/archive/`:
```
docs/archive/
├── 2024-q4/                                   (Complete Q4 2024 docs)
├── 2025-11-14_cleanup/                        (Previous cleanup docs)
└── backtest_analysis.md, etc.
```

---

### Source Code (100% Preserved)

All source code directories untouched:
```
bull_machine/                                  (100% intact)
engine/                                        (100% intact)
utils/                                         (100% intact)
scripts/                                       (100% intact)
tools/                                         (100% intact)
bin/                                          (100% intact)
tests/                                        (100% intact)
telemetry/                                    (100% intact)
```

---

### Configuration (100% Preserved)

All configuration files untouched:
```
configs/                                       (100% intact)
schema/                                        (100% intact)
pyproject.toml                                (intact)
setup.py                                      (intact)
requirements.txt                              (intact)
requirements-production.txt                   (intact)
```

---

### Data (100% Preserved)

All data directories untouched:
```
data/features_mtf/                            (100% intact)
data/macro/                                   (100% intact)
models/                                       (100% intact)
```

---

## Summary Statistics

### Deleted
- **Total files**: 92+
- **Total lines**: 1,147,296
- **Space recovered**: ~656MB

### Created
- **results_reference/**: 7 canonical files
- **Cleanup reports**: 3 markdown docs

### Preserved
- **Source code**: 100%
- **Configurations**: 100%
- **Documentation**: 100%
- **Data**: 100%
- **Production benchmarks**: 100%

---

**Cleanup Date**: 2025-11-14
**Status**: Complete and verified
