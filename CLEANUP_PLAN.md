# Bull Machine Repository Cleanup Plan
**Date:** 2025-11-14
**Status:** Ready for execution
**Estimated Space Recovery:** ~1.2 GB

## Executive Summary

This repository has accumulated significant bloat from rapid R&D development:
- **312 markdown files** (mostly in root)
- **895 JSON files** (scattered across root, results/, reports/)
- **1.0 GB results/** directory with experimental outputs
- **392 MB logs/** directory with debug logs
- **116 root-level markdown** files that should be archived
- **21 root-level Python scripts** that should be in bin/ or deleted
- Inadequate .gitignore allowing future bloat accumulation

---

## Current Repository Structure Analysis

### Disk Usage (Top Directories)
```
1.0G    results/          # Experimental backtest outputs
392M    logs/             # Debug and execution logs
109M    scripts/          # Mixed production/experimental scripts
81M     data/             # Feature stores and cached data
22M     telemetry/        # Large JSON masks (should not be in git)
20M     reports/          # Optuna trials and optimization results
8.7M    archive/          # Already archived content
7.6M    docs/             # Documentation (needs reorganization)
2.0M    bin/              # Production scripts (needs curation)
1.7M    models/           # ML artifacts
1.4M    dist/             # Build artifacts (should be ignored)
1.3M    engine/           # Production code (keep)
1.1M    bull_machine/     # Production code (keep)
772K    tests/            # Test suite (keep)
772K    configs/          # Configuration files (needs organization)
```

### File Count Summary
- **312** Markdown files total
- **116** Markdown files in root (should be ~3-5)
- **895** JSON files total
- **119** JSON files in results/
- **102** JSON files in reports/
- **21** Python scripts in root
- **10** Log/CSV/TXT/prof files in root

---

## Problems Identified

### 1. Root Directory Pollution (116 MD files, 21 PY files)
**Current root files that don't belong:**
- 70+ status/phase reports (MVP_PHASE*.md, PR*.md, SESSION*.md)
- 20+ analysis documents (ARCHETYPE*.md, OPTIMIZATION*.md)
- 15+ implementation plans (*_PLAN.md, *_STATUS.md)
- Test scripts (test_*.py, sweep_*.py, analyze_*.py)
- Experiment outputs (*.json, *.log, *.csv)

**Should only have:**
- README.md
- CHANGELOG.md
- LICENSE (if applicable)
- setup.py
- requirements.txt
- pytest.ini
- pyproject.toml
- Makefile
- .gitignore
- .pre-commit-config.yaml

### 2. Results Directory Bloat (1.0 GB)
**Large experimental subdirectories:**
- bench_v2/ (613 MB) - V2 benchmark results
- macro_fix_validation/ (222 MB) - Validation runs
- bench_v2_frontier/ (167 MB) - Frontier exploration
- 30+ optuna experiment folders (optuna_trap_*, optuna_step5_*)
- Hundreds of hybrid_signals_*.jsonl files (timestamped experiments)
- health_summary_*.json files (test artifacts)

### 3. Logs Directory Bloat (392 MB)
**Large log files:**
- bear_archetypes_*.log (83 MB each, 5+ files)
- Hundreds of timestamped backtest logs
- Debug and validation logs that should be temporary

### 4. Inadequate .gitignore
**Current .gitignore misses:**
- Experimental results folders (optuna_*, bench_v2*, frontier_*)
- Telemetry data (layer_masks.json - 101 MB in scripts/research/)
- Profile files (*.prof, *_profile.txt)
- Root-level test/sweep scripts
- Dist/ and build artifacts
- .mypy_cache/, .ruff_cache/, .pytest_cache/

### 5. Bin Directory Needs Curation
**Experimental scripts to archive:**
- backfill_*.py (4 files - one-time data migrations)
- fix_*.py (experimental fixes)
- diagnose_*.py (debug scripts)
- test_*.py (should be in tests/)

### 6. Configs Directory Needs Organization
**Current structure is scattered:**
- 15+ version subdirectories (v10_bases, v141, v142, v150, v160, v170, v171, v18, v185, v186, v19, v2, v3_replay_2024)
- Mixed production/experimental configs
- No clear "production" vs "experimental" separation

---

## Cleanup Strategy

### Phase 1: Enhance .gitignore (Prevent Future Bloat)
### Phase 2: Archive Root-Level Documentation
### Phase 3: Clean Results & Logs Directories
### Phase 4: Curate Bin Scripts
### Phase 5: Organize Configs
### Phase 6: Remove Build Artifacts
### Phase 7: Git Cleanup & Validation

---

## Detailed Action Plan

### Phase 1: Enhance .gitignore

**Add these patterns:**
```gitignore
# Build artifacts
build/
dist/
*.egg-info/

# Cache directories
.mypy_cache/
.ruff_cache/
.pytest_cache/
.cache/
__pycache__/

# Data directories (already covered but reinforce)
data/**/*
!data/.gitkeep
telemetry/**/*
!telemetry/.gitkeep

# Results & Experiments
results/**/*
!results/.gitkeep
!results/README.md

# Logs
logs/**/*
!logs/.gitkeep
!logs/README.md

# Reports (Optuna trials, optimization outputs)
reports/**/*
!reports/.gitkeep
!reports/README.md

# Root-level experimental outputs
/*_backtest_results*.json
/*_results*.json
/*_validation*.json
/*_optimization*.json
/*_fine_grid.json
/*_comparison.csv
/*.log
/*.prof
/*_profile.txt
/sweep_*.json
/hybrid_signals_*.jsonl
/health_summary_*.json
/portfolio_summary_*.json
/decision_log.jsonl
/fusion_*.jsonl
/signal_blocks.jsonl

# Root-level test/experiment scripts
/test_*.py
/sweep_*.py
/analyze_*.py
/download_*.py
/check_*.py
/monitor_*.sh
/optimize_all.sh

# Temporary analysis files
/institutional_testing_results.json
/config_patch_*.json
/exit_cfg_applied.json
/engine_factory.py
/bull_machine_config.py

# Model artifacts
models/**/*
!models/.gitkeep
!models/README.md

# Profiles
profiles/**/*

# Archive (optional - if you want to version control)
# archive/

# OS and IDE
.DS_Store
.idea/
.vscode/
*.swp
*.swo
```

### Phase 2: Archive Root-Level Documentation

**Create archive structure:**
```
docs/
├── archive/
│   ├── 2024-q4/
│   │   ├── mvp_phases/        # All MVP_PHASE*.md files
│   │   ├── pull_requests/     # All PR*.md files
│   │   ├── sessions/          # All SESSION*.md files
│   │   ├── archetype_work/    # All ARCHETYPE*.md files
│   │   ├── optimization/      # All OPTIMIZATION*.md, *_FINDINGS.md
│   │   ├── implementations/   # All *_PLAN.md, *_STATUS.md
│   │   └── legacy/            # Older reports
│   └── README.md              # Index of archived content
├── analysis/                  # Current analysis docs
├── reports/                   # Current reports
└── technical/                 # Technical documentation
```

**Move these root MD files to docs/archive/2024-q4/:**

**MVP Phases (12 files) → docs/archive/2024-q4/mvp_phases/**
- MVP_PHASE1_BLOCKER.md
- MVP_PHASE1_ROOT_CAUSE.md
- MVP_PHASE1_STATUS.md
- MVP_PHASE1.1_RESULTS.md
- MVP_PHASE2_*.md (10 files)
- MVP_PHASE4_*.md (4 files)
- MVP_ROADMAP.md

**Pull Requests (4 files) → docs/archive/2024-q4/pull_requests/**
- PR1_REVIEWER_NOTE.md
- PR6A_PROGRESS.md
- PR6A_STATUS.md
- PR_A_PARITY_DIAGNOSIS.md

**Sessions (2 files) → docs/archive/2024-q4/sessions/**
- SESSION_SUMMARY_2025-10-14.md
- SESSION_SUMMARY_2025-11-06.md

**Archetype Work (10 files) → docs/archive/2024-q4/archetype_work/**
- ARCHETYPE_IMPLEMENTATION_PLAN.md
- ARCHETYPE_PATHS_ANALYSIS.md
- ARCHETYPE_PATHS_LOCATIONS.md
- ARCHETYPE_WIRING_DIAGNOSIS.md
- COMPREHENSIVE_ARCHETYPE_AUDIT.md
- README_ARCHETYPE_ANALYSIS.md
- BEAR_ARCHETYPES_PHASE1_IMPLEMENTATION.md
- BEAR_ARCHETYPES_ZERO_MATCHES_DIAGNOSIS.md
- BEAR_ARCHITECTURE_EXECUTIVE_SUMMARY.md

**Optimization & Analysis (15 files) → docs/archive/2024-q4/optimization/**
- OPTIMIZATION_QUICKSTART.md
- OPTIMIZATION_TOOLS_GUIDE.md
- EXIT_OPTIMIZATION_PLAN.md
- OPTIMAL_CONFIG_BTC_2024.md
- OPTIMAL_CONFIGS_ALL_ASSETS_2024.md
- PERFORMANCE_OPTIMIZATION_FINDINGS.md
- OPTIMIZER_SIGNAL_GENERATION_ANALYSIS.md
- SPY_OPTIMIZER_FINDINGS.md
- SPY_ADAPTIVE_MAXHOLD_ANALYSIS.md
- TRAP_OPTIMIZATION_FAILURE_ANALYSIS.md
- BACKTEST_V2_OPTIMIZATION_REPORT.md
- OB_HIGH_OPTIMIZATION_SUMMARY.md
- WIRING_FIX_PLAN.md
- WIRING_FIX_STATUS.md

**Implementation Plans & Status (15 files) → docs/archive/2024-q4/implementations/**
- PHASE0_BRANCH_INTEGRATION_REPORT.md
- PHASE1_ALIAS_FIX_COMPLETE.md
- PHASE1_COMPLETE.md
- PHASE1_IMPLEMENTATION_PLAN.md
- PHASE2_STATUS.md
- BULL_MACHINE_V2_IMPLEMENTATION_PLAN.md
- REGIME_ROUTING_IMPLEMENTATION_PLAN.md
- KNOWLEDGE_V2_INTEGRATION.md
- KNOWLEDGE_V2_TESTING_STATUS.md
- TESTING_KNOWLEDGE_V2.md
- COMPLETE_KNOWLEDGE_ARCHITECTURE.md
- IMPLEMENTATION_ROADMAP.md

**Cleanup & Validation (4 files) → docs/archive/2024-q4/cleanup/**
- CLEANUP_EXECUTION_SUMMARY.md
- CLEANUP_REPORT.md
- OB_HIGH_COVERAGE_FIX_REPORT.md
- VALIDATION_CRITICAL_BUG_FOUND.md

**Configuration & Features (8 files) → docs/archive/2024-q4/features/**
- FEATURE_STORE_CONTENTS.md
- ML_FEATURE_INVENTORY.md
- ML_META_OPTIMIZER_ARCHITECTURE.md
- ML_ROADMAP.md
- CODE_REVIEW_IMPROVEMENTS.md
- ENHANCED_EXIT_STRATEGIES_DESIGN.md
- V2_CLEANUP_PLAN.md
- V2_CLEANUP_UPGRADED.md

**Replay & Validation (5 files) → docs/archive/2024-q4/validation/**
- REPLAY_EXIT_STRATEGY_ANALYSIS.md
- REPLAY_PARITY_RESOLUTION.md
- REPLAY_VALIDATION_SETUP.md
- HYBRID_RUNNER_VALIDATION.md
- HYBRID_VALIDATION_2024.md
- V19_3YEAR_VALIDATION_FINAL.md
- V1.8.1_TRUE_FUSION_COMPLETE.md

**Status & Progress (10 files) → docs/archive/2024-q4/status/**
- FINAL_STATUS.md
- NEXT_STEPS.md
- BASELINE_METRICS.md
- HANDOFF_NEXT_STEPS.md
- READY_TO_RUN.md
- WHILE_YOU_SLEPT.md
- FRONTIER_EXPLORATION_STATUS.md
- FRONTIER_UPDATE_TRIAL5.md
- FULL_BACKTEST_RESULTS_ANALYSIS.md
- PF20_RECOVERY_STATUS.md
- SCORE_PROPAGATION_BUG_FIX_REPORT.md

**Optuna & Routing (6 files) → docs/archive/2024-q4/optuna/**
- OPTUNA_PROGRESS_UPDATE.md
- OPTUNA_STATUS_2025_11_10.md
- OPTUNA_VALIDATION_PLAN.md
- STEP5_OPTUNA_SPEC.md
- ROUTER_V10_ANALYSIS_AND_RECOMMENDATIONS.md
- MASTER_OPTIMIZATION_ROADMAP.md

**Phase Progress (3 files) → docs/archive/2024-q4/phases/**
- PHASE_1_PROGRESS.md
- PHASE_1_QUICK_START.md

**Paper Trading (2 files) → docs/archive/2024-q4/paper_trading/**
- PAPER_TRADING_SETUP.md
- PAPER_TRADING_STATUS.md

**Audit & Reference (2 files) → docs/archive/2024-q4/audit/**
- AUDIT_INDEX.md
- AUDIT_QUICK_REFERENCE.md

**Keep in root (production docs):**
- README.md
- CHANGELOG.md
- DATA_DOWNLOAD_README.md
- DERIVATIVES_FETCH_GUIDE.md
- COMPREHENSIVE_SYSTEM_AUDIT_AND_MVP_ROADMAP.md (current roadmap)

**Keep in docs/ (current documentation):**
- docs/BEAR_FEATURE_PIPELINE_*.md (6 files - current work)
- docs/BEAR_MARKET_ANALYSIS_2022.md
- docs/BEAR_PATTERNS_*.md (3 files)
- docs/CLEANUP_COMPLETE.md
- docs/DOCUMENTATION_COMPLETE_SUMMARY.md
- docs/FEATURE_PIPELINE_AUDIT.md
- docs/FUNDING_RATES_EXPLAINED.md
- docs/OI_CHANGE_FAILURE_DIAGNOSIS.md
- docs/REGIME_ROUTING_*.md (2 files)
- docs/S5_*.md (2 files)

### Phase 3: Clean Results & Logs Directories

**Results cleanup (keep only production baselines):**

**DELETE (experimental runs - not referenced by production code):**
```bash
results/optuna_*                    # All Optuna experiment folders (~500 MB)
results/bench_v2_frontier/          # 167 MB
results/macro_fix_validation/       # 222 MB (keep summary only)
results/macro_fix_sanity/           # 44 KB
results/frontier_exploration/       # 1.1 MB
results/router_v10_*/               # Multiple validation folders
results/tiered_tests/
results/trap_validation/
results/20251001_203255/
results/*_backtest_results*.json    # Hundreds of timestamped results
results/hybrid_signals_*.jsonl      # Hundreds of signal dumps
results/health_summary_*.json       # Test artifacts
results/portfolio_summary_*.json    # Test artifacts
results/fusion_debug.jsonl          # 13 MB debug dump
results/fusion_validation.jsonl     # 3.7 MB
results/signal_blocks.jsonl         # 4.9 MB
results/decision_log.jsonl
results/open_fail.jsonl
results/open_ok.jsonl
results/*.log                       # Log files in results
```

**KEEP (production benchmarks):**
```bash
results/bench_v2/                   # 613 MB - Current production benchmarks
results/bear_patterns/              # Current bear pattern work
results/archive/                    # Already archived
results/.gitkeep
results/README.md (create to document structure)
```

**Logs cleanup:**

**DELETE (debug/experimental logs):**
```bash
logs/bear_archetypes_*.log          # 83 MB each, 5+ files (~415 MB)
logs/btc_3year_*.log
logs/BTC_confluence_*.log
logs/btc_exit_opt.log
logs/baseline_*.log
logs/backfill_*.log
logs/*_backtest_*.log
logs/*_validation_*.log
logs/*_feature_store_*.log
```

**KEEP:**
```bash
logs/paper_trading/                 # Production paper trading logs
logs/archive/                       # Already archived
logs/.gitkeep
logs/README.md (create)
```

**Create results/README.md:**
```markdown
# Results Directory

This directory contains production benchmark results and current experiment outputs.

## Structure
- `bench_v2/` - Production v2 benchmarks
- `bear_patterns/` - Bear market pattern analysis
- `archive/` - Historical results

## Guidelines
- Experimental results should be gitignored
- Only production-validated benchmarks should be committed
- Use timestamped subdirectories for experiments: `YYYYMMDD_experiment_name/`
```

### Phase 4: Curate Bin Scripts

**Move to bin/archive/experimental/ (one-time migrations):**
```bash
bin/backfill_liquidity_score.py
bin/backfill_liquidity_score_optimized.py
bin/backfill_ob_high.py
bin/backfill_ob_high_optimized.py
bin/backfill_missing_macro_features.py
bin/fix_oi_change_pipeline.py
bin/test_ob_high_optimization.py
```

**Move to bin/archive/diagnostics/ (debug scripts):**
```bash
bin/diagnose_eth_runtime.py
bin/debug_adaptive_logic.py
bin/check_pr3_nonzero_rates.py
```

**Move to tests/ (test scripts in bin):**
```bash
bin/test_feature_store_scores.py → tests/integration/
```

**Keep in bin/ (production scripts):**
- bin/backtest_knowledge_v2.py
- bin/backtest_router_v10*.py
- bin/build_feature_store*.py
- bin/build_macro_dataset.py
- bin/build_mtf_feature_store.py
- bin/build_wyckoff_cache.py
- bin/bull_machine_cli.py
- bin/analyze_*.py (analysis tools)
- bin/compare_*.py (comparison tools)
- All other production utilities

**Create bin/README.md:**
```markdown
# Bin Scripts

Production scripts for the Bull Machine trading system.

## Categories
- **Backtesting**: backtest_*.py
- **Feature Engineering**: build_*.py, add_*.py
- **Analysis**: analyze_*.py
- **Optimization**: optimize_*.py, consolidate_*.py
- **CLI**: bull_machine_cli.py

## Archive
- `archive/experimental/` - One-time data migrations
- `archive/diagnostics/` - Debug and diagnostic scripts
```

### Phase 5: Organize Configs

**Proposed structure:**
```
configs/
├── production/
│   ├── frozen/                 # Move from configs/frozen/
│   ├── live/                   # Move from configs/live/
│   ├── paper_trading/          # Move from configs/paper_trading/
│   ├── mvp_*.json              # Move production MVPs
│   ├── regime_routing_production_v1.json
│   └── bear_archetypes_phase1.json
├── experimental/
│   ├── adaptive/               # Move from configs/adaptive/
│   ├── sweep/                  # Move from configs/sweep/
│   └── knowledge_v2/           # Move from configs/knowledge_v2/
├── archive/
│   ├── v10_bases/
│   ├── v141/ v142/ v150/ v160/ v170/ v171/
│   ├── v18/ v185/ v186/ v19/ v2/
│   └── v3_replay_2024/
├── schema/                     # Keep as is
├── deltas/                     # Keep as is
└── README.md
```

**Keep in configs/ root:**
- production/
- experimental/
- archive/
- schema/
- deltas/
- stock/
- README.md

### Phase 6: Remove Build Artifacts & Cache

**DELETE:**
```bash
dist/                           # 1.4 MB build artifacts
.mypy_cache/                    # Type checking cache
.ruff_cache/                    # Linter cache
.pytest_cache/                  # Test cache
bull_machine.egg-info/          # Package metadata
*.pyc                           # Compiled Python
__pycache__/                    # Python cache
.DS_Store                       # macOS metadata
```

### Phase 7: Clean Root-Level Experimental Files

**DELETE from root:**

**JSON Results:**
```bash
btc_backtest_results_*.json (4 files)
btc_fine_grid.json
btc_results_comparison.csv
btc_results.json
btc_v19_*.json (5 files)
eth_production_backtest_results.json
sweep_results_*.json (12 files)
institutional_testing_results.json
config_patch_ml.json
exit_cfg_applied.json
optimization_results_v19.json
```

**Python Scripts:**
```bash
test_archetype_debug.py
test_boms_diagnostic.py
test_boms_4h.py
test_feature_store_scores.py
test_fusion_windowing.py
test_hooks_firing.py
test_macro_extraction.py
test_macro_loading.py
test_optimization.py
analyze_threshold_sensitivity.py
sweep_parameters.py
sweep_hybrid_params.py
sweep_thresholds.py
check_pr3_nonzero_rates.py
download_vix_2024.py
engine_factory.py
bull_machine_config.py
```

**Shell Scripts:**
```bash
monitor_and_compile_results.sh
optimize_all.sh
WATCH_CRITICAL_PROCESSES.sh
RUN_TESTS.sh
```

**Logs/CSV/TXT:**
```bash
q3_2024_hybrid.log
q3_2024_validation.log
q3_2024_validation_results.txt
full_year_test_output.log
monitor_output.log
threshold_sensitivity_sweep.csv
```

**Misc:**
```bash
conftest.py (only if duplicate of tests/conftest.py)
.cleanup_plan.txt
```

---

## Execution Scripts

### Script 1: cleanup_repository.sh (Main cleanup script)

See next section for complete script.

### Script 2: validate_cleanup.sh (Post-cleanup validation)

See next section for validation script.

---

## Safety Measures

1. **Backup Before Cleanup**
   ```bash
   # Create git tag for rollback
   git tag pre-cleanup-2025-11-14

   # Create tarball backup of entire repo
   cd ..
   tar -czf Bull-machine-backup-2025-11-14.tar.gz Bull-machine-/
   ```

2. **Dry Run Mode**
   - Script supports `--dry-run` flag
   - Shows what would be deleted without actually deleting

3. **Incremental Execution**
   - Script can be run phase by phase
   - Each phase creates a commit for easy rollback

4. **Validation Checks**
   - Verify no production code is broken
   - Check import statements still resolve
   - Run test suite after cleanup

---

## Expected Outcomes

### Space Recovery
- **Before:** ~1.9 GB repository size
- **After:** ~650 MB repository size
- **Recovery:** ~1.25 GB (66% reduction)

### File Reduction
- **Root MD files:** 116 → 5
- **Root PY files:** 21 → 0
- **Total JSON files:** 895 → ~100
- **Results folder:** 1.0 GB → ~620 MB
- **Logs folder:** 392 MB → ~10 MB

### Structure Improvement
```
Bull-machine/
├── README.md
├── CHANGELOG.md
├── setup.py
├── requirements.txt
├── pytest.ini
├── pyproject.toml
├── Makefile
├── .gitignore (enhanced)
├── .pre-commit-config.yaml
├── bin/                        # Curated production scripts
├── bull_machine/               # Production code
├── configs/                    # Organized (prod/exp/archive)
├── data/                       # Gitignored feature stores
├── docs/                       # Organized documentation
│   ├── archive/                # Historical docs by quarter
│   ├── analysis/
│   ├── reports/
│   └── technical/
├── engine/                     # Production code
├── logs/                       # Gitignored logs
├── models/                     # Gitignored ML artifacts
├── reports/                    # Gitignored optimization results
├── results/                    # Gitignored experiment outputs
├── tests/                      # Test suite
└── scripts/                    # Research scripts (organized)
```

---

## Post-Cleanup Maintenance

### .gitignore Enforcement
- Add pre-commit hook to prevent committing ignored patterns
- Regular audits (monthly) for new bloat

### Documentation Policy
1. **Session summaries** → docs/archive/YYYY-QQ/sessions/
2. **Implementation plans** → docs/archive/YYYY-QQ/implementations/
3. **Analysis reports** → docs/archive/YYYY-QQ/analysis/
4. **Keep in root:** Only README.md, CHANGELOG.md, current roadmap

### Results Policy
1. **Experiments** → results/YYYYMMDD_experiment_name/ (gitignored)
2. **Production benchmarks** → results/production/benchmark_name/
3. **Archive after 90 days** → results/archive/YYYY-QQ/

### Bin Scripts Policy
1. **One-time migrations** → bin/archive/experimental/
2. **Production utilities** → bin/
3. **Test scripts** → tests/
4. **Research scripts** → scripts/research/

---

## Risk Assessment

### Low Risk
- Deleting experimental results (not referenced)
- Removing cache directories
- Archiving old documentation
- Cleaning logs

### Medium Risk
- Moving bin scripts (validate imports)
- Reorganizing configs (validate references)

### High Risk
- None (all production code and configs preserved)

### Mitigation
- Git tag for rollback: `pre-cleanup-2025-11-14`
- Tarball backup: `Bull-machine-backup-2025-11-14.tar.gz`
- Incremental commits per phase
- Test suite validation after each phase

---

## Timeline

- **Phase 1 (gitignore):** 15 minutes
- **Phase 2 (docs):** 30 minutes
- **Phase 3 (results/logs):** 45 minutes
- **Phase 4 (bin):** 30 minutes
- **Phase 5 (configs):** 30 minutes
- **Phase 6 (artifacts):** 15 minutes
- **Phase 7 (root files):** 30 minutes
- **Validation:** 30 minutes

**Total:** ~3.5 hours for complete cleanup

---

## Next Steps

1. Review this plan
2. Create backup: `git tag pre-cleanup-2025-11-14`
3. Run cleanup script with `--dry-run`
4. Review dry-run output
5. Execute cleanup script
6. Run validation script
7. Commit changes
8. Update team documentation

---

## Questions for User

1. Should `archive/` directory be kept in git or gitignored?
2. Are there any specific experimental results that should be preserved?
3. Should we keep any specific bin scripts that aren't obviously production?
4. Do you want to version control `results/bench_v2/` (613 MB) or move to external storage?
5. Should telemetry/ directory be completely gitignored or keep some files?
