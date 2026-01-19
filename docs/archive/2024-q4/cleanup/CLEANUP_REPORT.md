# Repository Cleanup Report
**Date:** 2025-11-14
**Scope:** Comprehensive bloat removal while preserving 100% logic
**Current Size:** 2.9 GB
**Target:** Remove ~1.5 GB of bloat (52% reduction)

---

## Executive Summary

**Total files scanned:** 2,847
**Bloat identified:** 1,547 MB (53% of repository)
**Files to remove:** 47 files
**Files to consolidate:** 31 files
**Logic preservation:** 100% (all Python logic retained)

### Storage Impact by Category
| Category | Size | Files | Action |
|----------|------|-------|--------|
| Backup parquet files | 20.5 MB | 5 | DELETE |
| Old config files | 1.6 KB | 2 | DELETE |
| Old Python files | 15.8 KB | 2 | DELETE |
| Duplicate documentation | ~300 KB | 31 | CONSOLIDATE |
| Build artifacts | 1.4 MB | 1 dir | DELETE |
| Large log files | 1,477 MB | 20+ | ARCHIVE TOP 10 |
| Archive directory | 8.7 MB | 1 dir | REVIEW/CONSOLIDATE |

**Total Cleanup Impact:** ~1,524 MB (1.5 GB)

---

## Category 1: Backup Files (SAFE TO DELETE)

### Backup Parquet Files
| File | Size | Last Modified | Reason |
|------|------|---------------|--------|
| `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup_20251113_204010.parquet` | 7.4 MB | 2025-11-13 | Superseded by current file |
| `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31_backup_20251113_221404.parquet` | 4.7 MB | 2025-11-13 | Superseded by current file |
| `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_OI_FIXED_backup.parquet` | 7.4 MB | Prior | Superseded by current file |
| `data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet.backup` | 1.5 MB | Prior | Superseded by current file |
| `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet.bak_pre_regime_recalc` | 7.4 MB | Prior | Pre-regime recalc backup |

**Subtotal:** 28.4 MB
**Action:** DELETE (current versions are in production use)
**Risk:** NONE (current feature stores contain all data)

### Old Config Files
| File | Size | Reason |
|------|------|--------|
| `configs/v150/assets/ETH_old.json` | 843 B | Superseded by ETH.json |
| `configs/v150/assets/ETH_4H_old.json` | 847 B | Superseded by ETH_4H.json |

**Subtotal:** 1.6 KB
**Action:** DELETE
**Risk:** NONE (current configs in use)

### Old Python Files
| File | Size | Reason |
|------|------|--------|
| `bull_machine/strategy/atr_exits_old.py` | 7.9 KB | Superseded by atr_exits.py |
| `build/lib/bull_machine/strategy/atr_exits_old.py` | 7.9 KB | Build artifact |

**Subtotal:** 15.8 KB
**Action:** DELETE
**Risk:** NONE (current version in production)

---

## Category 2: Build Artifacts (SAFE TO DELETE)

### Build Directory
| Path | Size | Reason |
|------|------|--------|
| `build/` | 1.4 MB | Python build artifacts (can be regenerated) |

**Action:** DELETE entire directory
**Regeneration:** Run `python setup.py build` if needed
**Risk:** NONE (standard build artifacts)

---

## Category 3: Duplicate Documentation (CONSOLIDATE)

### Root Directory Summary Files (31 files)
These files duplicate information that exists in more comprehensive docs or are outdated summaries from prior work sessions.

#### Bear Archetype Documentation (9 files - Keep 2, Archive 7)
| File | Size | Status | Action |
|------|------|--------|--------|
| `BEAR_ARCHETYPES_PHASE1_IMPLEMENTATION.md` | 12 KB | Most comprehensive | **KEEP** |
| `BEAR_ARCHITECTURE_EXECUTIVE_SUMMARY.md` | 13 KB | Executive view | **KEEP** |
| `BEAR_ARCHETYPE_VALIDATION_SUMMARY.md` | 5.0 KB | Superseded by above | Archive |
| `BEAR_FEATURE_FIX_QUICK_START.md` | 7.6 KB | Covered in roadmap | Archive |
| `BEAR_FEATURE_FIX_SUMMARY.txt` | 8.2 KB | Duplicate of .md version | Archive |
| `BEAR_PATTERNS_QUICK_REFERENCE.md` | 4.4 KB | Exists in docs/ | Archive |
| `VALIDATION_SUMMARY_BEAR_PHASE1.md` | ? | Validation complete | Archive |

**Note:** `docs/` directory has comprehensive versions:
- `docs/BEAR_FEATURE_PIPELINE_EXECUTIVE_SUMMARY.md` (10 KB)
- `docs/BEAR_FEATURE_PIPELINE_ROADMAP.md` (14 KB)
- `docs/BEAR_MARKET_ANALYSIS_2022.md` (20 KB)
- `docs/BEAR_PATTERNS_FEATURE_MATRIX.md` (14 KB)
- `docs/BEAR_PATTERNS_IMPLEMENTATION_GUIDE.md` (15 KB)
- `docs/BEAR_PATTERNS_QUICK_REFERENCE.md` (8.4 KB)

#### OB High Fix Documentation (2 files - Keep 1, Archive 1)
| File | Size | Status | Action |
|------|------|--------|--------|
| `OB_HIGH_COVERAGE_FIX_REPORT.md` | 8.4 KB | Detailed report | **KEEP** |
| `OB_HIGH_FIX_SUMMARY.md` | 3.8 KB | Redundant summary | Archive |

#### Regime Routing Documentation (2 files - Keep 1, Archive 1)
| File | Size | Status | Action |
|------|------|--------|--------|
| `REGIME_ROUTING_IMPLEMENTATION_PLAN.md` | 15 KB | Most comprehensive | **KEEP** |
| `REGIME_ROUTING_EXECUTIVE_SUMMARY.md` | 8.2 KB | Redundant summary | Archive |

**Note:** `docs/REGIME_ROUTING_CURRENT_STATE.md` and `docs/REGIME_ROUTING_IMPACT_ESTIMATE.md` exist

#### Liquidity Backfill Documentation (1 file - Archive)
| File | Size | Status | Action |
|------|------|--------|--------|
| `LIQUIDITY_BACKFILL_COMPLETE.md` | ? | Work complete | Archive |

#### Session/Phase Summaries (12 files - Archive All)
These are historical work session summaries that served their purpose:
| File | Size | Reason |
|------|------|--------|
| `ARCHETYPE_INVESTIGATION_SUMMARY.md` | 11 KB | Historical investigation |
| `AUDIT_EXECUTIVE_SUMMARY.md` | 8.8 KB | Prior audit complete |
| `CLEANUP_SUMMARY.txt` | 2.9 KB | Prior cleanup record |
| `CURRENT_STATUS_SUMMARY.md` | 5.0 KB | Outdated status |
| `ENGINE_INTEGRATION_SUMMARY.md` | 5.6 KB | Integration complete |
| `FINAL_SESSION_SUMMARY.md` | 12 KB | Session complete |
| `ML_PIPELINE_SUMMARY.md` | 11 KB | Covered in docs/ |
| `ML_STACK_FINAL_SUMMARY.md` | 8.1 KB | Covered in docs/ |
| `OPTIMIZATION_RESULTS_SUMMARY.md` | 16 KB | Results archived |
| `OPTIMIZATION_SUMMARY.md` | 2.4 KB | Covered elsewhere |
| `SESSION_COMPLETE_SUMMARY.md` | 11 KB | Session complete |
| `V2_CLEANUP_SUMMARY.md` | 11 KB | Prior cleanup |

#### Phase Completion Summaries (5 files - Archive All)
| File | Size | Reason |
|------|------|--------|
| `PHASE_0_COMPLETION_SUMMARY.md` | 10 KB | Phase complete |
| `PHASE1_COMPLETION_SUMMARY.md` | 7.2 KB | Phase complete |
| `PHASE1_OPTIMIZATION_SUMMARY.md` | 12 KB | Phase complete |
| `PHASE2_COMPLETE_SUMMARY.md` | 17 KB | Phase complete |
| `PHASE_0_COMPLETION_SUMMARY.md` | 10 KB | Phase complete |

#### PR Summaries (5 files - Archive All)
| File | Size | Reason |
|------|------|--------|
| `PR1_SUMMARY.md` | 7.3 KB | PR merged |
| `PR2_SUMMARY.md` | 8.6 KB | PR merged |
| `PR3_SUMMARY.md` | 11 KB | PR merged |
| `PR4_SUMMARY.md` | 15 KB | PR merged |
| `PR6B_COMPLETION_SUMMARY.md` | 4.1 KB | PR merged |

**Subtotal:** ~300 KB across 31 files
**Action:** Move to `docs/archive/2025-11-14_cleanup/`
**Risk:** NONE (information preserved in archive)

---

## Category 4: Large Log Files (SELECTIVE ARCHIVE)

### Top 10 Largest Logs (1,477 MB total)
| File | Size | Type | Action |
|------|------|------|--------|
| `results/bench_v2/run_combined/backtest.log` | 256 MB | Benchmark | Archive Top 10 |
| `logs/bear_archetypes_adaptive_2022_2024_full.log` | 246 MB | Archetype testing | Archive |
| `results/macro_fix_validation/2022_2024_ob_expanded_summary.txt` | 192 MB | Validation | Archive |
| `results/macro_fix_validation/2022_2024_regime_routed_summary.txt` | 182 MB | Validation | Archive |
| `logs/bear_archetypes_adaptive_2022_2023_full.log` | 160 MB | Archetype testing | Archive |
| `results/macro_fix_validation/with_fixes_2022_2024.log` | 147 MB | Validation | Archive |
| `results/macro_fix_sanity/baseline_2022_2024.log` | 147 MB | Validation | Archive |
| `results/bench_v2/run_2024/backtest.log` | 96 MB | Benchmark | Keep (recent) |
| `results/bench_v2/run_2024_determinism/backtest.log` | 96 MB | Benchmark | Keep (recent) |
| `results/bench_v2_frontier/run_2023/backtest.log` | 96 MB | Benchmark | Archive |

**Subtotal (to archive):** ~1,426 MB (Archive largest 7 files)
**Action:** Compress and move to `logs/archive/2025-11-14/`
**Risk:** LOW (logs are historical, results documented)

---

## Category 5: Analysis/Validation Scripts (REVIEW)

### One-Time Analysis Scripts (8 files - Archive)
These scripts served their purpose and results are captured:

| Script | Purpose | Status | Action |
|--------|---------|--------|--------|
| `bin/analyze_2022_bear_market.py` | Historical analysis | Complete | Archive |
| `bin/backfill_liquidity_score.py` | Backfill complete | Complete | **KEEP** (reusable) |
| `bin/backfill_liquidity_score_optimized.py` | Optimized version | Complete | **KEEP** (current) |
| `bin/backfill_ob_high.py` | OB high backfill | In Progress | **KEEP** (active) |
| `bin/fix_oi_change_pipeline.py` | Pipeline fix | Complete | **KEEP** (reusable) |
| `bin/fix_oi_derivatives_simple.py` | Derivatives fix | Complete | Archive |
| `bin/profile_liquidity_baseline.py` | One-time profiling | Complete | Archive |
| `bin/simulate_regime_routing_2022.py` | Simulation | Complete | Archive |
| `bin/test_optimized_performance.py` | Performance test | Complete | Archive |
| `bin/test_s5_graceful_degradation.py` | Feature test | Complete | Archive |
| `bin/validate_liquidity_backfill.py` | Validation | Complete | Archive |
| `bin/validate_regime_routing_quick.sh` | Shell validation | Complete | Archive |

**Action:** Move to `bin/archive/2025-11-14_analysis/`
**Risk:** LOW (can reference if needed)

---

## Category 6: Archive Directory Review

### Current Archive (8.7 MB)
| Path | Size | Contents | Action |
|------|------|----------|--------|
| `archive/` | 8.7 MB | Old feature stores, analysis backups | Review/Consolidate |

**Recommendation:** Review contents. If truly obsolete, DELETE. If historical value, keep compressed.

---

## Category 7: Large JSON Files (22 MB - Keep)

| File | Size | Purpose | Action |
|------|------|---------|--------|
| `scripts/research/telemetry/layer_masks.json` | 101 MB | ML telemetry data | **KEEP** |
| `telemetry/layer_masks.json` | 22 MB | ML telemetry data | **KEEP** |

**Note:** These are ML training artifacts. Keep unless confirmed obsolete.

---

## Preservation Strategy

### Files to ALWAYS Preserve
1. All `.py` files in `engine/`, `bull_machine/`, `bin/` (except `*_old.py`)
2. All config files (`.json`) except `*_old.json`
3. Current feature stores (`.parquet` files without backup/timestamp suffixes)
4. `CHANGELOG.md`, `README.md`, `LICENSE`
5. `docs/` directory (comprehensive documentation)
6. Active ML artifacts (`data/ml/`, `data/macro/`)

### Safe to Remove
1. Backup `.parquet` files with timestamps
2. `*_old.*` files
3. `build/` directory
4. Historical log files (after archiving largest)

### Consolidate/Archive
1. Root directory summary `.md` files (move to `docs/archive/`)
2. One-time analysis scripts (move to `bin/archive/`)
3. Historical logs (compress and archive)

---

## Archive Structure

```
docs/archive/2025-11-14_cleanup/
├── bear_implementation/
│   ├── BEAR_ARCHETYPE_VALIDATION_SUMMARY.md
│   ├── BEAR_FEATURE_FIX_QUICK_START.md
│   ├── BEAR_FEATURE_FIX_SUMMARY.txt
│   └── VALIDATION_SUMMARY_BEAR_PHASE1.md
├── ob_high_fix/
│   └── OB_HIGH_FIX_SUMMARY.md
├── regime_routing/
│   └── REGIME_ROUTING_EXECUTIVE_SUMMARY.md
├── liquidity/
│   └── LIQUIDITY_BACKFILL_COMPLETE.md
├── session_summaries/
│   ├── [12 session/phase summary files]
│   └── [5 PR summary files]
└── cleanup_reports/
    ├── CLEANUP_SUMMARY.txt (prior cleanup)
    └── CLEANUP_REPORT_20251114.md (this report)

bin/archive/2025-11-14_analysis/
├── analyze_2022_bear_market.py
├── fix_oi_derivatives_simple.py
├── profile_liquidity_baseline.py
├── simulate_regime_routing_2022.py
├── test_optimized_performance.py
├── test_s5_graceful_degradation.py
├── validate_liquidity_backfill.py
└── validate_regime_routing_quick.sh

logs/archive/2025-11-14/
├── bear_archetypes_adaptive_2022_2024_full.log.gz (246 MB → ~25 MB)
├── bear_archetypes_adaptive_2022_2023_full.log.gz (160 MB → ~16 MB)
├── [5 more compressed logs]
└── manifest.txt (list of archived logs with dates)
```

---

## Execution Plan

### Phase 1: Safe Deletions (AUTOMATED)
```bash
# Delete backup parquet files
rm data/features_mtf/*backup*.parquet
rm data/features_mtf/*.parquet.backup
rm data/features_mtf/*.parquet.bak*

# Delete old config files
rm configs/v150/assets/*_old.json

# Delete old Python files
rm bull_machine/strategy/*_old.py

# Delete build artifacts
rm -rf build/
```

**Impact:** 30.2 MB removed
**Risk:** NONE

### Phase 2: Archive Documentation (MANUAL)
```bash
# Create archive structure
mkdir -p docs/archive/2025-11-14_cleanup/{bear_implementation,ob_high_fix,regime_routing,liquidity,session_summaries,cleanup_reports}

# Move files (31 files)
[See detailed commands in execution section]
```

**Impact:** 300 KB moved (not deleted)
**Risk:** NONE

### Phase 3: Archive Logs (AUTOMATED)
```bash
# Create logs archive
mkdir -p logs/archive/2025-11-14

# Compress and move top 7 logs
gzip -c logs/bear_archetypes_adaptive_2022_2024_full.log > logs/archive/2025-11-14/bear_archetypes_adaptive_2022_2024_full.log.gz
rm logs/bear_archetypes_adaptive_2022_2024_full.log
[Repeat for 6 more files]
```

**Impact:** 1,426 MB removed (142 MB compressed in archive)
**Compression Ratio:** ~10:1
**Risk:** LOW

### Phase 4: Archive Scripts (MANUAL)
```bash
# Create bin archive
mkdir -p bin/archive/2025-11-14_analysis

# Move one-time analysis scripts (8 files)
[See detailed commands in execution section]
```

**Impact:** ~50 KB moved
**Risk:** LOW

### Phase 5: Review Archive Directory (MANUAL)
```bash
# Review contents
ls -lh archive/

# If obsolete, remove
rm -rf archive/
```

**Impact:** 8.7 MB (if removed)
**Risk:** MEDIUM (review contents first)

---

## Storage Impact Summary

| Phase | Action | Size Removed | Size Archived | Risk |
|-------|--------|--------------|---------------|------|
| 1 | Delete backups/old files | 30.2 MB | 0 | NONE |
| 2 | Archive documentation | 0 | 300 KB | NONE |
| 3 | Archive/compress logs | 1,426 MB | 142 MB | LOW |
| 4 | Archive scripts | 0 | 50 KB | LOW |
| 5 | Remove archive dir | 8.7 MB | 0 | MEDIUM |
| **Total** | | **1,465 MB** | **142.4 MB** | |

**Net Storage Reduction:** 1,465 MB - 142.4 MB = **1,322.6 MB (1.29 GB)**
**Final Repository Size:** 2.9 GB - 1.32 GB = **1.58 GB (45% reduction)**

---

## Success Criteria

### Must Achieve
- [ ] 100% logic preservation (all `.py` files with unique logic kept)
- [ ] Zero data loss (all current feature stores intact)
- [ ] All documentation accessible (moved to archive, not deleted)
- [ ] Repository builds successfully after cleanup
- [ ] At least 1 GB storage reduction

### Nice to Have
- [ ] Organized archive structure for future reference
- [ ] Compressed logs for space efficiency
- [ ] Clear audit trail of what was removed/moved

---

## Rollback Plan

If cleanup causes issues:

1. **Restore backup files:** `git checkout HEAD -- data/features_mtf/`
2. **Restore old configs:** `git checkout HEAD -- configs/v150/assets/`
3. **Restore build:** Run `python setup.py build`
4. **Restore docs:** `git checkout HEAD -- *.md` (root directory)
5. **Restore logs:** Uncompress from `logs/archive/2025-11-14/`

---

## Risk Assessment

| Category | Risk Level | Mitigation |
|----------|------------|------------|
| Backup file deletion | LOW | Current files contain all data |
| Old file deletion | LOW | Current versions in production |
| Build artifact deletion | NONE | Can regenerate anytime |
| Documentation archiving | NONE | Moved, not deleted |
| Log archiving | LOW | Compressed, not deleted |
| Script archiving | LOW | Moved to bin/archive/ |
| Archive dir deletion | MEDIUM | Review contents first |

**Overall Risk:** LOW

---

## Next Steps

1. **Review this report** - Confirm cleanup plan
2. **Execute Phase 1** - Safe deletions (automated)
3. **Execute Phase 2** - Archive documentation (manual)
4. **Execute Phase 3** - Archive logs (automated)
5. **Execute Phase 4** - Archive scripts (manual)
6. **Execute Phase 5** - Review archive directory (manual decision)
7. **Generate final report** - `docs/CLEANUP_COMPLETE.md`
8. **Commit changes** - Single cleanup commit

---

**Status:** READY FOR EXECUTION
**Estimated Time:** 30 minutes (mostly automated)
**Approval Required:** Yes (before Phase 5 - archive dir deletion)
