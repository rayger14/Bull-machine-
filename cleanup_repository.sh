#!/bin/bash

# ============================================================================
# Bull Machine Repository Cleanup Script
# Date: 2025-11-14
# Purpose: Professional cleanup of repository bloat from R&D development
# ============================================================================

set -e  # Exit on error
# set -x  # Uncomment for debug output

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}"
DRY_RUN=false
VERBOSE=false
BACKUP_CREATED=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

execute() {
    local cmd="$1"
    local description="$2"

    if [ "$DRY_RUN" = true ]; then
        log_warning "[DRY RUN] Would execute: $cmd"
        if [ "$VERBOSE" = true ] && [ -n "$description" ]; then
            log_info "  → $description"
        fi
    else
        if [ "$VERBOSE" = true ]; then
            log_info "Executing: $cmd"
            if [ -n "$description" ]; then
                log_info "  → $description"
            fi
        fi
        eval "$cmd"
    fi
}

git_mv_safe() {
    local src="$1"
    local dst="$2"

    if [ ! -e "$src" ]; then
        log_warning "Source does not exist: $src"
        return 1
    fi

    # Create destination directory if needed
    local dst_dir="$(dirname "$dst")"
    execute "mkdir -p '$dst_dir'" "Create directory: $dst_dir"

    # Move file
    execute "git mv '$src' '$dst'" "Move: $src → $dst"
}

delete_file() {
    local file="$1"

    if [ ! -e "$file" ]; then
        if [ "$VERBOSE" = true ]; then
            log_warning "File does not exist (already deleted?): $file"
        fi
        return 0
    fi

    execute "git rm -f '$file'" "Delete: $file"
}

delete_directory() {
    local dir="$1"

    if [ ! -d "$dir" ]; then
        if [ "$VERBOSE" = true ]; then
            log_warning "Directory does not exist (already deleted?): $dir"
        fi
        return 0
    fi

    execute "git rm -rf '$dir'" "Delete directory: $dir"
}

create_backup() {
    if [ "$BACKUP_CREATED" = true ]; then
        log_info "Backup already created, skipping..."
        return 0
    fi

    log_section "Creating Backup"

    # Create git tag
    local tag_name="pre-cleanup-$(date +%Y-%m-%d)"
    log_info "Creating git tag: $tag_name"
    execute "git tag '$tag_name'" "Create rollback tag"

    # Show current repo size
    local repo_size=$(du -sh "$REPO_ROOT" | cut -f1)
    log_info "Current repository size: $repo_size"

    BACKUP_CREATED=true
    log_success "Backup created successfully"
}

phase_header() {
    log_section "Phase $1: $2"
}

phase_complete() {
    local phase_num="$1"
    local phase_name="$2"

    if [ "$DRY_RUN" = false ]; then
        log_info "Committing Phase $phase_num..."
        git add -A
        git commit -m "chore(cleanup): Phase $phase_num - $phase_name

- $3

Part of repository cleanup initiative (2025-11-14)
Estimated space recovery: $4" || log_warning "No changes to commit for Phase $phase_num"
    fi

    log_success "Phase $phase_num complete: $phase_name"
}

# ----------------------------------------------------------------------------
# Phase Functions
# ----------------------------------------------------------------------------

phase1_gitignore() {
    phase_header "1" "Enhance .gitignore"

    log_info "Backing up current .gitignore"
    execute "cp .gitignore .gitignore.backup" "Backup .gitignore"

    log_info "Installing enhanced .gitignore"
    if [ -f ".gitignore.new" ]; then
        execute "mv .gitignore.new .gitignore" "Install new .gitignore"
        log_success "Enhanced .gitignore installed"
    else
        log_error ".gitignore.new not found! Please create it first."
        return 1
    fi

    phase_complete "1" "Enhanced .gitignore" \
        "Updated .gitignore with comprehensive patterns to prevent future bloat" \
        "Prevention of future accumulation"
}

phase2_archive_docs() {
    phase_header "2" "Archive Root-Level Documentation"

    # Create archive structure
    log_info "Creating docs/archive structure..."
    execute "mkdir -p docs/archive/2024-q4/{mvp_phases,pull_requests,sessions,archetype_work,optimization,implementations,cleanup,features,validation,status,optuna,phases,paper_trading,audit}"

    # MVP Phases
    log_info "Archiving MVP phase documents..."
    for file in MVP_PHASE*.md MVP_ROADMAP.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/mvp_phases/$file"
    done

    # Pull Requests
    log_info "Archiving PR documents..."
    for file in PR*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/pull_requests/$file"
    done

    # Sessions
    log_info "Archiving session summaries..."
    for file in SESSION_SUMMARY*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/sessions/$file"
    done

    # Archetype Work
    log_info "Archiving archetype documents..."
    for file in ARCHETYPE*.md COMPREHENSIVE_ARCHETYPE_AUDIT.md README_ARCHETYPE_ANALYSIS.md \
                BEAR_ARCHETYPES_PHASE1_IMPLEMENTATION.md BEAR_ARCHETYPES_ZERO_MATCHES_DIAGNOSIS.md \
                BEAR_ARCHITECTURE_EXECUTIVE_SUMMARY.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/archetype_work/$file"
    done

    # Optimization
    log_info "Archiving optimization documents..."
    for file in OPTIMIZATION*.md EXIT_OPTIMIZATION_PLAN.md OPTIMAL_CONFIG*.md \
                PERFORMANCE_OPTIMIZATION_FINDINGS.md OPTIMIZER_SIGNAL_GENERATION_ANALYSIS.md \
                SPY_*.md TRAP_OPTIMIZATION_FAILURE_ANALYSIS.md BACKTEST_V2_OPTIMIZATION_REPORT.md \
                OB_HIGH_OPTIMIZATION_SUMMARY.md WIRING_FIX*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/optimization/$file"
    done

    # Implementations
    log_info "Archiving implementation plans..."
    for file in PHASE0_BRANCH_INTEGRATION_REPORT.md PHASE1_*.md PHASE2_STATUS.md \
                BULL_MACHINE_V2_IMPLEMENTATION_PLAN.md REGIME_ROUTING_IMPLEMENTATION_PLAN.md \
                KNOWLEDGE_V2*.md TESTING_KNOWLEDGE_V2.md COMPLETE_KNOWLEDGE_ARCHITECTURE.md \
                IMPLEMENTATION_ROADMAP.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/implementations/$file"
    done

    # Cleanup
    log_info "Archiving cleanup documents..."
    for file in CLEANUP_EXECUTION_SUMMARY.md CLEANUP_REPORT.md OB_HIGH_COVERAGE_FIX_REPORT.md \
                VALIDATION_CRITICAL_BUG_FOUND.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/cleanup/$file"
    done

    # Features
    log_info "Archiving feature documents..."
    for file in FEATURE_STORE_CONTENTS.md ML_FEATURE_INVENTORY.md ML_META_OPTIMIZER_ARCHITECTURE.md \
                ML_ROADMAP.md CODE_REVIEW_IMPROVEMENTS.md ENHANCED_EXIT_STRATEGIES_DESIGN.md \
                V2_CLEANUP*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/features/$file"
    done

    # Validation
    log_info "Archiving validation documents..."
    for file in REPLAY*.md HYBRID_RUNNER_VALIDATION.md HYBRID_VALIDATION_2024.md \
                V19_3YEAR_VALIDATION_FINAL.md V1.8.1_TRUE_FUSION_COMPLETE.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/validation/$file"
    done

    # Status
    log_info "Archiving status documents..."
    for file in FINAL_STATUS.md NEXT_STEPS.md BASELINE_METRICS.md HANDOFF_NEXT_STEPS.md \
                READY_TO_RUN.md WHILE_YOU_SLEPT.md FRONTIER_*.md FULL_BACKTEST_RESULTS_ANALYSIS.md \
                PF20_RECOVERY_STATUS.md SCORE_PROPAGATION_BUG_FIX_REPORT.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/status/$file"
    done

    # Optuna
    log_info "Archiving optuna documents..."
    for file in OPTUNA*.md STEP5_OPTUNA_SPEC.md ROUTER_V10_ANALYSIS_AND_RECOMMENDATIONS.md \
                MASTER_OPTIMIZATION_ROADMAP.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/optuna/$file"
    done

    # Phases
    log_info "Archiving phase documents..."
    for file in PHASE_1*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/phases/$file"
    done

    # Paper Trading
    log_info "Archiving paper trading documents..."
    for file in PAPER_TRADING*.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/paper_trading/$file"
    done

    # Audit
    log_info "Archiving audit documents..."
    for file in AUDIT_INDEX.md AUDIT_QUICK_REFERENCE.md; do
        [ -f "$file" ] && git_mv_safe "$file" "docs/archive/2024-q4/audit/$file"
    done

    # Create archive README
    cat > docs/archive/README.md << 'EOF'
# Documentation Archive

Historical documentation organized by quarter.

## Structure
- `2024-q4/` - Q4 2024 documentation
  - `mvp_phases/` - MVP development phases
  - `pull_requests/` - PR-related documentation
  - `sessions/` - Session summaries
  - `archetype_work/` - Archetype development
  - `optimization/` - Optimization reports
  - `implementations/` - Implementation plans
  - `cleanup/` - Cleanup reports
  - `features/` - Feature documentation
  - `validation/` - Validation reports
  - `status/` - Status updates
  - `optuna/` - Optuna optimization
  - `phases/` - Phase progress
  - `paper_trading/` - Paper trading setup
  - `audit/` - System audits

## Policy
- Documentation older than current quarter should be moved here
- Organize by quarter: YYYY-QQ/
- Keep current roadmap and critical docs in root
EOF
    execute "git add docs/archive/README.md" "Add archive README"

    phase_complete "2" "Archive Documentation" \
        "Moved 100+ markdown files from root to docs/archive/2024-q4/" \
        "~10 MB"
}

phase3_clean_results_logs() {
    phase_header "3" "Clean Results & Logs Directories"

    cd "$REPO_ROOT"

    # Clean results
    log_info "Cleaning results directory..."

    # Delete experimental optuna folders
    log_info "  Removing Optuna experiment folders..."
    for dir in results/optuna_*; do
        [ -d "$dir" ] && delete_directory "$dir"
    done

    # Delete other experimental folders
    log_info "  Removing experimental result folders..."
    delete_directory "results/bench_v2_frontier"
    delete_directory "results/macro_fix_validation"
    delete_directory "results/macro_fix_sanity"
    delete_directory "results/frontier_exploration"
    delete_directory "results/tiered_tests"
    delete_directory "results/trap_validation"
    delete_directory "results/20251001_203255"

    for dir in results/router_v10_*; do
        [ -d "$dir" ] && delete_directory "$dir"
    done

    # Delete timestamped result files
    log_info "  Removing timestamped result files..."
    for pattern in "hybrid_signals_*.jsonl" "health_summary_*.json" "portfolio_summary_*.json" \
                   "BTC_*_backtest_*.json" "ETH_*_backtest_*.json"; do
        find results/ -maxdepth 1 -name "$pattern" -type f -exec git rm -f {} \; 2>/dev/null || true
    done

    # Delete large debug files
    delete_file "results/fusion_debug.jsonl"
    delete_file "results/fusion_validation.jsonl"
    delete_file "results/signal_blocks.jsonl"
    delete_file "results/decision_log.jsonl"
    delete_file "results/open_fail.jsonl"
    delete_file "results/open_ok.jsonl"
    delete_file "results/daily_aggregate_results.json"
    delete_file "results/frontier_exploration.log"

    # Delete ML training files
    delete_file "results/btc_ml_training.json"
    delete_file "results/eth_ml_training.json"

    # Delete candidate files
    delete_file "results/BTC_2025_candidates.jsonl"
    delete_file "results/ETH_2025_candidates.jsonl"

    # Delete log files in results
    find results/ -name "*.log" -type f -exec git rm -f {} \; 2>/dev/null || true

    # Create results README
    cat > results/README.md << 'EOF'
# Results Directory

Production benchmark results and experiment outputs.

## Structure
- `bench_v2/` - Production v2 benchmarks (keep)
- `bear_patterns/` - Bear market pattern analysis (keep)
- `archive/` - Historical results (keep)

## Guidelines
- Experimental results are gitignored by default
- Only commit production-validated benchmarks
- Use timestamped subdirectories: `YYYYMMDD_experiment_name/`
- Archive old results to `archive/YYYY-QQ/` after 90 days

## Gitignored Patterns
All results are gitignored except for specific production benchmarks.
See `.gitignore` for details.
EOF
    execute "git add results/README.md" "Add results README"

    # Clean logs
    log_info "Cleaning logs directory..."

    # Delete large bear archetype logs
    for file in logs/bear_archetypes*.log; do
        [ -f "$file" ] && delete_file "$file"
    done

    # Delete other debug logs
    for pattern in "btc_3year*.log" "BTC_confluence*.log" "btc_exit_opt.log" \
                   "baseline*.log" "backfill*.log" "*_backtest*.log" \
                   "*_validation*.log" "*_feature_store*.log"; do
        find logs/ -maxdepth 1 -name "$pattern" -type f -exec git rm -f {} \; 2>/dev/null || true
    done

    # Create logs README
    cat > logs/README.md << 'EOF'
# Logs Directory

Runtime and debug logs.

## Structure
- `paper_trading/` - Paper trading execution logs
- `archive/` - Historical logs

## Guidelines
- All logs are gitignored by default
- Only critical logs should be committed
- Archive logs after 30 days
- Use log rotation for long-running processes

## Gitignored
All logs in this directory are gitignored except for structure.
See `.gitignore` for details.
EOF
    execute "git add logs/README.md" "Add logs README"

    phase_complete "3" "Clean Results & Logs" \
        "Removed experimental results (~400 MB) and debug logs (~350 MB)" \
        "~750 MB"
}

phase4_curate_bin() {
    phase_header "4" "Curate Bin Scripts"

    cd "$REPO_ROOT/bin"

    # Create archive structure
    execute "mkdir -p archive/experimental archive/diagnostics"

    # Move experimental scripts
    log_info "Moving experimental scripts to archive..."
    for script in backfill_liquidity_score.py backfill_liquidity_score_optimized.py \
                  backfill_ob_high.py backfill_ob_high_optimized.py \
                  backfill_missing_macro_features.py fix_oi_change_pipeline.py \
                  test_ob_high_optimization.py; do
        [ -f "$script" ] && git_mv_safe "$script" "archive/experimental/$script"
    done

    # Move diagnostic scripts
    log_info "Moving diagnostic scripts to archive..."
    for script in diagnose_eth_runtime.py debug_adaptive_logic.py check_pr3_nonzero_rates.py; do
        [ -f "$script" ] && git_mv_safe "$script" "archive/diagnostics/$script"
    done

    # Create bin README
    cat > README.md << 'EOF'
# Bin Scripts

Production scripts for the Bull Machine trading system.

## Categories
- **Backtesting**: `backtest_*.py` - Run backtests
- **Feature Engineering**: `build_*.py`, `add_*.py` - Build feature stores
- **Analysis**: `analyze_*.py` - Analyze results and performance
- **Optimization**: `consolidate_*.py` - Optimization utilities
- **CLI**: `bull_machine_cli.py` - Command-line interface

## Archive
- `archive/experimental/` - One-time data migrations and backfill scripts
- `archive/diagnostics/` - Debug and diagnostic scripts

## Guidelines
- Production scripts only in root bin/
- Test scripts belong in tests/
- Research scripts belong in scripts/research/
- Archive one-time scripts after use
EOF
    execute "git add README.md" "Add bin README"

    cd "$REPO_ROOT"

    phase_complete "4" "Curate Bin Scripts" \
        "Moved experimental and diagnostic scripts to bin/archive/" \
        "Organization"
}

phase5_organize_configs() {
    phase_header "5" "Organize Configs"

    cd "$REPO_ROOT/configs"

    # Create new structure
    execute "mkdir -p production/frozen production/live production/paper_trading"
    execute "mkdir -p experimental/adaptive experimental/sweep experimental/knowledge_v2"
    execute "mkdir -p archive"

    # Move to production
    log_info "Organizing production configs..."
    [ -d "frozen" ] && execute "mv frozen/* production/frozen/" && execute "rmdir frozen"
    [ -d "live" ] && execute "mv live/* production/live/" && execute "rmdir live"
    [ -d "paper_trading" ] && execute "mv paper_trading/* production/paper_trading/" && execute "rmdir paper_trading"

    # Move production MVPs
    for file in mvp_*.json regime_routing_production_v1.json bear_archetypes_phase1.json; do
        [ -f "$file" ] && execute "git mv '$file' production/"
    done

    # Move to experimental
    log_info "Organizing experimental configs..."
    [ -d "adaptive" ] && execute "mv adaptive/* experimental/adaptive/" && execute "rmdir adaptive"
    [ -d "sweep" ] && execute "mv sweep/* experimental/sweep/" && execute "rmdir sweep"
    [ -d "knowledge_v2" ] && execute "mv knowledge_v2/* experimental/knowledge_v2/" && execute "rmdir knowledge_v2"

    # Move version dirs to archive
    log_info "Archiving old version configs..."
    for dir in v10_bases v141 v142 v150 v160 v170 v171 v18 v185 v186 v19 v2 v3_replay_2024; do
        [ -d "$dir" ] && execute "git mv '$dir' archive/"
    done

    # Create configs README
    cat > README.md << 'EOF'
# Configurations

Bull Machine configuration files organized by purpose.

## Structure
- `production/` - Production configurations
  - `frozen/` - Frozen production configs
  - `live/` - Live trading configs
  - `paper_trading/` - Paper trading configs
  - `mvp_*.json` - MVP configurations
- `experimental/` - Experimental configurations
  - `adaptive/` - Adaptive strategy experiments
  - `sweep/` - Parameter sweep configs
  - `knowledge_v2/` - Knowledge v2 experiments
- `archive/` - Historical version configs
- `schema/` - Configuration schemas
- `deltas/` - Configuration deltas
- `stock/` - Stock market configs

## Guidelines
- Production configs in `production/`
- Experiments in `experimental/`
- Archive old versions to `archive/`
- Never modify frozen configs
- Use deltas for variations
EOF
    execute "git add README.md" "Add configs README"

    cd "$REPO_ROOT"

    phase_complete "5" "Organize Configs" \
        "Reorganized configs into production/experimental/archive structure" \
        "Organization"
}

phase6_remove_artifacts() {
    phase_header "6" "Remove Build Artifacts & Cache"

    cd "$REPO_ROOT"

    log_info "Removing build artifacts..."
    delete_directory "dist"
    delete_directory "bull_machine.egg-info"

    log_info "Removing cache directories..."
    delete_directory ".mypy_cache"
    delete_directory ".ruff_cache"
    # Note: .pytest_cache should already be in .gitignore

    log_info "Removing Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    log_info "Removing OS artifacts..."
    find . -name ".DS_Store" -delete 2>/dev/null || true

    phase_complete "6" "Remove Build Artifacts" \
        "Removed dist/, cache directories, and compiled Python files" \
        "~2 MB"
}

phase7_clean_root() {
    phase_header "7" "Clean Root-Level Experimental Files"

    cd "$REPO_ROOT"

    # Delete JSON results
    log_info "Removing JSON result files..."
    for file in btc_backtest_results_*.json btc_fine_grid.json btc_results_comparison.csv \
                btc_results.json btc_v19_*.json eth_production_backtest_results.json \
                sweep_results_*.json institutional_testing_results.json config_patch_ml.json \
                exit_cfg_applied.json optimization_results_v19.json; do
        [ -f "$file" ] && delete_file "$file"
    done

    # Delete Python scripts
    log_info "Removing experimental Python scripts..."
    for script in test_archetype_debug.py test_boms_diagnostic.py test_boms_4h.py \
                  test_feature_store_scores.py test_fusion_windowing.py test_hooks_firing.py \
                  test_macro_extraction.py test_macro_loading.py test_optimization.py \
                  analyze_threshold_sensitivity.py sweep_parameters.py sweep_hybrid_params.py \
                  sweep_thresholds.py check_pr3_nonzero_rates.py download_vix_2024.py \
                  engine_factory.py bull_machine_config.py; do
        [ -f "$script" ] && delete_file "$script"
    done

    # Delete shell scripts
    log_info "Removing experimental shell scripts..."
    for script in monitor_and_compile_results.sh optimize_all.sh WATCH_CRITICAL_PROCESSES.sh \
                  RUN_TESTS.sh; do
        [ -f "$script" ] && delete_file "$script"
    done

    # Delete logs and CSV files
    log_info "Removing log and CSV files..."
    for file in q3_2024_hybrid.log q3_2024_validation.log q3_2024_validation_results.txt \
                full_year_test_output.log monitor_output.log threshold_sensitivity_sweep.csv; do
        [ -f "$file" ] && delete_file "$file"
    done

    # Delete misc files
    log_info "Removing miscellaneous files..."
    for file in .cleanup_plan.txt conftest.py; do
        [ -f "$file" ] && delete_file "$file"
    done

    # Delete weird git artifacts
    for item in shasum rev-parse git HEAD; do
        [ -e "$item" ] && execute "rm -rf '$item'" "Remove: $item"
    done

    phase_complete "7" "Clean Root Files" \
        "Removed 50+ experimental files from root directory" \
        "~400 MB"
}

show_summary() {
    log_section "Cleanup Summary"

    local repo_size=$(du -sh "$REPO_ROOT" | cut -f1)
    log_info "Final repository size: $repo_size"

    log_info "Root markdown files: $(find "$REPO_ROOT" -maxdepth 1 -name "*.md" | wc -l)"
    log_info "Root Python files: $(find "$REPO_ROOT" -maxdepth 1 -name "*.py" | wc -l)"

    log_success "Cleanup complete!"
    echo ""
    log_info "Next steps:"
    echo "  1. Review changes: git status"
    echo "  2. Run tests: pytest"
    echo "  3. Verify imports: python -c 'import bull_machine; import engine'"
    echo "  4. Push changes: git push origin \$(git branch --show-current)"
    echo ""
    log_info "Rollback if needed:"
    echo "  git reset --hard pre-cleanup-$(date +%Y-%m-%d)"
}

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --dry-run       Show what would be done without making changes
    --verbose       Show detailed output
    --phase N       Run only phase N (1-7)
    --help          Show this help message

Phases:
    1. Enhance .gitignore
    2. Archive root-level documentation
    3. Clean results & logs directories
    4. Curate bin scripts
    5. Organize configs
    6. Remove build artifacts
    7. Clean root-level experimental files

Examples:
    $0 --dry-run                  # Preview all changes
    $0 --phase 1                  # Run only phase 1
    $0 --verbose                  # Run with detailed output
    $0                            # Run full cleanup

EOF
}

main() {
    local run_phase=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                log_warning "DRY RUN MODE - No changes will be made"
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --phase)
                run_phase="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Verify we're in a git repo
    if [ ! -d ".git" ]; then
        log_error "Not a git repository!"
        exit 1
    fi

    log_section "Bull Machine Repository Cleanup"
    log_info "Date: $(date)"
    log_info "Repository: $REPO_ROOT"

    if [ "$DRY_RUN" = true ]; then
        log_warning "Running in DRY RUN mode - no changes will be made"
    fi

    # Create backup
    create_backup

    # Run phases
    if [ -n "$run_phase" ]; then
        log_info "Running only Phase $run_phase"
        case $run_phase in
            1) phase1_gitignore ;;
            2) phase2_archive_docs ;;
            3) phase3_clean_results_logs ;;
            4) phase4_curate_bin ;;
            5) phase5_organize_configs ;;
            6) phase6_remove_artifacts ;;
            7) phase7_clean_root ;;
            *)
                log_error "Invalid phase number: $run_phase"
                usage
                exit 1
                ;;
        esac
    else
        log_info "Running all phases"
        phase1_gitignore
        phase2_archive_docs
        phase3_clean_results_logs
        phase4_curate_bin
        phase5_organize_configs
        phase6_remove_artifacts
        phase7_clean_root
    fi

    show_summary
}

main "$@"
