#!/bin/bash

# ============================================================================
# Bull Machine Cleanup Validation Script
# Date: 2025-11-14
# Purpose: Validate repository cleanup and ensure nothing critical was broken
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}"

errors=0
warnings=0
successes=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((successes++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((warnings++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((errors++))
}

log_section() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

check_file_exists() {
    local file="$1"
    local msg="$2"

    if [ -f "$file" ]; then
        log_success "$msg: $file exists"
        return 0
    else
        log_error "$msg: $file is missing!"
        return 1
    fi
}

check_dir_exists() {
    local dir="$1"
    local msg="$2"

    if [ -d "$dir" ]; then
        log_success "$msg: $dir exists"
        return 0
    else
        log_error "$msg: $dir is missing!"
        return 1
    fi
}

check_file_not_exists() {
    local file="$1"
    local msg="$2"

    if [ ! -f "$file" ]; then
        log_success "$msg: $file removed"
        return 0
    else
        log_warning "$msg: $file still exists"
        return 1
    fi
}

# ============================================================================
# Validation Tests
# ============================================================================

test_essential_files() {
    log_section "Test 1: Essential Files Present"

    check_file_exists "README.md" "Root README"
    check_file_exists "CHANGELOG.md" "Changelog"
    check_file_exists "setup.py" "Setup script"
    check_file_exists "requirements.txt" "Requirements"
    check_file_exists ".gitignore" "Gitignore"
    check_file_exists "pytest.ini" "Pytest config"
    check_file_exists "pyproject.toml" "Project config"
}

test_essential_dirs() {
    log_section "Test 2: Essential Directories Present"

    check_dir_exists "bull_machine" "Bull Machine package"
    check_dir_exists "engine" "Engine package"
    check_dir_exists "bin" "Bin scripts"
    check_dir_exists "configs" "Configs"
    check_dir_exists "docs" "Documentation"
    check_dir_exists "tests" "Tests"
}

test_cleanup_executed() {
    log_section "Test 3: Cleanup Executed"

    # Check root is clean
    local root_md_count=$(find "$REPO_ROOT" -maxdepth 1 -name "*.md" | wc -l)
    local root_py_count=$(find "$REPO_ROOT" -maxdepth 1 -name "*.py" | wc -l)

    log_info "Root markdown files: $root_md_count (should be ~5)"
    if [ "$root_md_count" -le 10 ]; then
        log_success "Root markdown count acceptable"
    else
        log_warning "Root has too many markdown files ($root_md_count)"
    fi

    log_info "Root Python files: $root_py_count (should be 0)"
    if [ "$root_py_count" -eq 0 ]; then
        log_success "Root Python files cleaned"
    else
        log_warning "Root still has Python files ($root_py_count)"
    fi

    # Check experimental files removed
    check_file_not_exists "test_archetype_debug.py" "Test script"
    check_file_not_exists "sweep_parameters.py" "Sweep script"
    check_file_not_exists "btc_fine_grid.json" "Experimental result"
    check_file_not_exists "optimization_results_v19.json" "Optimization result"
}

test_docs_archived() {
    log_section "Test 4: Documentation Archived"

    check_dir_exists "docs/archive/2024-q4" "Archive structure"
    check_dir_exists "docs/archive/2024-q4/mvp_phases" "MVP phases archive"
    check_dir_exists "docs/archive/2024-q4/optimization" "Optimization archive"
    check_dir_exists "docs/archive/2024-q4/implementations" "Implementations archive"

    check_file_exists "docs/archive/README.md" "Archive README"
}

test_results_cleaned() {
    log_section "Test 5: Results Directory Cleaned"

    check_dir_exists "results" "Results directory"
    check_file_exists "results/README.md" "Results README"

    # Check experimental folders removed
    if [ ! -d "results/optuna_trap_v2_DIAGNOSTIC" ]; then
        log_success "Experimental optuna folders removed"
    else
        log_warning "Optuna experiment folders still exist"
    fi

    # Check production benchmarks preserved
    if [ -d "results/bench_v2" ] || [ -d "results/bear_patterns" ]; then
        log_success "Production benchmarks preserved"
    else
        log_warning "Production benchmarks may have been removed"
    fi
}

test_logs_cleaned() {
    log_section "Test 6: Logs Directory Cleaned"

    check_dir_exists "logs" "Logs directory"
    check_file_exists "logs/README.md" "Logs README"

    # Check large logs removed
    if [ ! -f "logs/bear_archetypes_adaptive_2024_full.log" ]; then
        log_success "Large debug logs removed"
    else
        log_warning "Large debug logs still exist"
    fi
}

test_bin_curated() {
    log_section "Test 7: Bin Scripts Curated"

    check_dir_exists "bin/archive/experimental" "Experimental archive"
    check_dir_exists "bin/archive/diagnostics" "Diagnostics archive"
    check_file_exists "bin/README.md" "Bin README"

    # Check production scripts preserved
    check_file_exists "bin/backtest_knowledge_v2.py" "Production backtest script"
    check_file_exists "bin/build_feature_store_v2.py" "Production feature script"

    # Check experimental scripts moved
    if [ -f "bin/archive/experimental/backfill_liquidity_score.py" ]; then
        log_success "Experimental scripts archived"
    else
        log_warning "Experimental scripts may not be archived"
    fi
}

test_configs_organized() {
    log_section "Test 8: Configs Organized"

    check_dir_exists "configs/production" "Production configs"
    check_dir_exists "configs/experimental" "Experimental configs"
    check_dir_exists "configs/archive" "Config archive"
    check_file_exists "configs/README.md" "Configs README"

    # Check production configs
    if [ -d "configs/production/frozen" ]; then
        log_success "Frozen configs in production/"
    else
        log_warning "Frozen configs not in production/"
    fi
}

test_gitignore_enhanced() {
    log_section "Test 9: Gitignore Enhanced"

    if grep -q "optuna_\*" .gitignore; then
        log_success "Gitignore has optuna patterns"
    else
        log_warning "Gitignore missing optuna patterns"
    fi

    if grep -q "bench_v2_\*" .gitignore; then
        log_success "Gitignore has bench patterns"
    else
        log_warning "Gitignore missing bench patterns"
    fi

    if grep -q "results/\*\*/\*" .gitignore; then
        log_success "Gitignore has results patterns"
    else
        log_warning "Gitignore missing results patterns"
    fi
}

test_artifacts_removed() {
    log_section "Test 10: Build Artifacts Removed"

    if [ ! -d "dist" ]; then
        log_success "dist/ removed"
    else
        log_warning "dist/ still exists"
    fi

    if [ ! -d ".mypy_cache" ]; then
        log_success ".mypy_cache/ removed"
    else
        log_warning ".mypy_cache/ still exists"
    fi

    if [ ! -d ".ruff_cache" ]; then
        log_success ".ruff_cache/ removed"
    else
        log_warning ".ruff_cache/ still exists"
    fi
}

test_python_imports() {
    log_section "Test 11: Python Imports Work"

    log_info "Testing Python imports..."

    if python3 -c "import bull_machine" 2>/dev/null; then
        log_success "bull_machine package imports"
    else
        log_error "bull_machine package import failed!"
    fi

    if python3 -c "import engine" 2>/dev/null; then
        log_success "engine package imports"
    else
        log_error "engine package import failed!"
    fi

    if python3 -c "from bull_machine.backtest import Backtest" 2>/dev/null; then
        log_success "bull_machine submodules import"
    else
        log_warning "Some bull_machine submodules may have issues"
    fi
}

test_production_configs() {
    log_section "Test 12: Production Configs Accessible"

    # Check if production configs can be found
    if [ -d "configs/production/frozen" ] && [ "$(ls -A configs/production/frozen)" ]; then
        log_success "Frozen configs accessible"
    else
        log_warning "Frozen configs may be inaccessible"
    fi

    # Check for MVP configs
    if ls configs/production/mvp_*.json 1> /dev/null 2>&1; then
        log_success "MVP configs in production/"
    else
        log_warning "MVP configs not found in production/"
    fi
}

test_git_status() {
    log_section "Test 13: Git Status"

    log_info "Git status:"
    git status --short | head -20

    local untracked=$(git ls-files --others --exclude-standard | wc -l)
    log_info "Untracked files: $untracked"

    if [ "$untracked" -lt 20 ]; then
        log_success "Untracked files count acceptable"
    else
        log_warning "High number of untracked files ($untracked)"
    fi
}

test_repo_size() {
    log_section "Test 14: Repository Size"

    local repo_size=$(du -sh "$REPO_ROOT" | cut -f1)
    log_info "Current repository size: $repo_size"

    local results_size=$(du -sh "$REPO_ROOT/results" 2>/dev/null | cut -f1 || echo "0")
    local logs_size=$(du -sh "$REPO_ROOT/logs" 2>/dev/null | cut -f1 || echo "0")

    log_info "Results directory: $results_size"
    log_info "Logs directory: $logs_size"

    log_success "Size check complete"
}

test_critical_files() {
    log_section "Test 15: Critical Production Files"

    # Check production code files
    check_file_exists "bull_machine/__init__.py" "Bull Machine init"
    check_file_exists "engine/__init__.py" "Engine init"

    # Check key production scripts
    check_file_exists "bin/backtest_knowledge_v2.py" "Main backtest script"
    check_file_exists "bin/build_feature_store_v2.py" "Feature store builder"

    # Check current documentation
    check_file_exists "COMPREHENSIVE_SYSTEM_AUDIT_AND_MVP_ROADMAP.md" "Current roadmap"

    # Check docs files that should be kept
    if ls docs/BEAR_*.md 1> /dev/null 2>&1; then
        log_success "Current bear pattern docs preserved"
    else
        log_warning "Bear pattern docs may be missing"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_section "Bull Machine Cleanup Validation"
    log_info "Date: $(date)"
    log_info "Repository: $REPO_ROOT"

    cd "$REPO_ROOT"

    # Run all tests
    test_essential_files
    test_essential_dirs
    test_cleanup_executed
    test_docs_archived
    test_results_cleaned
    test_logs_cleaned
    test_bin_curated
    test_configs_organized
    test_gitignore_enhanced
    test_artifacts_removed
    test_python_imports
    test_production_configs
    test_git_status
    test_repo_size
    test_critical_files

    # Summary
    log_section "Validation Summary"

    log_info "Passed: $successes"
    log_info "Warnings: $warnings"
    log_info "Failures: $errors"

    echo ""
    if [ $errors -eq 0 ]; then
        log_success "Validation PASSED - Cleanup successful!"
        echo ""
        log_info "Next steps:"
        echo "  1. Run full test suite: pytest"
        echo "  2. Test a backtest: bin/backtest_knowledge_v2.py"
        echo "  3. Review git status: git status"
        echo "  4. Commit if satisfied: git push"
        return 0
    else
        log_error "Validation FAILED - $errors error(s) found"
        echo ""
        log_info "Please review errors above and fix before committing"
        return 1
    fi
}

main "$@"
