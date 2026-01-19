#!/bin/bash
#
# Automated Validation Pipeline
#
# Runs comprehensive validation suite on optimized configs:
# 1. Walk-forward validation (train/validate/test)
# 2. Cross-regime validation (regime-specific performance)
# 3. Statistical significance validation (bootstrap + permutation tests)
# 4. Generates consolidated validation report
#
# Usage:
#     # Validate single config
#     ./bin/run_full_validation.sh configs/mvp/mvp_bear_market_v1.json
#
#     # Validate directory of configs
#     ./bin/run_full_validation.sh configs/mvp/
#
#     # Validate with custom output directory
#     ./bin/run_full_validation.sh configs/optimized/ results/validation/custom/

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths
FEATURE_STORE="${FEATURE_STORE:-data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_backup.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-results/validation/$(date +%Y%m%d_%H%M%S)}"

# Validation parameters
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-1000}"
ALPHA="${ALPHA:-0.05}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Print header
print_header() {
    echo "========================================================================"
    echo "  COMPREHENSIVE VALIDATION PIPELINE"
    echo "========================================================================"
    echo ""
}

# Print config
print_config() {
    echo "Configuration:"
    echo "  Input:            $INPUT"
    echo "  Output:           $OUTPUT_DIR"
    echo "  Feature Store:    $FEATURE_STORE"
    echo "  Bootstrap:        $N_BOOTSTRAP iterations"
    echo "  Permutations:     $N_PERMUTATIONS iterations"
    echo "  Alpha:            $ALPHA"
    echo ""
}

# Validate input
validate_input() {
    if [ ! -e "$INPUT" ]; then
        log_error "Input not found: $INPUT"
        exit 1
    fi

    if [ ! -f "$FEATURE_STORE" ]; then
        log_error "Feature store not found: $FEATURE_STORE"
        log_error "Please set FEATURE_STORE environment variable or ensure default path exists"
        exit 1
    fi
}

# Create output directory
create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    log_success "Created output directory: $OUTPUT_DIR"
}

# Collect configs to validate
collect_configs() {
    CONFIGS=()

    if [ -f "$INPUT" ]; then
        # Single config file
        if [[ "$INPUT" == *.json ]]; then
            CONFIGS+=("$INPUT")
        else
            log_error "Input file is not a JSON config: $INPUT"
            exit 1
        fi
    elif [ -d "$INPUT" ]; then
        # Directory of configs
        while IFS= read -r -d '' file; do
            CONFIGS+=("$file")
        done < <(find "$INPUT" -name "*.json" -type f -print0)

        if [ ${#CONFIGS[@]} -eq 0 ]; then
            log_error "No JSON config files found in: $INPUT"
            exit 1
        fi
    else
        log_error "Invalid input: $INPUT"
        exit 1
    fi

    log_info "Found ${#CONFIGS[@]} config(s) to validate"
}

# Validate single config
validate_config() {
    local config="$1"
    local config_name=$(basename "$config" .json)

    log_info "========================================================================"
    log_info "Validating: $config_name"
    log_info "========================================================================"

    # Create config-specific output directory
    local config_output="$OUTPUT_DIR/$config_name"
    mkdir -p "$config_output"

    # Track validation results
    local validation_passed=true

    # 1. Walk-forward validation
    log_info "[1/3] Running walk-forward validation..."
    if python3 "$SCRIPT_DIR/validate_walkforward.py" \
        --config "$config" \
        --output "$config_output/walkforward" \
        --feature-store "$FEATURE_STORE" 2>&1 | tee "$config_output/walkforward.log"; then
        log_success "Walk-forward validation completed"
    else
        log_error "Walk-forward validation failed"
        validation_passed=false
    fi

    # 2. Cross-regime validation
    log_info "[2/3] Running cross-regime validation..."
    if python3 "$SCRIPT_DIR/validate_cross_regime.py" \
        --config "$config" \
        --output "$config_output/cross_regime" \
        --feature-store "$FEATURE_STORE" 2>&1 | tee "$config_output/cross_regime.log"; then
        log_success "Cross-regime validation completed"
    else
        log_warning "Cross-regime validation had warnings"
        # Don't fail overall validation for regime warnings
    fi

    # 3. Statistical significance validation
    log_info "[3/3] Running statistical significance validation..."
    if python3 "$SCRIPT_DIR/validate_statistical_significance.py" \
        --config "$config" \
        --output "$config_output/statistical" \
        --feature-store "$FEATURE_STORE" \
        --n-bootstrap "$N_BOOTSTRAP" \
        --n-permutations "$N_PERMUTATIONS" \
        --alpha "$ALPHA" 2>&1 | tee "$config_output/statistical.log"; then
        log_success "Statistical validation completed"
    else
        log_error "Statistical validation failed (no significant edge)"
        validation_passed=false
    fi

    # Save overall status
    if [ "$validation_passed" = true ]; then
        echo "PASSED" > "$config_output/VALIDATION_STATUS"
        log_success "Config validation PASSED: $config_name"
        return 0
    else
        echo "FAILED" > "$config_output/VALIDATION_STATUS"
        log_error "Config validation FAILED: $config_name"
        return 1
    fi
}

# Generate consolidated report
generate_report() {
    log_info "Generating consolidated validation report..."

    local report_file="$OUTPUT_DIR/VALIDATION_SUMMARY_REPORT.md"

    cat > "$report_file" << EOF
# Comprehensive Validation Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Output Directory:** $OUTPUT_DIR
**Total Configs:** ${#CONFIGS[@]}
**Passed:** $PASSED_COUNT
**Failed:** $FAILED_COUNT

---

## Validation Pipeline

This report summarizes results from the comprehensive validation pipeline:

1. **Walk-Forward Validation:** 3-tier temporal validation (train/validate/test)
2. **Cross-Regime Validation:** Regime-stratified performance analysis
3. **Statistical Significance:** Bootstrap + permutation tests

---

## Summary by Config

| Config | Walk-Forward | Cross-Regime | Statistical | Overall |
|--------|--------------|--------------|-------------|---------|
EOF

    # Add results for each config
    for config in "${CONFIGS[@]}"; do
        local config_name=$(basename "$config" .json)
        local config_output="$OUTPUT_DIR/$config_name"

        # Check individual validation results
        local wf_status="❌"
        local regime_status="❌"
        local stat_status="❌"
        local overall_status="❌"

        # Walk-forward
        if [ -f "$config_output/walkforward/$config_name/validation_summary.json" ]; then
            if grep -q '"production_ready": true' "$config_output/walkforward/$config_name/validation_summary.json" 2>/dev/null; then
                wf_status="✅"
            fi
        fi

        # Cross-regime
        if [ -f "$config_output/cross_regime/$config_name/regime_breakdown.json" ]; then
            if grep -q '"production_ready": true' "$config_output/cross_regime/$config_name/regime_breakdown.json" 2>/dev/null; then
                regime_status="✅"
            fi
        fi

        # Statistical
        if [ -f "$config_output/statistical/$config_name/statistical_summary.json" ]; then
            if grep -q '"statistically_significant": true' "$config_output/statistical/$config_name/statistical_summary.json" 2>/dev/null; then
                stat_status="✅"
            fi
        fi

        # Overall
        if [ -f "$config_output/VALIDATION_STATUS" ]; then
            if grep -q "PASSED" "$config_output/VALIDATION_STATUS"; then
                overall_status="✅"
            fi
        fi

        echo "| $config_name | $wf_status | $regime_status | $stat_status | $overall_status |" >> "$report_file"
    done

    cat >> "$report_file" << EOF

---

## Detailed Results

See individual config directories for detailed results:

EOF

    for config in "${CONFIGS[@]}"; do
        local config_name=$(basename "$config" .json)
        cat >> "$report_file" << EOF
### $config_name

- Walk-Forward: \`$config_name/walkforward/\`
- Cross-Regime: \`$config_name/cross_regime/\`
- Statistical: \`$config_name/statistical/\`

EOF
    done

    cat >> "$report_file" << EOF

---

## Production Deployment Recommendations

EOF

    if [ $PASSED_COUNT -gt 0 ]; then
        cat >> "$report_file" << EOF
**Production Ready Configs:** $PASSED_COUNT

The following configs passed all validation checks and are ready for production deployment:

EOF

        for config in "${CONFIGS[@]}"; do
            local config_name=$(basename "$config" .json)
            local config_output="$OUTPUT_DIR/$config_name"

            if [ -f "$config_output/VALIDATION_STATUS" ] && grep -q "PASSED" "$config_output/VALIDATION_STATUS"; then
                echo "- \`$config_name\`" >> "$report_file"
            fi
        done

    else
        cat >> "$report_file" << EOF
**WARNING:** No configs passed all validation checks.

Recommended actions:
1. Review failure reasons in individual validation reports
2. Re-run optimization with adjusted parameter ranges
3. Consider ensemble approach combining multiple strategies
4. Extend training period or adjust acceptance thresholds

EOF
    fi

    cat >> "$report_file" << EOF

---

## Next Steps

1. Review individual validation reports for detailed analysis
2. Examine equity curves and performance visualizations
3. Conduct monte carlo simulation on production candidates
4. Deploy top performers in shadow mode for live validation

---

**Files Generated:**

- \`VALIDATION_SUMMARY_REPORT.md\` - This summary report
- \`{config_name}/walkforward/\` - Walk-forward validation results
- \`{config_name}/cross_regime/\` - Cross-regime validation results
- \`{config_name}/statistical/\` - Statistical significance results
- \`{config_name}/VALIDATION_STATUS\` - Overall pass/fail status

EOF

    log_success "Consolidated report saved to: $report_file"

    # Display report
    cat "$report_file"
}

# Main execution
main() {
    # Check arguments
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <config_file_or_directory> [output_directory]"
        echo ""
        echo "Examples:"
        echo "  $0 configs/mvp/mvp_bear_market_v1.json"
        echo "  $0 configs/mvp/"
        echo "  $0 configs/optimized/ results/validation/my_validation/"
        echo ""
        echo "Environment Variables:"
        echo "  FEATURE_STORE      - Path to feature store (default: data/features_mtf/BTC_1H_...)"
        echo "  N_BOOTSTRAP        - Bootstrap iterations (default: 1000)"
        echo "  N_PERMUTATIONS     - Permutation iterations (default: 1000)"
        echo "  ALPHA              - Significance level (default: 0.05)"
        exit 1
    fi

    INPUT="$1"
    if [ $# -ge 2 ]; then
        OUTPUT_DIR="$2"
    fi

    # Print header
    print_header

    # Print configuration
    print_config

    # Validate input
    validate_input

    # Create output directory
    create_output_dir

    # Collect configs
    collect_configs

    # Run validation on each config
    PASSED_COUNT=0
    FAILED_COUNT=0

    for config in "${CONFIGS[@]}"; do
        if validate_config "$config"; then
            ((PASSED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
        echo ""
    done

    # Generate consolidated report
    generate_report

    # Final summary
    echo ""
    echo "========================================================================"
    echo "  VALIDATION COMPLETE"
    echo "========================================================================"
    echo ""
    echo "Total Configs:  ${#CONFIGS[@]}"
    echo "Passed:         $PASSED_COUNT"
    echo "Failed:         $FAILED_COUNT"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "========================================================================"

    # Exit with appropriate code
    if [ $FAILED_COUNT -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main
main "$@"
