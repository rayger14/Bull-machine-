#!/bin/bash

###############################################################################
# QUANT LAB VALIDATION PROTOCOL - MASTER SCRIPT
#
# Runs all 9 validation steps to ensure archetype engine testing is correct.
#
# Usage:
#   bash bin/validate_archetype_engine.sh --full
#   bash bin/validate_archetype_engine.sh --steps 1-3
#   bash bin/validate_archetype_engine.sh --step 4
###############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation state
VALIDATION_REPORT="validation_report_$(date +%Y%m%d_%H%M%S).txt"
STEPS_PASSED=0
STEPS_FAILED=0
TOTAL_STEPS=9

# Logging functions
log_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

log_success() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    echo "[PASS] $1" >> "$VALIDATION_REPORT"
    ((STEPS_PASSED++))
}

log_failure() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    echo "[FAIL] $1" >> "$VALIDATION_REPORT"
    ((STEPS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
    echo "[WARNING] $1" >> "$VALIDATION_REPORT"
}

log_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

# Initialize report
init_report() {
    cat > "$VALIDATION_REPORT" << EOF
========================================
QUANT LAB VALIDATION PROTOCOL REPORT
Generated: $(date)
========================================

EOF
}

# Step 1: Confirm Feature Store Coverage
step_1_feature_coverage() {
    log_header "STEP 1: Confirm Feature Store Coverage"

    if [ -f "bin/audit_archetype_pipeline.py" ]; then
        log_info "Running feature coverage audit..."

        if python bin/audit_archetype_pipeline.py > /tmp/step1_output.txt 2>&1; then
            # Check for minimum coverage
            coverage=$(grep -oP 'Overall: \K[0-9.]+' /tmp/step1_output.txt || echo "0")

            if (( $(echo "$coverage >= 98.0" | bc -l) )); then
                log_success "Feature coverage: ${coverage}% (≥ 98% required)"
                cat /tmp/step1_output.txt >> "$VALIDATION_REPORT"
                return 0
            else
                log_failure "Feature coverage: ${coverage}% (< 98%)"
                cat /tmp/step1_output.txt >> "$VALIDATION_REPORT"
                return 1
            fi
        else
            log_failure "Feature audit script failed"
            cat /tmp/step1_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/audit_archetype_pipeline.py not found - creating stub"
        create_audit_script
        log_info "Please run validation again after implementing audit_archetype_pipeline.py"
        return 1
    fi
}

# Step 2: Validate Feature Name Mapping
step_2_feature_mapping() {
    log_header "STEP 2: Validate Feature Name Mapping"

    if [ -f "bin/verify_feature_mapping.py" ]; then
        log_info "Verifying feature name mappings..."

        if python bin/verify_feature_mapping.py > /tmp/step2_output.txt 2>&1; then
            # Check for mapping errors
            if grep -q "feature not found" /tmp/step2_output.txt; then
                log_failure "Feature mapping errors detected"
                cat /tmp/step2_output.txt >> "$VALIDATION_REPORT"
                return 1
            else
                log_success "All feature mappings verified"
                cat /tmp/step2_output.txt >> "$VALIDATION_REPORT"
                return 0
            fi
        else
            log_failure "Feature mapping verification failed"
            cat /tmp/step2_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/verify_feature_mapping.py not found"
        return 1
    fi
}

# Step 3: Confirm Domain Engines Are ON
step_3_domain_engines() {
    log_header "STEP 3: Confirm Domain Engines Are ON"

    if [ -f "bin/check_domain_engines.py" ]; then
        log_info "Checking domain engine status..."

        if python bin/check_domain_engines.py --s1 --s4 --s5 > /tmp/step3_output.txt 2>&1; then
            # Check for all engines enabled
            enabled_count=$(grep -c "✓ ENABLED" /tmp/step3_output.txt || echo "0")

            if [ "$enabled_count" -ge 18 ]; then
                log_success "All 18 domain engines enabled (6 engines × 3 systems)"
                cat /tmp/step3_output.txt >> "$VALIDATION_REPORT"
                return 0
            else
                log_failure "Only $enabled_count/18 engines enabled"
                cat /tmp/step3_output.txt >> "$VALIDATION_REPORT"
                return 1
            fi
        else
            log_failure "Domain engine check failed"
            cat /tmp/step3_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/check_domain_engines.py not found"
        return 1
    fi
}

# Step 4: Confirm Archetype NOT Falling Back to Tier1
step_4_no_fallback() {
    log_header "STEP 4: Confirm Archetype NOT Falling Back to Tier1"

    if [ -f "bin/check_tier1_fallback.py" ]; then
        log_info "Checking for Tier1 fallback behavior..."

        if python bin/check_tier1_fallback.py --test-period 2022-05-01:2022-08-01 > /tmp/step4_output.txt 2>&1; then
            # Check fallback percentage
            fallback_pct=$(grep -oP 'Fallback: \K[0-9.]+%' /tmp/step4_output.txt || echo "100%")
            fallback_num=$(echo "$fallback_pct" | sed 's/%//')

            if (( $(echo "$fallback_num < 30" | bc -l) )); then
                log_success "Fallback trades: ${fallback_pct} (< 30% required)"
                cat /tmp/step4_output.txt >> "$VALIDATION_REPORT"
                return 0
            else
                log_failure "Fallback trades: ${fallback_pct} (≥ 30%)"
                cat /tmp/step4_output.txt >> "$VALIDATION_REPORT"
                return 1
            fi
        else
            log_failure "Tier1 fallback check failed"
            cat /tmp/step4_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/check_tier1_fallback.py not found"
        return 1
    fi
}

# Step 5: Confirm OI/Funding Are Loaded Properly
step_5_oi_funding_data() {
    log_header "STEP 5: Confirm OI/Funding Are Loaded Properly"

    log_info "Checking funding data..."
    if [ -f "bin/check_funding_data.py" ]; then
        python bin/check_funding_data.py > /tmp/step5a_output.txt 2>&1 || true
    else
        echo "check_funding_data.py not found" > /tmp/step5a_output.txt
    fi

    log_info "Checking OI data..."
    if [ -f "bin/check_oi_data.py" ]; then
        python bin/check_oi_data.py > /tmp/step5b_output.txt 2>&1 || true
    else
        echo "check_oi_data.py not found" > /tmp/step5b_output.txt
    fi

    # Check both outputs
    funding_ok=false
    oi_ok=false

    if grep -q "< 20% null" /tmp/step5a_output.txt; then
        funding_ok=true
    fi

    if grep -q "< 20% null" /tmp/step5b_output.txt; then
        oi_ok=true
    fi

    cat /tmp/step5a_output.txt >> "$VALIDATION_REPORT"
    cat /tmp/step5b_output.txt >> "$VALIDATION_REPORT"

    if [ "$funding_ok" = true ] && [ "$oi_ok" = true ]; then
        log_success "Funding and OI data properly loaded"
        return 0
    else
        log_failure "Missing or incomplete funding/OI data"
        return 1
    fi
}

# Step 6: Reproduce Short-Window Behavior
step_6_chaos_windows() {
    log_header "STEP 6: Reproduce Short-Window Behavior (Plumbing Sanity)"

    if [ -f "bin/test_chaos_windows.py" ]; then
        log_info "Testing chaos windows (Terra, FTX, CPI)..."

        if python bin/test_chaos_windows.py --s4 > /tmp/step6_output.txt 2>&1; then
            # Check for non-zero trades in each window
            terra_trades=$(grep -oP 'Terra.*: \K[0-9]+' /tmp/step6_output.txt || echo "0")
            ftx_trades=$(grep -oP 'FTX.*: \K[0-9]+' /tmp/step6_output.txt || echo "0")

            if [ "$terra_trades" -gt 0 ] && [ "$ftx_trades" -gt 0 ]; then
                log_success "Chaos windows producing trades (Terra: $terra_trades, FTX: $ftx_trades)"
                cat /tmp/step6_output.txt >> "$VALIDATION_REPORT"
                return 0
            else
                log_failure "Zero trades in chaos windows (Terra: $terra_trades, FTX: $ftx_trades)"
                cat /tmp/step6_output.txt >> "$VALIDATION_REPORT"
                return 1
            fi
        else
            log_failure "Chaos window test failed"
            cat /tmp/step6_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/test_chaos_windows.py not found"
        return 1
    fi
}

# Step 7: Apply Optimized Calibrations
step_7_optimized_calibrations() {
    log_header "STEP 7: Apply OPTIMIZED CALIBRATIONS"

    if [ -f "bin/apply_optimized_calibrations.py" ]; then
        log_info "Applying Optuna-derived calibrations..."

        if python bin/apply_optimized_calibrations.py --s1 --s4 --s5 > /tmp/step7_output.txt 2>&1; then
            # Verify optimized flag in configs
            configs_optimized=0
            for config in configs/s1_v2_production.json configs/s4_optimized_oos_test.json configs/s5_production.json; do
                if [ -f "$config" ] && grep -q '"optimized": true' "$config"; then
                    ((configs_optimized++))
                fi
            done

            if [ "$configs_optimized" -eq 3 ]; then
                log_success "All 3 configs have optimized calibrations"
                cat /tmp/step7_output.txt >> "$VALIDATION_REPORT"
                return 0
            else
                log_failure "Only $configs_optimized/3 configs optimized"
                cat /tmp/step7_output.txt >> "$VALIDATION_REPORT"
                return 1
            fi
        else
            log_failure "Calibration application failed"
            cat /tmp/step7_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/apply_optimized_calibrations.py not found"
        # Check if configs already have optimized flag
        log_info "Checking if configs already optimized..."
        configs_optimized=0
        for config in configs/s1_v2_production.json configs/s4_optimized_oos_test.json; do
            if [ -f "$config" ] && grep -q '"optimized": true' "$config"; then
                ((configs_optimized++))
            fi
        done

        if [ "$configs_optimized" -ge 1 ]; then
            log_warning "Some configs already optimized - proceeding"
            return 0
        else
            return 1
        fi
    fi
}

# Step 8: Full-Period Validation
step_8_full_validation() {
    log_header "STEP 8: Full-Period Validation (The REAL Test)"

    if [ -f "bin/run_archetype_suite.py" ]; then
        log_info "Running full validation (train/test/OOS)..."
        log_info "This may take 30-60 minutes..."

        if python bin/run_archetype_suite.py --periods train,test,oos > /tmp/step8_output.txt 2>&1; then
            # Parse performance metrics
            s4_pf=$(grep -oP 'S4.*Test PF: \K[0-9.]+' /tmp/step8_output.txt || echo "0")
            s1_pf=$(grep -oP 'S1.*Test PF: \K[0-9.]+' /tmp/step8_output.txt || echo "0")
            s5_pf=$(grep -oP 'S5.*Test PF: \K[0-9.]+' /tmp/step8_output.txt || echo "0")

            # Check minimum acceptable performance
            s4_ok=false
            s1_ok=false
            s5_ok=false

            if (( $(echo "$s4_pf >= 2.2" | bc -l) )); then s4_ok=true; fi
            if (( $(echo "$s1_pf >= 1.8" | bc -l) )); then s1_ok=true; fi
            if (( $(echo "$s5_pf >= 1.6" | bc -l) )); then s5_ok=true; fi

            cat /tmp/step8_output.txt >> "$VALIDATION_REPORT"

            if [ "$s4_ok" = true ] && [ "$s1_ok" = true ] && [ "$s5_ok" = true ]; then
                log_success "Performance meets minimums (S4: $s4_pf, S1: $s1_pf, S5: $s5_pf)"
                return 0
            else
                log_failure "Performance below minimums (S4: $s4_pf [need ≥2.2], S1: $s1_pf [need ≥1.8], S5: $s5_pf [need ≥1.6])"
                return 1
            fi
        else
            log_failure "Full validation suite failed"
            cat /tmp/step8_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/run_archetype_suite.py not found"
        return 1
    fi
}

# Step 9: Compare Against Baselines
step_9_baseline_comparison() {
    log_header "STEP 9: Compare Against Baselines (Final Truth)"

    if [ -f "bin/compare_archetypes_vs_baselines.py" ]; then
        log_info "Running baseline comparison..."

        if python bin/compare_archetypes_vs_baselines.py > /tmp/step9_output.txt 2>&1; then
            cat /tmp/step9_output.txt >> "$VALIDATION_REPORT"

            # Check if any archetype beats or competes with baselines
            if grep -q "Scenario A: Clear Winners" /tmp/step9_output.txt; then
                log_success "Archetypes beat baselines - ready for deployment"
                return 0
            elif grep -q "Scenario B: Competitive" /tmp/step9_output.txt; then
                log_success "Archetypes competitive - suitable for hybrid deployment"
                return 0
            else
                log_failure "Archetypes underperform baselines"
                return 1
            fi
        else
            log_failure "Baseline comparison failed"
            cat /tmp/step9_output.txt >> "$VALIDATION_REPORT"
            return 1
        fi
    else
        log_warning "bin/compare_archetypes_vs_baselines.py not found"
        return 1
    fi
}

# Helper: Create audit script stub
create_audit_script() {
    cat > bin/audit_archetype_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""
Feature coverage audit stub.
TODO: Implement full feature store audit.
"""
print("Overall: 0.0% coverage")
print("NOTE: audit_archetype_pipeline.py is a stub - implement feature audit logic")
exit(1)
EOF
    chmod +x bin/audit_archetype_pipeline.py
}

# Main execution
main() {
    local run_steps="all"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                run_steps="all"
                shift
                ;;
            --steps)
                run_steps="$2"
                shift 2
                ;;
            --step)
                run_steps="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--full | --steps 1-3 | --step 4]"
                exit 1
                ;;
        esac
    done

    init_report

    log_header "QUANT LAB VALIDATION PROTOCOL"
    log_info "Report: $VALIDATION_REPORT"

    # Run requested steps
    case $run_steps in
        all|1-9)
            step_1_feature_coverage || true
            step_2_feature_mapping || true
            step_3_domain_engines || true
            step_4_no_fallback || true
            step_5_oi_funding_data || true
            step_6_chaos_windows || true
            step_7_optimized_calibrations || true
            step_8_full_validation || true
            step_9_baseline_comparison || true
            ;;
        1-3)
            step_1_feature_coverage || true
            step_2_feature_mapping || true
            step_3_domain_engines || true
            ;;
        4-6)
            step_4_no_fallback || true
            step_5_oi_funding_data || true
            step_6_chaos_windows || true
            ;;
        7-9)
            step_7_optimized_calibrations || true
            step_8_full_validation || true
            step_9_baseline_comparison || true
            ;;
        1) step_1_feature_coverage || true ;;
        2) step_2_feature_mapping || true ;;
        3) step_3_domain_engines || true ;;
        4) step_4_no_fallback || true ;;
        5) step_5_oi_funding_data || true ;;
        6) step_6_chaos_windows || true ;;
        7) step_7_optimized_calibrations || true ;;
        8) step_8_full_validation || true ;;
        9) step_9_baseline_comparison || true ;;
        *)
            echo "Invalid steps specification: $run_steps"
            exit 1
            ;;
    esac

    # Final summary
    log_header "VALIDATION SUMMARY"

    echo -e "\nSteps Passed: ${GREEN}$STEPS_PASSED${NC}"
    echo -e "Steps Failed: ${RED}$STEPS_FAILED${NC}"
    echo -e "Total Steps:  $TOTAL_STEPS\n"

    # Determine overall status
    if [ "$STEPS_FAILED" -eq 0 ] && [ "$STEPS_PASSED" -eq "$TOTAL_STEPS" ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ 100% VALIDATED - READY FOR PRODUCTION${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo "" >> "$VALIDATION_REPORT"
        echo "VERDICT: 100% VALIDATED" >> "$VALIDATION_REPORT"
        exit 0
    elif [ "$STEPS_PASSED" -ge 7 ]; then
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}⚠ PARTIAL VALIDATION - REVIEW REQUIRED${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo "" >> "$VALIDATION_REPORT"
        echo "VERDICT: PARTIAL" >> "$VALIDATION_REPORT"
        exit 1
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}✗ VALIDATION FAILED - FIX ERRORS${NC}"
        echo -e "${RED}========================================${NC}"
        echo "" >> "$VALIDATION_REPORT"
        echo "VERDICT: FAILED" >> "$VALIDATION_REPORT"
        exit 1
    fi
}

# Run main
main "$@"
