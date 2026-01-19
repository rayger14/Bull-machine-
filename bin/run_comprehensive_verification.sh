#!/bin/bash
################################################################################
# Comprehensive Verification Test Suite
#
# Runs all verification tests to prove production readiness:
# 1. Domain engine gate fix (S1_core vs S1_full)
# 2. Feature store quality
# 3. OI graceful degradation
# 4. Safety checks
# 5. Performance regression
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$PROJECT_ROOT/verification_reports_$TIMESTAMP"

mkdir -p "$REPORT_DIR"

echo "================================================================================"
echo "COMPREHENSIVE VERIFICATION TEST SUITE"
echo "================================================================================"
echo "Timestamp: $(date)"
echo "Report directory: $REPORT_DIR"
echo ""

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    local output_file="$REPORT_DIR/${test_name}.log"

    echo "--------------------------------------------------------------------------------"
    echo "TEST: $test_name"
    echo "--------------------------------------------------------------------------------"
    echo "Command: $test_command"
    echo "Output: $output_file"
    echo ""

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if eval "$test_command" > "$output_file" 2>&1; then
        echo "✅ PASSED: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo "❌ FAILED: $test_name"
        echo "See log: $output_file"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

################################################################################
# TEST 1: Feature Store Quality
################################################################################
echo ""
echo "================================================================================"
echo "TEST 1: Feature Store Quality"
echo "================================================================================"

run_test "feature_store_quality" \
    "python3 $SCRIPT_DIR/verify_feature_store_quality.py"

################################################################################
# TEST 2: Domain Engine Gate Fix (Critical)
################################################################################
echo ""
echo "================================================================================"
echo "TEST 2: Domain Engine Gate Fix (S1_core vs S1_full)"
echo "================================================================================"

run_test "domain_gate_fix" \
    "python3 $SCRIPT_DIR/verify_domain_wiring.py"

################################################################################
# TEST 3: OI/Funding Graceful Degradation
################################################################################
echo ""
echo "================================================================================"
echo "TEST 3: OI/Funding Graceful Degradation (2022 Data)"
echo "================================================================================"

# Create test config for S4 on 2022 data
cat > "$REPORT_DIR/s4_2022_test.json" <<'EOF'
{
  "systems": ["S4"],
  "strategy_configs": {
    "S4": {
      "archetype": "funding_divergence",
      "min_funding_divergence": 0.002,
      "use_oi_fallback": true
    }
  },
  "risk_per_trade": 0.02,
  "use_market_context": true,
  "date_range": {
    "start": "2022-01-01",
    "end": "2023-01-01"
  }
}
EOF

run_test "oi_graceful_degradation_s4" \
    "python3 $PROJECT_ROOT/bull_machine/tools/backtest.py --config $REPORT_DIR/s4_2022_test.json" || true

# Create test config for S5 on 2022 data
cat > "$REPORT_DIR/s5_2022_test.json" <<'EOF'
{
  "systems": ["S5"],
  "strategy_configs": {
    "S5": {
      "archetype": "long_squeeze",
      "min_squeeze_intensity": 0.7,
      "use_oi_fallback": true
    }
  },
  "risk_per_trade": 0.02,
  "use_market_context": true,
  "date_range": {
    "start": "2022-01-01",
    "end": "2023-01-01"
  }
}
EOF

run_test "oi_graceful_degradation_s5" \
    "python3 $PROJECT_ROOT/bull_machine/tools/backtest.py --config $REPORT_DIR/s5_2022_test.json" || true

################################################################################
# TEST 4: Safety Checks (Vetoes)
################################################################################
echo ""
echo "================================================================================"
echo "TEST 4: Safety Checks - Permissive vs Strict"
echo "================================================================================"

# Create permissive config
cat > "$REPORT_DIR/permissive_test.json" <<'EOF'
{
  "systems": ["S1"],
  "strategy_configs": {
    "S1": {
      "archetype": "liquidity_vacuum",
      "min_liquidity_score": 0.1,
      "min_volume_surge": 0.5,
      "min_oi_change": 0.001
    }
  },
  "risk_per_trade": 0.02,
  "use_market_context": true,
  "feature_flags": {
    "use_safety_vetoes": true
  }
}
EOF

run_test "safety_permissive" \
    "python3 $PROJECT_ROOT/bull_machine/tools/backtest.py --config $REPORT_DIR/permissive_test.json" || true

# Create strict config
cat > "$REPORT_DIR/strict_test.json" <<'EOF'
{
  "systems": ["S1"],
  "strategy_configs": {
    "S1": {
      "archetype": "liquidity_vacuum",
      "min_liquidity_score": 0.9,
      "min_volume_surge": 3.0,
      "min_oi_change": 0.1
    }
  },
  "risk_per_trade": 0.02,
  "use_market_context": true
}
EOF

run_test "safety_strict" \
    "python3 $PROJECT_ROOT/bull_machine/tools/backtest.py --config $REPORT_DIR/strict_test.json" || true

################################################################################
# TEST 5: Performance Regression
################################################################################
echo ""
echo "================================================================================"
echo "TEST 5: Performance Regression"
echo "================================================================================"

# Time a standard backtest
cat > "$REPORT_DIR/performance_test.json" <<'EOF'
{
  "systems": ["S1"],
  "strategy_configs": {
    "S1": {
      "archetype": "liquidity_vacuum",
      "min_liquidity_score": 0.6,
      "min_volume_surge": 1.5
    }
  },
  "risk_per_trade": 0.02,
  "use_market_context": true
}
EOF

START_TIME=$(date +%s)
run_test "performance_baseline" \
    "python3 $PROJECT_ROOT/bull_machine/tools/backtest.py --config $REPORT_DIR/performance_test.json" || true
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "Performance test runtime: ${RUNTIME}s"

if [ $RUNTIME -lt 60 ]; then
    echo "✅ Performance: Runtime ${RUNTIME}s < 60s target"
else
    echo "⚠️  Performance: Runtime ${RUNTIME}s exceeds 60s target"
fi

################################################################################
# SUMMARY REPORT
################################################################################
echo ""
echo "================================================================================"
echo "VERIFICATION SUMMARY"
echo "================================================================================"
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

# Generate JSON summary
cat > "$REPORT_DIR/summary.json" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "performance_runtime_seconds": $RUNTIME,
  "report_directory": "$REPORT_DIR"
}
EOF

echo "Summary report: $REPORT_DIR/summary.json"
echo ""

# Production readiness decision
if [ $FAILED_TESTS -eq 0 ]; then
    echo "================================================================================"
    echo "✅ PRODUCTION READY"
    echo "================================================================================"
    echo "All verification tests passed."
    echo "System is ready for deployment."
    echo ""
    exit 0
else
    echo "================================================================================"
    echo "❌ NOT PRODUCTION READY"
    echo "================================================================================"
    echo "Some tests failed. Review logs in: $REPORT_DIR"
    echo ""
    exit 1
fi
