#!/bin/bash
# Knowledge v2.0 A/B/C Testing - Execution Script
# Run this to execute the full Q3 2024 test suite

set -e  # Exit on error

echo "======================================================================="
echo "Knowledge v2.0 A/B/C Testing Suite"
echo "======================================================================="
echo ""
echo "This will run 3 tests:"
echo "  1. BASELINE   - Knowledge v2.0 disabled"
echo "  2. SHADOW     - Knowledge v2.0 logs only (validates integration)"
echo "  3. ACTIVE     - Knowledge v2.0 modifies decisions"
echo ""
echo "Expected runtime: 30-45 minutes (with on-the-fly feature computation)"
echo "Feature store already built: data/features_v2/ETH_1H_2024-07-01_to_2024-09-30.parquet"
echo ""
echo "======================================================================="
echo ""

# Create output directory
mkdir -p reports/v2_ab_test

# Run tests
echo "Starting A/B/C tests..."
echo "Output will be logged to reports/v2_ab_test/run.log"
echo ""

nohup python3 -u bin/compare_knowledge_v2_abc.py \
  --asset ETH \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --configs configs/knowledge_v2/ETH_baseline.json,configs/knowledge_v2/ETH_shadow_mode.json,configs/knowledge_v2/ETH_v2_active.json \
  --output reports/v2_ab_test \
  > reports/v2_ab_test/run.log 2>&1 &

PID=$!
echo "Tests started in background (PID: $PID)"
echo ""
echo "To monitor progress:"
echo "  tail -f reports/v2_ab_test/run.log"
echo ""
echo "To check individual test logs:"
echo "  tail -f reports/v2_ab_test/ETH_baseline.log"
echo "  tail -f reports/v2_ab_test/ETH_shadow_mode.log"
echo "  tail -f reports/v2_ab_test/ETH_v2_active.log"
echo ""
echo "When complete, view results:"
echo "  cat reports/v2_ab_test/comparison_report.txt"
echo ""
echo "======================================================================="
