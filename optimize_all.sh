#!/bin/bash
# Bull Machine v1.8.6 - Batch Optimization Script
# Optimize all assets (BTC, ETH, SOL) in parallel

YEARS=3
MODE="grid"  # quick, grid, or walkforward
OUTPUT_DIR="results/optimization_$(date +%Y%m%d_%H%M%S)"

echo "🎯 Bull Machine v1.8.6 Batch Optimizer"
echo "Mode: $MODE"
echo "Years: $YEARS"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to optimize single asset
optimize_asset() {
    local asset=$1
    echo "🚀 Optimizing $asset..."

    python bin/optimize_v18.py \
        --mode "$MODE" \
        --asset "$asset" \
        --years "$YEARS" \
        --output "$OUTPUT_DIR/${asset}_optimization.json" \
        > "$OUTPUT_DIR/${asset}_log.txt" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ $asset optimization complete"

        # Analyze results
        python bin/analyze_optimization.py \
            "$OUTPUT_DIR/${asset}_optimization.json" \
            > "$OUTPUT_DIR/${asset}_analysis.txt" 2>&1

        echo "📊 $asset analysis complete"
    else
        echo "❌ $asset optimization failed (see ${asset}_log.txt)"
    fi
}

# Run optimizations in parallel (background jobs)
for asset in BTC ETH SOL; do
    optimize_asset "$asset" &
done

# Wait for all background jobs
wait

echo ""
echo "{'='*60}"
echo "BATCH OPTIMIZATION COMPLETE"
echo "{'='*60}"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View analysis:"
echo "  cat $OUTPUT_DIR/BTC_analysis.txt"
echo "  cat $OUTPUT_DIR/ETH_analysis.txt"
echo "  cat $OUTPUT_DIR/SOL_analysis.txt"
echo ""
echo "Compare best configs:"
echo "  grep 'RANK #1' $OUTPUT_DIR/*_analysis.txt -A 15"
