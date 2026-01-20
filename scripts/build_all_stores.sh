#!/usr/bin/env bash
#
# Build MTF feature stores for all assets (BTC, ETH, SPY, TSLA)
#
# Usage:
#   bash scripts/build_all_stores.sh [year]
#   Default: 2024 full year
#

set -euo pipefail

YEAR=${1:-2024}
START_DATE="${YEAR}-01-01"
END_DATE="${YEAR}-12-31"

OUTPUT_DIR="data/features_v2"
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Building MTF Feature Stores for All Assets"
echo "================================================================================"
echo "Period: $START_DATE to $END_DATE"
echo "Output: $OUTPUT_DIR/"
echo ""

# BTC (24/7 crypto)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1/4: Building BTC feature store..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bin/build_mtf_feature_store.py \
  --asset BTC \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --out "$OUTPUT_DIR/BTC_1H_${YEAR}.parquet"

# ETH (24/7 crypto)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2/4: Building ETH feature store..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bin/build_mtf_feature_store.py \
  --asset ETH \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --out "$OUTPUT_DIR/ETH_1H_${YEAR}.parquet"

# SPY (RTH only: 9:30-16:00 ET)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3/4: Building SPY feature store (RTH only)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bin/build_mtf_feature_store.py \
  --asset SPY \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --rth \
  --out "$OUTPUT_DIR/SPY_1H_${YEAR}.parquet"

# TSLA (RTH only: 9:30-16:00 ET)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4/4: Building TSLA feature store (RTH only)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bin/build_mtf_feature_store.py \
  --asset TSLA \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --rth \
  --out "$OUTPUT_DIR/TSLA_1H_${YEAR}.parquet"

echo ""
echo "================================================================================"
echo "✅ All Feature Stores Built Successfully"
echo "================================================================================"
echo ""
ls -lh "$OUTPUT_DIR/"*.parquet
echo ""

# Verify each store
echo "Verifying feature stores..."
for asset in BTC ETH SPY TSLA; do
  echo ""
  echo "Verifying ${asset}..."
  python3 scripts/verify_features.py "$OUTPUT_DIR/${asset}_1H_${YEAR}.parquet" || true
done

echo ""
echo "================================================================================"
echo "Build Complete"
echo "================================================================================"
