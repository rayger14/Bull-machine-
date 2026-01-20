#!/bin/bash
# Run full-year replays for all three assets

echo "================================================================================"
echo "Running Replay Validation Tests - All Assets"
echo "================================================================================"

echo ""
echo "🔹 BTC 2024 Replay..."
python3 bin/live/replay_runner.py \
  --asset BTC \
  --features data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/BTC_2024_best.json \
  --start 2024-01-01 --end 2024-12-31 \
  --speed 60 \
  --output reports/replay/BTC_2024_60x.json \
  2>&1 | grep -vE "^INFO:"

echo ""
echo "🔹 ETH 2024 Replay..."
python3 bin/live/replay_runner.py \
  --asset ETH \
  --features data/features_mtf/ETH_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/ETH_2024_best.json \
  --start 2024-01-01 --end 2024-12-31 \
  --speed 60 \
  --output reports/replay/ETH_2024_60x.json \
  2>&1 | grep -vE "^INFO:"

echo ""
echo "🔹 SPY 2024 Replay..."
python3 bin/live/replay_runner.py \
  --asset SPY \
  --features data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/SPY_2024_equity_tuned.json \
  --start 2024-01-01 --end 2024-12-31 \
  --speed 60 \
  --output reports/replay/SPY_2024_60x.json \
  2>&1 | grep -vE "^INFO:"

echo ""
echo "================================================================================"
echo "All replays complete! Results written to reports/replay/"
echo "================================================================================"
