#!/usr/bin/env bash
# Orchestrates the full-history live-path rebuild after klines are downloaded.
#  1. Prebuild macro + derivatives caches ONCE (avoids 8-worker race)
#  2. Launch 8 parallel chunk workers
#  3. (stitch + V14 build run separately after workers finish)
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Prebuilding caches (single process) ==="
rm -f data/cache/macro_daily_history.parquet   # rebuilt with 2017 start
python3 - << 'PYEOF'
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/rebuild'); sys.path.insert(0, 'scripts/data')
import pandas as pd
from replay_segment import build_macro_history, build_derivatives_history, KLINES
macro = build_macro_history()
print(f"macro cache: {len(macro)} days {macro.index[0]} -> {macro.index[-1]}")
klines = pd.read_parquet(KLINES)
klines.index = klines.index.tz_localize(None)
deriv = build_derivatives_history(klines["close"])
print(f"derivatives cache: {len(deriv)} hours {deriv.index[0]} -> {deriv.index[-1]}")
PYEOF

echo "=== Launching 8 chunk workers ==="
for i in 1 2 3 4 5 6 7 8; do
  nohup python3 scripts/rebuild/replay_segment.py --chunk $i/8 \
    > results/rebuild/chunk_$i.log 2>&1 &
done
sleep 5
echo "workers running: $(pgrep -f 'replay_segment.py --chunk' | wc -l | tr -d ' ')"
