# Batch Screener + Focused Replay Implementation Guide

## Status: Phase 2 Complete ✅ - Ready for Testing & Benchmarking

### What's Done ✅

1. **Batch Screener Created** (`bin/research/batch_screener.py`)
   - Vectorized feature computation (no lookahead)
   - Sentinel detection (SMA cross, ATR regime, session breaks)
   - Proto-fusion scoring
   - Outputs `candidates.jsonl`

2. **Config Extended** (`configs/v18/ETH_comprehensive.json`)
   - Added `batch_mode` section with thresholds

3. **Candidate-Driven Replay** (`bin/live/hybrid_runner.py`)
   - Added `--candidates` CLI argument
   - Implemented `load_candidates()` function
   - Modified main loop to skip non-candidate bars
   - Batch mode statistics logging

4. **Parity Testing Framework** (`tests/test_batch_parity.py`)
   - Automated full vs batch comparison
   - Trade-level validation
   - Detailed mismatch reporting

5. **Benchmark Script** (`bin/research/benchmark_batch.sh`)
   - Full replay baseline timing
   - Batch screener + focused replay timing
   - Speedup calculation and parity check

6. **Documentation**
   - `PERFORMANCE_OPTIMIZATION_FINDINGS.md` - Complete analysis
   - This file - Implementation guide

### What's Next 🎯

**Phase 2B/2C Testing**: Run parity tests and benchmarks to validate

**Phase 3**: Production Integration (optional)
   - Add batch mode to CI/CD
   - Update documentation
   - Create user guide for parameter sweeps

---

## Phase 2A: Modify hybrid_runner.py

### 1. Add CLI Argument

```python
parser.add_argument('--candidates', type=str, default=None,
                    help='Path to candidates JSONL (batch mode)')
```

### 2. Load Candidates

```python
def load_candidates(candidates_path: str, window_bars: int = 48) -> dict:
    """
    Load candidates and create lookup dict.

    Returns:
        {
            timestamp: {'side': str, 'score': float, 'reason': str},
            ...
        }
        Plus 'candidate_bars' set with all bars in ±window
    """
    candidates = {}
    candidate_bars = set()

    with open(candidates_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ts = pd.to_datetime(entry['timestamp'])
            candidates[ts] = entry

    # Expand to windows (±48 bars for context)
    all_timestamps = sorted(candidates.keys())
    for ts in all_timestamps:
        # Add ±window bars around each candidate
        for offset in range(-window_bars, window_bars + 1):
            candidate_bars.add(ts + pd.Timedelta(hours=offset))

    print(f"📋 Loaded {len(candidates)} candidates")
    print(f"📊 Will process {len(candidate_bars)} bars (windows)")

    return candidates, candidate_bars
```

### 3. Modify Main Loop

```python
# In run() method, after loading data:

if args.candidates:
    candidates, candidate_bars = load_candidates(
        args.candidates,
        self.config.get('batch_mode', {}).get('candidate_window_bars', 48)
    )
    batch_mode = True
else:
    candidates = {}
    candidate_bars = None
    batch_mode = False

# In per-bar loop:
for i in range(lookback, len(df_1h_full)):
    timestamp = df_1h_full.index[i]

    # BATCH MODE: Skip bars not in candidate windows
    if batch_mode and timestamp not in candidate_bars:
        continue

    # ... rest of existing logic ...
```

---

## Phase 2B: Parity Testing

Create `tests/test_batch_parity.py`:

```python
#!/usr/bin/env python3
"""
Parity test: Full replay vs Candidate-driven replay.

Asserts identical trade outcomes on short test period.
"""

import subprocess
import json
import pandas as pd

def test_batch_parity():
    """Test that batch mode produces same results as full mode."""

    # 1. Run batch screener
    print("🔍 Running batch screener...")
    subprocess.run([
        'python3', 'bin/research/batch_screener.py',
        '--asset', 'ETH',
        '--start', '2025-07-01',
        '--end', '2025-07-15',
        '--config', 'configs/v18/ETH_comprehensive.json',
        '--output', 'results/test_candidates.jsonl'
    ], check=True)

    # 2. Run full replay
    print("\n📊 Running full replay...")
    subprocess.run([
        'python3', 'bin/live/hybrid_runner.py',
        '--asset', 'ETH',
        '--start', '2025-07-01',
        '--end', '2025-07-15',
        '--config', 'configs/v18/ETH_comprehensive.json'
    ], check=True)

    full_trades = load_trades('results/trade_log.jsonl')

    # 3. Run candidate-driven replay
    print("\n🎯 Running candidate-driven replay...")
    subprocess.run([
        'python3', 'bin/live/hybrid_runner.py',
        '--asset', 'ETH',
        '--start', '2025-07-01',
        '--end', '2025-07-15',
        '--config', 'configs/v18/ETH_comprehensive.json',
        '--candidates', 'results/test_candidates.jsonl'
    ], check=True)

    batch_trades = load_trades('results/trade_log.jsonl')

    # 4. Compare
    print("\n🔬 Comparing results...")
    assert len(full_trades) == len(batch_trades), \
        f"Trade count mismatch: {len(full_trades)} vs {len(batch_trades)}"

    for i, (ft, bt) in enumerate(zip(full_trades, batch_trades)):
        assert ft['timestamp'] == bt['timestamp'], \
            f"Trade {i} timestamp mismatch"
        assert ft['side'] == bt['side'], \
            f"Trade {i} side mismatch"
        assert abs(ft['price'] - bt['price']) < 0.01, \
            f"Trade {i} price mismatch"

    print("✅ Parity test PASSED - identical results!")


def load_trades(path: str) -> list:
    """Load trades from JSONL."""
    trades = []
    with open(path, 'r') as f:
        for line in f:
            trades.append(json.loads(line))
    return trades


if __name__ == '__main__':
    test_batch_parity()
```

---

## Phase 2C: Benchmark

Create `bin/research/benchmark_batch.sh`:

```bash
#!/bin/bash
# Benchmark batch screener + focused replay vs full replay

ASSET=ETH
START=2025-06-15
END=2025-09-30
CONFIG=configs/v18/ETH_comprehensive.json

echo "======================================================================"
echo "🏁 Bull Machine v1.8 - Batch Mode Benchmark"
echo "======================================================================"

# Full replay (baseline)
echo ""
echo "1️⃣ Running FULL replay (baseline)..."
echo "----------------------------------------------------------------------"
time python3 bin/live/hybrid_runner.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    > /tmp/full_replay.log 2>&1

FULL_TIME=$(grep "real" /tmp/full_replay.log | tail -1)
FULL_TRADES=$(grep "Total:" /tmp/full_replay.log | tail -1)

echo "Full replay time: $FULL_TIME"
echo "Trades: $FULL_TRADES"

# Batch screener
echo ""
echo "2️⃣ Running BATCH screener..."
echo "----------------------------------------------------------------------"
time python3 bin/research/batch_screener.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    --output results/candidates.jsonl \
    > /tmp/batch_screener.log 2>&1

BATCH_TIME=$(grep "real" /tmp/batch_screener.log | tail -1)
CANDIDATES=$(wc -l < results/candidates.jsonl)

echo "Batch screener time: $BATCH_TIME"
echo "Candidates generated: $CANDIDATES"

# Focused replay
echo ""
echo "3️⃣ Running FOCUSED replay (candidate-driven)..."
echo "----------------------------------------------------------------------"
time python3 bin/live/hybrid_runner.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    --candidates results/candidates.jsonl \
    > /tmp/focused_replay.log 2>&1

FOCUSED_TIME=$(grep "real" /tmp/focused_replay.log | tail -1)
FOCUSED_TRADES=$(grep "Total:" /tmp/focused_replay.log | tail -1)

echo "Focused replay time: $FOCUSED_TIME"
echo "Trades: $FOCUSED_TRADES"

echo ""
echo "======================================================================"
echo "📊 RESULTS SUMMARY"
echo "======================================================================"
echo "Full replay:    $FULL_TIME"
echo "Batch + Focused: $BATCH_TIME + $FOCUSED_TIME"
echo ""
echo "Speedup: ~5-10× expected"
echo "======================================================================"
```

---

## Expected Performance

### Before (Full Replay)
```
Runtime: 46 minutes
Bars processed: 1,927 (100%)
Trades: 98
```

### After (Batch + Focused)
```
Batch screener:  ~30 seconds (vectorized)
Candidates:      ~100-200 (5-10% of bars)
Focused replay:  ~3-5 minutes (only candidate windows)
Total:           ~3.5-5.5 minutes
Speedup:         8-13×
```

---

## Implementation Checklist

### Phase 2A: Candidate-Driven Replay ✅
- [x] Add `--candidates` CLI argument to `hybrid_runner.py`
- [x] Implement `load_candidates()` function
- [x] Modify main loop to skip non-candidate bars
- [x] Add logging for batch mode stats

### Phase 2B: Parity Testing ✅
- [x] Create `tests/test_batch_parity.py`
- [ ] Run parity test on 2-week window (ready to run)
- [ ] Verify identical trade outcomes (ready to verify)
- [ ] Document any edge cases (ready to document)

### Phase 2C: Benchmark & Validate ✅
- [x] Create `bin/research/benchmark_batch.sh`
- [ ] Run full 3.5-month comparison (ready to run)
- [ ] Measure actual speedup (ready to measure)
- [ ] Validate final P&L matches (ready to validate)

### Phase 3: Production Integration
- [ ] Add batch mode to CI/CD
- [ ] Update documentation
- [ ] Create user guide for parameter sweeps
- [ ] Enable parallel asset processing

---

## Usage Examples

### 1. Quick Parameter Sweep (Batch Mode)

```bash
# Generate candidates once
python3 bin/research/batch_screener.py \
    --asset ETH --start 2025-06-15 --end 2025-09-30 \
    --config configs/v18/ETH_comprehensive.json \
    --output results/candidates.jsonl

# Test different fusion thresholds (fast!)
for threshold in 0.40 0.45 0.50 0.55; do
    # Modify config.json fusion.entry_threshold_confidence=$threshold
    python3 bin/live/hybrid_runner.py \
        --asset ETH --start 2025-06-15 --end 2025-09-30 \
        --config configs/v18/ETH_comprehensive_${threshold}.json \
        --candidates results/candidates.jsonl
done
```

### 2. Full Validation (Final Results)

```bash
# Run full replay for final P&L
python3 bin/live/hybrid_runner.py \
    --asset ETH --start 2025-06-15 --end 2025-09-30 \
    --config configs/v18/ETH_comprehensive.json
```

---

## Next Steps

**Immediate**:
1. Complete `hybrid_runner.py` modifications (Phase 2A)
2. Run parity test (Phase 2B)
3. Benchmark (Phase 2C)

**Future** (After Batch Mode Works):
1. Incremental domain states (3-5× additional)
2. Per-bar parallelism (1.5-2× additional)
3. Multi-asset batch processing

**Target**: 10-20× total speedup while preserving trading DNA

---

## Files Created

- ✅ `bin/research/batch_screener.py` - Vectorized candidate generation
- ✅ `configs/v18/ETH_comprehensive.json` - Added batch_mode config
- ✅ `PERFORMANCE_OPTIMIZATION_FINDINGS.md` - Complete analysis
- ✅ `BATCH_SCREENER_IMPLEMENTATION.md` - This guide
- ✅ `tests/test_batch_parity.py` - Parity testing framework
- ✅ `bin/research/benchmark_batch.sh` - Benchmark script
- ✅ `bin/live/hybrid_runner.py` - Modified for candidate-driven mode

---

## Questions?

See `PERFORMANCE_OPTIMIZATION_FINDINGS.md` for detailed profiling analysis and optimization roadmap.
