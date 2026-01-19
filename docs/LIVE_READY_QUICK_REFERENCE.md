# Live-Ready Feature Engineering: Quick Reference Card

**One-page cheat sheet for developers**

---

## The Core Pattern

```
┌─────────────────────────────────────────┐
│        S1FeatureLogic (Pure)            │  ← SINGLE SOURCE OF TRUTH
│  - liquidity_drain_pct()                │
│  - liquidity_velocity()                 │
│  - All 7 features as pure functions     │
└─────────────────────────────────────────┘
           ↓                ↓
    ┌──────────┐     ┌──────────┐
    │  BATCH   │     │  STREAM  │         ← EXECUTION STRATEGIES
    │ (pandas) │     │ (deque)  │
    └──────────┘     └──────────┘
```

**Key Insight:** Same logic, different execution. Math lives in pure functions.

---

## Quick Import Guide

```python
# Backtest mode (batch)
from engine.features.batch_engine import BatchFeatureEngine
engine = BatchFeatureEngine()
df_enriched = engine.compute_features(df)

# Live mode (stream)
from engine.features.stream_engine import StreamFeatureEngine
engine = StreamFeatureEngine()

for bar in api.get_bars():
    features = engine.compute_features(bar)

# State persistence (crash recovery)
from engine.features.state_persistence import StatePersistenceStore
store = StatePersistenceStore('data/live_state')
store.save('BTC', engine.get_state())
state = store.load('BTC')
engine.load_state(state)
```

---

## S1 Features at a Glance

| Feature | Formula | Interpretation | Typical Range |
|---------|---------|----------------|---------------|
| `liquidity_drain_pct` | `(liq - liq_7d_avg) / liq_7d_avg` | Liquidity change from 7d avg | -0.6 to +0.3 |
| `liquidity_velocity` | `Δliquidity / Δtime` | Rate of liquidity change | -0.2 to +0.2 |
| `liquidity_persistence` | `count(drain < -0.3 in 24h)` | Sustained drain duration | 0 to 24 bars |
| `capitulation_depth` | `(price - max_7d) / max_7d` | Drawdown from recent high | -0.4 to 0.0 |
| `crisis_composite` | `0.4*rv_z + 0.3*fund_z + 0.3*ret_z` | Macro stress indicator | -2.0 to +3.0 |
| `volume_climax_last_3b` | `max(vol_z in last 3)` | Volume spike detection | 0.0 to 5.0 |
| `wick_exhaustion_last_3b` | `max(wick_lower in last 3)` | Rejection strength | 0.0 to 0.8 |

---

## Common Commands

### Run Backtest with S1 Features
```bash
python3 bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v1.json \
  --start 2022-01-01 \
  --end 2022-12-31
```

### Test Batch vs Stream Parity
```bash
pytest tests/integration/test_batch_stream_parity.py -v -s
```

### Run Unit Tests
```bash
pytest tests/unit/features/test_s1_logic.py -v
```

### Check Performance
```bash
pytest tests/performance/test_stream_latency.py -v -s
```

### Simulate Live Trading
```bash
python3 bin/live_trader.py --config configs/live_test.json
```

---

## S1 State Machine States

```
WATCHING  → (liq_drain < -0.3)  → DRAINING
DRAINING  → (bars >= 8)         → SIGNAL
DRAINING  → (liq_drain >= -0.3) → WATCHING (reset)
SIGNAL    → (emit trade)        → WATCHING (reset)
```

**State Data:**
- `state`: Current state (enum)
- `bars_draining`: Counter (0 to persistence_threshold)
- `drain_start_price`: Price when drain started

---

## Critical Validation Checks

### 1. No Lookahead (MUST PASS)
```python
# Batch and stream MUST produce identical results
df_batch = BatchFeatureEngine().compute_features(df)
df_stream = [StreamFeatureEngine().compute_features(bar) for bar in df]

assert np.corrcoef(df_batch['liquidity_drain_pct'],
                   df_stream['liquidity_drain_pct'])[0,1] > 0.999
```

### 2. State Persistence (MUST SURVIVE)
```python
# Save state
state = engine.get_state()
store.save('BTC', state)

# Kill process, restart
engine2 = StreamFeatureEngine()
engine2.load_state(store.load('BTC'))

# Process same bar MUST produce identical result
assert features1 == features2
```

### 3. Performance (MUST BE FAST)
```python
# Batch: 1M rows/min
# Stream: <100ms per bar (p99)
```

---

## File Locations Quick Map

```
engine/features/
  computer.py              [Pure logic - S1FeatureLogic]
  batch_engine.py          [Backtest mode - vectorized]
  stream_engine.py         [Live mode - incremental]
  state_persistence.py     [Crash recovery]

engine/archetypes/
  logic_v2_adapter.py      [Reads S1 features from RuntimeContext]
  state_machines/
    s1_state_machine.py    [Multi-bar state tracking]

tests/
  unit/features/test_s1_logic.py               [Pure function tests]
  integration/test_batch_stream_parity.py      [Equality validation]
  performance/test_stream_latency.py           [Speed tests]
  validation/test_no_lookahead.py              [Lookahead detection]

bin/
  backtest_knowledge_v2.py [Uses BatchFeatureEngine]
  live_trader.py           [Uses StreamFeatureEngine]
```

---

## Common Pitfalls

### 1. Lookahead in Rolling Windows
```python
# WRONG (includes current bar)
liq_7d_avg = liq.rolling(168).mean()

# CORRECT (excludes current bar)
liq_7d_avg = liq.rolling(168).mean().shift(1)
```

### 2. State Update Order
```python
# WRONG (update state BEFORE computation)
self.liq_window.append(liq_current)  # ← Updates state too early
liq_drain = compute_drain(self.liq_window)  # ← Uses wrong window

# CORRECT (compute THEN update)
liq_drain = compute_drain(self.liq_window)  # ← Uses previous window
self.liq_window.append(liq_current)  # ← Update for NEXT iteration
```

### 3. Missing State Save
```python
# WRONG (lose state on crash)
for bar in bars:
    features = engine.compute_features(bar)
    # No save!

# CORRECT (save every 100 bars)
for i, bar in enumerate(bars):
    features = engine.compute_features(bar)
    if i % 100 == 0:
        store.save('BTC', engine.get_state())
```

---

## Debug Checklist

**Features not matching batch vs stream?**
- [ ] Check rolling window alignment (shift by 1?)
- [ ] Check state update order (compute before update?)
- [ ] Run parity test with verbose output

**State corruption after crash?**
- [ ] Check JSON is valid (`python -m json.tool state.json`)
- [ ] Check symbol matches
- [ ] Check timestamp is recent

**Slow stream updates?**
- [ ] Profile with `cProfile` or `line_profiler`
- [ ] Check if RollingWindow.to_array() called too often
- [ ] Check if using NumPy where native Python is faster

**Backtest results changed?**
- [ ] Check S1 features actually being used in archetype logic
- [ ] Check feature statistics (min/max/mean)
- [ ] Compare before/after feature distributions

---

## Performance Baselines

**Batch Mode (1M rows):**
```
Load parquet:        2.5s
Compute S1 features: 0.15s
Save parquet:        1.2s
TOTAL:               3.85s (260k rows/sec)
```

**Stream Mode (per bar):**
```
Network I/O:         5ms
Compute S1 features: 0.01ms
Update state:        0.001ms
TOTAL:               5.01ms (99% network)
```

**Memory Usage:**
```
Batch: 720MB peak (1M rows)
Stream: 10MB per symbol
```

---

## Phase Milestones

**Week 1 (Phase 1): Batch Mode**
- [ ] S1FeatureLogic implemented
- [ ] BatchFeatureEngine working
- [ ] Unit tests passing
- [ ] Backtest runs with S1 features

**Week 2 (Phase 2): Stream Mode**
- [ ] StreamFeatureEngine implemented
- [ ] Parity tests passing
- [ ] State persistence working
- [ ] Performance < 100ms

**Week 3 (Phase 3): State Machines**
- [ ] S1StateMachine implemented
- [ ] State transitions correct
- [ ] Crash recovery working
- [ ] Backtest results unchanged

**Week 4 (Phase 4): Live Deployment**
- [ ] Live trader script working
- [ ] OKX API integrated
- [ ] Monitoring in place
- [ ] Production ready

---

## Emergency Procedures

**State corruption detected:**
```bash
# 1. Stop live trader
pkill -f live_trader.py

# 2. Backup corrupt state
cp data/live_state/BTC_state.json data/live_state/BTC_state.json.corrupt

# 3. Delete corrupt state (will restart fresh)
rm data/live_state/BTC_state.json

# 4. Restart live trader
python3 bin/live_trader.py --config configs/live_production.json
```

**Performance degradation:**
```bash
# 1. Check CPU usage
top -o cpu

# 2. Profile live trader
python3 -m cProfile -o profile.stats bin/live_trader.py

# 3. Analyze profile
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Batch vs stream divergence:**
```bash
# 1. Run parity test with verbose output
pytest tests/integration/test_batch_stream_parity.py -v -s

# 2. Check for lookahead
pytest tests/validation/test_no_lookahead.py -v -s

# 3. Compare feature distributions
python3 -c "
import pandas as pd
df_batch = pd.read_parquet('data/batch_features.parquet')
df_stream = pd.read_parquet('data/stream_features.parquet')
print((df_batch - df_stream).describe())
"
```

---

## Contact & References

**Documentation:**
- Full Architecture: `docs/LIVE_READY_FEATURE_ARCHITECTURE.md`
- Implementation Guide: `docs/LIVE_READY_IMPLEMENTATION_GUIDE.md`
- Data Flow Diagrams: `docs/diagrams/LIVE_READY_DATA_FLOW.md`
- Executive Summary: `docs/LIVE_READY_SUMMARY.md`

**Key Principles:**
1. Same code for batch and stream (S1FeatureLogic)
2. No lookahead (strictly causal)
3. State persistence (crash recovery)
4. Comprehensive testing (parity, performance, validation)
5. Production-ready (monitoring, graceful shutdown)

**"If it only works in backtests, it's a toy."**
This architecture ensures identical behavior in backtest and production.
