# Bull Machine Implementation Roadmap

**Current State**: v1.7.3 merged to main - Live feeds + macro context integration validated
**Next Milestone**: v1.8 Hybrid (fast signals + periodic fusion) → v1.9 Numba (full engines optimized)

---

## Timeline Overview

```
v1.7.3 (DONE) ────→ v1.8-alpha ────→ v1.8-beta ────→ v1.8-rc ────→ v1.8.0 ────→ v1.9.0
  Oct 7, 2025     Oct 8 (+8h)     Oct 9 (+6h)     Oct 10 (+10h)   Oct 11      Nov 1 (+17h)
                  Infrastructure   Multi-asset    Profitability   Paper 6mo   Full engines
```

**Total Development Time**:
- v1.8: ~25-35 hours
- v1.9: ~17 hours
- **Combined**: ~42-52 hours (can be completed in 2-3 weeks part-time)

---

## v1.8: Hybrid Approach (Fast Signals + Periodic Fusion)

### Branch Strategy

```bash
# Create v1.8 branch from current state
git checkout feature/v1.7.3-live
git checkout -b feature/v1.8-hybrid
git push -u origin feature/v1.8-hybrid
```

### Phase 1: Infrastructure (8-12 hours)

**Goal**: Wire hybrid runner with 3 execution modes

**Tasks**:
1. ✅ Create `bin/live/hybrid_runner.py` (DONE)
2. ✅ Create `configs/v18/BTC_conservative.json` (DONE)
3. ⏳ Implement `_run_full_fusion()` method
4. ⏳ Add determinism test (`tests/test_determinism.py`)
5. ⏳ Add safety guards (loss streak, ATR throttle)
6. ⏳ Run baseline test

**Baseline Test**:
```bash
python bin/live/hybrid_runner.py \
  --asset BTC \
  --start 2025-08-01 \
  --end 2025-10-01 \
  --config configs/v18/BTC_conservative.json

# Relaxed targets (Phase 1):
# - PF > 0.8 (breakeven)
# - Trades ≥ 10
# - DD ≤ 15%
# - Determinism: 2 runs identical
```

**Deliverables**:
- [ ] `bin/live/hybrid_runner.py` (functional)
- [ ] `configs/v18/` (3 modes: advisory, prefilter, confirm)
- [ ] `tests/test_determinism.py`
- [ ] Baseline results in `results/v18/baseline/`

### Phase 1.5: Multi-Asset Validation (4-6 hours)

**Goal**: Tune config to hit Phase 1 targets, validate across assets

**Tasks**:
1. Tune BTC config (PF > 1.2, ≥30 trades, DD ≤10%)
2. Create ETH config
3. Create SOL config
4. Run multi-asset validation
5. Analyze disagreement patterns (fast vs fusion)

**Multi-Asset Test**:
```bash
# Run all assets in parallel
for asset in BTC ETH SOL; do
  python bin/live/hybrid_runner.py \
    --asset $asset \
    --start 2025-01-01 \
    --end 2025-10-01 \
    --config configs/v18/${asset}_conservative.json &
done
wait

# Aggregate results
python scripts/analysis/aggregate_results.py results/v18/
```

**Pass Criteria**:
- BTC: PF > 1.2, ≥30 trades
- ETH: PF > 1.2, ≥25 trades
- SOL: PF > 1.0, ≥20 trades (more volatile)
- All: DD ≤10%

**Deliverables**:
- [ ] Tuned configs for BTC/ETH/SOL
- [ ] Multi-asset results report
- [ ] Fast vs fusion agreement analysis

### Phase 2: Profitability Filters (10-15 hours)

**Goal**: Add filters to achieve PF ≥1.5

**Tasks**:

#### 2a. Pullback Timing (3 hours)
**File**: `bin/live/fast_signals.py`

```python
def check_pullback_timing(close, ma20, config):
    """Don't chase extremes."""
    pullback_bars = config['entries']['pullback_bars']
    pullback_pct = config['entries']['pullback_pct']

    highest_recent = close.tail(pullback_bars).max()
    lowest_recent = close.tail(pullback_bars).min()

    if close.iloc[-1] > ma20:  # Uptrend
        return close.iloc[-1] <= highest_recent * (1 - pullback_pct)
    else:  # Downtrend
        return close.iloc[-1] >= lowest_recent * (1 + pullback_pct)
```

#### 2b. Structure-Based Stops (4 hours)
**File**: `bin/live/paper_trading.py`

```python
def calculate_structure_stop(df, side, config):
    """Use swing highs/lows for stops."""
    atr_k = config['exits']['atr_k']
    lookback = 20

    if side == 'long':
        swing_low = df['Low'].tail(lookback).min()
        atr = calculate_atr(df, period=14)
        stop = swing_low - atr * atr_k
    else:  # short
        swing_high = df['High'].tail(lookback).max()
        atr = calculate_atr(df, period=14)
        stop = swing_high + atr * atr_k

    return stop
```

#### 2c. Scale-Out + Trail (2 hours)
**File**: `bin/live/paper_trading.py`

```python
def manage_position(position, current_price, config):
    """Scale out at 1R, trail remainder."""
    entry = position['entry_price']
    risk = abs(position['stop_loss'] - entry)

    # Take 50% at 1R
    if not position['tp1_hit']:
        target_1r = entry + risk if position['side'] == 'long' else entry - risk

        if (position['side'] == 'long' and current_price >= target_1r) or \
           (position['side'] == 'short' and current_price <= target_1r):
            close_partial(position, pct=0.5)
            position['tp1_hit'] = True
            position['stop_loss'] = entry  # Move to breakeven

    # Trail remainder
    if position['tp1_hit']:
        atr = calculate_atr(df, period=14)
        trail_distance = atr * config['exits']['trail_atr_k']

        if position['side'] == 'long':
            new_stop = current_price - trail_distance
            position['stop_loss'] = max(position['stop_loss'], new_stop)
        else:
            new_stop = current_price + trail_distance
            position['stop_loss'] = min(position['stop_loss'], new_stop)
```

#### 2d. Momentum Exhaustion Filter (4 hours)
**File**: `bin/live/fast_signals.py`

```python
def check_momentum_exhaustion(df_1h, df_4h):
    """Avoid catching reversals at extremes."""
    # RSI divergence
    price_high_recent = df_4h['Close'].tail(10).max()
    price_high_prev = df_4h['Close'].tail(20).head(10).max()

    rsi = calculate_rsi(df_4h, period=14)
    rsi_at_recent = rsi.iloc[-1]
    rsi_at_prev = rsi.tail(20).head(10).max()

    # Bearish divergence: price higher but RSI lower
    if price_high_recent > price_high_prev and rsi_at_recent < rsi_at_prev - 5:
        return True  # Exhausted, avoid long

    # Volume decline
    vol_recent = df_4h['Volume'].tail(5).mean()
    vol_prev = df_4h['Volume'].tail(15).head(10).mean()

    if vol_recent < vol_prev * 0.7:
        return True  # Momentum fading

    return False
```

**Re-Test**:
```bash
# Run with all filters enabled
python bin/live/hybrid_runner.py \
  --asset BTC \
  --start 2024-10-01 \
  --end 2025-10-01 \
  --config configs/v18/BTC_aggressive.json

# Target: PF ≥1.5, ≥100 trades, DD ≤10%
```

**Deliverables**:
- [ ] Pullback filter implemented
- [ ] Structure-based stops implemented
- [ ] Scale-out + trail implemented
- [ ] Momentum exhaustion filter implemented
- [ ] Phase 2 results (all assets)
- [ ] Performance comparison report

### Phase 3: Final Validation (2 hours)

**Tasks**:
1. Run determinism test (2 runs, diff outputs)
2. Validate macro veto rate (5-15%)
3. Generate summary report
4. Tag release `v1.8.0`

**Deliverables**:
- [ ] Determinism proof (diff = 0)
- [ ] Macro veto analysis
- [ ] v1.8.0 release notes
- [ ] Merge to main

**Merge Command**:
```bash
git checkout feature/v1.8-hybrid
git tag v1.8.0
git push origin v1.8.0

# PR to main
gh pr create --title "feat(v1.8.0): Hybrid approach with fast signals + periodic fusion" \
  --body "$(cat docs/v18_RELEASE_NOTES.md)"
```

---

## v1.8.0 → 6-Month Paper Trading

**Duration**: 6 months (as per NEXT_STEPS.md)

**Goal**: Validate positive expectancy in live market conditions

**Setup**:
```bash
# Deploy to paper trading server
python bin/live/hybrid_runner.py \
  --asset BTC \
  --mode live \
  --config configs/v18/BTC_conservative.json \
  --start-date today

# Monitor daily
tail -f results/live_signals_BTC_*.jsonl
```

**Monthly Review**:
- [ ] Month 1: PF > 1.0, no catastrophic losses
- [ ] Month 2: PF > 1.2, win rate stabilizing
- [ ] Month 3: PF > 1.3, drawdown controlled
- [ ] Month 4: PF > 1.4, consistent profitability
- [ ] Month 5: PF > 1.5, ready for small live capital
- [ ] Month 6: PF > 1.5, full validation complete

**Go/No-Go Decision** (Month 6):
- ✅ GO: PF ≥1.5, DD ≤10%, 6 consecutive profitable months → Deploy live with $1K
- ❌ NO-GO: Any metric fails → Back to Phase 2, add more filters

---

## v1.9: Numba Optimization (Full Engines)

### Branch Strategy

```bash
# Create v1.9 branch from main
git checkout main
git pull origin main  # After v1.8.0 merged
git checkout -b feature/v1.9-numba
git push -u origin feature/v1.9-numba
```

### Phase 1: Profiling (2 hours)

**Goal**: Identify exact bottlenecks

**Tasks**:
```bash
# Profile current engines
python -m cProfile -o profile.stats bin/live/hybrid_runner.py \
  --asset BTC --start 2025-09-01 --end 2025-09-30 \
  --config configs/v18/BTC_conservative.json

# Analyze
pip install snakeviz
snakeviz profile.stats
```

**Expected Hotspots**:
1. `calc_adx` - 15% of time
2. `detect_wyckoff_phase` - 25%
3. `HOBDetector.detect_hob` - 30%
4. `SMCEngine.analyze` - 20%

**Deliverables**:
- [ ] Profile report (`docs/v19/profiling_report.md`)
- [ ] Bottleneck priority list

### Phase 2: Core Indicators (4 hours)

**Goal**: Numba-optimize ADX, RSI, SMA

**Tasks**:
1. Create `engine/momentum/momentum_numba.py`
2. Implement `calc_adx_numba()`
3. Implement `calc_rsi_numba()`
4. Implement `calc_sma_numba()`
5. Add fallback logic in `engine/momentum/momentum.py`
6. Test accuracy (Numba output == pandas output)

**Test**:
```bash
pytest tests/test_numba_accuracy.py -v

# Expected:
# test_adx_accuracy: PASSED (max error < 0.001)
# test_rsi_accuracy: PASSED (max error < 0.001)
# test_sma_accuracy: PASSED (max error < 0.001)
```

**Deliverables**:
- [ ] `engine/momentum/momentum_numba.py`
- [ ] Accuracy tests passing
- [ ] Performance tests (`<1ms per calculation`)

### Phase 3: Domain Engines (6 hours)

**Goal**: Numba-optimize Wyckoff, HOB, SMC

**Tasks**:
1. Create `engine/wyckoff/wyckoff_numba.py`
2. Create `engine/liquidity/hob_numba.py`
3. Create `engine/smc/smc_numba.py`
4. Refactor pandas → numpy in core loops
5. Add `@jit(nopython=True)` decorators
6. Test accuracy

**Performance Target**:
- Wyckoff: <2ms (vs 80ms) = 40x faster
- HOB: <5ms (vs 120ms) = 24x faster
- SMC: <8ms (vs 150ms) = 19x faster

**Deliverables**:
- [ ] Numba-optimized domain engines
- [ ] Accuracy tests passing
- [ ] Performance benchmarks

### Phase 4: Integration (3 hours)

**Goal**: Wire Numba engines into hybrid runner

**Tasks**:
1. Update `hybrid_runner.py` to use Numba engines
2. Add config flag `use_numba: true`
3. Create `configs/v19/BTC_full_engines.json`
4. Run 1-year backtest

**Test**:
```bash
time python bin/live/hybrid_runner.py \
  --asset BTC \
  --start 2024-10-01 \
  --end 2025-10-01 \
  --config configs/v19/BTC_full_engines.json

# Expected: <60 seconds (vs 3+ minutes)
```

**Deliverables**:
- [ ] Numba integration complete
- [ ] 1-year backtest <60s
- [ ] Full engines enabled

### Phase 5: Validation (2 hours)

**Goal**: Verify signal quality with full engines

**Tasks**:
1. Run BTC/ETH/SOL 1-year backtests
2. Compare PF vs v1.8 (should be ≥v1.8)
3. Verify signal count (should be similar)
4. Tag release `v1.9.0`

**Success Criteria**:
- PF ≥1.8 (vs ≥1.5 in v1.8)
- DD ≤2% (vs ≤10% in v1.8)
- Return +3-5% (vs +1-3% in v1.8)

**Deliverables**:
- [ ] v1.9 performance report
- [ ] v1.8 vs v1.9 comparison
- [ ] v1.9.0 release notes
- [ ] Merge to main

---

## Summary

### Key Milestones

| Version | Goal | Timeline | Deliverable |
|---------|------|----------|-------------|
| v1.7.3 | ✅ Live feeds + macro | Oct 7, 2025 | Merged to main |
| v1.8.0-alpha | Infrastructure | Oct 8 (+8h) | Hybrid runner |
| v1.8.0-beta | Multi-asset | Oct 9 (+6h) | BTC/ETH/SOL configs |
| v1.8.0-rc | Profitability | Oct 10 (+10h) | Filters + validation |
| v1.8.0 | Paper trading | Oct 11 | 6-month validation |
| v1.9.0 | Full engines | Nov 1 (+17h) | Numba optimization |

### Decision Points

1. **After v1.8-alpha**: If PF < 0.8 → Revisit signal logic
2. **After v1.8-beta**: If any asset PF < 1.0 → Asset-specific tuning
3. **After v1.8-rc**: If PF < 1.5 → Add more filters
4. **After 6 months**: If PF < 1.5 → NO-GO on live trading
5. **After v1.9**: If performance <60s → Consider Cython

### Risk Mitigation

- **v1.8 fast signals fail**: Fall back to btc_simple_backtest.py logic (proven)
- **v1.8 PF < 1.5**: Extend Phase 2 with more filters (liquidation levels, sentiment)
- **v1.9 Numba too slow**: Use Cython as backup (similar performance)
- **6-month paper fails**: Extend to 12 months, add regime detection

---

## Next Immediate Action

```bash
# Commit current work
git add -A
git commit -m "docs: add v1.8/v1.9 implementation roadmap

- Create hybrid_runner.py with 3 execution modes
- Add v1.8 config template (BTC_conservative.json)
- Document Numba optimization plan (V1.9_NUMBA_OPTIMIZATION.md)
- Add IMPLEMENTATION_ROADMAP.md with detailed timeline
- Branch strategy: v1.8-hybrid, v1.9-numba"

# Push to GitHub
git push origin feature/v1.7.3-live

# Create v1.8 branch
git checkout -b feature/v1.8-hybrid
git push -u origin feature/v1.8-hybrid

# Start Phase 1 work
# ... (implement _run_full_fusion, determinism test, safety guards)
```

**Ready to proceed with Phase 1 implementation!**
