# Bull Machine v1.7.3 - Next Steps for Profitability

## Current Status (2025-10-07)

### ✅ What's Working
- **v1.7.3 merged to main** - Live feeds + macro context integration validated
- **Test suite stable** - 318 passing, 45 xfailed (documented)
- **Determinism confirmed** - Reproducible results
- **Data pipeline operational** - CSV replay, MTF alignment, macro integration

### ⚠️ What's Not Working  
- **Paper trading generates 0 trades** - Fusion engine with mock modules produces no signals
- **Simple backtest marginally unprofitable** - BTC 1-year: -0.76%, 3 trades, 33.3% win rate

## BTC Backtest Results (Oct 2024 - Oct 2025)

**Configuration:**
- Entry threshold: 0.45 confidence
- Risk: 5% per trade
- Stop loss: 8% (fixed)
- Take profit: 15% (fixed)
- ADX filter: >20 (trending markets only)

**Results:**
- **Starting**: $10,000
- **Final**: $9,924  
- **Return**: -0.76%
- **Trades**: 3 (1 win, 2 losses)
- **Win Rate**: 33.3%
- **Profit Factor**: 0.07
- **Best**: +1.35% (SHORT 2025-09-20)
- **Worst**: -8.00% (2 stop losses)

## Key Findings

### 1. ADX Filter Working
- All 3 trades occurred in trending markets (ADX >20)
- Achieved first winning trade vs earlier 0% win rate
- Filter prevents choppy market entries

### 2. Entry Timing Issue
- Both losses hit stop immediately
- Need pullback entry logic to avoid buying extremes
- Current SMA crossover too lagging

### 3. Risk/Reward Imbalance
- Avg win: 1.35% vs Avg loss: 8.00%
- Need 85%+ win rate OR 6:1 R:R to be profitable
- Fixed % stops don't respect market structure

### 4. Fusion Engine Limitation
- Paper trading (fusion engine) = 0 signals
- Simple backtest (price action) = 3 signals
- **Action**: Need to implement actual domain modules OR use simple backtest approach

## Immediate Action Plan

### Phase 1: Fix Signal Generation (Priority 1)

**Option A - Quick** (Recommended for testing):
```bash
# Use simple backtest logic in paper trading
# Copy calc_adx, generate_signal from btc_simple_backtest.py
# to bin/live/paper_trading.py
```

**Option B - Proper** (For production):
```bash
# Implement actual domain modules (not mocks)
# Wyckoff, liquidity, momentum detection
# ~40-80 hours of work
```

### Phase 2: Add Profitability Filters (Priority 2)

**1. Pullback Entry Timing**
```python
# Only enter after price mean-reverts 0.3-0.8×ATR toward SMA
entries.pullback_atr_min = 0.3
entries.pullback_atr_max = 0.8
```

**2. Structure-Based Stops**
```python
# Stop at last swing ± ATR instead of fixed %
exits.sl_mode = "structure_atr"
exits.sl_atr_mult = 1.2
```

**3. Scale-Out Strategy**
```python
# Take 50% at 1R, trail remainder
exits.tp1_r = 1.0
exits.tp1_pct = 0.5
exits.trail_atr = 1.0
```

**4. Momentum Exhaustion Filter**
```python
# Avoid catching reversals
# Check RSI divergence, volume decline
entries.check_momentum_exhaustion = true
```

### Phase 3: Extended Testing (Priority 3)

**Multi-Asset Validation:**
```bash
# Run 1-year backtests
python3 scripts/backtests/v173/btc_simple_backtest.py  # BTC
python3 scripts/backtests/v173/eth_simple_backtest.py  # ETH  
python3 scripts/backtests/v173/sol_simple_backtest.py  # SOL
```

**Success Criteria:**
- ✅ 100+ trades per asset
- ✅ Profit Factor >1.2
- ✅ Win Rate >50% OR R:R >3:1
- ✅ Max DD <15%
- ✅ Sharpe Ratio >1.0

## Timeline Estimate

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Copy simple backtest logic to paper trading | 2 hours | P0 |
| 1 | Test paper trading generates signals | 30 min | P0 |
| 2 | Add pullback entry timing | 3 hours | P1 |
| 2 | Implement structure-based stops | 4 hours | P1 |
| 2 | Add scale-out strategy | 2 hours | P1 |
| 2 | Add momentum exhaustion filter | 4 hours | P1 |
| 3 | Multi-asset 1-year backtests | 2 hours | P2 |
| 3 | 6-month paper trading validation | ongoing | P2 |

**Total Dev Time**: ~20 hours
**Total Validation Time**: 6 months

## Decision Point

**Do NOT go live until:**
1. ✅ Paper trading generates signals (currently 0)
2. ✅ 100+ backtest trades (currently 3)
3. ✅ Profit Factor >1.2 (currently 0.07)
4. ✅ 6 months paper trading with positive expectancy
5. ✅ Multi-asset validation (BTC/ETH/SOL all profitable)

**Current Recommendation**: Implement Phase 1 + Phase 2, then re-evaluate.
