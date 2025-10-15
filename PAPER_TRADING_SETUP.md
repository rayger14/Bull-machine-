# Paper Trading Setup Plan - Full Historical Validation

**Date**: 2025-10-14
**Status**: Data extraction complete, ready for comprehensive backtest
**Goal**: Validate ML stack on 2022-2025 data, extract best configs, launch paper trading

---

## Current Status ✅

### Data Extraction Complete
- **BTC**: 33,166 hourly bars (2022-01-01 to 2025-10-14) via CCXT ✅
- **ETH**: 33,166 hourly bars (2022-01-01 to 2025-10-14) via CCXT ✅
- **SPY**: Available in TradingView exports (BATS_SPY, 60_9f7f8.csv) ✅
- **Macro Data**: VIX, DXY, MOVE, TOTAL/TOTAL2/TOTAL3 integrated ✅

### What We Have
- ✅ Full 3-year historical OHLCV data
- ✅ Real macro indicators with realistic variance
- ✅ ML stack validated on 2024 (+11.2pp improvement)
- ✅ Feature store builder working
- ✅ Optimizer with regime adaptation ready

### What's Blocking Full 3-Year Backtest
- ⚠️ Feature store currently only loads 2024 data (15,550 bars)
- ⚠️ Need to point loader to CCXT data (33,166 bars) instead of TradingView
- ⚠️ OR need complete TradingView exports from 2022-2025

---

## Recommended Path Forward

### Option 1: Use CCXT Data (Fastest)
Modify the feature store builder to load from the CCXT CSV files we just downloaded:

```bash
# Data is ready at:
data/raw/binance/BTCUSD_1h_2022-2025.csv  (33,166 bars)
data/raw/binance/ETHUSD_1h_2022-2025.csv  (33,166 bars)

# Need to update loader in:
engine/io/tradingview_loader.py
```

**Pros**: Data already downloaded, just need to update loader
**Cons**: Need to modify loader to support CSV format
**Time**: ~30 minutes

### Option 2: Use Current 2024 Data for Production (Quickest)
Since we've already validated the ML stack on 2024 with proven +11.2pp improvement:

```bash
# Launch paper trading NOW with validated 2024 configs:
BTC: threshold=0.70, wyckoff=0.25, momentum=0.30
ETH: Similar optimal config
SPY: Run quick optimization on available data
```

**Pros**: Already validated, can start trading immediately
**Cons**: Missing 2022-2023 validation data
**Time**: ~5 minutes to launch

### Option 3: Export Fresh TradingView Data
Export complete 2022-2025 charts from TradingView manually:

**Required exports** (all need 1H resolution, 2022-01-01 to today):
- COINBASE:BTCUSD
- COINBASE:ETHUSD
- BATS:SPY

**Time**: ~15 minutes manual export

---

## Recommendation: Hybrid Approach

**Phase 1: Launch Paper Trading NOW** (5 minutes)
- Use validated 2024 configs
- Start collecting live performance data
- Monitor ML stack in real market conditions

**Phase 2: Complete Historical Validation** (30 minutes)
- Fix CCXT data loading or get fresh TradingView exports
- Run full 3-year backtests (2022-2024)
- Compare across bear market (2022), bull run (2023), chop (2024)
- Refine configs based on multi-year results

**Phase 3: Update Paper Trading** (ongoing)
- Apply refined configs from 3-year validation
- Monitor performance delta
- Document learnings

---

## Paper Trading Infrastructure

### What's Available
```
bin/live/hybrid_runner.py - Live trading engine
configs/v18/BTC_conservative.json - Base config template
```

### What We Need to Create
```
configs/paper_trading/
├── BTC_ML_optimized.json      # Best config from validation
├── ETH_ML_optimized.json      # Best ETH config
├── SPY_ML_optimized.json      # Best SPY config
└── paper_trading_params.json  # Paper trading settings
```

### Configuration Template
```json
{
  "asset": "BTC",
  "exchange": "coinbase",
  "timeframe": "1h",
  "starting_balance": 10000,
  "position_size_pct": 0.02,

  "fusion_threshold": 0.70,
  "wyckoff_weight": 0.25,
  "momentum_weight": 0.30,
  "smc_weight": 0.20,
  "hob_weight": 0.15,
  "temporal_weight": 0.10,

  "regime_adaptation": true,
  "ml_stack_enabled": true,

  "risk_management": {
    "max_position_size": 0.05,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 4.0
  },

  "logging": {
    "log_level": "INFO",
    "trade_log": "logs/paper_trading/BTC_trades.jsonl",
    "performance_log": "logs/paper_trading/BTC_performance.csv"
  }
}
```

---

## Launch Commands

### Quick Start (Use 2024 Validated Configs)

```bash
# BTC Paper Trading
PYTHONHASHSEED=0 python3 bin/live/hybrid_runner.py \
  --asset BTC \
  --mode paper \
  --config configs/paper_trading/BTC_ML_optimized.json \
  --start-date now \
  > logs/paper_btc.log 2>&1 &

# ETH Paper Trading
PYTHONHASHSEED=0 python3 bin/live/hybrid_runner.py \
  --asset ETH \
  --mode paper \
  --config configs/paper_trading/ETH_ML_optimized.json \
  --start-date now \
  > logs/paper_eth.log 2>&1 &

# SPY Paper Trading
PYTHONHASHSEED=0 python3 bin/live/hybrid_runner.py \
  --asset SPY \
  --mode paper \
  --config configs/paper_trading/SPY_ML_optimized.json \
  --start-date now \
  > logs/paper_spy.log 2>&1 &

# Monitor all
tail -f logs/paper_*.log
```

### Full Validation First (Recommended)

```bash
# 1. Run 3-year exhaustive backtests
PYTHONHASHSEED=0 python3 bin/optimize_v19.py \
  --asset BTC --mode exhaustive --regime true \
  --start 2022-01-01 --end 2024-12-31 \
  --workers 8 \
  --output reports/v19/BTC_3year_exhaustive_ML.json

# 2. Extract best config
python3 bin/analyze_optimization.py reports/v19/BTC_3year_exhaustive_ML.json --top 1

# 3. Create paper trading config with best params
# (use output from step 2)

# 4. Launch paper trading with validated config
```

---

## Monitoring & Metrics

### Real-Time Monitoring
```bash
# Watch trade execution
tail -f logs/paper_trading/BTC_trades.jsonl | jq '.'

# Performance dashboard
watch -n 60 'python3 tools/paper_trading_dashboard.py'

# Compare to baseline
python3 tools/compare_paper_vs_backtest.py
```

### Key Metrics to Track
- **Daily PnL** - Track vs $10k starting balance
- **Trade Count** - Should be ~47% lower than baseline
- **Win Rate** - Target 60-65%
- **Profit Factor** - Target >1.0
- **Sharpe Ratio** - Target >0.1
- **Max Drawdown** - Monitor vs backtest expectations
- **Regime Distribution** - Log what regimes are detected
- **ML Decisions** - Track threshold adjustments, risk multipliers

### Success Criteria (30 Days)
- ✅ Positive cumulative PnL
- ✅ Profit factor > 1.0
- ✅ Win rate > 55%
- ✅ Max drawdown < backtest worst case
- ✅ Sharpe ratio > 0
- ✅ Trade selectivity maintained (low trade count)

---

## Risk Management

### Paper Trading Limits
- **Max Position Size**: 5% of balance per trade
- **Daily Loss Limit**: -2% of starting balance (stop trading for day)
- **Weekly Loss Limit**: -5% of starting balance (review strategy)
- **Max Open Positions**: 1 per asset (no pyramiding initially)

### Kill Switches
```bash
# Emergency stop all paper trading
pkill -f "hybrid_runner.py"

# Pause BTC only
kill -STOP $(pgrep -f "hybrid_runner.*BTC")

# Resume BTC
kill -CONT $(pgrep -f "hybrid_runner.*BTC")
```

---

## Next Steps Decision Matrix

| Scenario | Action | Timeline |
|----------|--------|----------|
| **Need results ASAP** | Launch paper trading with 2024 configs NOW | 5 min |
| **Want full validation** | Fix CCXT loader, run 3-year backtests, then launch | 2-3 hours |
| **Conservative approach** | Export fresh TV data, validate 3 years, then launch | 3-4 hours |
| **Hybrid (recommended)** | Launch paper NOW, validate in parallel | 5 min + 3 hours |

---

## Files Created This Session

### Data Tools
- `tools/process_total_marketcap.py` - TOTAL/TOTAL2/TOTAL3 processor
- `tools/merge_total_to_macro.py` - Macro feature merger
- `tools/fetch_macro_yfinance.py` - Yahoo Finance data fetcher
- `tools/get_ccxt_data.py` - CCXT historical data fetcher (used)

### Updated Feature Stores
- `data/macro/BTC_macro_features.parquet` - Real VIX/DXY/MOVE/TOTAL
- `data/macro/ETH_macro_features.parquet` - Real VIX/DXY/MOVE/TOTAL
- `data/features/v18/BTC_1H.parquet` - 15,550 bars (2024 data)
- `data/features/v18/ETH_1H.parquet` - 33,067 bars (mixed data)

### Historical Data Downloaded
- `data/raw/binance/BTCUSD_1h_2022-2025.csv` - 33,166 bars ✅
- `data/raw/binance/ETHUSD_1h_2022-2025.csv` - 33,166 bars ✅

### Reports & Documentation
- `reports/v19/FINAL_2024_ML_VALIDATION.md` - Complete 2024 analysis
- `reports/v19/TOTAL_INTEGRATION_SUMMARY.md` - Data integration details
- `PAPER_TRADING_SETUP.md` - This file

---

## Recommended Immediate Action

**My recommendation**: Launch paper trading NOW with validated 2024 configs while we complete the 3-year validation in parallel.

### Command to Execute Right Now:

```bash
# Create paper trading configs directory
mkdir -p configs/paper_trading
mkdir -p logs/paper_trading

# Create BTC config (using validated 2024 optimal parameters)
cat > configs/paper_trading/BTC_ML_optimized.json <<'EOF'
{
  "asset": "BTC",
  "fusion_threshold": 0.70,
  "wyckoff_weight": 0.25,
  "momentum_weight": 0.30,
  "smc_weight": 0.20,
  "hob_weight": 0.15,
  "temporal_weight": 0.10,
  "regime_adaptation": true,
  "starting_balance": 10000,
  "position_size_pct": 0.02
}
EOF

# Launch BTC paper trading
PYTHONHASHSEED=0 nohup python3 bin/live/hybrid_runner.py \
  --asset BTC \
  --config configs/paper_trading/BTC_ML_optimized.json \
  --start $(date +%Y-%m-%d) \
  > logs/paper_trading/btc.log 2>&1 &

echo "Paper trading launched! Monitor with:"
echo "  tail -f logs/paper_trading/btc.log"
```

**This gets you live data collection starting NOW while we finalize the full historical validation.**

---

**Ready to proceed?** Let me know if you want to:
1. Launch paper trading immediately with current configs
2. Fix the CCXT data loading first
3. Both in parallel (recommended)
