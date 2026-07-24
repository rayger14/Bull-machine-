# Bull Machine v1.8.6 - Temporal Intelligence + Complete v1.8 Stack

[![CI](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml/badge.svg)](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-50%20unit%20tests%20passing-success)](https://github.com/rayger14/Bull-machine-/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%7C%20mypy-blue)](https://github.com/rayger14/Bull-machine-/actions)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://python.org)
[![Version](https://img.shields.io/badge/version-1.8.6-green)](https://github.com/rayger14/Bull-machine-/releases/tag/v1.8.6)
[![Production](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/rayger14/Bull-machine-)

**Institutional-grade algorithmic trading framework** with **Temporal Intelligence**, **True Domain Fusion**, **Smart Exits**, and **Macro Context Integration** supporting **BTC, ETH, SOL, SPY**.

**Branch Protection:** ✅ Required CI checks must pass before merging to main

**Status:** v1.8.6 production-ready ✅ | 50+ unit tests passing ✅ | Ready for live paper trading ✅

---

## 🚀 What's New in v1.8.6: Temporal Intelligence

### 🎯 Complete v1.8 Stack
- **v1.8.1**: True Fusion - Real domain engines (Wyckoff, SMC, HOB, Momentum)
- **v1.8.2**: Smart Exits - Regime-adaptive stops, liquidity trap protection
- **v1.8.3**: Dynamic Position Sizing - Fusion score × ADX adaptive sizing
- **v1.8.4**: Batch Mode - Window merging with auto-fallback for performance
- **v1.8.5**: Enhanced Logic - Negative Fibs, Fourier filter, Event tagging, Narrative detection
- **v1.8.6**: Temporal Intelligence - Gann cycles, Square of 9, LPPLS blowoff detection

### 🔮 Temporal Intelligence Module
- **Gann Cycles** - 30/60/90 day ACF vibrations for cycle detection
- **Square of 9** - Sacred geometry price level proximity scoring
- **Gann Angles** - 1×1, 2×1 support/resistance analysis
- **Thermo-floor** - Mining cost floor calculation (energy + difficulty)
- **Log Premium** - Difficulty-based time multiplier for value assessment
- **Logistic Bid** - Institutional re-accumulation probability
- **LPPLS Blowoff** - Power law bubble detection with veto capability

### 📊 Production Validation
- **Unit Tests**: 50+ tests (all passing) - Temporal (25), v1.8.5 modules (25)
- **Full Year 2025**: 7 trades total, +1.95% combined, 57% WR, 5.8% max DD
- **Integration**: All domain engines working together (12,837 lines of production code)
- **Configs**: 3-tier system (Conservative 0.70, Moderate 0.60, Fast 0.60+opts)

*"Temporal cycles meet institutional domain analysis."* - v1.8.6 Philosophy


## 🖥️ Frontend Strategy Workbench

A lightweight UI is available to visualize the fusion decision flow (domain consensus, macro vetoes, threshold gating), now with config snapshot loading, presets, and decision trace output:

```bash
python -m http.server 8080
# open http://localhost:8080/ui/trader_dashboard.html
```

See `docs/frontend/FRONTEND_UI_GUIDE.md` for details.

---

## 🚀 What's New in v1.7.3: Live Feeds + Macro Context Integration

### 🎯 Live Trading Pipeline
- **Mock Feed Runner** - CSV replay with MTF alignment for validation (168 ETH, 97 SOL, 263 BTC signals)
- **Paper Trading** - Realistic execution simulation with P&L tracking (30-day ETH: 697 bars, $10K→$10K)
- **Shadow Mode** - Log-only signal tracking for live monitoring without execution
- **Health Monitoring** - Macro veto rate (5-15%), SMC 2+ hit (≥30%), continuous validation

### 🏥 Macro Context System
- **VIX Hysteresis Guards** - On=22.0, Off=18.0 with proper state memory to prevent threshold thrashing
- **Macro Veto Integration** - Suppression flag with veto_strength calculation (85% threshold)
- **Fire Rate Monitoring** - Rolling window (100 bars) veto engagement tracking
- **Greenlight Signals** - Positive macro confirmation (VIX calm <18, DXY bullish >100)
- **Stock Market Context** - SPY/QQQ support for equity correlation analysis

### ✅ Production Validation
- **Test Suite**: 318 passed, 0 failed, 0 errors (45 xfailed with documentation)
- **Determinism**: 2 independent runs identical (48 signals each)
- **Backtest Parity**: 8 trades, -0.4% return, 62.5% win rate
- **Mock Feeds**: All assets validated (ETH/SOL/BTC)
- **Paper Trading**: 30-day clean execution

### 🔧 Critical Fixes
- Fixed VIX parameter passing to mtf_confluence() (vix_now + vix_prev)
- Added VIXHysteresis.previous_value tracking for proper hysteresis memory
- Fixed OHLCV column case sensitivity (Close vs close) throughout MTF engine
- Added None/NaN handling for VIX values with safe defaults
- Fixed MacroPulse fire_rate_stats initialization (TypeError)
- Fixed CVD dict/float type mismatch in orderflow

*"Live validation meets institutional macro analysis."* - v1.7.3 Philosophy

---

## 🚀 What's New in v1.7.2: Institutional Repository + Asset Adapter Architecture

### 🏛️ Professional Repository Organization
- **Clean Directory Structure** - Organized `/bin/`, `/scripts/research/`, `/tests/`, `/docs/` for institutional standards
- **Production Entry Points** - 5 dedicated executables in `/bin/` for professional deployment
- **Comprehensive Documentation** - Institutional-grade documentation in `/docs/` with detailed structure guide
- **Test Consolidation** - All tests organized in `/tests/` with robust validation framework
- **Root Directory Cleanup** - Reduced from 45 to 3 Python files for professional appearance

### 🌐 Universal Asset Adapter Architecture
- **Multi-Asset Support** - ETH, SOL, XRP, BTC, SPY with adaptive configuration system
- **Asset-Specific Profiling** - Automated parameter optimization for each asset class
- **Universal Backtesting** - Consistent framework across all supported assets
- **Adaptive Configuration** - Dynamic parameter adjustment based on asset characteristics
- **Cross-Asset Validation** - Comprehensive testing across multiple asset classes

### 🚀 Production Features
- **Professional CLI Interfaces** - 5 production-ready command-line tools
- **Institutional Testing Suite** - Comprehensive validation with robust error handling
- **Multi-Asset Backtesting** - Unified framework supporting diverse asset classes
- **Asset Profiler System** - Automated configuration generation for new assets
- **Enhanced Error Handling** - Improved reliability and debugging capabilities

### 📊 Repository Transformation
- **Before**: 45 Python files in root, scattered experimental code, debug directories
- **After**: 3 Python files in root, organized structure, professional appearance
- **Benefits**: Team collaboration ready, code audit compliant, institutional standards

*"Professional organization meets universal asset adaptability."* - v1.7.2 Philosophy

---

## ⚡ Quick Start (v1.7.3)

### Production CLI Interfaces
```bash
# Main CLI interface for all operations
python bin/bull_machine_cli.py --help

# ETH production backtesting
python bin/production_backtest.py

# Multi-asset adaptive backtesting (v1.7.2)
python bin/run_adaptive_backtest.py --asset ETH

# Asset profiling and configuration generation
python bin/run_multi_asset_profiler.py --asset SOL

# Institutional testing and validation
python bin/run_institutional_testing.py --all
```

### 🚀 Live Trading (v1.7.3) - NEW!

**Three-tier live pipeline: Mock → Shadow → Paper**

```bash
# Mock Feed (CSV replay)
bull-live-mock --asset ETH --start 2025-05-01 --end 2025-06-15 --config configs/live/presets/ETH_conservative.json

# Shadow Mode (log signals only, no orders)
bull-live-shadow --asset BTC --duration 2 --config configs/live/presets/BTC_vanilla.json

# Paper Trading (simulate fills, PnL, risk)
bull-live-paper --asset SOL --start 2025-08-01 --end 2025-09-30 --balance 25000 --config configs/live/presets/SOL_tuned.json
```

**Live Features:**
- ✅ **Right-edge enforcement** - No future leak, VIX hysteresis (on=22, off=18)
- ✅ **Health band monitoring** - Macro veto 5-15%, SMC ≥2-hit ≥30%, HOB ≤30%
- ✅ **Realistic execution** - 10bps fees, 5bps slippage, 2bps spread
- ✅ **Delta caps** - Macro ±0.10, Momentum ±0.06, HOB ±0.05, HPS ±0.03
- ✅ **Asset presets** - ETH/BTC/SOL configurations optimized for live trading

**⚠️ v1.7.3 Scope:** Mock/Shadow/Paper modes only. NO real exchange connections, MCP servers, or persistent execution services.

### Production Configuration
```python
# Frozen parameters achieving 12.76% returns (configs/v160/rc/ETH_production_v162.json)
{
  "risk_pct": 0.075,          # 7.5% risk allocation
  "entry_threshold": 0.3,     # Confluence threshold
  "min_active_domains": 3,    # Minimum domains required
  "sl_atr_multiplier": 1.4,   # Stop loss sizing
  "tp_atr_multiplier": 2.5    # Take profit target
}
```

### 📁 Institutional Directory Structure
```
bull-machine/
├── bin/                    # 🚀 Production Executables
│   ├── bull_machine_cli.py         # Main CLI interface
│   ├── production_backtest.py      # ETH production backtesting
│   ├── run_adaptive_backtest.py    # Multi-asset system (v1.7.2)
│   ├── run_institutional_testing.py # Institutional validation
│   ├── run_multi_asset_profiler.py  # Asset profiling system
│   └── live/                        # 📡 Live Trading Pipeline (v1.7.3)
│       ├── live_mock_feed.py        # CSV replay mock feed
│       ├── shadow_live.py           # Shadow mode (signals only)
│       ├── paper_trading.py         # Paper trading simulation
│       ├── adapters.py              # Data streaming adapters
│       ├── execution_sim.py         # Execution & PnL simulation
│       └── health_monitor.py        # Health band monitoring
├── bull_machine/           # 🔧 Core Production Package (112 files)
│   ├── backtest/           # Backtesting framework
│   ├── core/               # Core trading logic
│   ├── modules/            # Engine modules
│   ├── signals/            # Signal generation
│   └── strategy/           # Strategy implementations
├── engine/                 # 🌐 Asset Adapter Architecture (v1.7.2)
│   ├── adapters/           # Universal asset adapters
│   ├── context/            # Market context analysis
│   ├── fusion/             # Multi-domain fusion
│   ├── smc/               # Smart Money Concepts
│   └── timeframes/         # Multi-timeframe alignment
├── configs/                # ⚙️ Configuration Management
│   ├── v171/              # v1.7.1 production configs
│   ├── adaptive/          # v1.7.2 asset-specific configs
│   └── live/presets/      # v1.7.3 live trading presets
│       ├── ETH_conservative.json # Conservative ETH live config
│       ├── BTC_vanilla.json      # Standard BTC live config
│       └── SOL_tuned.json        # Optimized SOL live config
├── profiles/               # 📊 Asset Profiles (v1.7.2)
├── tests/                  # 🧪 Comprehensive Test Suite
├── scripts/research/       # 🔬 Research & Development
├── docs/                   # 📚 Documentation & Reports
└── results/archive/        # 📈 Organized Results Archive
```

### 📋 Configuration Path Reference

| Configuration Type | Path | Description |
|-------------------|------|-------------|
| **Production Configs** | `configs/v160/rc/ETH_production_v162.json` | Frozen production parameters (12.76% returns) |
| **Asset Configs** | `configs/v160/assets/{ETH,BTC,SPY}.json` | Asset-specific parameters |
| **Adaptive Configs** | `configs/adaptive/COINBASE_{BTCUSD,ETHUSD,SOLUSD,XRPUSD}_config.json` | Multi-asset adaptive configurations |
| **v1.7.1 Configs** | `configs/v171/{context,exits,fusion,liquidity,momentum,risk}.json` | Modular system configurations |
| **v1.7.0 Configs** | `configs/v170/assets/ETH_v17_*.json` | Historical calibration and tuning |
| **Legacy Configs** | `configs/v15x/` | Version 1.5x backward compatibility |
| **Profile Configs** | `configs/v14x/profile_{aggressive,balanced,conservative}.json` | Risk profile templates |

---

## 🏗️ Core Features (v1.7.2)

### 🌐 Universal Asset Adapter Architecture
- **Multi-Asset Support**: ETH, SOL, XRP, BTC, SPY with unified framework
- **Asset-Specific Optimization**: Automated parameter tuning for each asset class
- **Adaptive Configuration System**: Dynamic adjustment based on asset characteristics
- **Cross-Asset Validation**: Comprehensive testing across all supported assets

### 🏛️ Professional Repository Organization
- **Clean Architecture**: Proper separation of production vs research code
- **Production CLI Tools**: 5 dedicated command-line interfaces in `/bin/`
- **Institutional Documentation**: Comprehensive structure in `/docs/`
- **Test Consolidation**: Organized testing framework in `/tests/`

### 🔮 Advanced Trading Features
- **5-Domain Confluence System**: Wyckoff, Liquidity, Momentum, Temporal, and Fusion domains
  - Entry refinement with Order Block (OB) / Fair Value Gap (FVG) alignment
  - Exit targets at extensions (1.272–1.618 for exhaustion)
  - Risk anchoring (invalidation cuts if no reaction at cluster)
- **Time Clusters**: Overlapping Fib bar counts (21, 34, 55, 89, 144) from pivots
  - Pressure zones where moves must resolve (not predictive)
  - Aligns with Wyckoff Phase C (spring/shakeout) or D (markup/breakout)
  - Boosts fusion score when paired with liquidity/structure

### 🧙‍♂️ Oracle Whisper System
- **Soul-layer wisdom** triggered by high-confluence events
- **Price-Time Confluence**: *"Symmetry detected. Time and price converge"*
- **Premium/Discount Zones**: *"Fib levels divide reality: premium, equilibrium, discount"*
- **Temporal Pressure**: *"Time is pressure, not prediction. Fib clusters show when a move must resolve"*
- **CVD Divergences**: *"Hidden intent revealed. Bears exhaust as bulls accumulate"*

### 📊 Enhanced CVD & Orderflow Analysis
- **Cumulative Volume Delta (CVD)** with IamZeroIka's slope analysis
- **Divergence Detection**: Price vs volume intent misalignment
- **Break of Structure (BOS)** with 1/3 body close validation
- **Liquidity Capture Analysis (LCA)** for smart money detection
- **Intent Nudging** via volume confirmation and confluence counting

### 🏛️ Cross-Asset Optimization
- **SPY Config**: Lower thresholds for institutional equity markets
  - M1_TH=0.55 (vs 0.65 crypto), vol_override_atr_pct=0.04
  - Resolves "0 trades" issue with enhanced sensitivity
- **ETH Config**: Enhanced with temporal fibs + price clusters
  - Maintains crypto-optimized thresholds with cluster amplification

### ⚙️ 9-Layer Enhanced Confluence System
1. **Wyckoff Structure** - Traditional accumulation/distribution phases
2. **M1/M2 Wyckoff** - Spring/shakeout (M1) and markup/re-accumulation (M2) detection
3. **Liquidity Analysis** - Order blocks, FVGs, and liquidity sweeps
4. **Structure** - Support/resistance and trend analysis
5. **Momentum** - RSI, MACD, and momentum divergences
6. **Volume** - Volume profile and confirmation analysis
7. **Context** - Market regime and volatility context
8. **Fibonacci Clusters** - Price-time confluence zones (v1.6.1)
9. **MTF Synchronization** - Multi-timeframe bias alignment
- **Advanced Liquidity Analysis**
  Enhanced pHOB detection, sweep reclaim logic, tick-size guards, dynamic FOLP scoring.
- **Intelligent Signal Filtering**
  EQ magnet detection, desync handling, dynamic threshold bumps (±0.05-0.10).
- **Volatility-Scaled Risk Management**
  Adaptive position sizing, swing stops with ATR guardrails, complete TP ladder system.
- **Production-Validated Performance**
  Tested on 8,248+ historical bars across multiple assets and timeframes.
- **Comprehensive Testing Suite**
  MTF sync validation, PnL simulation, production analysis framework.  

---

## 📦 Installation  

Clone the repo and install dependencies:  
```bash
git clone https://github.com/rayger14/Bull-machine-.git
cd Bull-machine-
pip install -r requirements.txt
```

Or install in editable mode:  
```bash
pip install -e .
```

---

## ▶️ Usage

**v1.6.2 Production System (Recommended):**
```bash
# Complete 5-domain confluence backtest
python run_complete_confluence_system.py

# Institutional tearsheet generation
python generate_institutional_tearsheet.py

# Multi-stage optimization framework
python run_stage_a_complete.py    # Grid search optimization
python run_stage_b_optimization.py # Bayesian optimization
python run_stage_c_validation.py  # Walk-forward validation
```

**Risk Parameter Analysis:**
```bash
# Test different risk levels for institutional targets
python test_risk_scaling.py

# Extended PnL scaling analysis
python run_extended_pnl_scaling.py
```

**v1.6.2 Production Settings:**
- **Entry Threshold:** 0.3 (institutional-grade confluence requirement)
- **Min Active Domains:** 3 of 5 (Wyckoff, Liquidity, Momentum, Temporal, Fusion)
- **Risk Allocation:** 7.5% per trade (optimal for 8-15% annual targets)
- **Stop Loss:** 1.4x ATR with trailing functionality
- **Take Profit:** 2.5x ATR for institutional risk-reward ratios
- **Cooldown Period:** 7 days for optimal trade frequency

---

## 📊 CSV Format  

Your CSV must contain:  
- **Required:** `open, high, low, close`  
- **Optional:** `timestamp|datetime|date|time`, `volume`  

**Example:**  
```csv
timestamp,open,high,low,close,volume
2023-01-01,100,105,95,102,1500
2023-01-02,102,107,97,104,2000
```

---

## 📋 Example Output

```
Bull Machine 1.5.0 Starting...
Config version: 1.5.0
7-Layer Confluence System: Enabled
MTF Sync: ENABLED

==========================================
Running Multi-Timeframe Analysis...
==========================================
HTF Range: 95.50 - 105.50 (mid: 100.50)
HTF Bias: long (confirmed: True, strength: 0.75)
MTF Bias: long (confirmed: True, strength: 0.70)
Nested Confluence: ✓ (3 LTF levels)
EQ Magnet: Inactive

MTF Decision: ALLOW
Alignment Score: 83.3%

Running Wyckoff analysis...
   accumulation regime, phase C, bias long
   Confidence: phase=0.75, trend=0.80

Running Liquidity analysis...
   Overall Score: 0.65, Pressure: bullish
   FVGs: 3, OBs: 2, Sweeps: 1

Running Structure analysis...
   BOS Strength: 0.70, CHoCH: confirmed

Running Momentum analysis...
   Score: 0.55, RSI: 58, EMA Slope: positive

Running Volume analysis...
   Score: 0.60, Expansion: detected

Running Context analysis...
   Score: 0.50, Zone: premium (0.618)

Running Advanced Fusion Engine...
   Module Scores: W:0.77 L:0.65 S:0.70 M:0.55 V:0.60 C:0.50
   Fusion Score: 0.413 (threshold: 0.35)
   Signal: LONG with confidence 0.413
   TTL: 20 bars

=== TRADE PLAN GENERATED (1.5.0 MTF) ===
Direction: LONG (MTF Approved)
Entry: 45.67
Stop: 43.89
Size: 0.5618
Risk: $100.00 (1.00%)
Expected R: 2.50
MTF Quality: 1.2x (alignment bonus)
Take Profits:
  TP1: 47.45 (1R)
  TP2: 49.23 (2R)
  TP3: 52.79 (4R)

✅ HIGH CONVICTION SIGNAL - All timeframes aligned
```

---

---

## 📈 Performance Results (v1.6.2)

### Institutional-Grade Validation
**Validation Period:** 2024-01-01 to 2024-12-31
**Asset:** ETH/USD Multi-Timeframe (1H/4H/1D)

| Metric | Value | Institutional Benchmark | Status |
|--------|-------|------------------------|--------|
| **Total Return** | 12.76% | 8-15% Target | ✅ ACHIEVED |
| **Win Rate** | 62.5% | >55% Target | ✅ ACHIEVED |
| **Profit Factor** | 2.07 | >1.5 Target | ✅ ACHIEVED |
| **Maximum Drawdown** | 8.34% | <10% Target | ✅ ACHIEVED |
| **Sharpe Ratio** | 0.57 | >0.5 Target | ✅ ACHIEVED |
| **Total Trades** | 8 | Adequate Sample | ✅ ACHIEVED |

### Production Performance (2024)
- **Starting Capital:** $100,000
- **Ending Capital:** $112,762
- **Profit/Loss:** +$12,762
- **Best Trade:** +59.38% return
- **Worst Trade:** -36.44% return
- **Average Trade:** +1.69% return
- **Risk Per Trade:** 7.5% allocation
- **Trading Frequency:** 0.67 trades/month

### Multi-Year Track Record (2022-2024)
- **Total Returns:** 16.4% over 2+ years
- **Maximum Drawdown:** 8.4% (consistently below 10%)
- **Market Cycles Tested:** Bull, bear, and sideways conditions
- **Cross-Validation:** Multiple timeframes and market regimes

---

## ⚠️ Limitations (v1.6.2)
- Historical data backtesting only — live trading integration in development
- Requires multi-timeframe data (1H, 4H, 1D) for optimal performance
- Institutional-grade validation complete, but recommend extensive paper trading before live deployment
- Optimization framework requires substantial computational resources for large parameter sweeps

---

## 🔮 Roadmap

- **v1.3.0** ✅ Enhanced Liquidity + 6-Layer Confluence (COMPLETE)
- **v1.4.0** ✅ Multi-Timeframe Sync + 7-Layer System (COMPLETE)
- **v1.5.0** ✅ Advanced Exit System + Quality Floor + Telemetry (COMPLETE)
- **v1.6.1** ✅ Fibonacci Clusters + Cross-Asset Optimization (COMPLETE)
- **v1.6.2** ✅ Institutional-Grade Production Release (CURRENT)
- **v1.7.x** → Live Exchange Integration (Binance, Coinbase Pro) + Real-time monitoring
- **v2.x** → Machine Learning Layer (pattern recognition, adaptive thresholds)
- **Beyond** → Multi-asset portfolio management, sentiment integration  

---

## 📊 Production Testing (v1.6.2)

Run institutional-grade analysis and optimization:

```bash
# Complete 5-domain confluence system
python run_complete_confluence_system.py

# Multi-stage optimization framework
python run_stage_a_complete.py      # Stage A: Grid search
python run_stage_b_optimization.py  # Stage B: Bayesian optimization
python run_stage_c_validation.py    # Stage C: Walk-forward validation

# Professional tearsheet generation
python generate_institutional_tearsheet.py

# Risk parameter scaling analysis
python test_risk_scaling.py
python run_extended_pnl_scaling.py

# Production monitoring deployment
python deploy_production_monitoring.py
```

**Institutional Validation Results:**
- **Optimization Framework:** 121 parameter combinations tested (66 Stage A + 50 Stage B + 5 risk levels)
- **Data Coverage:** 2024 complete year with multi-timeframe validation
- **Quality Gates:** All institutional metrics achieved (8-15% returns, <10% drawdown, >55% win rate)
- **Production Ready:** Frozen configuration with git-tracked reproducibility
- **Scaling Validated:** Linear projection testing up to $10M AUM

---

## 🧪 Development  

Run tests (if included):  
```bash
pytest tests/ --cov=bull_machine
```

Code quality checks:  
```bash
black bull_machine/
flake8 bull_machine/
mypy bull_machine/
```

---

## 📜 License  

MIT License — for educational purposes only.  
⚠️ **Warning:** This is experimental software. Do not use for live trading without thorough testing.  
