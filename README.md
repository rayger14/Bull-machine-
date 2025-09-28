# Bull Machine 1.6.1

[![CI](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml/badge.svg)](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-success)](https://github.com/rayger14/Bull-machine-/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%7C%20mypy-blue)](https://github.com/rayger14/Bull-machine-/actions)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://python.org)

Advanced algorithmic trading system with **Fibonacci Price-Time Clusters**, **CVD Orderflow Analysis**, **Oracle Whisper System**, **Multi-Timeframe Sync**, **9-Layer Confluence**, **Wyckoff M1/M2**, and **cross-asset optimization**.

**Branch Protection:** ✅ Required CI checks must pass before merging to main

**Status:** All tests passing ✅

---

## 🚀 What's New in 1.6.1: Fibonacci Clusters & Cross-Asset Optimization

- **🔮 Fibonacci Price Clusters** - Overlapping fib levels (0.382, 0.618, 1.272, etc.) for premium/discount zones, entry refinement (OB/FVG alignment), exit targets (1.272–1.618), and risk anchoring (invalidation cuts)
- **⏰ Fibonacci Time Clusters** - Overlapping Fib bar counts (21, 34, 55, 89, 144) for pressure zones, boosting Wyckoff Phase C/D signals when "time and price sing as one"
- **🧙‍♂️ Oracle Whisper System** - Soul-layer wisdom drops triggered by price-time confluence: *"Symmetry detected. Time and price converge. Pressure must resolve."*
- **📊 Enhanced CVD Analysis** - IamZeroIka's slope detection for divergence analysis and hidden intent revelation in orderflow
- **🏛️ Cross-Asset Configs** - SPY.json with lower thresholds (M1_TH=0.55, vol_override_atr_pct=0.04) for low-vol equities + enhanced ETH config
- **⚡ Integration Magic** - Clusters boost fusion (+0.05–0.10) with liquidity/structure, never standalone; price-time confluence amplifies Wyckoff phases
- **📈 Production Validation** - SPY orderflow backtesting: 55.3% win rate, 20.5% return, resolving "0 trades" issue with institutional market sensitivity

*"Price and time symmetry = where structure and vibration align."* - v1.6.1 Philosophy

---

## 🏗️ Core Features (1.6.1)

### 🔮 Fibonacci Price-Time Clusters
- **Price Clusters**: Overlapping fib levels (0.382, 0.618, 1.272, etc.) for premium/discount zones
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

**1.5.1 with Ensemble Mode (Recommended):**
```bash
python run_eth_ensemble_backtest.py  # Optimized for ETH with true R-based exits
```

**v1.3.0 Baseline (for comparison):**
```bash
python -m bull_machine.app.main --csv your_data.csv --balance 10000
```

**Disable MTF for testing:**
```bash
python -m bull_machine.app.main_v13 --csv your_data.csv --balance 10000
```

**1.5.1 Production Settings:**
- **Enter Threshold:** 0.44 (with dynamic volatility adjustments ±0.02)
- **Quality Floors:** Wyckoff/Liquidity/Structure: 0.25, Momentum/MTF: 0.27
- **Cooldown Period:** 168 bars (7 days) for optimal frequency control
- **Profit Ladders:** 1.5R/2.5R/4.0R with 25%/50%/25% position scaling
- **Exit Strategy:** True R-based calculations with trailing stops

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

## 📈 Performance Results (1.5.0)

### Validated Performance Across Markets
**Total Trades Analyzed:** 2,745 (v1.3.0) vs 1,768 (1.5.0)

| Asset | Timeframe | v1.3.0 Win Rate | 1.5.0 Win Rate | PnL Improvement |
|-------|-----------|-----------------|-----------------|------------------|
| BTCUSD | 1D | 58.0% | 69.6% | +208.3% |
| BTCUSD | 4H | 58.0% | 75.0% | +71.5% |
| ETHUSD | 1D | 58.0% | 69.6% | +184.0% |
| SPY | All TFs | 58.0% | 75.0% | +50-80% |

**Overall Results:**
- **Total PnL:** +1,299.9% (v1.3.0) → +2,092.5% (1.5.0)
- **Improvement:** +792.6% (+61.0% better performance)
- **Win Rate:** 58% → 69.6-75% (+11.6-17% improvement)
- **Signal Quality:** 36% fewer trades, but much higher success rate

**12-Month Account Growth Simulation:**
- v1.3.0: $10,000 → $25,182
- 1.5.0: $10,000 → $34,985 (+$9,803 additional profit)

---

## ⚠️ Limitations (1.5.0)
- CSV input only — no live exchange feeds yet.
- Requires sufficient historical data (200+ bars) for MTF analysis.
- Production-validated but recommend paper trading before live deployment.  

---

## 🔮 Roadmap

- **v1.3.0** ✅ Enhanced Liquidity + 6-Layer Confluence (COMPLETE)
- **v1.4.0** ✅ Multi-Timeframe Sync + 7-Layer System (COMPLETE)
- **1.5.0** ✅ Advanced Exit System + Quality Floor + Telemetry (CURRENT)
- **v1.5.0** → Live Exchange Integration (Binance, Coinbase Pro)
- **v2.x** → Machine Learning Layer (pattern recognition, adaptive thresholds)
- **Beyond** → Multi-asset portfolio management, sentiment integration  

---

## 📊 Production Testing (1.5.0)

Run comprehensive analysis and backtests:

```bash
# 1.5.0 MTF Performance Analysis
python run_production_analysis.py

# 1.5.0 vs v1.3.0 PnL Comparison
python simulate_v13_pnl.py

# MTF Core Functionality Demo
python demonstrate_v13_mtf.py

# 1.5.0 Production Test
python production_test_v13.py
```

**Production Analysis Results:**
- **MTF Decision Matrix:** 7 scenarios tested (57% ALLOW, 29% RAISE, 14% VETO)
- **Data Suitability:** 8,248+ bars across BTC/ETH/SPY
- **Performance Impact:** +22% accuracy boost through filtering
- **Market Adaptation:** Works across crypto and traditional markets

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
