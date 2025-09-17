# Bull Machine v1.2.1

[![CI](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml/badge.svg)](https://github.com/rayger14/Bull-machine-/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-9%2F9%20passing-success)](https://github.com/rayger14/Bull-machine-/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%7C%20mypy-blue)](https://github.com/rayger14/Bull-machine-/actions)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://python.org)

Algorithmic trading signal generator combining **Wyckoff structure**, **Liquidity analysis** (Fair Value Gaps & Order Blocks), plus **Dynamic TTL** and advanced **risk planning**.

**Branch Protection:** ✅ Required CI checks must pass before merging to main

---

## 🚀 Features (v1.2.1)
- **6-Layer Confluence System**
  Wyckoff + Liquidity + Structure + Momentum + Volume + Context analysis with weighted fusion.
- **Advanced Liquidity Analysis**
  Enhanced pHOB detection, sweep reclaim logic, tick-size guards, dynamic FOLP scoring.
- **Intelligent Fusion Engine**
  Optimized vetoes, volatility shock detection, trend filters with configurable thresholds.
- **Volatility-Scaled Risk Management**
  Adaptive position sizing, swing stops with ATR guardrails, complete TP ladder system.
- **Production Configuration**
  Validated optimal settings: ETH 4H (77% win rate), BTC Daily, COIN 1H performance.
- **Comprehensive Testing**
  9/9 test suite with CI/CD pipeline (ruff, mypy, pytest) ensuring code quality.  

---

## 📦 Installation  

Clone the repo and install dependencies:  
```bash
git clone https://github.com/yourusername/bull_machine.git
cd bull_machine
pip install -r requirements.txt
```

Or install in editable mode:  
```bash
pip install -e .
```

---

## ▶️ Usage

Run with production configuration (v1.2.1):
```bash
python -m bull_machine.app.main --csv your_data.csv --balance 10000 --config config/production.json
```

Or use default settings:
```bash
python -m bull_machine.app.main --csv your_data.csv --balance 10000
```

Production settings (config/production.json):
- **Enter Threshold:** 0.35
- **Volatility Shock:** 4.0σ
- **Trend Alignment:** 0.60

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
Bull Machine v1.2.1 Starting...
Config version: 1.2.1
6-Layer Confluence System: Enabled

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

=== TRADE PLAN GENERATED (v1.2.1) ===
Direction: LONG
Entry: 45.67
Stop: 43.89
Size: 0.5618
Risk: $100.00 (1.00%)
Expected R: 2.50
Take Profits:
  TP1: 47.45 (1R)
  TP2: 49.23 (2R)
  TP3: 52.79 (4R)
```

---

## ⚠️ Limitations (v1.2.1)
- CSV input only — no live exchange feeds yet.
- Single timeframe analysis (multi-TF sync planned in v1.2.2).
- Simplified heuristics — strong foundation but requires further refinement for institutional-grade performance.  

---

## 🔮 Roadmap

- **v1.2.1** ✅ Enhanced Liquidity + 6-Layer Confluence (COMPLETE)
- **v1.2.2** → Multi-Timeframe Sync (D1→4H→1H bias gates)
- **v1.3.0** → Advanced Fusion & Veto Systems
- **v1.4.0** → Professional Backtesting Framework
- **v2.x** → Candle wisdom (wicks, traps, advanced OBs)
- **Beyond** → Live feeds, temporal clustering, sentiment integration  

---

## 📊 Backtesting (Alpha)

Run backtests on historical data to evaluate performance:

```bash
# Basic backtest on BTC daily data
python production_test_v121.py

# Custom CSV backtest
python backtest_v121_btc.py --csv btc_4h_data.csv

# PnL analysis
python pnl_analysis_v121.py --csv your_data.csv --balance 10000
```

**Note:** Backtesting is in alpha. Results include:
- Win rate and R-multiple statistics
- Trade-by-trade breakdown
- Total PnL and expectancy calculations

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
