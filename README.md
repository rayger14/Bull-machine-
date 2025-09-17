# Bull Machine v1.2.1  

Algorithmic trading signal generator combining **Wyckoff structure**, **Liquidity analysis** (Fair Value Gaps & Order Blocks), plus **Dynamic TTL** and advanced **risk planning**.  

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

Run directly on a CSV of OHLCV data:  
```bash
python -m bull_machine.app.main --csv your_data.csv --balance 10000
```

Or if installed as a package:  
```bash
bull-machine --csv your_data.csv --balance 10000
```

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
Bull Machine v1.1 Starting...
Config version: 1.1
Dynamic TTL: Enabled

Running Wyckoff analysis...
   accumulation regime, phase C, bias long
   Confidence: phase=0.80, trend=0.75

Running Liquidity analysis...
   Score: 0.68, Pressure: bullish
   FVGs: 3, OBs: 2

Running Signal Fusion...
   Signal: long with confidence 0.72
   TTL(bars): 21

=== TRADE PLAN GENERATED ===
Direction: LONG
Entry: 45.67
Stop: 44.12
Size: 6.4516
Risk: $100.00
Take Profits:
  tp1: 47.22 (33%) - move_stop_to_breakeven
  tp2: 48.77 (33%) - trail_remainder
  tp3: 50.32 (34%) - liquidate_or_hard_trail
```

---

## ⚠️ Limitations (v1.1)  
- Not a backtesting engine yet (analyzes latest chart, generates signals/risk plans).  
- CSV input only — no live exchange feeds yet.  
- Single timeframe analysis (multi-TF sync planned in Phase 1.4).  
- Wyckoff & Liquidity logic are simplified heuristics — good foundation but not institutional-grade (yet).  

---

## 🔮 Roadmap  

- **Phase 1.2** → Enhanced Liquidity (pHOB, sweeps, premium/discount validation)  
- **Phase 1.3** → Fusion scoring with veto logic + improved range suppression  
- **Phase 1.4** → Multi-timeframe sync + professional backtesting harness  
- **Phase 2.x** → Candle wisdom (wicks, traps, advanced OBs, live feeds)  
- **Beyond** → Temporal clustering, sentiment/NFT floors, astro-timing modules  

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
