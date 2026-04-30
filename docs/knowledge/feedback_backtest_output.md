---
name: backtest_output_format
description: Always include data range, starting equity, and avg risk per trade in backtest output
type: feedback
---

Always include data range (start-end dates), starting equity ($100K), and average risk per trade in backtest results output.

**Why:** User wants full context on every test run without having to ask.

**How to apply:** When printing or reporting backtest results, always show: date range, initial cash, commission rate, slippage, and avg risk per trade alongside PF/PnL/Sharpe/MaxDD.
