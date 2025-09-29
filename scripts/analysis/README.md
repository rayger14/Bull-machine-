# Analysis Scripts

Performance analysis and result inspection tools.

## Scripts

- `analyze_2024_eth_performance.py` - ETH 2024 performance analysis
- `analyze_2024_trades_only.py` - Trade-specific analysis
- `analyze_optimization_results.py` - Optimization result analysis
- `analyze_spy_results.py` - SPY performance analysis

## Usage

These scripts analyze backtest results and optimization outputs:

```bash
python scripts/analysis/analyze_2024_eth_performance.py
python scripts/analysis/analyze_optimization_results.py
```

For new analysis, prefer the CLI:
```bash
python bull_machine_cli.py --mode tearsheet --config configs/v160/rc/ETH_production_v162.json
```