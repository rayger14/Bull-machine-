# Backtest Scripts

These scripts provide specialized backtest functionality. For most use cases, prefer the unified CLI:

```bash
# Instead of run_complete_confluence_system.py
python bull_machine_cli.py --mode confluence --asset ETH

# Instead of run_full_ensemble_backtests.py
python bull_machine_cli.py --mode ensemble --asset ETH

# Instead of run_btc_ensemble_backtest.py
python bull_machine_cli.py --mode ensemble --asset BTC
```

## Scripts in this directory

- `run_complete_confluence_system.py` - Complete 5-domain confluence backtest
- `run_full_ensemble_backtests.py` - Multi-asset ensemble backtesting
- `run_true_ensemble_backtests.py` - True ensemble with validation
- `run_btc_ensemble_backtest.py` - BTC-specific ensemble backtest
- `run_btc_comprehensive_backtest.py` - BTC comprehensive analysis
- `run_eth_ensemble_backtest.py` - ETH ensemble backtest
- `run_eth_ensemble_backtest_v160.py` - ETH v1.6.0 specific backtest
- `run_sol_comprehensive_backtest.py` - SOL comprehensive backtest

## Migration to CLI

These scripts will gradually be replaced by CLI modes for better maintainability and consistency.