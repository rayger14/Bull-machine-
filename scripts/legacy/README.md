# Legacy Scripts

**⚠️ ARCHIVED - Use `bull_machine_cli.py` instead**

These scripts are kept for historical provenance and reference. They represent earlier versions of the Bull Machine system (v1.4.x - v1.5.x) and specific experimental approaches.

## Recommended Migration

Instead of using these legacy scripts, use the unified CLI:

```bash
# Modern approach
python bull_machine_cli.py --mode confluence --asset ETH --config configs/v160/rc/ETH_production_v162.json
```

## Legacy Scripts (v1.4.x - v1.5.x)

- `run_eth_backtest_v14.py` - v1.4 ETH backtest
- `run_eth_backtest_v142.py` - v1.4.2 ETH backtest
- `run_eth_backtest_final.py` - v1.4 final ETH backtest
- `run_eth_simple_backtest.py` - Simple ETH backtest
- `run_eth_mtf_dollar_backtest.py` - Multi-timeframe dollar backtest
- `run_demo_backtest.py` - Demo backtest script
- `run_v141_baseline_backtest.py` - v1.4.1 baseline
- `run_v142_corrected_backtest.py` - v1.4.2 corrected version
- `run_v142_demo_backtest.py` - v1.4.2 demo
- `run_v150_real_data_backtest.py` - v1.5.0 real data
- `run_v150_standardized_backtest.py` - v1.5.0 standardized
- `run_eth_v160_enhanced_backtest.py` - v1.6.0 enhanced

## Historical Context

These scripts represent the evolution of Bull Machine from basic backtesting to the current institutional-grade framework. They're preserved for:

- Historical reference and research
- Debugging and comparison purposes
- Understanding system evolution
- Compliance and audit trails

**Do not use for production trading or new development.**