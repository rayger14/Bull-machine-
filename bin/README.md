# Bin Scripts

Production scripts for the Bull Machine trading system.

## Categories
- **Backtesting**: `backtest_*.py` - Run backtests
- **Feature Engineering**: `build_*.py`, `add_*.py` - Build feature stores
- **Analysis**: `analyze_*.py` - Analyze results and performance
- **Optimization**: `consolidate_*.py` - Optimization utilities
- **CLI**: `bull_machine_cli.py` - Command-line interface

## Archive
- `archive/experimental/` - One-time data migrations and backfill scripts
- `archive/diagnostics/` - Debug and diagnostic scripts

## Guidelines
- Production scripts only in root bin/
- Test scripts belong in tests/
- Research scripts belong in scripts/research/
- Archive one-time scripts after use
