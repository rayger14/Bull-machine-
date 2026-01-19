# Reference Results

This directory contains a curated set of canonical backtest results that serve as ground truth for the Bull Machine system.

## Purpose

- **Regression Testing**: Verify system behavior remains consistent across code changes
- **Performance Baseline**: Document historical performance metrics
- **Knowledge Preservation**: Keep essential experimental results without bloating the repo

## Structure

```
results_reference/
├── btc/                    # Bitcoin baseline results
├── eth/                    # Ethereum baseline results  
├── sol/                    # Solana baseline results
├── bear_market/            # Bear market pattern validation
└── system_validation/      # Integration tests and validation runs
```

## Files Kept

Only the most representative, stable results are kept here:
- One canonical result per major asset/regime combination
- Critical validation results (e.g., 2022 bear market validation)
- System integration test outputs

## What's NOT Here

Transient experimental outputs are regenerated locally and excluded via .gitignore:
- `*_backtest_results*.json` (daily experimental runs)
- `health_summary_*.json` (runtime health checks)
- `hybrid_signals_*.jsonl` (signal debugging logs)
- `fusion_*.jsonl` (fusion layer debug)
- `sweep_results_*.json` (parameter sweeps)

## Regenerating Results

To reproduce any of these results, use the configs in `configs/` with:
```bash
python bin/run_backtest.py --config configs/[asset]_baseline.json
```
