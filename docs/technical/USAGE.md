# Bull Machine v1.6.2 - Usage Guide

## Quick Start

The Bull Machine CLI is your single entry point for all operations:

```bash
# Default 5-domain confluence backtest
python bull_machine_cli.py --config configs/v160/rc/ETH_production_v162.json

# Help and all available modes
python bull_machine_cli.py --help
```

## Available Modes

### 1. Confluence Backtest (Default)
Complete 5-domain confluence system with institutional validation.

```bash
python bull_machine_cli.py --mode confluence \
  --config configs/v160/rc/ETH_production_v162.json \
  --start 2024-01-01 --end 2024-12-31
```

### 2. Ensemble Backtesting
Multi-strategy ensemble with validation across timeframes.

```bash
python bull_machine_cli.py --mode ensemble \
  --asset ETH --config configs/v160/assets/ETH.json
```

### 3. Multi-Stage Optimization
Institutional-grade parameter optimization framework.

```bash
# Stage A: Grid Search (66 combinations)
python bull_machine_cli.py --mode stage-a --asset ETH --config configs/v160/assets/ETH.json

# Stage B: Bayesian Optimization (50 refinements)
python bull_machine_cli.py --mode stage-b --asset ETH --config configs/v160/assets/ETH.json

# Stage C: Walk-Forward Validation
python bull_machine_cli.py --mode stage-c --asset ETH --config configs/v160/assets/ETH.json
```

### 4. Risk Parameter Scaling
Optimize risk allocation for institutional return targets (8-15% annual).

```bash
python bull_machine_cli.py --mode risk-scaling \
  --asset ETH --config configs/v160/assets/ETH.json
```

### 5. Signal Weight Optimization
Optimize domain weighting and confluence thresholds.

```bash
python bull_machine_cli.py --mode weight-opt \
  --asset ETH --config configs/v160/assets/ETH.json
```

### 6. Professional Tearsheet
Generate institutional-grade performance reports.

```bash
python bull_machine_cli.py --mode tearsheet \
  --config configs/v160/rc/ETH_production_v162.json
```

## Configuration

### Production Configuration
Use the frozen production config for validated results:
```bash
--config configs/v160/rc/ETH_production_v162.json
```

### Development Configuration
Use asset-specific configs for experimentation:
```bash
--config configs/v160/assets/ETH.json
--config configs/v160/assets/BTC.json
--config configs/v160/assets/SPY.json
```

### Runtime Overrides
Override parameters without modifying config files:

```bash
# Environment variable override
BM_RTCFG="thresh0.3_min3_cd7_r0.075_sl1.4_tp2.5" python bull_machine_cli.py --config ...

# CLI parameter overrides
python bull_machine_cli.py --config ... --risk-pct 0.075 --start 2024-01-01 --end 2024-12-31
```

## Output Management

Results are automatically saved to the `reports/` directory (gitignored):

- **Backtest Results**: JSON output with clean metrics
- **Optimization Logs**: JSONL append-only logging
- **Tearsheets**: Professional PDF/PNG reports
- **Telemetry**: Detailed execution logs

## Script Organization

The repository is organized for maintainability:

```
/bull_machine_cli.py          # Single CLI entry point
/scripts/
  /backtests/                 # Specialized backtest runners
  /opt/                       # Optimization framework
  /legacy/                    # Archived v1.4.x scripts
/configs/v160/                # Configuration files
  /assets/                    # Asset-specific configs
  /rc/                        # Release candidate (frozen)
/docs/                        # Documentation
/tests/                       # Test suite
```

## Advanced Usage

### Quiet Mode (Grid Optimization)
```bash
python bull_machine_cli.py --mode confluence --quiet --seed 42 --config ...
```

### Custom Output Directory
```bash
python bull_machine_cli.py --out reports/custom_analysis --config ...
```

### Reproducible Results
```bash
python bull_machine_cli.py --seed 42 --config ... # Fixed seed for reproducibility
```

## Migration from Legacy Scripts

If you were using individual runner scripts, migrate to CLI modes:

| Legacy Script | New CLI Command |
|---------------|----------------|
| `run_complete_confluence_system.py` | `--mode confluence` |
| `run_full_ensemble_backtests.py` | `--mode ensemble` |
| `run_stage_a_complete.py` | `--mode stage-a` |
| `run_signal_weight_optimization.py` | `--mode weight-opt` |
| `test_risk_scaling.py` | `--mode risk-scaling` |
| `generate_institutional_tearsheet.py` | `--mode tearsheet` |

Legacy scripts are preserved in `scripts/legacy/` for reference.

## Best Practices

1. **Use Production Config** for validated results
2. **Set Seeds** for reproducible optimization
3. **Monitor Resources** during long optimizations
4. **Version Control** important configurations
5. **Review Tearsheets** before deployment decisions

## Support

For issues or questions:
- Check logs in `reports/` directory
- Review configuration examples in `configs/v160/`
- Consult legacy scripts in `scripts/legacy/` for reference
- Verify CI status for system health