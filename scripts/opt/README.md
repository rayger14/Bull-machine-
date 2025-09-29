# Optimization Scripts

Institutional-grade optimization framework with multi-stage validation.

## Preferred Usage (CLI)

```bash
# Stage A: Grid Search Optimization
python bull_machine_cli.py --mode stage-a --asset ETH

# Stage B: Bayesian Optimization
python bull_machine_cli.py --mode stage-b --asset ETH

# Stage C: Walk-Forward Validation
python bull_machine_cli.py --mode stage-c --asset ETH

# Signal Weight Optimization
python bull_machine_cli.py --mode weight-opt --asset ETH

# Risk Parameter Scaling
python bull_machine_cli.py --mode risk-scaling --asset ETH
```

## Scripts in this directory

- `run_stage_a_complete.py` - Grid search optimization (66 combinations)
- `run_stage_b_optimization.py` - Bayesian optimization refinement
- `run_stage_c_validation.py` - Walk-forward validation
- `run_signal_weight_optimization.py` - Signal weighting analysis
- `run_extended_pnl_scaling.py` - Risk parameter scaling
- `run_tuning_optimization.py` - Fine-tuning optimization
- `run_multi_asset_optimization.py` - Cross-asset optimization

## Optimization Framework

The optimization follows institutional-grade validation:
1. **Stage A**: Broad parameter grid search
2. **Stage B**: Bayesian refinement of promising regions
3. **Stage C**: Out-of-sample walk-forward validation
4. **Risk Scaling**: Institutional return target optimization (8-15% annual)

All optimizations include crash protection and resource monitoring.