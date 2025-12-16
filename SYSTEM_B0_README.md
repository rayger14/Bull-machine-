# System B0 - Baseline Conservative Strategy

**Production-Ready Deployment System**

## Quick Start

### 1. Validate System
```bash
python bin/validate_system_b0.py --quick
```

### 2. Run Backtest
```bash
python examples/baseline_production_deploy.py --mode backtest --period 2022-01-01:2024-09-30
```

### 3. Start Monitoring
```bash
python bin/monitor_system_b0.py --once
```

## System Overview

System B0 is a simple, robust baseline strategy:
- **Entry:** -15% drawdown from 30d high
- **Exit:** +8% profit target OR -2.5 ATR stop loss
- **Performance:** Test PF 3.17, WR 42.9%

## Files

```
configs/system_b0_production.json          # Configuration
examples/baseline_production_deploy.py     # Deployment script
bin/monitor_system_b0.py                   # Monitoring system
bin/validate_system_b0.py                  # Validation suite
docs/SYSTEM_B0_PRODUCTION_GUIDE.md         # Complete documentation
```

## Architecture

- **Data Integrity:** Validation at every stage
- **Security:** Configuration-based, no hardcoded credentials
- **Fault Tolerance:** Circuit breakers and graceful degradation
- **Observability:** Comprehensive logging and alerting
- **Testability:** Full validation suite with walk-forward tests

## Documentation

See `docs/SYSTEM_B0_PRODUCTION_GUIDE.md` for complete documentation including:
- Deployment procedures
- Monitoring guidelines
- Risk parameters
- Troubleshooting
- Emergency procedures

## Quick Commands

```bash
# Full validation
python bin/validate_system_b0.py

# Quick validation (essential tests only)
python bin/validate_system_b0.py --quick

# Backtest modes
python examples/baseline_production_deploy.py --mode backtest --period 2022-01-01:2024-12-31
python examples/baseline_production_deploy.py --mode backtest --period 2022-01-01:2022-12-31  # Bear
python examples/baseline_production_deploy.py --mode backtest --period 2023-01-01:2023-12-31  # Bull

# Monitoring
python bin/monitor_system_b0.py                  # Continuous monitoring
python bin/monitor_system_b0.py --interval 60    # Custom interval
python bin/monitor_system_b0.py --once           # Single check
```

## Performance Targets

| Metric | Target | Test Result |
|--------|--------|-------------|
| Profit Factor | >= 2.0 | 3.17 |
| Win Rate | >= 35% | 42.9% |
| Max Drawdown | <= 25% | ~18% |
| Total Trades (2022-2024) | >= 40 | 47 |

## Safety Features

- **Risk Management:** 2% per trade, 6% max portfolio exposure
- **Circuit Breakers:** Auto-stop on 30% drawdown or 10 consecutive losses
- **Kill Switch:** Manual emergency stop
- **Alert System:** Console, file, and webhook notifications

## Next Steps

1. Run validation suite: `python bin/validate_system_b0.py`
2. Review configuration: `configs/system_b0_production.json`
3. Read documentation: `docs/SYSTEM_B0_PRODUCTION_GUIDE.md`
4. Test deployment: `python examples/baseline_production_deploy.py --mode backtest --period 2024-01-01:2024-09-30`

---

**Status:** Production Ready  
**Version:** 1.0.0  
**Last Updated:** 2025-12-04
