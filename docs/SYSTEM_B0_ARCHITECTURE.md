# System B0 Architecture Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-12-04

## System Overview

System B0 is a production-ready deployment system for the Baseline-Conservative trading strategy. It implements a robust, fault-tolerant architecture with comprehensive observability and safety mechanisms.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM B0 ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  Configuration Layer │
└──────────┬───────────┘
           │
           │  configs/system_b0_production.json
           │  ┌─ Strategy Parameters (entry/exit thresholds)
           │  ├─ Risk Management (position sizing, limits)
           │  ├─ Execution Settings (timeframe, slippage)
           │  ├─ Monitoring Config (alerts, thresholds)
           │  └─ Emergency Settings (circuit breakers)
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT LAYER                                   │
│  examples/baseline_production_deploy.py                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ ConfigLoader   │  │ ProductionLogger│  │ DataManager     │              │
│  ├────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ - Load config  │  │ - Console logs  │  │ - Load data     │              │
│  │ - Validate     │  │ - File logs     │  │ - Validate      │              │
│  │ - Secure       │  │ - Severity      │  │ - Cache         │              │
│  └────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ RiskManager    │  │ BacktestEngine  │  │ SystemB0        │              │
│  ├────────────────┤  ├─────────────────┤  ├─────────────────┤              │
│  │ - Position     │  │ - Simulate      │  │ - Orchestrate   │              │
│  │   sizing       │  │   trades        │  │ - Run modes     │              │
│  │ - Circuit      │  │ - Calculate     │  │ - Report        │              │
│  │   breakers     │  │   metrics       │  │   results       │              │
│  │ - Limits       │  │ - Validate      │  │                 │              │
│  └────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           │  Modes: backtest | live_signal | paper_trading | live_trading
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MONITORING LAYER                                   │
│  bin/monitor_system_b0.py                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ MonitoringEngine                                                   │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ - check_market()        → MarketStatus                            │     │
│  │ - check_signal()        → Signal detection                        │     │
│  │ - check_position()      → PositionStatus                          │     │
│  │ - run_check()           → Complete status report                  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ AlertSystem                                                        │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ - Console alerts        → Immediate visibility                    │     │
│  │ - File alerts           → Persistent log (alerts.jsonl)           │     │
│  │ - Webhook alerts        → External integrations (Slack, etc.)     │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Dashboard                                                          │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │ - Market status         → Price, drawdown, distance to entry      │     │
│  │ - Position status       → PnL, stops, targets                     │     │
│  │ - Statistics            → Signals, checks, performance            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           │  Real-time updates every 5 minutes (configurable)
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          VALIDATION LAYER                                    │
│  bin/validate_system_b0.py                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ ValidationSuite                                                  │       │
│  ├──────────────────────────────────────────────────────────────────┤       │
│  │                                                                  │       │
│  │  Test 1: Extended Period Performance (2022-2024)                │       │
│  │  ├─ Check PF >= 2.0                                             │       │
│  │  ├─ Check WR >= 35%                                             │       │
│  │  └─ Check DD <= 25%                                             │       │
│  │                                                                  │       │
│  │  Test 2: Regime Performance Breakdown                           │       │
│  │  ├─ Bear Market (2022): PF >= 2.0                               │       │
│  │  └─ Bull Market (2023): PF >= 2.5                               │       │
│  │                                                                  │       │
│  │  Test 3: Walk-Forward Validation                                │       │
│  │  ├─ 6-month train, 3-month test windows                         │       │
│  │  └─ All windows positive                                        │       │
│  │                                                                  │       │
│  │  Test 4: Parameter Sensitivity                                  │       │
│  │  ├─ Vary buy_threshold ±20%                                     │       │
│  │  ├─ Vary profit_target ±25%                                     │       │
│  │  └─ All variations PF > 1.5                                     │       │
│  │                                                                  │       │
│  │  Test 5: Statistical Significance                               │       │
│  │  ├─ Sufficient trades (N >= 40)                                 │       │
│  │  └─ PF confidence interval excludes 1.0                         │       │
│  │                                                                  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  Output: ValidationReport with overall score and detailed results            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           │  Generates comprehensive validation report
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                             MODEL LAYER                                      │
│  engine/models/simple_classifier.py                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ BuyHoldSellClassifier (BaseModel)                                 │     │
│  ├────────────────────────────────────────────────────────────────────┤     │
│  │                                                                    │     │
│  │  Parameters:                                                       │     │
│  │  - buy_threshold: -0.15 (15% drawdown)                            │     │
│  │  - profit_target: 0.08 (8% gain)                                  │     │
│  │  - stop_atr_mult: 2.5 (2.5x ATR)                                  │     │
│  │                                                                    │     │
│  │  Methods:                                                          │     │
│  │  - fit(data)              → Calibration (optional)                │     │
│  │  - predict(bar, position) → Generate Signal                       │     │
│  │  - get_position_size()    → Calculate size                        │     │
│  │  - get_params()           → Return parameters                     │     │
│  │                                                                    │     │
│  │  Logic:                                                            │     │
│  │  Entry:  IF capitulation_depth < -15%    → LONG                   │     │
│  │  Exit:   IF profit >= 8%                 → CLOSE (profit_target)  │     │
│  │          IF price <= (entry - 2.5*ATR)   → CLOSE (stop_loss)      │     │
│  │                                                                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           │  Uses Signal and Position dataclasses
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Feature Store (engine/features/builder.py)                                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ Required Features:                                               │       │
│  │ - close, high, low, volume (OHLCV)                              │       │
│  │ - atr_14 (volatility)                                           │       │
│  │ - capitulation_depth (30d drawdown)                             │       │
│  │ - high_30d (rolling high)                                       │       │
│  │ - volume_z (optional, volume spike detection)                   │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  Data Sources:                                                               │
│  - Historical: Parquet files in data/feature_store/                          │
│  - Live: OKX API (primary), CCXT (backup)                                    │
│  - Cache: 60-second TTL                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Configuration Layer
**File:** `configs/system_b0_production.json`

**Responsibilities:**
- Store all system parameters
- Define risk limits and safety thresholds
- Configure monitoring and alerting
- Specify validation criteria

**Design Principles:**
- Single source of truth
- No hardcoded values
- Validation on load
- Easy to audit

### 2. Deployment Layer
**File:** `examples/baseline_production_deploy.py`

**Components:**

#### ConfigLoader
- Loads and validates configuration
- Ensures required sections present
- Type checking and range validation

#### ProductionLogger
- Structured logging to console and file
- Severity levels (INFO, WARNING, CRITICAL)
- Timestamp formatting
- Log rotation support

#### DataManager
- Loads historical data from feature store
- Validates required features
- Computes derived features (capitulation_depth, ATR)
- Implements caching for performance

#### RiskManager
- Position sizing (fixed risk method)
- Portfolio risk tracking
- Circuit breaker logic
- Trade history and state management

#### BacktestEngine
- Accurate trade simulation
- Slippage and commission modeling
- Exit condition checking
- Performance metric calculation

#### SystemB0
- Main orchestration class
- Mode selection (backtest, live_signal, paper_trading)
- Result reporting
- Performance validation

**Operational Modes:**
1. **Backtest:** Historical validation
2. **Live Signal:** Real-time signals, no execution
3. **Paper Trading:** Simulated execution
4. **Live Trading:** Real execution (not yet implemented)

### 3. Monitoring Layer
**File:** `bin/monitor_system_b0.py`

**Components:**

#### MonitoringEngine
- Real-time market status tracking
- Signal detection and logging
- Position monitoring (PnL, distances)
- Alert generation

#### AlertSystem
- Multiple channels (console, file, webhook)
- Severity-based routing
- Persistent alert log (JSON Lines)
- External integration support

#### Dashboard
- Real-time metrics display
- Market condition summary
- Position status (if open)
- Statistics and counters

**Alert Types:**
- **INFO:** Normal operations, position updates
- **WARNING:** Risk limits, signal rejections
- **CRITICAL:** Emergency stops, system errors

### 4. Validation Layer
**File:** `bin/validate_system_b0.py`

**Test Suite:**

1. **Extended Period Performance**
   - Full 2022-2024 backtest
   - Validate against targets (PF, WR, DD)
   - Score: weighted combination

2. **Regime Breakdown**
   - Bear market (2022) performance
   - Bull market (2023) performance
   - Regime-specific targets

3. **Walk-Forward Validation**
   - 6-month train, 3-month test
   - Rolling windows with 3-month step
   - All windows must be positive

4. **Parameter Sensitivity**
   - Vary key parameters ±20-25%
   - Check stability (PF > 1.5)
   - Ensure robustness

5. **Statistical Significance**
   - Minimum trade count validation
   - Confidence interval calculation
   - Ensure edge is real (CI excludes 1.0)

**Output:**
- ValidationReport with pass/fail per test
- Overall score (0-1)
- Summary recommendation
- Detailed results saved to JSON

### 5. Model Layer
**File:** `engine/models/simple_classifier.py`

**Class:** `BuyHoldSellClassifier`

**Strategy Logic:**

```python
# Entry Condition
if capitulation_depth < buy_threshold (-15%):
    if volume_spike (optional):
        ENTER LONG
        stop_loss = entry_price - (atr_14 * 2.5)

# Exit Conditions
if in_position:
    profit_pct = (current_price - entry_price) / entry_price
    
    if profit_pct >= profit_target (8%):
        EXIT → profit_target
    
    if current_price <= stop_loss:
        EXIT → stop_loss
```

**Position Sizing:**
```python
risk_amount = portfolio_value * risk_per_trade_pct
risk_per_unit = entry_price - stop_loss
position_size = risk_amount / risk_per_unit
```

## Data Flow

### Backtest Mode
```
1. Load config → ConfigLoader
2. Initialize model → BuyHoldSellClassifier
3. Load data → DataManager → FeatureStoreBuilder
4. Validate features → DataManager
5. For each bar:
   a. Generate signal → Model.predict()
   b. Check risk limits → RiskManager
   c. Simulate trade → BacktestEngine
   d. Update state → RiskManager
6. Calculate metrics → BacktestEngine
7. Validate performance → SystemB0
8. Generate report → ProductionLogger
```

### Monitoring Mode
```
1. Load config → ConfigLoader
2. Initialize engine → MonitoringEngine
3. Setup alerts → AlertSystem
4. Main loop:
   a. Load latest bar → DataManager (or live API)
   b. Check market → MonitoringEngine.check_market()
   c. Check signal → MonitoringEngine.check_signal()
   d. Check position → MonitoringEngine.check_position()
   e. Generate alerts → AlertSystem
   f. Update dashboard → MonitoringEngine.print_dashboard()
   g. Sleep (check_interval_seconds)
```

### Validation Mode
```
1. Load config → ConfigLoader
2. Initialize suite → ValidationSuite
3. For each test:
   a. Create isolated system → SystemB0
   b. Run backtest → SystemB0.run_backtest()
   c. Evaluate results → Test-specific logic
   d. Record outcome → TestResult
4. Calculate overall score → ValidationSuite
5. Generate report → ValidationReport
6. Save to JSON → File
```

## Security Architecture

### Configuration Security
- No credentials in code
- Environment variables for sensitive data
- Config file validation
- Safe default values

### Data Validation
- Required features checked
- Range validation for parameters
- Null/NaN detection
- Timestamp continuity checks

### Risk Controls
- Position size limits (absolute USD cap)
- Portfolio risk limits (percentage)
- Daily trade limits (prevent overtrading)
- Cooldown periods (reduce frequency)
- Circuit breakers (auto-stop)

### Error Handling
- Try-catch at every external interaction
- Graceful degradation (continue on non-critical errors)
- State preservation (logs, alerts)
- Recovery procedures documented

## Performance Characteristics

### Computational Complexity
- **Data loading:** O(N) where N = number of bars
- **Signal generation:** O(1) per bar
- **Backtest simulation:** O(N × T) where T = avg bars held
- **Validation suite:** O(N × V) where V = number of validation windows

### Memory Usage
- Feature store: ~500 MB for 3 years 1H data
- Backtest state: ~10 MB (trade history)
- Monitoring: ~1 MB (current state)

### Latency
- Signal generation: < 1ms
- Data load (cached): < 100ms
- Full backtest (2022-2024): ~5-10 seconds
- Validation suite (full): ~2-3 minutes

## Extensibility Points

### Adding New Modes
1. Implement mode handler in `SystemB0`
2. Add mode-specific configuration
3. Update CLI argument parser
4. Document in production guide

### Custom Alert Channels
1. Implement alert handler in `AlertSystem._send_alert()`
2. Add configuration for new channel
3. Update monitoring config schema

### Additional Validation Tests
1. Add test method to `ValidationSuite`
2. Call from `run_all_tests()`
3. Update report generation

### Parameter Optimization
1. Modify model parameters in config
2. Run validation suite to verify
3. Compare against baseline
4. Document changes

## Deployment Checklist

- [ ] All files present and executable
- [ ] Configuration reviewed and customized
- [ ] Feature store validated
- [ ] Validation suite passes
- [ ] Monitoring system tested
- [ ] Alert channels configured
- [ ] Risk limits appropriate
- [ ] Emergency procedures understood
- [ ] Documentation reviewed
- [ ] Deployment approved

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-04 | Initial production release |

---

**Maintained by:** Trading Systems Team  
**Review Cycle:** Monthly  
**Next Review:** 2025-01-04
