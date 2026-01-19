# ARCHETYPE EVALUATION REPORT: 2018-2024
================================================================================

**Evaluation Date**: 2026-01-16 11:23:06
**Dataset**: features_2018_2024_UPDATED.parquet (61,277 bars)
**Period**: 2018-01-01 to 2024-12-31 (7 years)
**Archetypes Tested**: 9

## EXECUTIVE SUMMARY
--------------------------------------------------------------------------------

- **✅ Production Ready**: 1 archetypes
- **⚠️ Needs Tuning**: 6 archetypes
- **❌ Broken**: 2 archetypes

## PRIORITIZATION MATRIX
--------------------------------------------------------------------------------

| Archetype | Name | Maturity | PF | Sharpe | Max DD | Trades/Yr | Status | Priority |
|-----------|------|----------|----|----|--------|-----------|--------|----------|
| A         | Spring/UTAD          | STUB         | 0.93   |  -0.36 |   19.3% |       723.0 | ❌ BROKEN           | P0       |
| S5        | Long Squeeze         | PRODUCTION   | 0.62   |  -2.74 |    3.3% |        70.2 | ❌ BROKEN           | P0       |
| S1        | Liquidity Vacuum     | PRODUCTION   | 1.34   |   1.83 |    0.5% |         5.4 | ⚠️ NEEDS TUNING    | P1       |
| H         | Trap Within Trend    | CALIBRATED   | 1.18   |   0.91 |    4.6% |       100.9 | ⚠️ NEEDS TUNING    | P1       |
| K         | Wick Trap Moneytaur  | CALIBRATED   | 1.18   |   0.91 |    3.9% |       117.8 | ⚠️ NEEDS TUNING    | P1       |
| S4        | Funding Divergence   | PRODUCTION   | 1.08   |   0.41 |    3.6% |       104.2 | ⚠️ NEEDS TUNING    | P1       |
| C         | BOS/CHOCH Reversal   | STUB         | 1.05   |   0.26 |   12.2% |       515.6 | ⚠️ NEEDS TUNING    | P1       |
| G         | Liquidity Sweep      | STUB         | 1.05   |   0.26 |   12.2% |       515.6 | ⚠️ NEEDS TUNING    | P1       |
| B         | Order Block Retest   | CALIBRATED   | 1.73   |   3.04 |    4.8% |       331.0 | ✅ PRODUCTION READY | P3       |

## STATUS DEFINITIONS
--------------------------------------------------------------------------------

- **✅ PRODUCTION READY**: PF >1.4, Sharpe >0.5, DD <25%
- **⚠️ NEEDS TUNING**: PF 1.0-1.4, acceptable DD <35%
- **❌ BROKEN**: PF <1.0 or DD >35% or zero signals

## DETAILED RESULTS BY ARCHETYPE
--------------------------------------------------------------------------------

### S1 - Liquidity Vacuum
**Maturity**: PRODUCTION
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 36
- Trades/Year: 5.4
- Win Rate: 52.8%
- Profit Factor: 1.34
- Sharpe Ratio: 1.83
- Max Drawdown: 0.5%
- Total PnL: $55.52
- Avg Trade: $1.54

**Period Breakdown**:
- 2018-2021: 29 trades, PF 1.41, WR 51.7%
- 2022-2024: 7 trades, PF 0.95, WR 57.1%

**Regime Breakdown**:
- crisis    :  31 trades, PF 1.08  , WR  48.4%
- risk_off  :   5 trades, PF 5.55  , WR  80.0%
- neutral   : No trades
- risk_on   : No trades

### S4 - Funding Divergence
**Maturity**: PRODUCTION
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 504
- Trades/Year: 104.2
- Win Rate: 52.2%
- Profit Factor: 1.08
- Sharpe Ratio: 0.41
- Max Drawdown: 3.6%
- Total PnL: $99.05
- Avg Trade: $0.20

**Period Breakdown**:
- 2018-2021: 20 trades, PF 6.81, WR 70.0%
- 2022-2024: 484 trades, PF 1.01, WR 51.4%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  :  90 trades, PF 1.04  , WR  46.7%
- neutral   : 414 trades, PF 1.09  , WR  53.4%
- risk_on   : No trades

### S5 - Long Squeeze
**Maturity**: PRODUCTION
**Status**: ❌ BROKEN

**Full Period (2018-2024)**:
- Trades: 195
- Trades/Year: 70.2
- Win Rate: 43.1%
- Profit Factor: 0.62
- Sharpe Ratio: -2.74
- Max Drawdown: 3.3%
- Total PnL: $-223.04
- Avg Trade: $-1.14

**Period Breakdown**:
- 2018-2021: No trades
- 2022-2024: 195 trades, PF 0.62, WR 43.1%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : 110 trades, PF 0.83  , WR  46.4%
- risk_on   :  85 trades, PF 0.36  , WR  38.8%

### H - Trap Within Trend
**Maturity**: CALIBRATED
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 703
- Trades/Year: 100.9
- Win Rate: 53.9%
- Profit Factor: 1.18
- Sharpe Ratio: 0.91
- Max Drawdown: 4.6%
- Total PnL: $528.10
- Avg Trade: $0.75

**Period Breakdown**:
- 2018-2021: 653 trades, PF 1.17, WR 54.1%
- 2022-2024: 49 trades, PF 1.49, WR 53.1%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : No trades
- risk_on   : 703 trades, PF 1.18  , WR  53.9%

### B - Order Block Retest
**Maturity**: CALIBRATED
**Status**: ✅ PRODUCTION READY

**Full Period (2018-2024)**:
- Trades: 2310
- Trades/Year: 331.0
- Win Rate: 57.8%
- Profit Factor: 1.73
- Sharpe Ratio: 3.04
- Max Drawdown: 4.8%
- Total PnL: $5055.26
- Avg Trade: $2.19

**Period Breakdown**:
- 2018-2021: 2269 trades, PF 1.73, WR 57.7%
- 2022-2024: 41 trades, PF 1.73, WR 61.0%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : No trades
- risk_on   : 2310 trades, PF 1.73  , WR  57.8%

### C - BOS/CHOCH Reversal
**Maturity**: STUB
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 3598
- Trades/Year: 515.6
- Win Rate: 55.4%
- Profit Factor: 1.05
- Sharpe Ratio: 0.26
- Max Drawdown: 12.2%
- Total PnL: $530.00
- Avg Trade: $0.15

**Period Breakdown**:
- 2018-2021: 2941 trades, PF 0.99, WR 55.4%
- 2022-2024: 656 trades, PF 1.47, WR 55.6%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : No trades
- risk_on   : 3598 trades, PF 1.05  , WR  55.4%

### K - Wick Trap Moneytaur
**Maturity**: CALIBRATED
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 821
- Trades/Year: 117.8
- Win Rate: 55.4%
- Profit Factor: 1.18
- Sharpe Ratio: 0.91
- Max Drawdown: 3.9%
- Total PnL: $561.21
- Avg Trade: $0.68

**Period Breakdown**:
- 2018-2021: 653 trades, PF 1.18, WR 56.0%
- 2022-2024: 167 trades, PF 1.17, WR 53.3%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : 118 trades, PF 1.07  , WR  53.4%
- risk_on   : 703 trades, PF 1.19  , WR  55.8%

### G - Liquidity Sweep
**Maturity**: STUB
**Status**: ⚠️ NEEDS TUNING

**Full Period (2018-2024)**:
- Trades: 3598
- Trades/Year: 515.6
- Win Rate: 55.4%
- Profit Factor: 1.05
- Sharpe Ratio: 0.26
- Max Drawdown: 12.2%
- Total PnL: $530.00
- Avg Trade: $0.15

**Period Breakdown**:
- 2018-2021: 2941 trades, PF 0.99, WR 55.4%
- 2022-2024: 656 trades, PF 1.47, WR 55.6%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : No trades
- risk_on   : 3598 trades, PF 1.05  , WR  55.4%

### A - Spring/UTAD
**Maturity**: STUB
**Status**: ❌ BROKEN

**Full Period (2018-2024)**:
- Trades: 5042
- Trades/Year: 723.0
- Win Rate: 52.2%
- Profit Factor: 0.93
- Sharpe Ratio: -0.36
- Max Drawdown: 19.3%
- Total PnL: $-1035.23
- Avg Trade: $-0.21

**Period Breakdown**:
- 2018-2021: 2939 trades, PF 0.98, WR 53.4%
- 2022-2024: 2100 trades, PF 0.85, WR 50.5%

**Regime Breakdown**:
- crisis    : No trades
- risk_off  : No trades
- neutral   : 1448 trades, PF 0.72  , WR  49.2%
- risk_on   : 3594 trades, PF 1.04  , WR  53.4%

## OPTIMIZATION RECOMMENDATIONS
--------------------------------------------------------------------------------

### P0 - CRITICAL (Broken, Fix First)
- **A** (Spring/UTAD): PF 0.93, DD 19.3% - Complete re-optimization required, ~2-3 days
- **S5** (Long Squeeze): PF 0.62, DD 3.3% - Complete re-optimization required, ~2-3 days

### P1 - HIGH (Needs Tuning)
- **S1** (Liquidity Vacuum): PF 1.34, Sharpe 1.83 - Parameter optimization, ~1-2 days
- **H** (Trap Within Trend): PF 1.18, Sharpe 0.91 - Parameter optimization, ~1-2 days
- **K** (Wick Trap Moneytaur): PF 1.18, Sharpe 0.91 - Parameter optimization, ~1-2 days
- **S4** (Funding Divergence): PF 1.08, Sharpe 0.41 - Parameter optimization, ~1-2 days
- **C** (BOS/CHOCH Reversal): PF 1.05, Sharpe 0.26 - Parameter optimization, ~1-2 days
- **G** (Liquidity Sweep): PF 1.05, Sharpe 0.26 - Parameter optimization, ~1-2 days

### P3 - LOW (Production Ready)
- **B** (Order Block Retest): PF 1.73, Sharpe 3.04 - No immediate action needed

## TIME ESTIMATES
--------------------------------------------------------------------------------

**Total Optimization Effort**:
- P0 (Broken): 2 × 2.5 days = 5.0 days
- P1 (Tuning): 6 × 1.5 days = 9.0 days
- P2 (Stubs): 0 × 4 days = 0 days
- **TOTAL: 14.0 days (~2.8 weeks)**
