# Quick Test Configs - Phase 1 Threshold Validation

## Purpose
These configs test varying strictness levels for S2 (Failed Rally) and S5 (Long Squeeze) archetypes during the 2022 bear market.

## Config Files

| Config | Expected Trades | S2 Fusion | S2 Wick | S2 RSI | S2 Vol Z | S5 Fusion | S5 Funding Z | S5 RSI | S5 Liquidity |
|--------|----------------|-----------|---------|--------|----------|-----------|--------------|--------|--------------|
| **ultra_strict.json** | 5-10 | 0.55 | 3.0 | 75 | 0.3 | 0.55 | 2.0 | 75 | 0.15 |
| **strict.json** | 15-20 | 0.50 | 2.5 | 72 | 0.4 | 0.50 | 1.7 | 72 | 0.18 |
| **moderate.json** | 25-35 | 0.45 | 2.2 | 70 | 0.5 | 0.45 | 1.4 | 70 | 0.22 |
| **relaxed.json** | 40-50 | 0.40 | 2.0 | 68 | 0.6 | 0.40 | 1.2 | 68 | 0.25 |
| **ultra_relaxed.json** | 60+ | 0.36 | 1.8 | 65 | 0.8 | 0.35 | 1.0 | 65 | 0.30 |

## Configuration Details

### All Configs Include:
- **Enabled Archetypes**: S2 (Failed Rally) + S5 (Long Squeeze)
- **Disabled Archetypes**: All others (A-M, S1, S3-S4, S6-S8)
- **Regime Override**: 2022 = "risk_off"
- **Routing Weights** (risk_off): Failed Rally = 2.0, Long Squeeze = 2.2

### Parameter Definitions

#### S2 (Failed Rally):
- **fusion_threshold**: Minimum structural fusion score (higher = stricter)
- **wick_ratio_min**: Minimum upper wick to body ratio (higher = stricter)
- **rsi_min**: Minimum RSI at rejection (higher = stricter)
- **vol_z_max**: Maximum volume z-score (lower = stricter)

#### S5 (Long Squeeze):
- **fusion_threshold**: Minimum structural fusion score (higher = stricter)
- **funding_z_min**: Minimum funding rate z-score (higher = stricter)
- **rsi_min**: Minimum RSI threshold (higher = stricter)
- **liquidity_max**: Maximum liquidity score (lower = stricter)

## Usage

Run backtests with each config to validate:
1. Trade count matches expected range
2. Quality of matches (win rate, avg R-multiple)
3. Pattern distribution across 2022
4. Optimal balance between selectivity and coverage

```bash
# Example backtest command
python bin/backtest_knowledge_v2.py \
  --config configs/quick_test/moderate.json \
  --symbol BTCUSDT \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --timeframe 1h
```

## Expected Outcomes

- **ultra_strict**: Very high quality, minimal coverage
- **strict**: High quality, low coverage
- **moderate**: Balanced quality and coverage (target zone)
- **relaxed**: Lower quality, good coverage
- **ultra_relaxed**: Lower quality, maximum coverage (baseline)

## Target Selection
The **moderate** config (25-35 trades) is the recommended starting point for Phase 1 optimization, balancing signal quality with sufficient sample size.
