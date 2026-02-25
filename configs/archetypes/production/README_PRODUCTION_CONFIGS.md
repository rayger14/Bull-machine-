# Production Archetype Configurations

**Created:** 2025-12-15
**Purpose:** Production-ready configs for all 16 archetypes with proper thresholds
**Location:** `/configs/archetypes/production/`

## Overview

This directory contains production-ready JSON configuration files for all 16 archetypes. These configs replace the smoke test configs (which had `thresholds=0` blocking all signals) with proper production thresholds designed to allow quality signals while maintaining selectivity.

## Key Design Principles

1. **Relaxed Thresholds**: `fusion_threshold` set to 0.22-0.40 (vs 0.0 in smoke tests) to allow signals
2. **All 6 Engines Enabled**: Every config has all domain engines active for maximum boost
3. **Regime-Aware Routing**: Weight multipliers vary by market regime for each archetype
4. **Production Risk Management**: Appropriate `base_risk_pct`, `atr_stop_mult`, and `cooldown_bars`
5. **Archetype-Specific Parameters**: Each config tailored to its detection strategy

## Configuration Files (16 Total)

### Bull Market Archetypes (11)

| Code | Name | File | Direction | Fusion Threshold | Key Focus |
|------|------|------|-----------|------------------|-----------|
| A | Spring | `archetype_a_spring.json` | long | 0.35 | Wyckoff springs at accumulation |
| B | Order Block Retest | `archetype_b_order_block_retest.json` | long | 0.30 | SMC order block continuations |
| C | Wick Trap | `archetype_c_wick_trap.json` | long | 0.30 | Bullish wick rejections |
| D | Failed Continuation | `archetype_d_failed_continuation.json` | long | 0.35 | Failed bearish pattern reversals |
| E | Volume Exhaustion | `archetype_e_volume_exhaustion.json` | long | 0.35 | Selling climax reversals |
| F | Exhaustion Reversal | `archetype_f_exhaustion_reversal.json` | long | 0.35 | Momentum exhaustion longs |
| G | Liquidity Sweep | `archetype_g_liquidity_sweep.json` | long | 0.30 | Liquidity grab reversals |
| H | Momentum Continuation | `archetype_h_momentum_continuation.json` | long | 0.28 | Uptrend pullback entries |
| K | Trap Within Trend | `archetype_k_trap_within_trend.json` | long | 0.32 | False breakdown in uptrends |
| L | Retest Cluster | `archetype_l_retest_cluster.json` | long | 0.38 | Multi-factor confluence zones |
| M | Confluence Breakout | `archetype_m_confluence_breakout.json` | long | 0.40 | High-quality breakouts |

### Bear Market Archetypes (3)

| Code | Name | File | Direction | Fusion Threshold | Key Focus |
|------|------|------|-----------|------------------|-----------|
| S1 | Liquidity Vacuum | `archetype_s1_liquidity_vacuum.json` | long | 0.54 | Crisis capitulation reversals |
| S4 | Funding Divergence | `archetype_s4_funding_divergence.json` | long | 0.78 | Negative funding + resilience |
| S5 | Long Squeeze | `archetype_s5_long_squeeze.json` | short | 0.45 | Overheated long liquidations |

### Neutral Archetypes (2)

| Code | Name | File | Direction | Fusion Threshold | Key Focus |
|------|------|------|-----------|------------------|-----------|
| S3 | Whipsaw | `archetype_s3_whipsaw.json` | neutral | 0.25 | Range-bound mean reversion |
| S8 | Volume Fade Chop | `archetype_s8_volume_fade_chop.json` | neutral | 0.22 | Low-volume fade trades |

## Threshold Strategy by Archetype Type

### Premium Quality (High Selectivity)
- **S1, S4, M**: `fusion_threshold >= 0.40`
- Low frequency, high quality
- Strongest confluence requirements
- Best for crisis/high-conviction setups

### Standard Quality (Balanced)
- **A, D, E, F, L**: `fusion_threshold = 0.35-0.38`
- Medium frequency
- Good balance of quality and quantity
- Core reversal strategies

### Active Trading (Higher Frequency)
- **B, C, G, H, K**: `fusion_threshold = 0.28-0.32`
- Higher trade frequency
- Continuation and trap patterns
- Better in trending markets

### Scalping (Opportunistic)
- **S3, S8**: `fusion_threshold = 0.22-0.25`
- Highest frequency
- Lower risk per trade
- Neutral regime specialists

## Domain Engine Configuration

All 16 configs have identical engine settings:

```json
{
  "feature_flags": {
    "enable_wyckoff": true,
    "enable_smc": true,
    "enable_temporal": true,
    "enable_hob": true,
    "enable_fusion": true,
    "enable_macro": true
  }
}
```

This ensures maximum signal boost from all 6 domain engines.

## Regime Routing Examples

### Bull Archetypes (A-M)
- **Risk-On**: 1.0-2.0x weight (favored in bull markets)
- **Neutral**: 1.0x weight (baseline)
- **Risk-Off**: 0.5-0.8x weight (reduced)
- **Crisis**: 0.2-0.5x weight (minimal)

### Bear Archetypes (S1, S4, S5)
- **Risk-On**: 0.0-0.5x weight (minimal/disabled)
- **Neutral**: 0.5-1.0x weight (moderate)
- **Risk-Off**: 1.0-2.2x weight (favored)
- **Crisis**: 1.5-2.5x weight (maximum)

### Neutral Archetypes (S3, S8)
- **Risk-On**: 0.2-0.3x weight (minimal)
- **Neutral**: 1.5-2.0x weight (maximum)
- **Risk-Off**: 0.5-0.8x weight (moderate)
- **Crisis**: 0.1x weight (disabled)

## Risk Management

### Conservative (Bear Archetypes)
- `base_risk_pct`: 0.015 (1.5%)
- `max_position_size_pct`: 0.15 (15%)
- `atr_stop_mult`: 2.2-3.0

### Standard (Bull Archetypes)
- `base_risk_pct`: 0.02 (2%)
- `max_position_size_pct`: 0.20 (20%)
- `atr_stop_mult`: 1.8-2.5

### Aggressive (Scalping)
- `base_risk_pct`: 0.008-0.01 (0.8-1%)
- `max_position_size_pct`: 0.10 (10%)
- `atr_stop_mult`: 1.0-1.5

## Usage Instructions

### Single Archetype Backtest
```bash
python bin/backtest_knowledge_v2.py \
  --config configs/archetypes/production/archetype_a_spring.json \
  --pair BTCUSDT \
  --timeframe 1h \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

### Portfolio Backtest (Multiple Archetypes)
Create a unified config with multiple archetypes enabled:
```json
{
  "version": "portfolio_bull_v1",
  "profile": "Bull Market Portfolio - All 11 Bull Archetypes",
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    "enable_C": true,
    ...
  }
}
```

### Quick Validation Test
```bash
# Test S1 on 2022 bear market
python bin/backtest_knowledge_v2.py \
  --config configs/archetypes/production/archetype_s1_liquidity_vacuum.json \
  --pair BTCUSDT \
  --timeframe 1h \
  --start-date 2022-01-01 \
  --end-date 2022-12-31
```

## Expected Behavior

### High-Quality Archetypes
- **S1**: 40-60 trades/year, 50-60% win rate
- **S4**: 12 trades/year, PF 2.22, 55.7% win rate
- **S5**: 9 trades/year, PF 1.86, 55.6% win rate
- **M**: 8-15 trades/year, high win rate expected

### Medium-Frequency Archetypes
- **A, D, E, F, L**: 20-40 trades/year
- **B, G, K**: 30-50 trades/year
- **H**: 40-60 trades/year (trend follower)

### Active Trading Archetypes
- **C**: 50-80 trades/year (wick traps)
- **S3**: 60-100 trades/year (range scalps)
- **S8**: 80-120 trades/year (fade trades)

## Validation Checklist

Before deploying any config to live trading:

- [ ] Run backtest on full 2020-2024 dataset
- [ ] Verify trade count is reasonable (not 0, not excessive)
- [ ] Check win rate is above 45%
- [ ] Ensure profit factor > 1.2
- [ ] Validate max drawdown < 30%
- [ ] Test on out-of-sample period (2024 H2)
- [ ] Review regime distribution of trades
- [ ] Confirm no config parsing errors

## Next Steps

1. **Individual Testing**: Run each archetype individually to establish baselines
2. **Regime Validation**: Test bear archetypes on 2022, bull on 2021, neutral on 2023
3. **Portfolio Construction**: Combine non-correlated archetypes for portfolio effect
4. **Optimization**: Run Optuna on promising archetypes to refine thresholds
5. **Walk-Forward**: Validate on rolling windows to test stability

## Troubleshooting

### No Trades Generated
- Check `fusion_threshold` - may be too high
- Verify domain engines are enabled
- Review regime routing weights
- Ensure feature data is available

### Too Many Trades
- Increase `fusion_threshold`
- Add `cooldown_bars`
- Increase `min_confidence`
- Tighten archetype-specific thresholds

### Low Win Rate
- Review false positives in trade log
- Tighten detection criteria
- Add confluence requirements
- Review exit strategy

## References

**Template Configs:**
- `/configs/variants/s1_full.json` - S1 production reference
- `/configs/variants/s4_full.json` - S4 production reference
- `/configs/variants/s5_full.json` - S5 production reference

**Documentation:**
- `ARCHETYPE_NAME_MAPPING_REFERENCE.md` - Archetype code reference
- `ARCHETYPE_MODEL_IMPLEMENTATION.md` - Technical implementation
- `docs/ARCHETYPE_SYSTEMS_PRODUCTION_GUIDE.md` - Production deployment

**Optimization:**
- `bin/optimize_archetype_regime_aware.py` - Single archetype optimization
- `bin/optuna_parallel_archetypes_v2.py` - Multi-archetype parallel optimization

---

**Status:** Production Ready
**Last Updated:** 2025-12-15
**Maintainer:** Bull Machine Development Team
