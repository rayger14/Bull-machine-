# Auto-Generated Regime-Specific Configurations

This directory contains scripts and configurations for generating regime-specific trading configs based on historical backtest performance.

## Overview

The system extracts optimal entry thresholds from winning trades in different market regimes (neutral, risk_on, risk_off, crisis) and generates customized configs for each regime.

## Scripts

### 1. `bin/extract_thresholds.py`

Analyzes backtest results to extract regime-specific thresholds.

**Input:**
- `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` - Feature data with regime column
- `results/validation/bear_2022_separated.csv` - Bear market backtest results
- `results/validation/bull_2024_separated.csv` - Bull market backtest results

**Output:**
- `configs/auto/hmm_thresholds.json` - Extracted thresholds for each regime

**Usage:**
```bash
python bin/extract_thresholds.py
```

**What it does:**
- Loads backtest results and filters for winning trades
- Maps trades to their corresponding market regime
- Calculates percentile-based thresholds:
  - `min_liquidity`: 10th percentile of liquidity_score
  - `fusion_threshold`: 15th percentile of fusion_score
  - `volume_z_min`: 20th percentile of volume_z
  - `funding_z_min`: 80th percentile for short entries (default: 1.5)

### 2. `configs/auto/generate_regime_config.py`

Generates regime-specific config files from base config and extracted thresholds.

**Input:**
- `configs/mvp/mvp_bull_market_v1.json` - Base configuration template
- `configs/auto/hmm_thresholds.json` - Extracted thresholds

**Output:**
- `configs/auto/config_regime_0.json` - Neutral regime config
- `configs/auto/config_regime_1.json` - Risk-on regime config
- `configs/auto/config_regime_2.json` - Risk-off regime config
- `configs/auto/config_regime_3.json` - Crisis regime config
- `configs/auto/config_index.json` - Index of generated configs

**Usage:**
```bash
python configs/auto/generate_regime_config.py
```

**What it does:**
- Loads base config and threshold data
- For each regime:
  - Applies regime-specific thresholds to all archetype patterns
  - Sets `regime_override` to force the specific regime
  - Adds metadata about extraction sample size and parameters
- Creates an index file for easy reference

## Workflow

1. **Extract Thresholds:**
   ```bash
   python bin/extract_thresholds.py
   ```

2. **Generate Configs:**
   ```bash
   python configs/auto/generate_regime_config.py
   ```

3. **Review Generated Configs:**
   ```bash
   cat configs/auto/config_index.json
   ```

4. **Run Backtests:**
   ```bash
   python bin/backtest_knowledge_v2.py \
     --config configs/auto/config_regime_0.json \
     --asset BTC \
     --timeframe 1H \
     --start 2022-01-01 \
     --end 2024-12-31
   ```

## Generated Config Structure

Each regime-specific config includes:

```json
{
  "version": "auto_regime_0_neutral",
  "description": "Auto-generated config for neutral regime...",
  "regime_classifier": {
    "regime_override": {
      "0": "neutral",
      "_comment": "Force all periods to neutral for regime-specific backtesting"
    }
  },
  "archetypes": {
    "thresholds": {
      "min_liquidity": 0.0,
      "trap_within_trend": {
        "fusion_threshold": 0.459,
        "volume_z_min": -0.299,
        ...
      }
    }
  },
  "_threshold_metadata": {
    "regime_id": 0,
    "regime_name": "neutral",
    "sample_size": 32,
    "fusion_threshold": 0.459,
    ...
  }
}
```

## Current Status

Based on the latest extraction (2025-11-16):

| Regime ID | Name      | Winning Trades | Fusion Threshold | Volume Z Min |
|-----------|-----------|----------------|------------------|--------------|
| 0         | neutral   | 32             | 0.459            | -0.299       |
| 1         | risk_on   | 0              | 0.400 (default)  | -1.000       |
| 2         | risk_off  | 0              | 0.400 (default)  | -1.000       |
| 3         | crisis    | 0              | 0.400 (default)  | -1.000       |

**Note:** Only neutral regime has sufficient winning trades. Other regimes use conservative defaults.

## Customization

To modify the extraction logic:

1. **Change percentiles:** Edit `calculate_regime_thresholds()` in `extract_thresholds.py`
2. **Add new thresholds:** Modify the threshold calculation and apply logic
3. **Use different base config:** Change `base_config_path` in `generate_regime_config.py`

## Troubleshooting

**Issue:** No winning trades for some regimes

- This is expected if backtests were run with `regime_override` forcing specific regimes
- Solution: Run backtests across full date range with dynamic regime detection

**Issue:** Liquidity scores are 0.0

- Check if liquidity_score feature is being calculated in feature pipeline
- Verify feature is included in backtest output

**Issue:** Config generation fails

- Ensure `extract_thresholds.py` ran successfully first
- Check that `hmm_thresholds.json` exists and is valid JSON

## Next Steps

1. **Collect More Data:** Run backtests without regime override to get samples across all regimes
2. **Refine Percentiles:** Experiment with different percentile thresholds (10/15/20 vs 5/10/15)
3. **Add Regime-Specific Features:** Include regime-specific indicators (VIX, funding rates)
4. **Validate Performance:** Compare regime-specific configs vs. unified config
