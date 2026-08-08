# Migration Guide: JSON to YAML Archetype Configs

**Date:** 2026-02-04
**Purpose:** Guide for migrating from monolithic JSON to isolated YAML configs

## Overview

This guide helps you migrate from:
- **Old:** Monolithic `bull_machine_production_v10.json` with shared fusion weights
- **New:** Isolated YAML files per archetype with per-archetype fusion weights

## Why Migrate?

### Problems with Monolithic JSON
1. **Shared Fusion Weights:** All archetypes forced to use same fusion weights
2. **Conflicting Optimizations:** Optimizing one archetype affects all others
3. **Poor Version Control:** Hard to track changes per archetype
4. **No A/B Testing:** Can't test archetype variations easily
5. **Large Merge Conflicts:** Changes to any archetype affect entire file

### Benefits of Isolated YAML
1. **Per-Archetype Fusion:** Each archetype optimizes its own fusion weights
2. **Independent Optimization:** Change one archetype without affecting others
3. **Clear Git History:** See exactly what changed per archetype
4. **Easy A/B Testing:** Copy YAML file and modify
5. **Better Organization:** One file per archetype, easy to navigate

## Migration Steps

### Step 1: Understand Current Structure

**Old Config (bull_machine_production_v10.json):**
```json
{
  "version": "v10_extended_holds",
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    "thresholds": {
      "spring": {
        "fusion_threshold": 0.18,
        "pti_score_threshold": 0.05,
        "disp_atr_multiplier": 0.2,
        "wick_lower_threshold": 0.2
      },
      "order_block_retest": {
        "fusion_threshold": 0.18,
        "boms_strength_min": 0.1,
        "wyckoff_min": 0.15
      }
    }
  },
  "fusion": {
    "weights": {
      "wyckoff": 0.5,
      "liquidity": 0.2,
      "momentum": 0.2,
      "smc": 0.1
    }
  }
}
```

**Problem:** Single fusion weights for all archetypes!

### Step 2: Extract Archetype Configs

For each archetype, create a YAML file with:
1. Thresholds from `archetypes.thresholds.<name>`
2. Fusion weights (initially estimated based on archetype type)
3. Exit logic (if custom, otherwise use defaults)
4. Position sizing (use defaults)
5. Regime preferences (estimate based on archetype behavior)

**Example: Spring (A)**

```yaml
# configs/archetypes/spring.yaml
name: spring
display_name: "Spring / UTAD"
aliases: ["A", "trap_reversal", "wyckoff_spring_utad"]
direction: long

# NEW: Per-archetype fusion weights
# Spring is Wyckoff-heavy, so give it high Wyckoff weight
fusion_weights:
  wyckoff: 0.60      # High for Wyckoff detection
  liquidity: 0.20    # Medium for sweep component
  momentum: 0.10     # Low
  smc: 0.10         # Low

# Thresholds from old config
thresholds:
  fusion_threshold: 0.18
  pti_score_threshold: 0.05
  disp_atr_multiplier: 0.2
  wick_lower_threshold: 0.2

# Exit logic (use defaults)
exit_logic:
  max_hold_hours: 168
  scale_out_enabled: true
  scale_out_levels: [1.0, 2.0, 3.0]
  scale_out_pcts: [0.3, 0.4, 0.3]
  trailing_start_r: 1.0
  trailing_atr_mult: 2.0

# Position sizing (use defaults)
position_sizing:
  risk_per_trade_pct: 0.02
  max_position_size_pct: 0.1
  atr_stop_mult: 2.5

# Regime preferences (estimate)
regime_preferences:
  risk_on: 0.5       # Works but not optimal
  neutral: 1.0       # Good
  risk_off: 1.5      # Better
  crisis: 2.0        # Best (counter-trend)

allowed_regimes: []  # All regimes

description: "PTI-based spring/UTAD detection"
priority: 1
enabled: true
```

### Step 3: Initial Fusion Weight Estimates

Use these guidelines for initial fusion weights (will be optimized later):

#### Wyckoff-Heavy Archetypes (0.50-0.70)
```yaml
# Spring (A), Trap Within Trend (H), Confluence Breakout (M)
fusion_weights:
  wyckoff: 0.60
  liquidity: 0.20
  momentum: 0.10
  smc: 0.10
```

#### Liquidity-Heavy Archetypes (0.40-0.50)
```yaml
# Liquidity Vacuum (S1), Liquidity Compression (E), Funding Divergence (S4)
# Long Squeeze (S5), Liquidity Sweep (G), Wick Trap (K)
fusion_weights:
  wyckoff: 0.25
  liquidity: 0.45
  momentum: 0.15
  smc: 0.15
```

#### Momentum-Heavy Archetypes (0.35-0.45)
```yaml
# Exhaustion Reversal (F), Volume Fade Chop (S8), Failed Continuation (D)
fusion_weights:
  wyckoff: 0.20
  liquidity: 0.20
  momentum: 0.45
  smc: 0.15
```

#### SMC-Heavy Archetypes (0.35-0.40)
```yaml
# Order Block Retest (B), FVG Continuation (C)
fusion_weights:
  wyckoff: 0.25
  liquidity: 0.20
  momentum: 0.15
  smc: 0.40
```

#### Balanced Archetypes (0.25-0.30)
```yaml
# Retest Cluster (L), Whipsaw (S3)
fusion_weights:
  wyckoff: 0.25
  liquidity: 0.30
  momentum: 0.25
  smc: 0.20
```

### Step 4: Convert Old Enable Flags

**Old:**
```json
{
  "archetypes": {
    "enable_A": true,
    "enable_B": true,
    "enable_C": false,
    "enable_S1": true
  }
}
```

**New:**
Set `enabled: true/false` in each YAML file:

```yaml
# spring.yaml
enabled: true

# fvg_continuation.yaml
enabled: false
```

### Step 5: Load and Test

```python
from engine.config.archetype_config_loader import load_archetype_configs

# Load new YAML configs
configs = load_archetype_configs("configs/archetypes/")

# Validate
for name, config in configs.items():
    print(f"{name}: {config['fusion_weights']}")
```

### Step 6: Validate Schema

```bash
python engine/config/archetype_config_loader.py \
  --dir configs/archetypes/ \
  --verbose
```

Expected output:
```
Loading 16 archetype configs from configs/archetypes/
Loaded spring (Spring / UTAD) - Fusion: W=0.60 L=0.20 M=0.10 S=0.10
Loaded order_block_retest (Order Block Retest) - Fusion: W=0.30 L=0.15 M=0.15 S=0.40
...
Successfully loaded 16 archetype configs

spring               | W=0.60 L=0.20 M=0.10 S=0.10 | long    | ENABLED
order_block_retest   | W=0.30 L=0.15 M=0.15 S=0.40 | long    | ENABLED
...

All configs valid ✓
```

## Code Integration

### Option 1: Backward Compatible (Recommended for Gradual Migration)

Keep existing code working while adding new functionality:

```python
from engine.config.archetype_config_loader import (
    load_archetype_configs,
    merge_with_base_config
)
import json

# Load old JSON config
with open("configs/bull_machine_production_v10.json") as f:
    base_config = json.load(f)

# Load new YAML configs
archetype_configs = load_archetype_configs("configs/archetypes/")

# Merge (YAML takes precedence)
merged_config = merge_with_base_config(archetype_configs, base_config)

# Use merged config with existing code
strategy = BullMachineStrategy(merged_config)
```

### Option 2: Direct YAML Loading (Recommended for New Code)

Use YAML configs directly:

```python
from engine.config.archetype_config_loader import load_archetype_configs

configs = load_archetype_configs("configs/archetypes/")

# Use per-archetype fusion weights
for archetype_name in configs:
    config = configs[archetype_name]

    if not config["enabled"]:
        continue

    # Get archetype-specific fusion weights
    fusion_weights = config["fusion_weights"]

    # Calculate fusion score with archetype-specific weights
    fusion_score = calculate_fusion(
        wyckoff_score * fusion_weights["wyckoff"] +
        liquidity_score * fusion_weights["liquidity"] +
        momentum_score * fusion_weights["momentum"] +
        smc_score * fusion_weights["smc"]
    )
```

## Fusion System Integration

### Old Fusion (Global Weights)

```python
# engine/fusion/fusion_engine.py (OLD)
class FusionEngine:
    def __init__(self, config):
        # Global weights for all archetypes
        self.weights = config["fusion"]["weights"]

    def calculate_fusion(self, domain_scores):
        return (
            domain_scores["wyckoff"] * self.weights["wyckoff"] +
            domain_scores["liquidity"] * self.weights["liquidity"] +
            domain_scores["momentum"] * self.weights["momentum"] +
            domain_scores["smc"] * self.weights["smc"]
        )
```

### New Fusion (Per-Archetype Weights)

```python
# engine/fusion/fusion_engine.py (NEW)
from engine.config.archetype_config_loader import load_archetype_configs

class FusionEngine:
    def __init__(self, config_dir="configs/archetypes/"):
        # Load per-archetype weights
        self.archetype_configs = load_archetype_configs(config_dir)

    def calculate_fusion(self, archetype_name, domain_scores):
        # Get archetype-specific weights
        weights = self.archetype_configs[archetype_name]["fusion_weights"]

        # Calculate fusion with archetype weights
        return (
            domain_scores["wyckoff"] * weights["wyckoff"] +
            domain_scores["liquidity"] * weights["liquidity"] +
            domain_scores["momentum"] * weights["momentum"] +
            domain_scores["smc"] * weights["smc"]
        )
```

## Testing Migration

### 1. Validate Config Loading

```python
import pytest
from engine.config.archetype_config_loader import load_archetype_configs

def test_load_configs():
    configs = load_archetype_configs("configs/archetypes/")

    # Check all 16 archetypes loaded
    assert len(configs) == 16

    # Check spring loaded correctly
    assert "spring" in configs
    spring = configs["spring"]

    # Check fusion weights sum to 1.0
    weight_sum = sum(spring["fusion_weights"].values())
    assert 0.99 <= weight_sum <= 1.01

    # Check required fields
    assert "name" in spring
    assert "thresholds" in spring
    assert "fusion_weights" in spring
```

### 2. Compare Fusion Scores

```python
def test_fusion_parity():
    """Test that new fusion matches old fusion when using global weights."""

    # Load old config
    with open("configs/bull_machine_production_v10.json") as f:
        old_config = json.load(f)
    old_weights = old_config["fusion"]["weights"]

    # Load new config
    new_configs = load_archetype_configs("configs/archetypes/")

    # Test fusion calculation
    domain_scores = {
        "wyckoff": 0.5,
        "liquidity": 0.3,
        "momentum": 0.6,
        "smc": 0.4
    }

    # Old fusion (global weights)
    old_fusion = (
        domain_scores["wyckoff"] * old_weights["wyckoff"] +
        domain_scores["liquidity"] * old_weights["liquidity"] +
        domain_scores["momentum"] * old_weights["momentum"] +
        domain_scores["smc"] * old_weights["smc"]
    )

    # New fusion (per-archetype weights)
    spring_weights = new_configs["spring"]["fusion_weights"]
    new_fusion = (
        domain_scores["wyckoff"] * spring_weights["wyckoff"] +
        domain_scores["liquidity"] * spring_weights["liquidity"] +
        domain_scores["momentum"] * spring_weights["momentum"] +
        domain_scores["smc"] * spring_weights["smc"]
    )

    # Should be different (that's the point!)
    assert old_fusion != new_fusion

    # But both should be valid
    assert 0.0 <= old_fusion <= 1.0
    assert 0.0 <= new_fusion <= 1.0
```

### 3. Backtest Comparison

```bash
# Run backtest with old config
python bin/nautilus_backtest.py \
  --config configs/bull_machine_production_v10.json \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output results/old_config.json

# Run backtest with new YAML configs
python bin/nautilus_backtest.py \
  --config-dir configs/archetypes/ \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output results/new_config.json

# Compare results
python bin/compare_backtest_results.py \
  --old results/old_config.json \
  --new results/new_config.json
```

## Rollback Plan

If migration causes issues:

### 1. Keep Old Config Working
Don't delete `bull_machine_production_v10.json` immediately.

### 2. Feature Flag
```python
USE_YAML_CONFIGS = os.getenv("USE_YAML_CONFIGS", "false").lower() == "true"

if USE_YAML_CONFIGS:
    configs = load_archetype_configs("configs/archetypes/")
else:
    with open("configs/bull_machine_production_v10.json") as f:
        configs = json.load(f)
```

### 3. Gradual Rollout
Migrate one archetype at a time:

```python
# Hybrid approach: Use YAML for some archetypes, JSON for others
yaml_archetypes = ["spring", "order_block_retest"]
yaml_configs = load_archetype_configs("configs/archetypes/")

for archetype in yaml_archetypes:
    # Use YAML config
    config = yaml_configs[archetype]
else:
    # Use old JSON config
    config = old_json_config["archetypes"]["thresholds"][archetype]
```

## Post-Migration Optimization

After migration, optimize each archetype independently:

```bash
# Optimize spring with new per-archetype fusion weights
python bin/optimize_archetype.py \
  --archetype spring \
  --optimize-fusion-weights \
  --metric sharpe \
  --n-trials 100

# Results saved to: configs/archetypes/spring_optimized.yaml
```

## Common Issues

### Issue 1: Fusion Weights Don't Sum to 1.0

**Error:**
```
WARNING: Fusion weights for spring sum to 0.95, expected ~1.0. Normalizing...
```

**Fix:**
Adjust weights in YAML file:
```yaml
fusion_weights:
  wyckoff: 0.60
  liquidity: 0.20
  momentum: 0.10
  smc: 0.10  # Sum = 1.00 ✓
```

### Issue 2: Missing Required Field

**Error:**
```
ValueError: Missing required field 'direction' in configs/archetypes/spring.yaml
```

**Fix:**
Add required field:
```yaml
direction: long
```

### Issue 3: Invalid Regime Name

**Error:**
```
ValueError: Invalid regime 'bull_market' in configs/archetypes/spring.yaml
```

**Fix:**
Use valid regime names:
```yaml
allowed_regimes: [risk_on, neutral, risk_off, crisis]
```

## Checklist

- [ ] Extract all archetype thresholds from JSON
- [ ] Create YAML files for all 16 archetypes
- [ ] Assign initial fusion weights per archetype
- [ ] Add exit logic to each archetype
- [ ] Add regime preferences
- [ ] Validate all configs with loader tool
- [ ] Update code to load YAML configs
- [ ] Run comparison backtests
- [ ] Document changes in git
- [ ] Optimize fusion weights per archetype
- [ ] Archive old JSON config

## Timeline

**Week 1:** Extract and create YAML files
**Week 2:** Integrate with code, test backward compatibility
**Week 3:** Run validation backtests
**Week 4:** Optimize per-archetype fusion weights
**Week 5:** Full production rollout

## Support

For issues or questions:
1. Check **SCHEMA.md** for config format
2. Check **README.md** for usage examples
3. Run validation: `python engine/config/archetype_config_loader.py`
4. Check logs for detailed error messages

---

**Created:** 2026-02-04
**Version:** 1.0
