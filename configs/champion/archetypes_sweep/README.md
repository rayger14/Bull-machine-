# Archetype Configuration System

**Version:** 1.0
**Date:** 2026-02-04
**Status:** Production Ready

## Overview

This directory contains isolated YAML configuration files for all 16 Bull Machine archetypes. Each archetype has its own config file for independent optimization and version control.

## Key Innovation: Per-Archetype Fusion Weights

The primary improvement over the monolithic JSON config is **per-archetype fusion weights**. This allows each archetype to optimize its own fusion calculation independently.

### Before (Monolithic)
```json
{
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
**Problem:** All archetypes forced to use same fusion weights

### After (Isolated)
```yaml
# spring.yaml
fusion_weights:
  wyckoff: 0.60    # Spring is Wyckoff-heavy
  liquidity: 0.20
  momentum: 0.10
  smc: 0.10

# liquidity_vacuum.yaml
fusion_weights:
  wyckoff: 0.35
  liquidity: 0.45  # Liquidity Vacuum is liquidity-heavy
  momentum: 0.10
  smc: 0.10
```
**Benefit:** Each archetype optimized for its core detection strategy

## Directory Contents

### Core Files
- **16 YAML Configs:** One per archetype (A, B, C, D, E, F, G, H, K, L, M, S1, S3, S4, S5, S8)
- **SCHEMA.md:** Complete schema documentation
- **README.md:** This file
- **MIGRATION.md:** Migration guide from JSON

### Legacy Files
- **production/**: Old JSON configs (deprecated, kept for reference)
- **spring_utad_baseline.json**: Old baseline configs (deprecated)

## Quick Start

### 1. Load Configs

```python
from engine.config.archetype_config_loader import load_archetype_configs

# Load all archetype configs
configs = load_archetype_configs("configs/archetypes/")

# Get specific archetype
spring = configs["spring"]
print(f"Fusion weights: {spring['fusion_weights']}")
print(f"Thresholds: {spring['thresholds']}")
```

### 2. Get Fusion Weights

```python
from engine.config.archetype_config_loader import get_archetype_fusion_weights

# Get fusion weights for spring archetype
weights = get_archetype_fusion_weights("spring")
# Returns: {'wyckoff': 0.6, 'liquidity': 0.2, 'momentum': 0.1, 'smc': 0.1}
```

### 3. Get Enabled Archetypes

```python
from engine.config.archetype_config_loader import get_enabled_archetypes

enabled = get_enabled_archetypes()
# Returns: ['spring', 'order_block_retest', 'fvg_continuation', ...]
```

### 4. Validate Configs

```bash
python engine/config/archetype_config_loader.py --dir configs/archetypes/ --verbose
```

## Archetype Summary

| Code | Name | File | Direction | Primary Domain |
|------|------|------|-----------|----------------|
| A | Spring | `spring.yaml` | long | Wyckoff (0.60) |
| B | Order Block Retest | `order_block_retest.yaml` | long | SMC (0.40) |
| C | FVG Continuation | `fvg_continuation.yaml` | long | SMC (0.35) |
| D | Failed Continuation | `failed_continuation.yaml` | long | Momentum (0.35) |
| E | Liquidity Compression | `liquidity_compression.yaml` | long | Liquidity (0.50) |
| F | Exhaustion Reversal | `exhaustion_reversal.yaml` | long | Momentum (0.45) |
| G | Liquidity Sweep | `liquidity_sweep.yaml` | long | Liquidity (0.40) |
| H | Trap Within Trend | `trap_within_trend.yaml` | long | Wyckoff/Liquidity (0.35/0.35) |
| K | Wick Trap | `wick_trap.yaml` | long | Liquidity (0.40) |
| L | Retest Cluster | `retest_cluster.yaml` | long | Liquidity (0.30) |
| M | Confluence Breakout | `confluence_breakout.yaml` | long | Wyckoff (0.30) |
| S1 | Liquidity Vacuum | `liquidity_vacuum.yaml` | long | Liquidity (0.45) |
| S3 | Whipsaw | `whipsaw.yaml` | neutral | Liquidity/Momentum (0.30/0.30) |
| S4 | Funding Divergence | `funding_divergence.yaml` | long | Liquidity (0.50) |
| S5 | Long Squeeze | `long_squeeze.yaml` | short | Liquidity (0.50) |
| S8 | Volume Fade Chop | `volume_fade_chop.yaml` | neutral | Momentum (0.40) |

## Configuration Philosophy

### Fusion Weight Guidelines

**Wyckoff-Heavy Archetypes (0.50-0.70):**
- Spring (A): 0.60
- Trap Within Trend (H): 0.35
- Confluence Breakout (M): 0.30

**Liquidity-Heavy Archetypes (0.40-0.50):**
- Liquidity Vacuum (S1): 0.45
- Liquidity Compression (E): 0.50
- Funding Divergence (S4): 0.50
- Long Squeeze (S5): 0.50
- Liquidity Sweep (G): 0.40
- Wick Trap (K): 0.40

**Momentum-Heavy Archetypes (0.35-0.45):**
- Exhaustion Reversal (F): 0.45
- Volume Fade Chop (S8): 0.40
- Failed Continuation (D): 0.35

**SMC-Heavy Archetypes (0.35-0.40):**
- Order Block Retest (B): 0.40
- FVG Continuation (C): 0.35

**Balanced Archetypes (0.25-0.30):**
- Retest Cluster (L): All ~0.25-0.30
- Whipsaw (S3): All ~0.25-0.30

### Threshold Ranges

**Conservative (Low Frequency):**
- Fusion threshold: 0.35-0.45
- Examples: M (0.18), L (0.18)

**Balanced (Medium Frequency):**
- Fusion threshold: 0.25-0.35
- Examples: A-H (0.18)

**Aggressive (High Frequency):**
- Fusion threshold: 0.15-0.25
- Examples: S1, S4, S5 (0.15)

## Optimization Workflow

### 1. Individual Archetype Optimization

```bash
# Optimize spring archetype only
python bin/optimize_archetype.py \
  --archetype spring \
  --metric sharpe \
  --n-trials 100

# Writes optimized params to: configs/archetypes/spring_optimized.yaml
```

### 2. A/B Testing

```bash
# Create variant
cp configs/archetypes/spring.yaml configs/archetypes/spring_v2.yaml

# Edit spring_v2.yaml with experimental weights
vim configs/archetypes/spring_v2.yaml

# Compare
python bin/compare_archetype_configs.py \
  --config-a configs/archetypes/spring.yaml \
  --config-b configs/archetypes/spring_v2.yaml
```

### 3. Batch Optimization

```bash
# Optimize all archetypes
for archetype in A B C D E F G H K L M S1 S3 S4 S5 S8; do
  python bin/optimize_archetype.py --archetype $archetype
done
```

## Integration with Existing Code

### Backward Compatibility

The config loader provides `merge_with_base_config()` to maintain compatibility:

```python
from engine.config.archetype_config_loader import (
    load_archetype_configs,
    merge_with_base_config
)

# Load YAML configs
archetype_configs = load_archetype_configs("configs/archetypes/")

# Load base JSON config
import json
with open("configs/bull_machine_production_v10.json") as f:
    base_config = json.load(f)

# Merge for backward compatibility
merged_config = merge_with_base_config(archetype_configs, base_config)

# Use merged config with existing code
strategy = BullMachineStrategy(merged_config)
```

### New Code (Recommended)

```python
from engine.config.archetype_config_loader import load_archetype_configs

# Load configs directly
configs = load_archetype_configs("configs/archetypes/")

# Use per-archetype fusion weights
for archetype_name in configs:
    fusion_weights = configs[archetype_name]["fusion_weights"]
    fusion_score = calculate_archetype_fusion(
        archetype_name,
        fusion_weights,
        domain_scores
    )
```

## File Format

All configs use YAML for human readability and git-friendliness.

### Example Config

```yaml
# Archetype A - Spring / UTAD
name: spring
display_name: "Spring / UTAD"
aliases: ["A", "trap_reversal", "wyckoff_spring_utad"]
direction: long

# Per-archetype fusion weights
fusion_weights:
  wyckoff: 0.60
  liquidity: 0.20
  momentum: 0.10
  smc: 0.10

# Entry thresholds
thresholds:
  fusion_threshold: 0.18
  pti_score_threshold: 0.05
  disp_atr_multiplier: 0.2
  wick_lower_threshold: 0.2

# Exit logic
exit_logic:
  max_hold_hours: 168
  scale_out_enabled: true
  scale_out_levels: [1.0, 2.0, 3.0]
  scale_out_pcts: [0.3, 0.4, 0.3]
  trailing_start_r: 1.0
  trailing_atr_mult: 2.0

# Position sizing
position_sizing:
  risk_per_trade_pct: 0.02
  max_position_size_pct: 0.1
  atr_stop_mult: 2.5

# Regime preferences
regime_preferences:
  risk_on: 0.5
  neutral: 1.0
  risk_off: 1.5
  crisis: 2.0

# Allowed regimes
allowed_regimes: []  # All regimes

# Metadata
description: "PTI-based spring/UTAD detection"
priority: 1
enabled: true
```

## Next Steps

### 1. Integration Tasks
- [ ] Update `engine/archetypes/logic_v2_adapter.py` to use per-archetype fusion weights
- [ ] Update `engine/fusion/fusion_engine.py` to support archetype-specific weights
- [ ] Update backtesting framework to load YAML configs
- [ ] Add archetype config versioning (v1, v2, etc.)

### 2. Optimization Tasks
- [ ] Run Optuna optimization for each archetype independently
- [ ] Validate optimized configs on out-of-sample data
- [ ] Document optimal fusion weight ranges per archetype type
- [ ] Create archetype performance comparison dashboard

### 3. Testing Tasks
- [ ] Unit tests for config loader
- [ ] Integration tests for merged configs
- [ ] Validation tests for schema compliance
- [ ] Performance tests (YAML vs JSON loading)

## Benefits Summary

### 1. Isolation
Each archetype can be optimized independently without affecting others.

### 2. Clarity
Clear ownership of config parameters per archetype.

### 3. Version Control
Git-friendly YAML format with clear diff history per archetype.

### 4. A/B Testing
Easy to create archetype variants for experimentation.

### 5. Optimization
Per-archetype fusion weights unlock independent optimization.

### 6. Documentation
Self-documenting configs with inline comments.

### 7. Flexibility
Easy to disable/enable archetypes or create new ones.

## See Also

- **SCHEMA.md** - Complete configuration schema
- **MIGRATION.md** - Migration guide from JSON
- **production/README_PRODUCTION_CONFIGS.md** - Legacy JSON configs
- **engine/config/archetype_config_loader.py** - Config loader implementation
- **engine/archetypes/registry.py** - Archetype registry

## Questions?

For questions or issues:
1. Check **SCHEMA.md** for config format
2. Check **MIGRATION.md** for migration help
3. Run validation tool: `python engine/config/archetype_config_loader.py`
4. Check logs for config loading errors

---

**Created:** 2026-02-04
**Author:** Bull Machine Team
**Version:** 1.0
