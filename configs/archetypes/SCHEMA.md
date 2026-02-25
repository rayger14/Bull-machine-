# Archetype Configuration Schema

**Version:** 1.0
**Date:** 2026-02-04
**Purpose:** Define schema for isolated archetype YAML configs

## Overview

Each archetype has its own YAML file in `configs/archetypes/` for:
- **Isolation**: Independent optimization without affecting other archetypes
- **Clarity**: Clear configuration ownership per archetype
- **Versioning**: Easy to track changes per archetype in git
- **A/B Testing**: Simple to create archetype variations

## Directory Structure

```
configs/archetypes/
├── spring.yaml                    # Archetype A
├── order_block_retest.yaml        # Archetype B
├── fvg_continuation.yaml          # Archetype C
├── failed_continuation.yaml       # Archetype D
├── liquidity_compression.yaml     # Archetype E
├── exhaustion_reversal.yaml       # Archetype F
├── liquidity_sweep.yaml           # Archetype G
├── trap_within_trend.yaml         # Archetype H
├── wick_trap.yaml                 # Archetype K
├── retest_cluster.yaml            # Archetype L
├── confluence_breakout.yaml       # Archetype M
├── liquidity_vacuum.yaml          # Archetype S1
├── whipsaw.yaml                   # Archetype S3
├── funding_divergence.yaml        # Archetype S4
├── long_squeeze.yaml              # Archetype S5
├── volume_fade_chop.yaml          # Archetype S8
└── production/                    # Legacy JSON configs (deprecated)
```

## Schema Definition

### Required Fields

```yaml
# Core Identity
name: string                       # Canonical name (snake_case)
display_name: string               # Human-readable name
direction: "long" | "short" | "neutral"

# Entry Thresholds
thresholds:
  fusion_threshold: float          # Minimum fusion score (0.0-1.0)
  # ... archetype-specific thresholds
```

### Optional Fields (with defaults)

```yaml
# Aliases for backward compatibility
aliases: [string, ...]             # ["A", "spring", ...]

# NEW: Per-archetype fusion weights
fusion_weights:
  wyckoff: float                   # Default: 0.25 (sum must = 1.0)
  liquidity: float                 # Default: 0.25
  momentum: float                  # Default: 0.25
  smc: float                       # Default: 0.25

# Exit Logic
exit_logic:
  max_hold_hours: int              # Default: 168 (7 days)
  scale_out_enabled: bool          # Default: true
  scale_out_levels: [float, ...]   # R-multiples [1.0, 2.0, 3.0]
  scale_out_pcts: [float, ...]     # Portions [0.3, 0.4, 0.3]
  trailing_start_r: float          # Default: 1.0
  trailing_atr_mult: float         # Default: 2.0

# Position Sizing
position_sizing:
  risk_per_trade_pct: float        # Default: 0.02 (2%)
  max_position_size_pct: float     # Default: 0.1 (10%)
  atr_stop_mult: float             # Default: 2.5

# Regime Preferences (multipliers)
regime_preferences:
  risk_on: float                   # Default: 1.0 (neutral)
  neutral: float                   # Default: 1.0
  risk_off: float                  # Default: 1.0
  crisis: float                    # Default: 1.0

# Regime Filtering
allowed_regimes: [string, ...]     # Default: [] (all allowed)
                                   # Options: ["risk_on", "neutral", "risk_off", "crisis"]

# Metadata
description: string                # Default: ""
priority: int                      # Default: 100 (lower = higher priority)
enabled: bool                      # Default: true
```

## Field Details

### 1. Core Identity

#### `name` (required)
- **Type:** string
- **Format:** snake_case
- **Purpose:** Canonical identifier for code
- **Example:** `"spring"`, `"order_block_retest"`

#### `display_name` (required)
- **Type:** string
- **Purpose:** Human-readable name for UI/logs
- **Example:** `"Spring / UTAD"`, `"Order Block Retest"`

#### `direction` (required)
- **Type:** enum
- **Values:** `"long"`, `"short"`, `"neutral"`
- **Purpose:** Trade direction bias
- **Note:** Only `"long_squeeze"` is currently `"short"`

#### `aliases`
- **Type:** list of strings
- **Purpose:** Backward compatibility with legacy letter codes
- **Example:** `["A", "spring", "wyckoff_spring_utad"]`

### 2. Fusion Weights (NEW - Key Innovation)

```yaml
fusion_weights:
  wyckoff: 0.60        # Primary domain
  liquidity: 0.20      # Secondary
  momentum: 0.10       # Tertiary
  smc: 0.10           # Supporting
```

**Purpose:** Per-archetype fusion calculation for independent optimization

**Guidelines:**
- **Sum must equal 1.0** (will auto-normalize if not)
- Higher weight = more influence on fusion score
- Reflects archetype's core detection strategy

**Domain Weight Ranges by Archetype Type:**

| Archetype Type | Primary Domain | Typical Weights |
|----------------|----------------|-----------------|
| Wyckoff (A, H) | wyckoff | 0.50-0.70 |
| Liquidity (S1, S4, G, K) | liquidity | 0.40-0.50 |
| Momentum (F, D) | momentum | 0.40-0.50 |
| SMC (B, C, M) | smc | 0.35-0.50 |
| Balanced (L, E) | mixed | 0.25-0.35 each |

### 3. Entry Thresholds

```yaml
thresholds:
  fusion_threshold: 0.18           # Universal threshold
  # Archetype-specific thresholds below:
  pti_score_threshold: 0.05        # Example: Spring
  boms_strength_min: 0.1           # Example: Order Block
  smc_score_min: 0.15              # Example: FVG
  # ... etc
```

**Common Thresholds:**
- `fusion_threshold`: Minimum fusion score (0.15-0.40 typical)
- `liquidity_score_min`: Minimum liquidity score
- `wyckoff_min`: Minimum Wyckoff score
- `momentum_min`: Minimum momentum score
- `smc_score_min`: Minimum SMC score

**Archetype-Specific:**
Each archetype has unique thresholds. See individual YAML files.

### 4. Exit Logic

```yaml
exit_logic:
  max_hold_hours: 168              # Maximum hold time
  scale_out_enabled: true          # Enable scale-out
  scale_out_levels: [1.0, 2.0, 3.0]   # R-multiples to scale
  scale_out_pcts: [0.3, 0.4, 0.3]     # % to exit at each level
  trailing_start_r: 1.0            # Start trailing at 1R
  trailing_atr_mult: 2.0           # Trail with 2x ATR
```

**Scale-Out Example:**
- At 1.0R profit: Exit 30% of position
- At 2.0R profit: Exit 40% of remaining (28% total)
- At 3.0R profit: Exit 30% of remaining (12.6% total)
- Remaining 29.4% rides with trailing stop

### 5. Position Sizing

```yaml
position_sizing:
  risk_per_trade_pct: 0.02         # 2% risk per trade
  max_position_size_pct: 0.1       # Max 10% of portfolio
  atr_stop_mult: 2.5               # Stop at 2.5x ATR
```

**Risk Calculation:**
```
position_size = (account_equity * risk_per_trade_pct) / (atr * atr_stop_mult)
position_size = min(position_size, account_equity * max_position_size_pct)
```

### 6. Regime Preferences

```yaml
regime_preferences:
  risk_on: 1.3       # 30% boost in risk-on
  neutral: 1.0       # No change in neutral
  risk_off: 0.7      # 30% penalty in risk-off
  crisis: 0.5        # 50% penalty in crisis
```

**Purpose:** Weight multipliers for regime routing

**Example:**
- Base fusion score: 0.40
- In risk_on: 0.40 * 1.3 = 0.52 (boosted)
- In crisis: 0.40 * 0.5 = 0.20 (penalized)

**Typical Patterns:**
- **Bull archetypes (A-M):** High in risk_on, low in crisis
- **Bear archetypes (S1, S3, S4):** High in crisis/risk_off, low in risk_on
- **Neutral archetypes (S3, S8):** High in neutral, low in trending

### 7. Allowed Regimes

```yaml
allowed_regimes: [risk_on, neutral]  # Only fire in these regimes
# OR
allowed_regimes: []                  # Fire in all regimes
```

**Purpose:** Hard filter - archetype won't fire outside allowed regimes

**Common Patterns:**
- Bull archetypes: `[risk_on, neutral]`
- Bear archetypes: `[risk_off, crisis]`
- Universal: `[]` (all allowed)

## Example: Complete Config

```yaml
# Archetype A - Spring / UTAD
name: spring
display_name: "Spring / UTAD"
aliases: ["A", "trap_reversal", "wyckoff_spring_utad"]
direction: long

# Per-archetype fusion weights
fusion_weights:
  wyckoff: 0.60        # Primary: Wyckoff spring detection
  liquidity: 0.20      # Secondary: Liquidity sweep component
  momentum: 0.10       # Tertiary: Momentum reversal
  smc: 0.10           # Supporting: SMC structure

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
  risk_on: 0.5       # Works but not optimal
  neutral: 1.0       # Good
  risk_off: 1.5      # Better
  crisis: 2.0        # Best (counter-trend)

# Allowed regimes
allowed_regimes: []  # Works in all regimes

# Metadata
description: "PTI-based spring/UTAD detection with displacement confirmation"
priority: 1
enabled: true
```

## Loading Configs

### Python API

```python
from engine.config.archetype_config_loader import (
    load_archetype_configs,
    get_archetype_fusion_weights,
    get_enabled_archetypes
)

# Load all configs
configs = load_archetype_configs("configs/archetypes/")

# Get specific archetype
spring = configs["spring"]
fusion_weights = spring["fusion_weights"]

# Get fusion weights only
weights = get_archetype_fusion_weights("spring")

# Get enabled archetypes
enabled = get_enabled_archetypes()
```

### CLI Validation

```bash
python engine/config/archetype_config_loader.py \
  --dir configs/archetypes/ \
  --verbose
```

## Migration from JSON

### Old (Monolithic JSON)

```json
{
  "archetypes": {
    "thresholds": {
      "spring": {
        "fusion_threshold": 0.18,
        "pti_score_threshold": 0.05
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

**Problem:** Single fusion weights for all archetypes

### New (Isolated YAML)

```yaml
# configs/archetypes/spring.yaml
name: spring
fusion_weights:
  wyckoff: 0.60
  liquidity: 0.20
  momentum: 0.10
  smc: 0.10
thresholds:
  fusion_threshold: 0.18
  pti_score_threshold: 0.05
```

**Benefit:** Per-archetype fusion optimization

## Optimization Workflow

### 1. Independent Optimization

```bash
# Optimize spring archetype only
python bin/optimize_archetype.py --archetype spring

# Results: configs/archetypes/spring_optimized.yaml
```

### 2. A/B Testing

```bash
# Create variant
cp configs/archetypes/spring.yaml configs/archetypes/spring_v2.yaml

# Edit spring_v2.yaml with different weights
# Compare performance
```

### 3. Version Control

```bash
git diff configs/archetypes/spring.yaml
# Shows only changes to spring archetype
```

## Best Practices

### 1. Fusion Weight Selection

**Start with archetype domain focus:**
- Wyckoff archetypes: `wyckoff: 0.5-0.7`
- Liquidity archetypes: `liquidity: 0.4-0.6`
- Momentum archetypes: `momentum: 0.4-0.6`
- SMC archetypes: `smc: 0.4-0.6`

**Then optimize with Optuna/grid search**

### 2. Threshold Tuning

**Conservative (low frequency):**
- `fusion_threshold: 0.35-0.45`
- Higher quality, fewer trades

**Balanced:**
- `fusion_threshold: 0.25-0.35`
- Medium frequency

**Aggressive (high frequency):**
- `fusion_threshold: 0.15-0.25`
- More trades, lower average quality

### 3. Regime Configuration

**Risk-on specialists (A-M):**
```yaml
regime_preferences:
  risk_on: 1.3
  neutral: 1.0
  risk_off: 0.7
  crisis: 0.5
allowed_regimes: [risk_on, neutral]
```

**Crisis specialists (S1, S3):**
```yaml
regime_preferences:
  risk_on: 0.5
  neutral: 0.8
  risk_off: 1.5
  crisis: 2.5
allowed_regimes: [risk_off, crisis]
```

## Validation Checklist

- [ ] All required fields present
- [ ] Fusion weights sum to 1.0
- [ ] Direction is valid (long/short/neutral)
- [ ] Thresholds are reasonable (0.0-1.0 for scores)
- [ ] Exit logic percentages sum to 1.0
- [ ] Regime preferences are positive floats
- [ ] Allowed regimes are valid
- [ ] No syntax errors (run validation tool)

## See Also

- `/configs/archetypes/*.yaml` - Individual archetype configs
- `/engine/config/archetype_config_loader.py` - Config loader
- `/configs/archetypes/production/README_PRODUCTION_CONFIGS.md` - Legacy configs
- `/engine/archetypes/registry.py` - Archetype registry
