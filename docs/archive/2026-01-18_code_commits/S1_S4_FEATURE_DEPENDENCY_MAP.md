# S1 & S4 Feature Dependency Map (Visual Reference)

**Quick reference for understanding what features each archetype needs**

---

## S1 LIQUIDITY VACUUM - Feature Dependency Tree

```
S1 Liquidity Vacuum Reversal
│
├─ HARD REQUIREMENTS (Pattern fails if missing)
│  ├─ OHLCV Data
│  │  ├─ open, high, low, close
│  │  └─ Used for: Candle analysis, drawdown calculation
│  │
│  ├─ liquidity_score (Feature Store)
│  │  └─ PRIMARY SIGNAL - Orderbook depth gauge
│  │
│  └─ V2 Runtime Features (Computed on-demand)
│     ├─ capitulation_depth (30d rolling high drawdown)
│     └─ crisis_composite (VIX + Funding + RV + Drawdown)
│
├─ CONFLUENCE SIGNALS (3-of-4 required in V2 mode)
│  ├─ volume_climax_last_3b (3-bar rolling max)
│  │  ├─ Derived from: volume_zscore (Feature Store)
│  │  └─ Threshold: > 0.50 (RAISED from 0.25)
│  │
│  ├─ wick_exhaustion_last_3b (3-bar rolling max)
│  │  ├─ Derived from: wick_lower_ratio (Runtime)
│  │  │  └─ Calculated from: open, high, low, close
│  │  └─ Threshold: > 0.60 (RAISED from 0.30)
│  │
│  ├─ capitulation_depth (see above)
│  │  └─ Threshold: < -0.20 (20% drawdown)
│  │
│  └─ crisis_composite (see above)
│     └─ Threshold: > 0.35 (0.40 misses FTX!)
│
├─ ENHANCEMENT SIGNALS (Optional - boost score)
│  ├─ liquidity_drain_pct (V2 Runtime)
│  │  ├─ KEY FIX for liquidity paradox
│  │  └─ Measures RELATIVE drain vs 7d average
│  │
│  ├─ liquidity_velocity (V2 Runtime)
│  │  └─ Speed of drain (active vs quiet)
│  │
│  ├─ liquidity_persistence (V2 Runtime)
│  │  └─ Consecutive bars with sustained stress
│  │
│  ├─ funding_Z (Feature Store)
│  │  └─ Negative funding = short squeeze fuel
│  │
│  ├─ VIX_Z, DXY_Z (Macro)
│  │  └─ Crisis context indicators
│  │
│  ├─ rsi_14 (Feature Store)
│  │  └─ Oversold condition
│  │
│  └─ atr_percentile (Feature Store)
│     └─ Volatility spike detection
│
└─ REGIME FILTER (Optional - can disable)
   ├─ regime_label (Regime Classifier)
   │  ├─ Allowed: ['risk_off', 'crisis']
   │  └─ Fallback: Infer from crisis_composite
   │
   └─ Drawdown Override
      └─ If capitulation_depth < -10%, bypass regime check
```

---

## S4 FUNDING DIVERGENCE - Feature Dependency Tree

```
S4 Funding Divergence (Short Squeeze)
│
├─ HARD REQUIREMENTS (Pattern fails if missing)
│  ├─ close (OHLCV)
│  │  └─ Used for: Price resilience calculation
│  │
│  ├─ funding_rate or funding_Z (Feature Store)
│  │  └─ PRIMARY SIGNAL - Negative funding = overcrowded shorts
│  │
│  └─ liquidity_score (Feature Store)
│     └─ Thin orderbook = amplified squeeze violence
│
├─ RUNTIME FEATURES (Computed on-demand)
│  ├─ funding_z_negative (24h rolling z-score)
│  │  ├─ Derived from: funding_rate
│  │  └─ Threshold: < -1.2 (more negative = more shorts)
│  │
│  ├─ price_resilience (12h lookback)
│  │  ├─ Calculation: (actual_change - expected_change) / 0.04
│  │  ├─ DIVERGENCE SIGNAL: Price NOT falling despite bearish funding
│  │  └─ Threshold: > 0.5 (price showing strength)
│  │
│  ├─ volume_quiet (boolean)
│  │  ├─ Derived from: volume_zscore < -0.5
│  │  └─ Coiled spring effect (calm before storm)
│  │
│  └─ s4_fusion_score (weighted combination)
│     ├─ funding_negative: 40% weight
│     ├─ price_resilience: 30% weight
│     ├─ volume_quiet: 15% weight
│     └─ liquidity_thin: 15% weight
│
└─ DETECTION LOGIC (Sequential gates)
   ├─ Gate 1: funding_z < -1.2 (REQUIRED)
   ├─ Gate 2: liquidity_score < 0.30 (REQUIRED)
   ├─ Gate 3: price_resilience >= 0.5 (REQUIRED if available)
   └─ Gate 4: s4_fusion_score >= 0.40 (Final threshold)
```

---

## Feature Source Mapping

### Feature Store (Pre-computed, must exist)
```
S1 Needs:
  ├─ liquidity_score       ⭐ CRITICAL
  ├─ volume_zscore         ⭐ CRITICAL
  ├─ funding_Z             ⚠️  Optional (enhances score)
  ├─ VIX_Z                 ⚠️  Optional (crisis context)
  ├─ DXY_Z                 ⚠️  Optional (crisis context)
  ├─ rsi_14                ⚠️  Optional (oversold)
  ├─ atr_20                ✓  Position sizing
  └─ atr_percentile        ⚠️  Optional (volatility)

S4 Needs:
  ├─ funding_rate/funding_Z ⭐ CRITICAL
  ├─ liquidity_score        ⭐ CRITICAL
  ├─ volume_zscore          ✓  For volume_quiet
  └─ atr_20                 ✓  Position sizing

Shared:
  ├─ liquidity_score        ⭐ BOTH patterns need this
  ├─ volume_zscore          ✓  Helpful for both
  ├─ funding_Z              ⭐ Critical for S4, optional for S1
  └─ regime_label           ⚠️  Optional (regime routing)
```

### Runtime Computed (NOT in feature store)
```
S1 Runtime Enrichment (12 features):
  ├─ V1 Features (4):
  │  ├─ wick_lower_ratio
  │  ├─ liquidity_vacuum_score
  │  ├─ volume_panic
  │  └─ crisis_context
  │
  └─ V2 Features (8):
     ├─ liquidity_drain_pct        ⭐ KEY FIX
     ├─ liquidity_velocity
     ├─ liquidity_persistence
     ├─ capitulation_depth          ⭐ CRITICAL GATE
     ├─ crisis_composite            ⭐ CRITICAL GATE
     ├─ volume_climax_last_3b       ⭐ CONFLUENCE SIGNAL
     ├─ wick_exhaustion_last_3b     ⭐ CONFLUENCE SIGNAL
     └─ liquidity_vacuum_fusion

S4 Runtime Enrichment (4 features):
  ├─ funding_z_negative      ⭐ CRITICAL GATE
  ├─ price_resilience        ⭐ CRITICAL GATE
  ├─ volume_quiet            ✓  Coiled spring detection
  └─ s4_fusion_score         ✓  Final score
```

---

## Detection Mode Comparison

### S1: Dual Mode (V2 preferred, V1 fallback)

**V2 MODE (Multi-bar Capitulation)**
```
IF (has V2 features):
  ├─ REGIME FILTER (optional)
  │  └─ regime in ['risk_off', 'crisis'] OR drawdown > 10%
  │
  ├─ HARD GATES (both required)
  │  ├─ capitulation_depth < -0.20
  │  └─ crisis_composite >= 0.35
  │
  └─ CONFLUENCE MODE
     ├─ Require 3-of-4 conditions:
     │  ├─ capitulation_depth < -0.20
     │  ├─ crisis_composite > 0.35
     │  ├─ volume_climax_3b > 0.50
     │  └─ wick_exhaustion_3b > 0.60
     │
     └─ Weighted Score >= 0.65
        └─ Weights: depth(20%), crisis(15%), vol(8%), wick(7%), liq(25%), other(25%)
```

**V1 MODE (Single-bar Fallback)**
```
ELSE (V2 features missing):
  ├─ liquidity_score < 0.20
  │
  ├─ AND (volume_zscore > 1.5 OR wick_lower_ratio > 0.28)
  │
  └─ AND liquidity_vacuum_fusion > 0.30
```

---

### S4: Single Mode (Runtime features required)

```
S4 DETECTION:
  ├─ Gate 1: funding_z < -1.2         ⭐ NEGATIVE extreme
  ├─ Gate 2: liquidity < 0.30         ⭐ Thin orderbook
  ├─ Gate 3: resilience >= 0.5        ⭐ Price strength (if available)
  └─ Gate 4: fusion_score >= 0.40     ✓  Final score

  Components:
  ├─ funding_negative (40%)   = (-funding_z - 1.0) / 2.0
  ├─ price_resilience (30%)   = Runtime calculated
  ├─ volume_quiet (15%)       = 1.0 if vol_z < -0.5 else 0.0
  └─ liquidity_thin (15%)     = 1.0 - (liquidity / 0.5)
```

---

## Critical Thresholds (Production Values)

### S1 Liquidity Vacuum
```
V2 Confluence Mode (configs/s1_v2_production.json):
  capitulation_depth_max:    -0.20      # 20% drawdown minimum
  crisis_composite_min:       0.35      # Catches FTX (0.34)
  confluence_min_conditions:  3         # 3-of-4 conditions
  confluence_threshold:       0.65      # 65% weighted score
  volume_climax_3b_min:       0.50      # RAISED from 0.25
  wick_exhaustion_3b_min:     0.60      # RAISED from 0.30

Regime Filter:
  allowed_regimes:           ['risk_off', 'crisis']
  drawdown_override_pct:      0.10      # 10% = bypass regime

V1 Fallback:
  liquidity_max:              0.20
  volume_z_min:               1.5
  wick_lower_min:             0.28
  fusion_threshold:           0.30
```

### S4 Funding Divergence
```
Optimized (configs/s4_optimized_oos_test.json):
  funding_z_max:             -1.976     # More negative than default
  resilience_min:             0.555     # Optimized
  liquidity_max:              0.348     # Optimized
  fusion_threshold:           0.7824    # Stricter than default 0.40

Runtime Params:
  funding_lookback:           24        # Hours
  price_lookback:             12        # Hours
```

---

## Integration Examples

### S1 Integration (V2 Mode)
```python
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    apply_liquidity_vacuum_enrichment
)

# BEFORE backtest
df_enriched = apply_liquidity_vacuum_enrichment(
    df,
    lookback=24,           # General lookback
    volume_lookback=24     # Volume z-score window
)

# Adds 12 runtime features:
# - wick_lower_ratio
# - liquidity_vacuum_score
# - volume_panic
# - crisis_context
# - liquidity_vacuum_fusion
# - liquidity_drain_pct          ⭐ V2 KEY FIX
# - liquidity_velocity
# - liquidity_persistence
# - capitulation_depth           ⭐ V2 CRITICAL
# - crisis_composite             ⭐ V2 CRITICAL
# - volume_climax_last_3b        ⭐ V2 CONFLUENCE
# - wick_exhaustion_last_3b      ⭐ V2 CONFLUENCE
```

### S4 Integration
```python
from engine.strategies.archetypes.bear.funding_divergence_runtime import (
    apply_s4_enrichment
)

# BEFORE backtest
df_enriched = apply_s4_enrichment(
    df,
    funding_lookback=24,    # Funding z-score window
    price_lookback=12,      # Resilience calculation window
    volume_lookback=24      # Volume quiet window
)

# Adds 4 runtime features:
# - funding_z_negative     ⭐ CRITICAL
# - price_resilience       ⭐ CRITICAL
# - volume_quiet           ✓  Boolean
# - s4_fusion_score        ✓  Final score
```

---

## Performance Targets

### S1 Liquidity Vacuum
```
Target Frequency:     10-15 trades/year (40-60 with relaxed thresholds)
Expected Win Rate:    50-60%
Major Events/Year:    3-4 capitulations caught
Trade Distribution:   Concentrated in bear markets
                      (2023 bull: 0 trades = CORRECT)

Historical Catches:
  ✓ LUNA May-12 (2022)    -80% crash → 25% bounce
  ✓ LUNA Jun-18 (2022)    Final capitulation
  ✓ FTX Nov-9 (2022)      Exchange collapse
  ✓ Japan Aug-5 (2024)    Flash crash

Known Misses (by design):
  ✗ SVB Mar-10            Moderate event, no volume climax
  ✗ Aug Flush Aug-17      Mild, regime uncertain
```

### S4 Funding Divergence
```
Target Frequency:     6-10 trades/year
Expected PF:          > 2.0 (violent squeezes)
Edge:                 Short squeeze violence in thin markets

BTC Examples:
  ✓ 2022-08-15    Funding -0.15% → +18% rally
  ✓ 2023-01-14    Negative funding + strength → +12%
```

---

## Optimization Priority

### S1 High-Impact Parameters (tune these first)
1. `confluence_threshold` (0.60-0.70)        → Trade frequency vs precision
2. `capitulation_depth_max` (-0.15 to -0.25) → Severity filter
3. `crisis_composite_min` (0.30-0.45)        → Macro sensitivity
4. `volume_climax_3b_min` (0.25-0.60)        → Exhaustion gate
5. `wick_exhaustion_3b_min` (0.30-0.70)      → Exhaustion gate

### S4 High-Impact Parameters
1. `fusion_threshold` (0.40-0.80)     → Trade frequency
2. `funding_z_max` (-1.0 to -2.5)     → Funding extreme
3. `resilience_min` (0.4-0.7)         → Divergence strength
4. `liquidity_max` (0.2-0.4)          → Orderbook thinness

---

**END OF VISUAL REFERENCE**
