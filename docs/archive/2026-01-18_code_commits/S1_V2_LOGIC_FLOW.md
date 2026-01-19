# S1 V2 Logic Flow Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     S1 _check_S1() Entry                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────┐
        │  Check for V2 Features in dataframe    │
        │  - capitulation_depth                   │
        │  - crisis_composite                     │
        │  - volume_climax_last_3b                │
        │  - wick_exhaustion_last_3b              │
        └────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ALL Present            ANY Missing
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│   V2 MODE      │      │   V1 FALLBACK  │
│  (Preferred)   │      │   (Compatible) │
└────────┬───────┘      └────────┬───────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Return (bool, float,  │
        │  dict) to caller       │
        └────────────────────────┘
```

## V2 Mode Logic (Multi-Bar Capitulation)

```
┌─────────────────────────────────────────────────────────────────┐
│                        V2 MODE START                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────┐
            │  HARD GATE 1: Capitulation Depth│
            │  capitulation_depth >= -0.20?   │
            │  (20%+ drawdown from 30d high)  │
            └───────────┬─────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                  YES       NO
                   │         │
                   │         └──► REJECT (insufficient drawdown)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  HARD GATE 2: Crisis Environment│
            │  crisis_composite >= 0.40?      │
            │  (VIX+funding+vol+drawdown)     │
            └───────────┬─────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                  YES       NO
                   │         │
                   │         └──► REJECT (not in crisis)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  OR GATE: Multi-Bar Exhaustion  │
            │  (Need at least ONE signal)     │
            └───────────┬─────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌────────────────────┐        ┌────────────────────┐
│ Volume Climax 3B?  │   OR   │ Wick Exhaustion 3B?│
│ vol_climax > 0.25  │        │ wick_exhaust > 0.30│
└────────┬───────────┘        └────────┬───────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                   ┌────┴────┐
                   │         │
         At Least ONE     NONE
                   │         │
                   │         └──► REJECT (no exhaustion signal)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  SOFT FUSION: Calculate Score   │
            │  - Weighted V2 features          │
            │  - Liquidity drain/velocity      │
            │  - Funding + RSI confluence      │
            └───────────┬─────────────────────┘
                        │
                        ▼
            ┌─────────────────────────────────┐
            │  Final Fusion Gate              │
            │  fusion_score >= 0.30?          │
            └───────────┬─────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                  YES       NO
                   │         │
                   │         └──► REJECT (score too low)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  ✓ PATTERN DETECTED              │
            │  Return (True, score, metadata) │
            └─────────────────────────────────┘
```

## V1 Fallback Logic (Single-Bar)

```
┌─────────────────────────────────────────────────────────────────┐
│                    V1 FALLBACK MODE START                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────┐
            │  HARD GATE: Liquidity Drain     │
            │  liquidity_score < 0.20?        │
            │  (absolute drain check)         │
            └───────────┬─────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                  YES       NO
                   │         │
                   │         └──► REJECT (liquidity not drained)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  OR GATE: Single-Bar Exhaustion │
            │  (Need at least ONE signal)     │
            └───────────┬─────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌────────────────────┐        ┌────────────────────┐
│ Volume Panic?      │   OR   │ Wick Rejection?    │
│ vol_zscore >= 1.5  │        │ wick_lower >= 0.28 │
└────────┬───────────┘        └────────┬───────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                   ┌────┴────┐
                   │         │
         At Least ONE     NONE
                   │         │
                   │         └──► REJECT (no exhaustion signal)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  SOFT FUSION: Calculate Score   │
            │  - Weighted V1 features          │
            │  - Funding + VIX + DXY           │
            │  - RSI + ATR confluence          │
            └───────────┬─────────────────────┘
                        │
                        ▼
            ┌─────────────────────────────────┐
            │  Final Fusion Gate              │
            │  fusion_score >= 0.30?          │
            └───────────┬─────────────────────┘
                        │
                   ┌────┴────┐
                   │         │
                  YES       NO
                   │         │
                   │         └──► REJECT (score too low)
                   │
                   ▼
            ┌─────────────────────────────────┐
            │  ✓ PATTERN DETECTED              │
            │  Return (True, score, metadata) │
            └─────────────────────────────────┘
```

## Feature Comparison Table

| Aspect | V2 Multi-Bar | V1 Single-Bar |
|--------|-------------|---------------|
| **Liquidity Check** | RELATIVE drain vs 7d avg | ABSOLUTE level |
| **Exhaustion Window** | Last 3 bars (rolling max) | Current bar only |
| **Drawdown Filter** | Required (>= 20% from high) | Not used |
| **Crisis Detection** | Composite (VIX+funding+vol+DD) | Simple (VIX+DXY) |
| **Velocity/Persistence** | Used in scoring | Not used |
| **Trade Frequency** | Lower (10-15/year) | Higher (15-25/year) |
| **Precision** | Higher (fewer false +) | Lower (more noise) |
| **Use Case** | True capitulations only | General liquidity drains |

## Key Improvements in V2

### 1. Fixes June 18, 2022 Detection Failure

**Problem (V1):**
```
June 18 absolute liquidity = 0.308 (looks normal)
V1 threshold = 0.20
0.308 > 0.20 → REJECT ✗
```

**Solution (V2):**
```
June 18 liquidity = 0.308
7-day average = 0.55
Drain % = (0.308 - 0.55) / 0.55 = -44%
V2 uses RELATIVE drain + other signals → DETECT ✓
```

### 2. Handles Temporal Separation of Signals

**Problem (V1):**
```
Bar 1: Huge wick, no volume spike
Bar 2: Volume spike, wick faded
V1 requires signals on SAME bar → May miss both
```

**Solution (V2):**
```
Bar 1: wick_lower = 0.42
Bar 2: wick_lower = 0.15, volume_z = 2.1
Bar 3: Entry candidate

V2 checks:
- volume_climax_last_3b = max(0, 2.1, 0.8) = 2.1 ✓
- wick_exhaustion_last_3b = max(0.42, 0.15, 0.10) = 0.42 ✓
→ DETECT ✓
```

### 3. Separates Noise from True Capitulations

**Problem (V1):**
```
Random -8% dip in sideways market
liquidity = 0.15 (drained)
volume_z = 1.7 (spike)
V1 → May detect (false positive)
```

**Solution (V2):**
```
Same dip:
capitulation_depth = -8% (>= -20% threshold) → REJECT
V2 requires >= 20% drawdown → Filters noise ✓
```

## Real-World Example: June 18, 2022 Bottom

### Timeline (3-hour bars)

```
Bar T-2:  High: $22,800, Close: $20,200, Wick: 18%
          volume_z = 1.2, liquidity = 0.35

Bar T-1:  High: $20,800, Close: $17,600, Wick: 35% ← Big wick!
          volume_z = 1.8, liquidity = 0.31
          wick_lower_ratio = 0.35

Bar T:    High: $19,100, Close: $18,200, Wick: 12%
          volume_z = 2.4 ← Volume spike!
          liquidity = 0.308
          wick_lower_ratio = 0.12
```

### V1 Logic (FAILS)
```
Checks Bar T only:
✗ liquidity = 0.308 >= 0.20 (REJECT - not drained enough)

OR

✗ volume_z = 2.4 >= 1.5 (PASS)
✗ wick_lower = 0.12 >= 0.28 (FAIL)
→ Only volume passes, misses best entry bar
```

### V2 Logic (PASSES)
```
Checks Bar T with 3-bar context:
✓ capitulation_depth = -44.7% >= -20% (PASS)
✓ crisis_composite = 0.617 >= 0.40 (PASS)
✓ volume_climax_3b = max(1.2, 1.8, 2.4) = 2.4 → 0.447 >= 0.25 (PASS)
✓ wick_exhaustion_3b = max(0.18, 0.35, 0.12) = 0.35 → 0.372 >= 0.30 (PASS)
✓ fusion_score = 0.58 >= 0.30 (PASS)
→ DETECT ✓ (All gates pass)
```

## Return Value Structure

### On Detection (True)

```python
(
    True,  # matched
    0.58,  # fusion score
    {
        "mode": "v2_multi_bar_capitulation",
        "mechanism": "liquidity_vacuum_capitulation_fade_v2",
        "components": {
            "capitulation_depth_score": 0.89,
            "crisis_environment": 0.62,
            "volume_climax_3b": 0.45,
            # ... all components
        },
        "weights": { ... },
        "gates_passed": {
            "capitulation_depth": -0.447,
            "crisis_composite": 0.617,
            "volume_climax_3b": 0.447,
            "wick_exhaustion_3b": 0.372,
            "has_volume_exhaustion": True,
            "has_wick_exhaustion": True
        }
    }
)
```

### On Rejection (False)

```python
(
    False,  # not matched
    0.0,    # score (or partial if failed fusion gate)
    {
        "reason": "v2_insufficient_drawdown",
        "capitulation_depth": -0.08,
        "threshold": -0.20,
        "note": "Need >= 20% drawdown from 30d high for true capitulation"
    }
)
```

## Optuna Integration

### Parameter Space

```python
# Hard gates (most impact on trade count)
'capitulation_depth_max': trial.suggest_float(..., -0.35, -0.10)
'crisis_composite_min': trial.suggest_float(..., 0.25, 0.55)

# Exhaustion thresholds (moderate impact)
'volume_climax_3b_min': trial.suggest_float(..., 0.15, 0.40)
'wick_exhaustion_3b_min': trial.suggest_float(..., 0.20, 0.45)

# Fusion threshold (fine-tuning)
'fusion_threshold': trial.suggest_float(..., 0.25, 0.45)
```

### Multi-Objective Optimization

```python
def objective(trial):
    config = build_config_with_trial_params(trial)
    results = run_backtest(config)

    pf = results['profit_factor']
    trade_count = results['trade_count']

    # Target 10-15 trades/year, penalize outside range
    trade_penalty = abs(trade_count - 12.5) / 12.5

    # Maximize PF, minimize trade deviation
    return pf * (1 - 0.2 * trade_penalty)
```

---

**Status:** COMPLETE ✓
**Date:** 2025-11-23
**Version:** S1 V2.0
