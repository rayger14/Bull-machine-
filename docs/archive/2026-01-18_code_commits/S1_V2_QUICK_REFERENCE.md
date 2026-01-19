# S1 V2 Quick Reference

## What Changed?

Updated `engine/archetypes/logic_v2_adapter.py:_check_S1()` to use multi-bar capitulation features instead of single-bar checks.

## V2 Detection Pattern

```
IF (capitulation_depth >= -20% AND crisis_composite >= 0.40):
    IF (volume_climax_3b > 0.25 OR wick_exhaustion_3b > 0.30):
        COMPUTE fusion_score
        IF fusion_score >= 0.30:
            TRIGGER S1 SIGNAL
```

## Key Features Used

### V2 Features (Required)
- `capitulation_depth` - Drawdown from 30d high
- `crisis_composite` - Composite crisis score (VIX + funding + vol + drawdown)
- `volume_climax_last_3b` - Max volume panic in last 3 bars
- `wick_exhaustion_last_3b` - Max wick rejection in last 3 bars

### V2 Features (Optional - Enhance Scoring)
- `liquidity_drain_pct` - Relative drain vs 7d average
- `liquidity_velocity` - Rate of liquidity drain
- `liquidity_persistence` - Consecutive bars with drain

### V1 Fallback (If V2 Missing)
- `liquidity_score` - Absolute liquidity level
- `volume_zscore` - Volume z-score
- `wick_lower_ratio` - Single-bar lower wick ratio

## Config Parameters

### Minimal V2 Config

```json
{
  "liquidity_vacuum": {
    "use_v2_logic": true,
    "capitulation_depth_max": -0.20,
    "crisis_composite_min": 0.40,
    "volume_climax_3b_min": 0.25,
    "wick_exhaustion_3b_min": 0.30,
    "fusion_threshold": 0.30
  }
}
```

## Threshold Quick Tune

**More trades (looser):**
```json
{
  "capitulation_depth_max": -0.15,
  "crisis_composite_min": 0.30,
  "volume_climax_3b_min": 0.20,
  "wick_exhaustion_3b_min": 0.25
}
```

**Higher precision (tighter):**
```json
{
  "capitulation_depth_max": -0.25,
  "crisis_composite_min": 0.50,
  "volume_climax_3b_min": 0.30,
  "wick_exhaustion_3b_min": 0.35
}
```

## Testing Commands

### 1. Enable V2 Features
```python
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import LiquidityVacuumRuntimeFeatures

enricher = LiquidityVacuumRuntimeFeatures()
df = enricher.enrich_dataframe(df)  # Adds V2 features
```

### 2. Run Backtest
```bash
python bin/backtest_knowledge_v2.py \
  --config configs/s1_v2_example_config.json \
  --start 2022-01-01 \
  --end 2022-12-31
```

### 3. Check Detection Mode
Look for log message:
- `[Liquidity Vacuum V2] First evaluation` = V2 mode active
- `[Liquidity Vacuum V1 (fallback)] First evaluation` = V1 fallback active

## Expected Results (2022)

With recommended thresholds:

| Event | Detection | Reason |
|-------|-----------|--------|
| LUNA (May 12) | PASS | depth=-38.4%, crisis=63.9%, wick_3b=48.9% |
| June 18 Bottom | PASS | depth=-44.7%, crisis=61.7%, vol_3b=44.7%, wick_3b=37.2% |
| FTX (Nov 9) | FAIL | crisis=30.3% (< 0.40 threshold) |

To catch FTX: Lower `crisis_composite_min` to `0.30` (trade-off: more noise)

## Optuna Optimization

```python
# Suggested parameter ranges
trial.suggest_float('capitulation_depth_max', -0.35, -0.10)
trial.suggest_float('crisis_composite_min', 0.25, 0.55)
trial.suggest_float('volume_climax_3b_min', 0.15, 0.40)
trial.suggest_float('wick_exhaustion_3b_min', 0.20, 0.45)
trial.suggest_float('fusion_threshold', 0.25, 0.45)
```

**Objective:** Maximize PF, target 10-15 trades/year

## Troubleshooting

**No trades detected?**
1. Check if V2 features present: `'capitulation_depth' in df.columns`
2. Check mode in logs: Should see "V2" not "V1 fallback"
3. Try looser thresholds (see "More trades" config above)

**Too many false positives?**
1. Tighten hard gates (see "Higher precision" config above)
2. Increase fusion_threshold (0.30 → 0.35)
3. Add regime gating (only in risk_off/crisis regimes)

**V1 fallback activating unexpectedly?**
1. Verify V2 features enriched: `enricher.enrich_dataframe(df)`
2. Check for NaN values in V2 features
3. Ensure features computed before backtest runs

## Next Steps

1. Add V2 enrichment to backtest pipeline
2. Run full 2022 validation
3. Optimize thresholds with Optuna
4. Deploy to production

## Files

- **Implementation:** `engine/archetypes/logic_v2_adapter.py` (lines 1219-1526)
- **V2 Features:** `engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py`
- **Example Config:** `configs/s1_v2_example_config.json`
- **Full Docs:** `S1_V2_IMPLEMENTATION_COMPLETE.md`

---

**Status:** COMPLETE ✓
**Date:** 2025-11-23
