# Temporal Fusion Layer - Quick Reference

**One-page guide for developers**

---

## What Is This?

The Bull Machine's **sense of time** - detects when multiple time cycles align to boost/penalize fusion weights.

**Philosophy**: *"Time is pressure, not prediction."*

---

## 4 Components

| Component | Weight | Measures | Score Range |
|-----------|--------|----------|-------------|
| **Fib Time** | 40% | Fib projections from Wyckoff events | 0.20-0.80 |
| **Gann Cycles** | 30% | Sacred vibration numbers | 0.20-0.90 |
| **Vol Cycles** | 20% | Compression/expansion | 0.10-0.90 |
| **Emotional** | 10% | RSI+funding psychology | 0.05-0.95 |

---

## Confluence → Adjustment

| Confluence | Meaning | Fusion Multiplier |
|-----------|---------|------------------|
| ≥0.85 | Strong alignment | ×1.15 (+15%) |
| ≥0.70 | Moderate alignment | ×1.10 (+10%) |
| 0.30-0.70 | Neutral | ×1.00 (no change) |
| ≤0.30 | Light misalignment | ×0.95 (-5%) |
| ≤0.15 | Strong misalignment | ×0.85 (-15%) |

---

## Quick Start

### 1. Enable in Config

```json
{
  "temporal_fusion": {
    "enabled": true
  }
}
```

### 2. Compute Features

```bash
python bin/compute_temporal_features.py \
    --input data/btc_1h.parquet \
    --output data/btc_1h_temporal.parquet
```

### 3. Validate

```bash
python bin/validate_temporal_confluence.py \
    --data data/btc_1h_temporal.parquet
```

### 4. Backtest

```bash
python bin/backtest_engine.py --config your_config.json
```

---

## Required Features

**Mandatory**:
- `close`, `volume`, `atr`, `rsi`

**Optional** (for best results):
- `wyckoff_sc`, `wyckoff_ar`, `wyckoff_st`, `wyckoff_sos`, `wyckoff_sow`
- `funding`, `atr_ma_20`

---

## Expected Results

| Event | Date | Expected Confluence |
|-------|------|-------------------|
| June 18, 2022 bottom | 2022-06-18 | High (>0.70) |
| LUNA collapse | 2022-05-12 | High (>0.70) |
| FTX collapse | 2022-11-09 | High (>0.70) |
| Nov 2021 top | 2021-11-10 | Low (<0.30) |

---

## Key Files

| File | Purpose |
|------|---------|
| `engine/temporal/temporal_fusion.py` | Core engine |
| `engine/archetypes/logic_v2_adapter.py` | Integration hook |
| `bin/compute_temporal_features.py` | Feature computation |
| `bin/validate_temporal_confluence.py` | Historical validation |
| `configs/temporal_fusion_config.json` | Configuration template |
| `docs/TEMPORAL_FUSION_LAYER.md` | Full documentation |

---

## Debug

Enable debug logging:

```python
import logging
logging.getLogger('engine.temporal').setLevel(logging.DEBUG)
```

Check logs for:
- `[TEMPORAL] Fusion adjusted: X → Y (confluence=Z)`
- `[TemporalFusion] confluence=X.XXX | fib=... gann=... vol=... emotional=...`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No temporal adjustment | Check `enabled: true` in config |
| All scores = 0.20 | Missing Wyckoff events, run detection first |
| Import error | Check `engine/temporal/temporal_fusion.py` exists |
| Missing features | Run `compute_temporal_features.py` first |

---

## Performance Expectations

- **Computation**: <1ms per bar (runtime)
- **Memory**: +5 columns (~40KB per 10K bars)
- **Profit Factor**: +5-10% vs baseline
- **Win Rate**: +2-5% vs baseline
- **Sharpe Ratio**: +0.1-0.2 vs baseline

---

## Component Details (Brief)

### Fib Time (40%)

```python
# Counts Fib hits from Wyckoff events
fib_levels = [13, 21, 34, 55, 89, 144]
score = 0.80 if hits>=3 else 0.60 if hits==2 else 0.40 if hits==1 else 0.20
```

### Gann Cycles (30%)

```python
# Detects vibration hits from SC
gann_vibrations = [3,7,9,12,21,36,45,72,90,144]
score = 0.90 if vib in [90,144] else 0.75 if vib in [45,72] else 0.60
```

### Vol Cycles (20%)

```python
# ATR ratio determines phase
atr_ratio = atr / atr_ma_20
score = 0.90 if ratio<0.75 else 0.10 if ratio>1.25 else 0.50
```

### Emotional (10%)

```python
# RSI + funding → psychology
score = 0.95 if rsi<25 and funding<-0.02 else \
        0.05 if rsi>75 and funding>0.03 else 0.50
```

---

## Example Adjustment

```python
# Before temporal layer
base_fusion = 0.65

# Temporal confluence
fib=0.80, gann=0.90, vol=0.90, emotional=0.95
confluence = 0.80×0.40 + 0.90×0.30 + 0.90×0.20 + 0.95×0.10 = 0.865

# Adjustment
multiplier = 1.15  # confluence >= 0.85
adjusted_fusion = 0.65 × 1.15 = 0.7475

# Result: Signal now passes (was 0.65 < 0.70 threshold before)
```

---

## Integration Point

```python
# In ArchetypeLogic.detect()
fusion_score = self._fusion(context.row)
fusion_score, _ = self._apply_wyckoff_event_boosts(context.row, fusion_score)

# TEMPORAL ADJUSTMENT ← HERE
if self.temporal_fusion_enabled:
    confluence = self.temporal_fusion_engine.compute_temporal_confluence(context)
    fusion_score = self.temporal_fusion_engine.adjust_fusion_weight(fusion_score, confluence)

# Continue with soft filters...
```

---

## Config Options

```json
{
  "temporal_fusion": {
    "enabled": true,
    "temporal_weights": {
      "fib_time": 0.40,
      "gann_cycles": 0.30,
      "volatility_cycles": 0.20,
      "emotional_cycles": 0.10
    },
    "temporal_adjustment_range": [0.85, 1.15],
    "fib_levels": [13, 21, 34, 55, 89, 144],
    "gann_vibrations": [3, 7, 9, 12, 21, 36, 45, 72, 90, 144],
    "fib_tolerance_bars": 3,
    "gann_tolerance_bars": 2,
    "vol_compression_threshold": 0.75,
    "vol_expansion_threshold": 1.25,
    "emotional_rsi_thresholds": {
      "extreme_fear": 25,
      "hope_lower": 35,
      "hope_upper": 45,
      "greed": 65,
      "extreme_greed": 75
    }
  }
}
```

---

## One-Liner Summary

**Temporal Fusion Layer**: Detects when 4 time systems align (Fib, Gann, Vol, Emotional) and adjusts fusion by ±5-15% for better entry timing.

---

*Built with wisdom. Time is geometry.*
