# Structure Module Feature Schema

This document defines all feature store columns added by the Structure module (Week 1).

## Internal vs External Structure (6 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| `internal_phase` | str | accumulation, distribution, markup, markdown, transition | Local structure phase (micro) |
| `external_trend` | str | bullish, bearish, range | HTF trend direction (macro) |
| `structure_alignment` | bool | True/False | Whether internal matches external |
| `conflict_score` | float | 0-1 | Divergence strength (early reversal signal) |
| `internal_strength` | float | 0-1 | Confidence in internal structure |
| `external_strength` | float | 0-1 | Confidence in external trend |

**Usage**:
```python
from engine.structure import detect_structure_state

state = detect_structure_state(df_1h, df_4h, df_1d, config)
features.update(state.to_dict())
```

**Fusion Impact**:
- `conflict_score > 0.6`: Entry threshold +0.05
- `conflict_score > 0.75`: Entry threshold +0.08

---

## BOMS Detection (7 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| `boms_detected` | bool | True/False | BOMS confirmed |
| `boms_direction` | str | bullish, bearish, none | BOMS direction |
| `boms_volume_surge` | float | 0-5+ | Volume ratio vs mean |
| `boms_fvg_present` | bool | True/False | Fair Value Gap left behind |
| `boms_confirmation` | int | 0-10+ | Bars since break |
| `boms_break_level` | float | price | Price level broken |
| `boms_displacement` | float | 0-1+ | Displacement beyond break |

**Usage**:
```python
from engine.structure import detect_boms

boms = detect_boms(df_4h, timeframe='4H', config)
features.update(boms.to_dict())
```

**Fusion Impact**:
- 4H/1D BOMS: Fusion +0.10
- 1H BOMS: Fusion +0.05
- Volume > 2.0x: Additional +0.02

---

## 1-2-3 Squiggle Pattern (8 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| `squiggle_stage` | int | 0-3 | 0=none, 1=BOS, 2=retest, 3=continuation |
| `squiggle_pattern_id` | str | - | Unique pattern identifier |
| `squiggle_direction` | str | bullish, bearish, none | Pattern direction |
| `squiggle_entry_window` | bool | True/False | Stage 2 entry window active |
| `squiggle_confidence` | float | 0-1 | Pattern quality score |
| `squiggle_bos_level` | float | price | BOS breakout level |
| `squiggle_retest_quality` | float | 0-1 | Retest precision |
| `squiggle_bars_since_bos` | int | 0-20+ | Time since BOS |

**Usage**:
```python
from engine.structure import detect_squiggle_123

pattern = detect_squiggle_123(df_4h, timeframe='4H', config)
features.update(pattern.to_dict())
```

**Fusion Impact**:
- Stage 2 (entry_window): Fusion +0.05
- High-quality retest (>0.8): Additional +0.02

---

## Range Outcomes (8 columns)

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| `range_outcome` | str | breakout, fakeout, rejection, range_bound, none | Classification |
| `range_outcome_direction` | str | bullish, bearish, neutral | Outcome direction |
| `range_outcome_confidence` | float | 0-1 | Classification confidence |
| `range_high` | float | price | Upper range boundary |
| `range_low` | float | price | Lower range boundary |
| `breakout_strength` | float | 0-1 | Displacement strength |
| `volume_confirmation` | bool | True/False | Volume supports outcome |
| `bars_in_range` | int | 0-50+ | Range duration |

**Usage**:
```python
from engine.structure import classify_range_outcome

outcome = classify_range_outcome(df_4h, timeframe='4H', config)
features.update(outcome.to_dict())
```

**Fusion Impact**:
- Confirmed breakout: Fusion +0.08
- Fakeout detected: Fusion -0.10 (avoid trap)
- Rejection: Fusion -0.05 (choppy)

---

## Summary

**Total New Columns**: 29

**Breakdown**:
- Internal/External Structure: 6 columns
- BOMS Detection: 7 columns
- 1-2-3 Squiggle Pattern: 8 columns
- Range Outcomes: 8 columns

**Integration Pattern**:

All structure modules follow the same pattern:

1. **Dataclass with `to_dict()`** - Clean feature export
2. **Main detector function** - Returns dataclass instance
3. **Fusion adjustment function** - Modifies fusion score/threshold
4. **Timeframe-aware** - Different thresholds for 1H/4H/1D

**Example Full Integration**:

```python
# In build_feature_store.py or fusion engine

from engine.structure import (
    detect_structure_state,
    detect_boms,
    detect_squiggle_123,
    classify_range_outcome
)

# Compute all structure features
structure_state = detect_structure_state(df_1h, df_4h, df_1d, config)
boms_signal = detect_boms(df_4h, timeframe='4H', config)
squiggle_pattern = detect_squiggle_123(df_4h, timeframe='4H', config)
range_outcome = classify_range_outcome(df_4h, timeframe='4H', config)

# Export to feature store
features.update(structure_state.to_dict())
features.update(boms_signal.to_dict())
features.update(squiggle_pattern.to_dict())
features.update(range_outcome.to_dict())

# Apply fusion adjustments
fusion_score, _, reasons = apply_structure_fusion_adjustment(fusion_score, structure_state, config)
fusion_score, _, reasons = apply_boms_fusion_boost(fusion_score, boms_signal, '4H', config)
fusion_score, _, reasons = apply_squiggle_fusion_boost(fusion_score, squiggle_pattern, config)
fusion_score, _, reasons = apply_range_fusion_adjustment(fusion_score, range_outcome, config)
```

---

**Version**: Week 1 Core Structure Implementation
**Status**: Complete
**Next**: Week 2 - Psychology & Volume layers (PTI, FRVP)
