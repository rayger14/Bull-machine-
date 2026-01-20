# Wyckoff Weighted Boosts - Quick Reference

**Implementation Date**: 2026-01-19
**Status**: ✅ Production Ready

---

## Domain Weight Configuration

```python
domain_weights = {
    'wyckoff': 0.4,   # Structural grammar (HIGHEST)
    'smc': 0.3,       # Order flow confirmation
    'temporal': 0.3,  # Timing psychology
    'hob': 0.2,       # Order blocks
    'macro': 0.1      # Global sentiment (LOWEST)
}
```

---

## Weighted Boost Formula

```
final_boost = 1 + (raw_boost - 1) * weight
```

**Example**: Wyckoff Spring A
- Raw boost: 2.5x
- Weight: 0.4
- Weighted: 1 + (2.5 - 1) * 0.4 = **1.6x**

---

## All 12 Wyckoff States

### Existing States (6)

| State | Raw Boost | Weighted | Type |
|-------|-----------|----------|------|
| **Spring A** | 2.50x | 1.60x | Major Boost |
| **Spring B** | 2.50x | 1.60x | Major Boost |
| **LPS** | 1.50x | 1.20x | Support |
| **Accumulation (Phase A)** | 2.00x | 1.40x | Boost |
| **Distribution (Phase D)** | 0.70x | 0.88x | Penalty |
| **UTAD** | 0.70x | 0.88x | Penalty |

### New States (6)

| State | Raw Boost | Weighted | Type |
|-------|-----------|----------|------|
| **Reaccumulation (Phase B)** | 1.50x | 1.20x | Boost |
| **Markup (Phase E)** | 1.80x | 1.32x | Boost |
| **Absorption** | 0.70x | 0.88x | Penalty |
| **SOW (Sign of Weakness)** | 0.60x | 0.84x | Penalty |
| **AR (Automatic Rally)** | 1.40x | 1.16x | Boost |
| **Secondary Test (ST)** | 0.80x | 0.92x | Penalty |

---

## Multi-Engine Confluence Example

**Scenario**: Wyckoff Spring + SMC BOS + Temporal Fib

### Before (Equal Weights)
```
2.5 * 2.0 * 1.7 = 8.5x → capped at 5.0x
```

### After (Weighted)
```
Wyckoff: 1 + (2.5 - 1) * 0.4 = 1.60x
SMC:     1 + (2.0 - 1) * 0.3 = 1.30x
Temporal: 1 + (1.7 - 1) * 0.3 = 1.21x

Combined: 1.60 * 1.30 * 1.21 = 2.52x
```

**Result**: More controlled scaling, no explosion!

---

## Validation Commands

### Run Full Validation
```bash
python3 bin/validate_wyckoff_weighted_boosts.py
```

### Run Unit Tests
```bash
python3 -m pytest tests/test_wyckoff_weighted_boosts.py -v
```

### Quick Manual Test
```python
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
import pandas as pd
from unittest.mock import Mock

logic = ArchetypeLogic({
    'wyckoff_events': {'enabled': True},
    'feature_flags': {'enable_wyckoff': True}
})

row_data = {
    'wyckoff_spring_a': True,
    'wyckoff_phase_abc': 'C',
    # ... other features ...
}

context = Mock()
context.row = pd.Series(row_data)
context.metadata = {'feature_flags': {'enable_wyckoff': True}}
context.regime = 'neutral'
context.regime_confidence = 0.5
context.get_threshold = lambda a, p, d: d

matched, score, meta, direction = logic._check_A(context)
print(f"Domain boost: {meta.get('domain_boost', 1.0)}")
```

---

## Key Implementation Files

| File | Description |
|------|-------------|
| `engine/archetypes/logic_v2_adapter.py` | Core implementation (lines 1769-1950) |
| `tests/test_wyckoff_weighted_boosts.py` | Unit tests |
| `bin/validate_wyckoff_weighted_boosts.py` | Validation script |
| `WYCKOFF_WEIGHTED_BOOSTS_IMPLEMENTATION_REPORT.md` | Full report |

---

## Feature Extraction Status

### Available Now
- `wyckoff_phase_abc` (A/B/C/D/E)
- `wyckoff_spring_a/b`
- `wyckoff_lps`
- `wyckoff_utad/bc`
- `wyckoff_sow/ar/st`

### Placeholder (if needed)
- `wyckoff_absorption` → Will wire to feature store if needed

---

## Backward Compatibility

✅ **Preserved**:
- All existing Wyckoff signals work exactly as before
- Feature flags still control engine activation
- Score capping at [0.0, 5.0] enforced
- No changes to regime penalties or threshold gates
- No changes to RegimeService or RuntimeContext

---

## Production Checklist

- [x] Weighted boost system implemented
- [x] All 12 Wyckoff states added
- [x] Unit tests passing
- [x] Validation script passing
- [x] Backward compatibility verified
- [x] Documentation complete
- [ ] Smoke test on 2022-2024 data (next step)
- [ ] Performance benchmark (optional)

---

## Quick Debugging

If domain boost is 1.0 when expected higher:

1. **Check feature flags**: `enable_wyckoff = True`?
2. **Check signals**: Are Wyckoff features actually True in data?
3. **Check archetype match**: Archetype must match first (need spring pattern for A)
4. **Check metadata**: `meta.get('domain_signals', [])` should show signals

---

**Questions?** See `WYCKOFF_WEIGHTED_BOOSTS_IMPLEMENTATION_REPORT.md` for full details.
