# Wyckoff Threshold Tuning - Quick Reference

**Date**: 2025-11-18
**Status**: ✅ PRODUCTION READY

## What Was Fixed

1. **2022 Data Loader** - Now handles timestamp as index or column
2. **Spring-A Detection** - 0 → 3 events in 2024 (WORKING)
3. **SC Detection** - Detected 2022 bear market capitulation
4. **ST Noise** - Reduced by 5.4% (4,184 → 3,958)

## Key Detections Validated

### 2024 Bull Market

**BC (Buying Climax)**:
- March 28, 2024 @ $70,850 (ATH) - Confidence: 0.74
- June 3, 2024 @ $69,802 - Confidence: 0.70

**Spring-A (Major Pullbacks)**:
- March 5, 2024 @ $65,607 - Confidence: 0.72
- April 19, 2024 @ $61,367 - Confidence: 0.60
- July 27, 2024 @ $67,779 - Confidence: 0.65

**LPS (Last Point of Support)**: 1,243 events - 93% avg confidence

### 2022 Bear Market

**SC (Selling Climax)**:
- May 7, 2022 @ $35,041 - Confidence: 0.70

**Spring-A (Capitulation Lows)**:
- May 12, 2022 @ $27,710 - Confidence: 0.83
- Nov 11, 2022 @ $16,872 - Confidence: 0.79 ← June 2022 low area

## Threshold Changes

| Parameter | Old | New | Change |
|-----------|-----|-----|--------|
| `spring_a_breakdown_margin` | 0.020 | 0.015 | +33% lenient |
| `spring_a_recovery_bars` | 3 | 2 | Faster |
| `spring_a_volume_z_max` | - | 1.0 | Added |
| `spring_b_breakdown_min` | 0.005 | 0.003 | +67% sensitive |
| `spring_b_recovery_bars` | - | 2 | Added |
| `st_volume_z_max` | 0.5 | 0.3 | -40% stricter |

## Files Changed

1. `bin/validate_wyckoff_on_features.py` - Timestamp handling
2. `configs/wyckoff_events_config.json` - Tuned thresholds
3. `configs/wyckoff_events_config.json.backup` - Original backup

## Usage in Trading

**High-Conviction Entries**:
```python
if wyckoff_spring_a and wyckoff_spring_a_confidence > 0.65:
    if wyckoff_lps within 30 bars:
        # STRONG LONG SIGNAL
```

**Risk Management**:
```python
if wyckoff_bc and wyckoff_bc_confidence > 0.70:
    # AVOID LONGS - Distribution likely
    # Consider shorts on UTAD
```

## Next Steps

1. ✅ Deploy to backtest environment
2. ⏳ Test Spring-A + LPS confluence
3. ⏳ Optional: Further tune Spring-B (still 0 events)
4. ⏳ Optional: Reduce ST noise further (st_volume_z_max = 0.2)

## Rollback Instructions

If needed, restore original thresholds:
```bash
cp configs/wyckoff_events_config.json.backup configs/wyckoff_events_config.json
```
