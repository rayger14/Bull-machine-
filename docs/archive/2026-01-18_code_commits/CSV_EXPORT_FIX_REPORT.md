# CSV Export Fix - Bear Archetypes Missing from Exports

**Date**: 2025-11-19
**Status**: ✅ FIXED
**Impact**: CRITICAL - Prevented accurate validation of bear archetypes

## Problem

Bear archetypes (S1-S8) were **firing correctly** during backtests, but **not being exported to CSV files**. This caused:

1. Phase 1 validation to incorrectly report "S2=0, S5=0"
2. Analysis scripts unable to count bear archetype performance
3. Optimization infrastructure unable to tune bear archetype thresholds

## Root Cause

File: `bin/backtest_knowledge_v2.py` line 2367

The `archetype_map` dictionary only contained 12 bull archetypes (A-M). When creating CSV records, bear archetypes were:
- ✓ Detected by the backtest engine (logs show "Trade X: archetype_failed_rally")
- ✗ Missing from the archetype_map
- ✗ Not included in exported CSV columns

## Fix

Added 8 bear archetype mappings to the archetype_map:

```python
archetype_map = {
    # Bull archetypes (A-M) - existing
    'trap': 'archetype_trap',
    'order_block_retest': 'archetype_retest',
    # ... 10 more bull archetypes

    # Bear archetypes (S1-S8) - ADDED
    'liquidity_vacuum': 'archetype_liquidity_vacuum',  # S1
    'failed_rally': 'archetype_failed_rally',  # S2
    'bear_ob_retest': 'archetype_bear_ob_retest',  # S3
    'distribution_climax': 'archetype_distribution_climax',  # S4
    'long_squeeze': 'archetype_long_squeeze',  # S5
    'bear_fvg_continuation': 'archetype_bear_fvg_continuation',  # S6
    'bear_compression': 'archetype_bear_compression',  # S7
    'bear_exhaustion_spike': 'archetype_bear_exhaustion_spike'  # S8
}
```

## Validation

**Before fix:**
- CSV columns: 47 (12 bull archetypes + 35 features)
- archetype_failed_rally: MISSING
- archetype_long_squeeze: MISSING

**After fix:**
- CSV columns: 56 (12 bull + 8 bear archetypes + 36 features)
- archetype_failed_rally: ✓ Present (418 trades in ultra_strict config)
- archetype_long_squeeze: ✓ Present (1 trade in ultra_strict config)

## Impact on Phase 1 Results

**Original Phase 1 results (INVALID)**:
```
ultra_strict: 524 trades, S2=0, S5=0, tier1=0
strict: 614 trades, S2=0, S5=0, tier1=0
moderate: 679 trades, S2=0, S5=0, tier1=0
relaxed: 742 trades, S2=0, S5=0, tier1=0
ultra_relaxed: 815 trades, S2=0, S5=0, tier1=0
```

**Corrected results (VALID - single config test)**:
```
ultra_strict (fusion=0.55): 524 trades
  - S2 (Failed Rally): 418 trades (79.8%)
  - S5 (Long Squeeze): 1 trade (0.2%)
  - Other: ~105 trades (20%)
```

## Next Steps

1. ✅ CSV export fixed
2. ✅ Validation script updated to count bear archetypes
3. 🔄 **TODO**: Re-run Phase 1 validation with all 5 configs
4. 🔄 **TODO**: Analyze S2 overtrading issue (418 trades at fusion=0.55 is too high)
5. 🔄 **TODO**: Investigate why S5 produces only 1 trade (threshold too strict?)

## Key Discovery

**S2 (Failed Rally) is HIGHLY SENSITIVE to thresholds.** At fusion=0.55 (ultra_strict), it produced 418 trades in 2022, far exceeding target of 5-10. This suggests:
- S2 detection logic is working
- Thresholds need significant adjustment
- May need fusion=0.70+ to achieve target trade count
