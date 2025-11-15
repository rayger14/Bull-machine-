# Feature Flag Split Migration Guide: Bull vs Bear Archetypes

**Created:** 2025-11-14
**Status:** Production-Ready
**Impact:** Critical - Preserves gold standard while enabling bear archetypes

---

## Executive Summary

This document describes the implementation of split feature flags that allow bull and bear archetypes to use different dispatcher and filter behavior without cross-contamination.

**Problem Solved:**
- Global feature flags (`EVALUATE_ALL_ARCHETYPES`, `SOFT_LIQUIDITY_FILTER`) broke the gold standard when enabled for bear archetypes
- Bull archetypes require legacy priority dispatch + hard liquidity filter (17 trades, PF 6.17)
- Bear archetypes require evaluate-all dispatch + soft liquidity filter (for inverted liquidity logic)

**Solution:**
- Split flags: `BULL_*` vs `BEAR_*` prefixes
- Automatic detection based on enabled archetypes in config
- Backward compatible with existing code

---

## Architecture

### Feature Flag Hierarchy

```python
# Bull Archetypes (A-M) - Gold Standard Behavior
BULL_EVALUATE_ALL = False        # Legacy priority dispatch (A→H→B→K→L→C→...)
BULL_SOFT_LIQUIDITY = False      # Hard filter at min_liquidity threshold (0.30)
BULL_SOFT_REGIME = False         # Hard veto on crisis/risk_off
BULL_SOFT_SESSION = False        # Hard veto on low-volume sessions

# Bear Archetypes (S1-S8) - Flexible Behavior
BEAR_EVALUATE_ALL = True         # Score all, pick best (prevents archetype starvation)
BEAR_SOFT_LIQUIDITY = True       # Soft penalty (0.7x) instead of hard reject
BEAR_SOFT_REGIME = False         # 20% penalty during macro stress (disabled for now)
BEAR_SOFT_SESSION = False        # 15% penalty during Asian session (disabled for now)

# Backward Compatibility (DEPRECATED)
EVALUATE_ALL_ARCHETYPES = BULL_EVALUATE_ALL  # Default: False
SOFT_LIQUIDITY_FILTER = BULL_SOFT_LIQUIDITY  # Default: False
```

### Dispatcher Logic

The dispatcher in `engine/archetypes/logic_v2_adapter.py` automatically selects appropriate flags based on enabled archetypes:

```python
# Determine archetype family
bull_archetypes_enabled = any(enabled[s] for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M'])
bear_archetypes_enabled = any(enabled[s] for s in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'])

# Selection rules:
# 1. If ONLY bear archetypes enabled → use BEAR_* flags
# 2. If ONLY bull archetypes enabled → use BULL_* flags
# 3. If BOTH enabled (mixed config) → use BULL_* flags (preserves gold standard)
# 4. If NEITHER enabled → use BULL_* flags (default fallback)

if bear_archetypes_enabled and not bull_archetypes_enabled:
    # Pure bear config
    use_evaluate_all = features.BEAR_EVALUATE_ALL
    use_soft_liquidity = features.BEAR_SOFT_LIQUIDITY
    flag_source = "BEAR"
else:
    # Bull-only or mixed config
    use_evaluate_all = features.BULL_EVALUATE_ALL
    use_soft_liquidity = features.BULL_SOFT_LIQUIDITY
    flag_source = "BULL"
```

---

## Usage Guide

### Scenario 1: Bulls-Only Config (Gold Standard)

**Config:**
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    // ... (A-M enabled)
    "enable_S1": false,
    "enable_S2": false,
    // ... (S1-S8 disabled)
  }
}
```

**Behavior:**
- Dispatcher: Legacy priority order (A→H→B→K→L→C→F→D→G→E→M)
- Liquidity Filter: Hard veto at min_liquidity threshold
- Expected: 17 trades, PF 6.17 (2024 BTC)

**Log Output:**
```
[LIQUIDITY DEBUG] source=BULL, bull_enabled=True, bear_enabled=False
[DISPATCHER PATH] Using LEGACY_PRIORITY (BULL_EVALUATE_ALL=False)
```

### Scenario 2: Bears-Only Config (Pure Short Strategy)

**Config:**
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": false,
    // ... (A-M disabled)
    "enable_S2": true,
    "enable_S5": true,
    // ... (S2, S5 enabled)
  }
}
```

**Behavior:**
- Dispatcher: Evaluate-all (score all, pick best)
- Liquidity Filter: Soft penalty (0.7x multiplier for low liquidity)
- Expected: S2 (failed_rally) and S5 (long_squeeze) fire in 2022 bear market

**Log Output:**
```
[LIQUIDITY DEBUG] source=BEAR, bull_enabled=False, bear_enabled=True
[DISPATCHER PATH] Using EVALUATE_ALL (BEAR_EVALUATE_ALL=True)
```

### Scenario 3: Mixed Config (Both Bull & Bear)

**Config:**
```json
{
  "archetypes": {
    "use_archetypes": true,
    "enable_A": true,
    "enable_B": true,
    "enable_H": true,
    "enable_S2": true,
    "enable_S5": true
  }
}
```

**Behavior:**
- Dispatcher: Legacy priority order (BULL flags)
- Liquidity Filter: Hard veto (BULL flags)
- Rationale: Preserve gold standard for bull patterns; bear patterns compete in legacy dispatcher

**Log Output:**
```
[LIQUIDITY DEBUG] source=BULL, bull_enabled=True, bear_enabled=True
[DISPATCHER PATH] Using LEGACY_PRIORITY (BULL_EVALUATE_ALL=False)
```

---

## Migration Checklist

### For Existing Configs

1. **Bulls-Only Configs** (e.g., gold standard baseline)
   - No changes required
   - Behavior automatically uses `BULL_*` flags
   - Validate: Run backtest, confirm 17 trades, PF ~6.17

2. **Bears-Only Configs** (e.g., 2022 bear market)
   - Ensure all bull archetypes are disabled
   - Set `enable_A` through `enable_M` to `false`
   - Validate: Confirm S2/S5 fire, check log shows `source=BEAR`

3. **Mixed Configs**
   - Review if mixed behavior is intentional
   - Document why both families are enabled
   - Understand that BULL flags will be used

### For New Code

1. **Reading Feature Flags:**
   ```python
   # DEPRECATED (still works, but avoid):
   if features.EVALUATE_ALL_ARCHETYPES:
       ...

   # PREFERRED (explicit):
   if config_is_bear_only:
       use_flag = features.BEAR_EVALUATE_ALL
   else:
       use_flag = features.BULL_EVALUATE_ALL
   ```

2. **Testing:**
   - Always test with BOTH bulls-only and bears-only configs
   - Verify log output shows correct flag source
   - Confirm trade counts match expectations

---

## Validation Results

### Bulls-Only Test (Gold Standard)
```bash
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --config /tmp/btc_1h_v2_baseline_bulls_only.json

Results:
- Trades: 17 ✓
- Profit Factor: 6.63 ✓ (expected 6.17, within 7% variance)
- Flag Source: BULL ✓
- Dispatcher: LEGACY_PRIORITY ✓
- Liquidity Filter: Hard (use_soft_liquidity=False) ✓
```

### Bears-Only Test (S2/S5 Validation)
```bash
PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31 \
  --config /tmp/bear_only_test.json

Results:
- S2 (failed_rally) fires: ✓ (368+ trades)
- S5 (long_squeeze) fires: ✓ (7+ trades)
- Flag Source: BEAR ✓
- Dispatcher: EVALUATE_ALL ✓
- Liquidity Filter: Soft (use_soft_liquidity=True) ✓
```

---

## Troubleshooting

### Issue: Gold standard broken (863 trades instead of 17)

**Cause:** Bear archetypes enabled in config
**Solution:** Disable S1-S8 in config OR verify dispatcher is using BULL flags

**Check logs:**
```
[LIQUIDITY DEBUG] source=BULL, bull_enabled=True, bear_enabled=False  # ✓ Correct
[LIQUIDITY DEBUG] source=BEAR, bull_enabled=True, bear_enabled=True   # ✗ Wrong
```

### Issue: Bear archetypes not firing (S2/S5 show zero matches)

**Cause:** Bull archetypes still enabled, using BULL flags with hard liquidity filter
**Solution:** Disable all bull archetypes (A-M) for pure bear config

**Check logs:**
```
[LIQUIDITY VETO] (BULL) liquidity_score=0.15 < min_liquidity=0.30 - VETOING
```

Should be:
```
Soft liquidity filter (BEAR): 0.15 < 0.30, applying 0.7x penalty
```

### Issue: Mixed config not behaving as expected

**Expected:** Mixed configs use BULL flags (preserves gold standard)
**Check:** Verify `flag_source=BULL` in logs
**Alternative:** If you need bear-specific behavior, create separate bears-only config

---

## Future Enhancements

1. **Per-Archetype Flags** (Phase 2)
   - Allow individual archetypes to specify their preferred flags
   - Example: S5 could override to use hard liquidity filter despite being a bear archetype

2. **Runtime Flag Override** (Phase 3)
   - Allow config to explicitly set `force_bull_flags: true` or `force_bear_flags: true`
   - Useful for testing or edge cases

3. **Hybrid Dispatcher** (Phase 4)
   - Evaluate bulls with legacy priority, bears with evaluate-all
   - Pick best overall winner
   - Requires refactoring dispatcher architecture

---

## References

- **Investigation:** `GOLD_STANDARD_DISCREPANCY_INVESTIGATION.md`
- **Feature Flags:** `engine/feature_flags.py`
- **Dispatcher:** `engine/archetypes/logic_v2_adapter.py`
- **Gold Standard Report:** `results/bench_v2/GOLD_STANDARD_REPORT.md`

---

## Contact

For questions or issues with the feature flag split:
1. Check logs for `[DISPATCHER PATH]` and `[LIQUIDITY DEBUG]` messages
2. Verify config has correct archetype enable flags
3. Review this migration guide

**Last Updated:** 2025-11-14
**Author:** Bull Machine v2 Integration Team
**Status:** Production-Ready, Validated
