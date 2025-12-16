# Archetype Cleanup Quick Reference

**Last Updated**: 2025-12-12
**Status**: COMPLETE

---

## At a Glance

| Category | Count | Archetypes | Action Required |
|----------|-------|-----------|----------------|
| **Ghost** | 3 | P, Q, N | Commented out in registry - DO NOT enable |
| **Stub** | 2 | S6, S7 | Remove from configs (always returns False) |
| **Deprecated** | 1 | S2 | Remove from BTC configs (poor performance) |
| **Fully Wired** | 3 | S1, S4, S5 | Use in production (8x-12x boost) |
| **Partially Wired** | 3 | A, B, H | Wire domain engines next sprint |
| **Unwired** | 13 | C,D,E,F,G,K,L,M,S3,S8 | Basic fusion only (future wiring) |

---

## Critical Issues

### ISSUE 1: Ghost Archetypes (P, Q, N)
- **Problem**: Documented in registry but no implementation exists
- **Risk**: Configs enabling them will silently fail or crash
- **Fix**: Commented out in `engine/archetypes/registry.py`
- **Action**: Verify no configs reference P, Q, or N

### ISSUE 2: Stub Archetypes (S6, S7)
- **Problem**: Method exists but always returns False
- **Risk**: Enabled in 70 configs, wasting computation
- **Fix**: Added deprecation warnings in code
- **Action**: Remove `enable_S6` and `enable_S7` from all configs

### ISSUE 3: Wiring Inconsistency
- **Problem**: Only 3/19 archetypes have full domain engine wiring
- **Risk**: Signal quality varies 8x-12x across portfolio
- **Fix**: Documented standardization pattern
- **Action**: Wire H, A, B next sprint (high priority)

---

## What Changed

### Files Created
```
DEPRECATED_ARCHETYPES.md               # Full documentation
ARCHETYPE_AUDIT_TABLE.csv             # Status table
ARCHETYPE_WIRING_CONSISTENCY_REPORT.md # Wiring analysis
GHOST_ARCHETYPE_CLEANUP_SUMMARY.md    # Completion report
ARCHETYPE_CLEANUP_QUICK_REFERENCE.md  # This file
```

### Files Modified
```
engine/archetypes/registry.py         # Commented out P, Q, N
engine/archetypes/logic_v2_adapter.py # Added deprecation warnings
```

### No Deletions
All deprecated code is commented out, not deleted.

---

## Config Cleanup Checklist

Before enabling ANY archetype in production:

```
[ ] NOT in ghost list (P, Q, N)
[ ] NOT in stub list (S6, S7)
[ ] NOT deprecated (S2 for BTC)
[ ] Has working _check_X() method
[ ] Check domain wiring status (S1/S4/S5 = best)
[ ] Backtest validation passed
[ ] Paper trading validation passed
```

---

## Domain Wiring Status

### FULL (Use in Production) ✓
- **S1** - Liquidity Vacuum: Wyckoff + SMC + Temporal + HOB + PTI
- **S4** - Funding Divergence: Wyckoff + SMC + Temporal + HOB
- **S5** - Long Squeeze: Wyckoff + SMC + Temporal + HOB

**Boost**: 8x - 12x realistic (95x theoretical)
**Quality**: Best signal quality, lowest false positives

### PARTIAL (Wire Next Sprint)
- **A** - Trap Reversal: Has PTI, needs full Wyckoff/SMC
- **B** - Order Block: Has BOS/BOMS, needs full SMC/Wyckoff
- **H** - Trap Within Trend: Has HTF, needs full domain engines

**Boost**: 2x - 4x
**Quality**: Medium, needs full wiring

### NONE (Basic Fusion Only)
- **C,D,E,F,G,K,L,M** - Bull archetypes
- **S3, S8** - Bear archetypes

**Boost**: 1.0x (no domain knowledge)
**Quality**: Weakest, highest false positives

---

## Quick Commands

### Check which archetypes are in your config
```bash
python -c "
import json
with open('configs/mvp/mvp_bull_market_v1.json') as f:
    config = json.load(f)
    enabled = [k.replace('enable_', '') for k in config.get('archetypes', {}) if k.startswith('enable_')]
    print('Enabled archetypes:', ', '.join(enabled))
"
```

### Validate all configs
```bash
python bin/validate_configs.py --all
```

### Test domain wiring
```bash
python bin/test_domain_wiring.py --archetype S1 --variant core,full
```

### Backtest comparison
```bash
python bin/backtest.py --config YOUR_CONFIG.json --output results.csv
```

---

## Migration Priority

### High Priority (Next Sprint)
Wire domain engines for:
1. **H** (99 configs) - Trap Within Trend
2. **A** (99 configs) - Trap Reversal
3. **B** (99 configs) - Order Block Retest

**Impact**: 297 configs get 8x-12x quality boost

### Medium Priority (Future)
Wire domain engines for:
4. **G** (99 configs) - Re-accumulate
5. **K** (99 configs) - Wick Trap
6. **M** (98 configs) - Ratio Coil Break

### Low Priority (Consider Deprecating)
7. **C,D,E,F,L** - Low usage or overlap with wired archetypes
8. **S3, S8** - Bear archetypes, wire if needed

---

## Code Pattern

### Correct (Fully Wired)
```python
def _check_S1(context):
    base_fusion = context.row.get('fusion_score', 0.0)
    domain_boost = 1.0

    # Apply domain boosts
    if wyckoff_spring_a:
        domain_boost *= 2.5
    if smc_bos_bullish:
        domain_boost *= 2.0

    # Boost BEFORE threshold check
    final_score = base_fusion * domain_boost

    if final_score < threshold:
        return False, 0.0, {"reason": "below_threshold"}

    # Structural vetoes
    if supply_overhead:
        return False, 0.0, {"veto": "supply"}

    return True, final_score, meta
```

### Incorrect (Unwired)
```python
def _check_C(context):
    fusion = context.row.get('fusion_score', 0.0)

    # No domain boost
    if fusion < threshold:
        return False

    return True
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Archetypes | 24 (A-M, S1-S8, P, Q, N) |
| Working | 19 (79%) |
| Fully Wired | 3 (16% of working) |
| Ghost/Stub | 5 (21% of total) |
| Configs Affected | 1,287 |
| Quality Gap | 8x-12x (wired vs unwired) |

---

## Documentation Index

### Core Documents
- **DEPRECATED_ARCHETYPES.md** - Comprehensive ghost/stub documentation
- **ARCHETYPE_AUDIT_TABLE.csv** - Complete status table
- **ARCHETYPE_WIRING_CONSISTENCY_REPORT.md** - Detailed wiring analysis
- **GHOST_ARCHETYPE_CLEANUP_SUMMARY.md** - Full completion report

### Code Locations
- **Registry**: `engine/archetypes/registry.py`
- **Detection**: `engine/archetypes/logic_v2_adapter.py`
- **Threshold**: `engine/archetypes/threshold_policy.py`
- **Feature Flags**: `engine/feature_flags.py`

### Examples
- **Fully Wired**: `logic_v2_adapter.py:1321` (S1)
- **Partially Wired**: `logic_v2_adapter.py:869` (A)
- **Unwired**: `logic_v2_adapter.py:1013` (C)

---

## FAQ

**Q: Why are P, Q, N missing?**
A: They were documented but never implemented. Now commented out in registry to prevent confusion.

**Q: Why do S6, S7 return False?**
A: They require external data (TOTAL3, yield curve) not in feature store. Stub implementations prevent crashes.

**Q: Should I remove S2?**
A: For BTC, yes (poor performance). For altcoins, maybe keep it. Use S4 instead for BTC.

**Q: Why only 3 fully wired archetypes?**
A: S1, S4, S5 were priority (bear market focus). Others need wiring in future sprints.

**Q: What's the domain boost range?**
A: Theoretical max is 95x (all engines firing). Realistic is 8x-12x. Unwired archetypes get 1.0x (no boost).

**Q: How do I wire an archetype?**
A: Follow the S1/S4/S5 pattern in wiring consistency report. Add Wyckoff, SMC, Temporal, HOB boosting before fusion gate.

---

## Contact

For questions about archetype cleanup:
- See: `DEPRECATED_ARCHETYPES.md` (comprehensive doc)
- See: `ARCHETYPE_WIRING_CONSISTENCY_REPORT.md` (wiring guide)
- See: `GHOST_ARCHETYPE_CLEANUP_SUMMARY.md` (completion report)
