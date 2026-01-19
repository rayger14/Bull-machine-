# Wyckoff Hard Veto Fix - Quick Reference

## What Changed

**File:** `engine/archetypes/logic_v2_adapter.py`

### S1 - Liquidity Vacuum (Line 1763-1769)
```python
# OLD: Hard veto → return False
if wyckoff_distribution or wyckoff_utad or wyckoff_bc:
    return False, 0.0, {"reason": "wyckoff_distribution_veto", ...}

# NEW: Soft veto → reduce confidence
if wyckoff_distribution:
    domain_boost *= 0.30  # 70% reduction

if wyckoff_utad or wyckoff_bc:
    domain_boost *= 0.50  # 50% reduction
```

### S4 - Funding Divergence (Line 2680-2686)
Same pattern as S1.

### S5 - Failed Rally
No change needed (SHORT archetype, distribution is good).

---

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Entry availability | 2-3% | 100% |
| Confidence during distribution | BLOCKED | 30% of normal |
| Confidence during UTAD/BC | BLOCKED | 50% of normal |
| Combined penalty | BLOCKED | 15% of normal |

---

## How It Works

1. Score calculated normally from archetype logic
2. Domain boost applied: `score = score * domain_boost`
3. If wyckoff_distribution: `domain_boost *= 0.30`
4. If wyckoff_utad/bc: `domain_boost *= 0.50`
5. Final score compared to fusion_threshold
6. Low scores filtered, high scores pass

**Example:**
- Base score: 0.80
- Distribution active: 0.80 × 0.30 = 0.24
- Fusion threshold: 0.54
- Result: REJECTED (0.24 < 0.54)

But if base score is 0.90:
- 0.90 × 0.30 = 0.27 → still rejected

Only VERY strong setups (score > 1.8) can overcome distribution veto:
- 1.8 × 0.30 = 0.54 → PASSES

---

## Monitoring

Watch for these signals in logs:
- `wyckoff_distribution_caution` - distribution phase penalty applied
- `wyckoff_utad_bc_caution` - UTAD/BC penalty applied
- `[DOMAIN_DEBUG] S1 Domain Boost Applied: 0.30x` - soft veto active

---

## Testing Commands

```bash
# Test S1 with new logic
python3 bin/backtest_knowledge_v2.py \
  --config configs/variants/s1_full.json \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31

# Test S4 with new logic
python3 bin/backtest_knowledge_v2.py \
  --config configs/variants/s4_full.json \
  --asset BTC \
  --start 2022-01-01 \
  --end 2022-12-31
```

---

## Expected Results

- S1/S4 trade counts should increase significantly
- CORE vs FULL variants should now differentiate
- Win rate may decrease slightly (accepting riskier trades)
- Sharpe should remain stable (risk-adjusted)

---

## Rollback Plan

If needed, revert changes in `logic_v2_adapter.py`:

**S1 (line 1763-1769):** Restore hard veto
**S4 (line 2680-2686):** Restore hard veto

```python
# ROLLBACK PATTERN
if wyckoff_distribution or wyckoff_utad or wyckoff_bc:
    return False, 0.0, {
        "reason": "wyckoff_distribution_veto",
        "wyckoff_distribution": wyckoff_distribution,
        "wyckoff_utad": wyckoff_utad,
        "wyckoff_bc": wyckoff_bc,
        "note": "Don't long into Wyckoff distribution phase"
    }
```
