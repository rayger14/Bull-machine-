# DOMAIN ENGINE WIRING - QUICK REFERENCE

## TL;DR

Domain engines (Wyckoff, SMC, Temporal, HOB, Macro, Fusion) are now wired into S1, S4, S5 archetype check functions. They modify archetype scores via multiplicative boosts/vetoes.

---

## Enable Domain Engines (Config)

Add to your config's `feature_flags` section:

```json
"feature_flags": {
  "enable_wyckoff": true,   // Wyckoff structural events
  "enable_smc": true,        // Smart Money Concepts
  "enable_temporal": true,   // Temporal confluence (PTI)
  "enable_hob": false,       // Higher Order Blocks (not yet implemented)
  "enable_macro": true,      // Macro regime filters
  "enable_fusion": false     // Meta-fusion (not yet implemented)
}
```

---

## What Gets Boosted/Vetoed

### S1 Liquidity Vacuum (Long Reversals)

| Domain | Trigger | Effect | Why |
|---|---|---|---|
| Wyckoff | Spring A/B detected | **+25%** score | Major capitulation confirmation |
| Wyckoff | Preliminary Support | **+15%** score | Early capitulation signal |
| SMC | Bullish structure (score > 0.5) | **+15%** score | Break of structure confirms |
| Temporal | PTI confluence true | **+10%** score | Time cluster adds conviction |
| Macro | Extreme crisis (>70%) | **-15%** score | Avoid falling knife |

### S4 Funding Divergence (Short Squeeze)

| Domain | Trigger | Effect | Why |
|---|---|---|---|
| Wyckoff | UTAD or SOW detected | **VETO** | Don't long into distribution |
| Wyckoff | Accumulation phase | **+20%** score | Accumulation amplifies squeeze |
| SMC | Bullish structure (score > 0.6) | **+15%** score | Liquidity sweep confirms |
| Temporal | PTI confluence true | **+10%** score | Time confluence adds conviction |

### S5 Long Squeeze (Short Cascades)

| Domain | Trigger | Effect | Why |
|---|---|---|---|
| Wyckoff | UTAD or distribution phase | **+20%** score | Distribution confirms top |
| Wyckoff | Sign of Weakness | **+10%** score | Weakness signal supports short |
| SMC | Bearish structure (score < -0.5) | **+15%** score | Supply zone confirms resistance |
| Temporal | PTI support (score < -0.5) | **VETO** | Don't short into support |
| Temporal | PTI resistance (score > 0.5) | **+10%** score | Resistance cluster confirms |

---

## Test Configs

### Baseline (No Domains)
```bash
python bin/backtest_single.py \
  --config configs/test/s1_core_only.json \
  --symbol BTCUSDT --start_date 2022-01-01 --end_date 2022-12-31
```

### Wyckoff Only
```bash
python bin/backtest_single.py \
  --config configs/test/s1_wyckoff_only.json \
  --symbol BTCUSDT --start_date 2022-01-01 --end_date 2022-12-31
```

### All Domains
```bash
python bin/backtest_single.py \
  --config configs/test/s1_all_domains.json \
  --symbol BTCUSDT --start_date 2022-01-01 --end_date 2022-12-31
```

---

## Verify Wiring is Active

Check trade metadata for:
```python
{
  "domain_boost": 1.25,  // Multiplier applied to score
  "domain_signals": [    // Which engines fired
    "wyckoff_spring_detected",
    "smc_bullish_structure",
    "temporal_confluence_detected"
  ]
}
```

---

## Feature Availability

| Feature | Status | Archetype |
|---|---|---|
| `wyckoff_spring_a/b` | ✅ Available | S1, S4 |
| `wyckoff_ps` | ✅ Available | S1 |
| `wyckoff_utad` | ✅ Available | S4, S5 |
| `wyckoff_sow` | ✅ Available | S4, S5 |
| `wyckoff_phase_abc` | ✅ Available | S4, S5 |
| `wyckoff_pti_confluence` | ✅ Available | S1, S4, S5 |
| `wyckoff_pti_score` | ✅ Available | S5 |
| `smc_score` | ✅ Available | S1, S4, S5 |
| `hob_demand_zone` | ❌ Placeholder | S1 |
| `hob_supply_zone` | ❌ Placeholder | S5 |

---

## Default Behavior (No Config Change)

All domain engines default to **OFF** (backward compatible):
- Existing configs work unchanged
- No domain boosts applied unless explicitly enabled
- Trade count/PF matches baseline

---

## Optimization Impact

Domain engines affect archetype quality, so re-optimization recommended:
1. Enable domain engines in config
2. Re-run archetype optimization
3. Expect different optimal thresholds (domain boosts change quality)

---

## Files Modified

- `/engine/archetypes/logic_v2_adapter.py` (S1, S4, S5 check functions)
- `/engine/feature_flags.py` (added 6 domain engine flags)
- `/configs/test/s1_*.json` (3 test configs for verification)

---

## Quick Boost Magnitude Reference

- **Strong:** +20-25% (Wyckoff major events)
- **Medium:** +15% (SMC structure shifts)
- **Weak:** +10% (Temporal confluence)
- **Penalty:** -15-20% (Macro extreme stress)
- **Veto:** Hard block (return False immediately)

---

## Next Steps

1. Run test configs to verify wiring works
2. Inspect trade metadata for `domain_boost`/`domain_signals`
3. Compare trade counts across variants (core vs wyckoff vs all)
4. Re-optimize archetypes with domains enabled
5. Add HOB features to registry (activate HOB placeholders)

---

**Version:** v1.0
**Date:** 2025-12-10
**Status:** Ready for Testing
