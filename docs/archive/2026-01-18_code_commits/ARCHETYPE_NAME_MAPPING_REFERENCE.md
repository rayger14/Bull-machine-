# Archetype Name Mapping Reference

**Quick reference for archetype ID → Name → Config Key mappings**

---

## Current State (Bull Machine v2)

### Letter → Runtime Name → Config Query Key

| Letter | Runtime Name | Config Query Key | Check Function | Registry Canonical | Status |
|--------|--------------|------------------|----------------|-------------------|--------|
| **A** | trap_reversal | `spring` | `_check_A` | wyckoff_spring_utad | ⚠️ Hard PTI gate |
| **B** | order_block_retest | `order_block_retest` | `_check_B` | order_block_retest | ✓ Working |
| **C** | fvg_continuation | `wick_trap` (!) | `_check_C` | bos_choch_reversal | ⚠️ Name mismatch |
| **D** | failed_continuation | `failed_continuation` | `_check_D` | failed_continuation | ✓ Working |
| **E** | liquidity_compression | `volume_exhaustion` (!) | `_check_E` | liquidity_compression | ⚠️ Name mismatch |
| **F** | expansion_exhaustion | `exhaustion_reversal` | `_check_F` | expansion_exhaustion | ✓ Working |
| **G** | re_accumulate | `liquidity_sweep` | `_check_G` | liquidity_sweep_reclaim | ⚠️ Strict defaults |
| **H** | trap_within_trend | `trap_within_trend` | `_check_H` | trap_within_trend | ✓ Working |
| **K** | wick_trap | `wick_trap_moneytaur` | `_check_K` | wick_trap_moneytaur | ⚠️ Name mismatch |
| **L** | volume_exhaustion | `volume_exhaustion` | `_check_L` | fakeout_real_move | ⚠️ Name mismatch |
| **M** | ratio_coil_break | `confluence_breakout` | `_check_M` | ratio_coil_break | ⚠️ Name mismatch |

### Bear Archetypes (Short-Biased)

| Letter | Runtime Name | Config Query Key | Check Function | Status |
|--------|--------------|------------------|----------------|--------|
| **S1** | breakdown | `breakdown` | `_check_S1` | ✓ Working |
| **S2** | failed_rally | `failed_rally` | `_check_S2` | ✓ Working |
| **S3** | whipsaw | `whipsaw` | `_check_S3` | Disabled |
| **S4** | distribution | `distribution` | `_check_S4` | Disabled |
| **S5** | long_squeeze | `long_squeeze` | `_check_S5` | ✓ Working |
| **S6** | alt_rotation_down | `alt_rotation_down` | `_check_S6` | Disabled |
| **S7** | curve_inversion | `curve_inversion` | `_check_S7` | Disabled |
| **S8** | volume_fade_chop | `volume_fade_chop` | `_check_S8` | Disabled |

---

## Optimizer Archetype Groups (INCORRECT MAPPINGS!)

From `bin/optuna_parallel_archetypes_v2.py`:

```python
ARCHETYPE_GROUPS = {
    'trap_within_trend': {
        'archetypes': ['A', 'G', 'K'],  # ❌ WRONG!
        'canonical': ['trap_within_trend', 'liquidity_sweep', 'spring'],
    },
    'order_block_retest': {
        'archetypes': ['B', 'H', 'L'],  # ❌ H should be separate!
        'canonical': ['order_block_retest', 'momentum_continuation', 'volume_exhaustion'],
    },
    'bos_choch': {
        'archetypes': ['C'],
        'canonical': ['wick_trap'],  # ❌ Name mismatch with registry!
    },
}
```

**Problems**:
1. **A, G, K grouped as "trap_within_trend"** but H is the actual trap_within_trend!
2. **H grouped with B, L** but H is fundamentally different (inverted liquidity logic)
3. **Canonical names don't match config query keys** (optimizer writes to wrong sections)

---

## Correct Optimizer Groups (PROPOSED FIX)

```python
ARCHETYPE_GROUPS = {
    'spring_utad': {
        'archetypes': ['A'],
        'canonical': ['spring'],  # Match _check_A query
        'description': 'PTI-based spring/UTAD reversal (requires PTI feature)',
    },
    'order_block_retest': {
        'archetypes': ['B'],
        'canonical': ['order_block_retest'],
        'description': 'BOMS + Wyckoff structure retest',
    },
    'fvg_continuation': {
        'archetypes': ['C'],
        'canonical': ['wick_trap'],  # Match _check_C query (despite name!)
        'description': 'FVG + displacement continuation',
    },
    'liquidity_sweep': {
        'archetypes': ['G'],
        'canonical': ['liquidity_sweep'],  # Match _check_G query
        'description': 'BOMS strength + high liquidity sweep',
    },
    'trap_within_trend': {
        'archetypes': ['H'],
        'canonical': ['trap_within_trend'],  # Match _check_H query
        'description': 'ADX trend + LOW liquidity trap (inverted logic)',
    },
    'wick_trap': {
        'archetypes': ['K'],
        'canonical': ['wick_trap_moneytaur'],  # Match _check_K query
        'description': 'Wick rejection + ADX + liquidity',
    },
    'volume_exhaustion': {
        'archetypes': ['L'],
        'canonical': ['volume_exhaustion'],  # Match _check_L query
        'description': 'Volume spike + extreme RSI',
    },
}
```

---

## Config Structure Examples

### What the Optimizer SHOULD Write

For archetype K (wick_trap):
```json
{
  "archetypes": {
    "enable_K": true,
    "thresholds": {
      "wick_trap_moneytaur": {
        "fusion_threshold": 0.44,
        "adx_threshold": 25.0,
        "liquidity_threshold": 0.30
      }
    },
    "wick_trap_moneytaur": {
      "archetype_weight": 1.1,
      "final_fusion_gate": 0.44,
      "cooldown_bars": 10
    }
  }
}
```

### What the Optimizer CURRENTLY Writes (BROKEN!)

```json
{
  "archetypes": {
    "enable_K": true,
    "thresholds": {
      "wick_trap": {  // ❌ Wrong name! _check_K queries 'wick_trap_moneytaur'
        "fusion_threshold": 0.44,
        "adx_threshold": 25.0
      }
    }
  }
}
```

---

## Quick Diagnosis Guide

### "Why isn't archetype X firing?"

1. **Check enable flag**: `enable_X: true` in config?
2. **Check dispatcher order**: Is X checked AFTER archetypes that always match? (early return problem)
3. **Check hard gates**: Does _check_X have feature requirements not in data?
4. **Check name mapping**: Does config query key match what optimizer writes?
5. **Check defaults**: Are default thresholds too strict/permissive?

### "Why is wick_trap dominating?"

1. **Early in priority order**: K is checked 4th (A, B, C, then K)
2. **Permissive defaults**: Requires ADX >= 25, liq >= 0.30 (commonly met)
3. **Simple boolean check**: No complex scoring, just pass/fail gates
4. **Legacy dispatch**: First match wins, K matches before H/L get a chance

### "Why didn't optimization improve performance?"

1. **Name mismatch**: Optimizer writes to different config key than check function queries
2. **Hard gates**: Optimization can't bypass feature requirements (e.g., PTI trap type)
3. **Zero-variance bug**: All trials use same defaults when config lookup fails
4. **Wrong archetype**: Optimizing A when you meant H (letter confusion)

---

## Migration Path (Long-Term)

### Phase 1: Document Current State ✓
- Create this reference guide
- Add warnings to optimizer about name mismatches

### Phase 2: Add Alias Resolution
```python
def get_threshold(self, archetype_name, param_name, default):
    # Resolve aliases: 'spring' → canonical name
    canonical = resolve_archetype_key(archetype_name, warn=False)
    # Look up using canonical name
```

### Phase 3: Standardize Check Functions
Update all `_check_X` to use canonical registry names:
```python
# Old: fusion_th = context.get_threshold('spring', 'fusion_threshold', 0.33)
# New: fusion_th = context.get_threshold('wyckoff_spring_utad', 'fusion_threshold', 0.33)
```

### Phase 4: Deprecate Letter Codes
- Migrate configs to use canonical slugs
- Add warnings when letter codes detected
- Update optimizer to use canonical slugs

### Phase 5: Class-Based Archetypes
Replace check functions with ArchetypePattern classes:
```python
class WyckoffSpringUtad(ArchetypePattern):
    canonical_name = 'wyckoff_spring_utad'
    query_key = 'wyckoff_spring_utad'  # Same as canonical
    letter_code = 'A'  # For backward compat

    def check(self, context: RuntimeContext) -> tuple:
        fusion_th = context.get_threshold(self.query_key, 'fusion_threshold', 0.33)
        # ...
```

---

## References

- **Registry**: `/engine/archetypes/registry.py` (canonical names, aliases, priorities)
- **Check Logic**: `/engine/archetypes/logic_v2_adapter.py` (archetype_map, _check_X functions)
- **Optimizer**: `/bin/optuna_parallel_archetypes_v2.py` (ARCHETYPE_GROUPS)
- **Threshold Resolver**: `/engine/runtime/context.py` (get_threshold method)
- **Investigation Report**: `/TRAP_WITHIN_TREND_ZERO_MATCHES_INVESTIGATION.md`
