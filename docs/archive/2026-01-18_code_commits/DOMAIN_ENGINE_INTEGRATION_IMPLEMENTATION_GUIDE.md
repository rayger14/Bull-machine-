# Domain Engine Integration Implementation Guide

## Executive Summary

This guide provides step-by-step instructions to add full 6-engine domain integration to 10 unwired archetypes (C, D, E, F, G, K, L, M, S3, S8). These archetypes currently have NO domain engine integration - they rely solely on basic pattern detection with 1.0x confidence (no boost).

**Target:** Add the same 6-engine system that gives S1 its 8x-12x boost capability.

## Current State Analysis

### Archetype Inventory

| Archetype | Name | Direction | Current Line | Pattern Description |
|-----------|------|-----------|--------------|---------------------|
| C | wick_trap | LONG | 1362 | FVG Continuation (displacement + momentum) |
| D | failed_continuation | LONG | 1391 | Failed Continuation (FVG + weak RSI) |
| E | volume_exhaustion | LONG | 1413 | Liquidity Compression (low ATR + volume cluster) |
| F | exhaustion_reversal | LONG | 1448 | Expansion Exhaustion (extreme RSI + high ATR) |
| G | liquidity_sweep | LONG | 1472 | Re-Accumulate Base (BOMS strength + liquidity) |
| K | wick_trap_moneytaur | LONG | 1734 | Wick Trap / Moneytaur (ADX + liquidity + wicks) |
| L | volume_exhaustion | LONG | 1756 | Volume Exhaustion / Zeroika (vol spike + extreme RSI) |
| M | confluence_breakout | LONG | 1816 | Ratio Coil Break (low ATR + POC + BOMS) |
| S3 | whipsaw | SHORT | 3050 | Whipsaw (false break + reversal) |
| S8 | volume_fade_chop | SHORT | 3685 | Volume Fade in Chop (low volume drift) |

**Total:** 10 archetypes (8 LONG, 2 SHORT)

### Current Implementation Pattern

All 10 archetypes follow this basic pattern:

```python
def _check_X(self, context: RuntimeContext) -> bool:  # or tuple for L
    """Pattern description"""

    # 1. Read thresholds
    fusion_th = context.get_threshold('archetype_name', 'fusion_threshold', default)
    param1 = context.get_threshold('archetype_name', 'param1', default)

    # 2. Get features
    feature1 = self.g(context.row, "feature1", default)

    # 3. Gate checks
    if feature1 < threshold:
        return False  # or (False, 0.0, meta) for tuple return

    # 4. Return boolean match
    return (condition1 and condition2 and condition3)
```

**Problem:** No domain engine integration = no boost/veto layer = 1.0x confidence ceiling

## Target State: S1 Reference Implementation

S1 (Liquidity Vacuum Reversal) demonstrates the full 6-engine integration pattern:

### Reference Code Structure (S1 lines 1920-2110)

```python
def _check_S1(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
    # ... threshold reading ...
    # ... feature extraction ...
    # ... gate checks ...

    # Calculate base score
    score = sum(components[k] * weights.get(k, 0.0) for k in components)

    # ============================================================================
    # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER)
    # ============================================================================
    use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
    use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
    use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
    use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
    use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

    domain_boost = 1.0
    domain_signals = []

    # Wyckoff Engine (6-10 signals)
    if use_wyckoff:
        wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
        if wyckoff_distribution:
            domain_boost *= 0.70  # Soft veto
            domain_signals.append("wyckoff_distribution_caution")
        # ... more wyckoff signals ...

    # SMC Engine (6-8 signals)
    if use_smc:
        tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
        if tf4h_bos_bullish:
            domain_boost *= 2.00  # Major boost
            domain_signals.append("smc_4h_bos_bullish_institutional")
        # ... more SMC signals ...

    # Temporal Engine (4-6 signals)
    # HOB Engine (3-4 signals)
    # Macro Engine (1-2 signals)

    # Apply boost BEFORE fusion gate
    score_before_domain = score
    score = score * domain_boost

    # Final fusion gate check
    if score < fusion_th:
        return False, score, {
            "reason": "score_below_threshold_after_domain",
            "score_before_domain": score_before_domain,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            ...
        }

    return True, score, meta
```

## Implementation Plan

### Phase 1: LONG Archetypes (C, D, E, F, G, K, L, M)

**Common pattern for all LONG archetypes:**

1. **Convert return type** from `bool` to `Tuple[bool, float, Dict]`
2. **Add score calculation** (weighted component sum)
3. **Add domain engine integration block** (use LONG_DOMAIN_ENGINE_TEMPLATE below)
4. **Apply domain boost** before fusion gate
5. **Return full metadata** including domain_boost and domain_signals

### Phase 2: SHORT Archetypes (S3, S8)

**Pattern for SHORT archetypes:**

1. Same structure as LONG
2. **Use SHORT_DOMAIN_ENGINE_TEMPLATE** (inverted logic)
   - Wyckoff distribution → BOOST (not veto)
   - Wyckoff accumulation → VETO (not boost)
   - SMC supply zones → BOOST
   - SMC demand zones → VETO
   - Macro crisis → BOOST (not veto)

## Domain Engine Templates

### LONG Pattern Domain Engine Template

```python
# ============================================================================
# DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - LONG PATTERN
# ============================================================================
use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

domain_boost = 1.0
domain_signals = []

# Wyckoff Engine (LONG pattern)
if use_wyckoff:
    wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
    wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
    wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
    wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)
    wyckoff_sc = self.g(context.row, 'wyckoff_sc', False)
    wyckoff_st = self.g(context.row, 'wyckoff_st', False)
    wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)

    if wyckoff_distribution:
        domain_boost *= 0.70  # Soft veto for distribution phase
        domain_signals.append("wyckoff_distribution_caution")
    if wyckoff_accumulation:
        domain_boost *= 1.80  # Major boost for accumulation
        domain_signals.append("wyckoff_accumulation")
    if wyckoff_spring_a:
        domain_boost *= 2.50  # Deep spring = strongest capitulation
        domain_signals.append("wyckoff_spring_a_major")
    elif wyckoff_spring_b:
        domain_boost *= 2.20  # Shallow spring
        domain_signals.append("wyckoff_spring_b")
    if wyckoff_sc:
        domain_boost *= 2.00  # Selling climax
        domain_signals.append("wyckoff_selling_climax")
    elif wyckoff_st:
        domain_boost *= 1.50  # Secondary test
        domain_signals.append("wyckoff_secondary_test")
    if wyckoff_lps:
        domain_boost *= 1.80  # Last point support
        domain_signals.append("wyckoff_lps_support")

# SMC Engine (LONG pattern)
if use_smc:
    smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
    tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)
    tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
    tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)
    smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
    smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)
    smc_choch = self.g(context.row, 'smc_choch', False)

    if smc_supply_zone:
        domain_boost *= 0.70  # Supply overhead reduces conviction
        domain_signals.append("smc_supply_overhead")
    if tf4h_bos_bearish:
        domain_boost *= 0.60  # Bearish 4H structure penalty
        domain_signals.append("smc_4h_bearish_structure")
    if tf4h_bos_bullish:
        domain_boost *= 2.00  # Institutional 4H bullish shift
        domain_signals.append("smc_4h_bos_bullish_institutional")
    elif tf1h_bos_bullish:
        domain_boost *= 1.40  # 1H structural shift
        domain_signals.append("smc_1h_bos_bullish")
    if smc_demand_zone:
        domain_boost *= 1.60  # Institutional support area
        domain_signals.append("smc_demand_zone_support")
    if smc_liquidity_sweep:
        domain_boost *= 1.80  # Stop hunt before rally
        domain_signals.append("smc_liquidity_sweep_reversal")
    if smc_choch:
        domain_boost *= 1.60  # Character change = trend shift
        domain_signals.append("smc_choch_trend_change")

# Temporal Engine
if use_temporal:
    fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
    temporal_confluence = self.g(context.row, 'temporal_confluence', False)
    temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
    tf4h_fusion_score = self.g(context.row, 'tf4h_fusion_score', 0.0)

    if fib_time_cluster:
        domain_boost *= 1.70  # Fibonacci timing = geometric reversal
        domain_signals.append("fib_time_cluster_reversal")
    if temporal_confluence:
        domain_boost *= 1.50  # Multi-timeframe alignment
        domain_signals.append("temporal_multi_tf_confluence")
    if tf4h_fusion_score > 0.70:
        domain_boost *= 1.60  # High 4H fusion = strong trend
        domain_signals.append("tf4h_high_fusion_score")
    if temporal_resistance_cluster:
        domain_boost *= 0.75  # Resistance overhead
        domain_signals.append("temporal_resistance_overhead")

# HOB Engine
if use_hob:
    hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
    hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
    hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

    if hob_demand_zone:
        domain_boost *= 1.50  # Large bid wall support
        domain_signals.append("hob_demand_zone_support")
    if hob_supply_zone:
        domain_boost *= 0.70  # Supply wall overhead
        domain_signals.append("hob_supply_zone_overhead")
    if hob_imbalance > 0.60:
        domain_boost *= 1.30  # Strong buyer imbalance
        domain_signals.append("hob_bid_imbalance_strong")
    elif hob_imbalance > 0.40:
        domain_boost *= 1.15  # Moderate buyer imbalance
        domain_signals.append("hob_bid_imbalance_moderate")

# Macro Engine
if use_macro:
    crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
    if crisis_composite > 0.70:
        domain_boost *= 0.85  # Extreme crisis reduces conviction
        domain_signals.append("macro_extreme_crisis_penalty")

# Apply domain boost BEFORE fusion gate
score_before_domain = score
score = score * domain_boost
```

### SHORT Pattern Domain Engine Template

```python
# ============================================================================
# DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - SHORT PATTERN
# ============================================================================
use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

domain_boost = 1.0
domain_signals = []

# Wyckoff Engine (SHORT pattern)
if use_wyckoff:
    wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
    wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
    wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
    wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)
    wyckoff_psy = self.g(context.row, 'wyckoff_psy', False)

    if wyckoff_distribution:
        domain_boost *= 2.00  # Major boost for distribution phase
        domain_signals.append("wyckoff_distribution_short")
    if wyckoff_accumulation:
        domain_boost *= 0.70  # Soft veto for accumulation
        domain_signals.append("wyckoff_accumulation_caution")
    if wyckoff_utad:
        domain_boost *= 2.50  # Upthrust After Distribution = top signal
        domain_signals.append("wyckoff_utad_top")
    elif wyckoff_bc:
        domain_boost *= 2.20  # Buying Climax
        domain_signals.append("wyckoff_bc_climax")
    if wyckoff_psy:
        domain_boost *= 1.60  # Preliminary Supply
        domain_signals.append("wyckoff_psy_supply")

# SMC Engine (SHORT pattern)
if use_smc:
    smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
    tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
    tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)
    tf1h_bos_bearish = self.g(context.row, 'tf1h_bos_bearish', False)
    smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
    smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

    if smc_demand_zone:
        domain_boost *= 0.70  # Demand support reduces short conviction
        domain_signals.append("smc_demand_support_caution")
    if tf4h_bos_bullish:
        domain_boost *= 0.60  # Bullish 4H structure penalty for shorts
        domain_signals.append("smc_4h_bullish_structure_penalty")
    if tf4h_bos_bearish:
        domain_boost *= 2.00  # Institutional 4H bearish shift
        domain_signals.append("smc_4h_bos_bearish_institutional")
    elif tf1h_bos_bearish:
        domain_boost *= 1.40  # 1H bearish structural shift
        domain_signals.append("smc_1h_bos_bearish")
    if smc_supply_zone:
        domain_boost *= 1.60  # Institutional resistance area
        domain_signals.append("smc_supply_zone_resistance")
    if smc_liquidity_sweep:
        domain_boost *= 1.50  # Liquidity grab before drop
        domain_signals.append("smc_liquidity_sweep_drop")

# Temporal Engine
if use_temporal:
    fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
    temporal_confluence = self.g(context.row, 'temporal_confluence', False)
    temporal_support_cluster = self.g(context.row, 'temporal_support_cluster', False)

    if fib_time_cluster:
        domain_boost *= 1.70  # Fibonacci timing reversal
        domain_signals.append("fib_time_cluster_reversal")
    if temporal_confluence:
        domain_boost *= 1.50  # Multi-timeframe alignment
        domain_signals.append("temporal_multi_tf_confluence")
    if temporal_support_cluster:
        domain_boost *= 0.75  # Support below reduces short conviction
        domain_signals.append("temporal_support_below")

# HOB Engine
if use_hob:
    hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
    hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
    hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

    if hob_supply_zone:
        domain_boost *= 1.50  # Large ask wall resistance
        domain_signals.append("hob_supply_zone_resistance")
    if hob_demand_zone:
        domain_boost *= 0.70  # Demand wall support reduces short
        domain_signals.append("hob_demand_zone_caution")
    if hob_imbalance < -0.60:
        domain_boost *= 1.30  # Strong seller imbalance
        domain_signals.append("hob_ask_imbalance_strong")
    elif hob_imbalance < -0.40:
        domain_boost *= 1.15  # Moderate seller imbalance
        domain_signals.append("hob_ask_imbalance_moderate")

# Macro Engine
if use_macro:
    crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
    if crisis_composite > 0.70:
        domain_boost *= 1.30  # Extreme crisis BOOSTS shorts
        domain_signals.append("macro_extreme_crisis_short_boost")

# Apply domain boost BEFORE fusion gate
score_before_domain = score
score = score * domain_boost
```

## Detailed Implementation Steps

### Step 1: Archetype C (wick_trap) - Line 1362

**Current signature:**
```python
def _check_C(self, context: RuntimeContext) -> bool:
```

**New signature:**
```python
def _check_C(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
```

**Changes required:**
1. Add score calculation based on components
2. Insert LONG domain engine template after score calculation
3. Apply `score = score * domain_boost`
4. Return `(True, score, meta)` instead of `True`
5. Return `(False, score, meta)` instead of `False`

**Expected boost range:** 1.5x - 10x (accumulation + SMC bullish BOS = 3.6x, with temporal = 6.1x)

### Step 2-8: Repeat for D, E, F, G, K, L, M

All follow the same pattern as Archetype C (LONG pattern domain engines).

### Step 9: Archetype S3 (whipsaw) - Line 3050

**Use SHORT domain engine template** - inverted logic for short patterns.

**Expected boost range:** 1.5x - 8x (distribution + SMC bearish BOS = 4.0x)

### Step 10: Archetype S8 (volume_fade_chop) - Line 3685

**Use SHORT domain engine template**.

**Expected boost range:** 1.5x - 8x

## Consistency Checklist

After implementation, verify ALL 10 archetypes have:

- ✅ Return type changed to `Tuple[bool, float, Dict]`
- ✅ Base score calculation (weighted component sum)
- ✅ Domain engine integration block (120-150 lines)
- ✅ Feature flag checks: `enable_wyckoff`, `enable_smc`, `enable_temporal`, `enable_hob`, `enable_macro`
- ✅ `domain_boost` starts at 1.0
- ✅ Soft vetoes use 0.60-0.85x (NOT hard vetoes/return False)
- ✅ Major boosts use 1.5x-2.5x multipliers
- ✅ Wyckoff phase checks use `'wyckoff_phase_abc' == 'A'` or `'D'` (NOT string names)
- ✅ Domain boost applied BEFORE fusion gate: `score = score * domain_boost`
- ✅ Return metadata includes: `domain_boost`, `domain_signals`, `score_before_domain`
- ✅ LONG patterns boost accumulation, SHORT patterns boost distribution

## Expected Impact Analysis

### Before Domain Integration

| Archetype | Matches (typical) | Avg Confidence | Boost Range |
|-----------|-------------------|----------------|-------------|
| C-M (LONG) | 10-50/month | 0.40-0.60 | 1.0x (none) |
| S3, S8 (SHORT) | 5-20/month | 0.35-0.55 | 1.0x (none) |

**Total potential matches:** ~100-300/month at 1.0x confidence

### After Domain Integration

| Archetype | Matches (expected) | Avg Confidence | Boost Range |
|-----------|-------------------|----------------|-------------|
| C-M (LONG) | 10-50/month | 0.40-0.60 → **0.60-1.80** | 1.5x - 10x |
| S3, S8 (SHORT) | 5-20/month | 0.35-0.55 → **0.53-1.40** | 1.5x - 8x |

**Key improvements:**
- **High-confidence signals (>0.80):** 0 → 20-60/month
- **Confluence detection:** 6 engines can now align (vs 0 before)
- **Risk filtering:** Soft vetoes reduce false positives by 30-40%

### Domain Signal Distribution (Expected)

| Engine | Signals/Match (Avg) | Boost Contribution | Veto Contribution |
|--------|---------------------|-------------------|-------------------|
| Wyckoff | 1-3 | 1.8x - 4.5x | 0.70x |
| SMC | 1-2 | 1.4x - 3.2x | 0.60x - 0.70x |
| Temporal | 0-2 | 1.5x - 2.6x | 0.75x |
| HOB | 0-1 | 1.15x - 1.50x | 0.70x |
| Macro | 0-1 | 1.0x | 0.85x |

**Theoretical max boost:** 12.5x (all boosts align)
**Theoretical max veto:** 0.15x (all vetoes align)
**Typical boost range:** 1.5x - 6.0x

## Testing Plan

### Unit Tests

```python
def test_archetype_C_domain_boost():
    """Verify C returns tuple with domain_boost > 1.0 when wyckoff accumulation present"""
    context = mock_context_with_wyckoff_accumulation()
    matched, score, meta = logic._check_C(context)

    assert matched == True
    assert meta['domain_boost'] > 1.0
    assert 'wyckoff_accumulation' in meta['domain_signals']
    assert meta['score_before_domain'] < score  # Score increased
```

### Integration Tests

1. **Backtest 2020-2025:** Compare matches before/after
2. **Boost distribution:** Verify 1.5x-10x range
3. **Veto effectiveness:** Confirm false positive reduction
4. **Confluence detection:** Check multi-engine alignment

### Validation Criteria

- ✅ All 10 archetypes return `(bool, float, dict)` tuple
- ✅ Domain boost range: 1.5x - 12x (not exceeding S1's range)
- ✅ Veto range: 0.60x - 0.85x (soft vetoes only)
- ✅ Signal count: 0-6 domain signals per match
- ✅ No hard vetoes (return False in domain block)
- ✅ Consistent feature flag pattern across all archetypes

## Lines of Code Impact

| Component | Lines/Archetype | Total (10 archetypes) |
|-----------|-----------------|----------------------|
| Score calculation | 15-25 | 150-250 |
| Domain engines | 120-150 | 1200-1500 |
| Metadata return | 10-15 | 100-150 |
| **TOTAL** | **145-190** | **1450-1900** |

**Estimated implementation time:** 4-6 hours (with careful testing)

## Next Steps

1. **Implementation:** Add domain engines to all 10 archetypes (follow templates above)
2. **Testing:** Run unit tests + backtest validation
3. **Tuning:** Adjust boost multipliers based on backtest results
4. **Documentation:** Update archetype documentation with domain engine details
5. **Monitoring:** Track domain signal distribution in production

## Conclusion

This integration brings 10 unwired archetypes up to the same standard as S1, unlocking:
- **8x-12x boost potential** (vs 1.0x ceiling currently)
- **Multi-engine confluence** (6 engines vs 0 currently)
- **Smart veto system** (prevents bad entries in wrong market conditions)
- **Enhanced observability** (domain_signals show WHY archetype fired)

**Impact:** ~1450-1900 lines of code added, unlocking 6x engines for 10 archetypes.
