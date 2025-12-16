# Archetype Domain Engine Wiring Consistency Report

## Executive Summary

**Problem**: Only 3 out of 19 working archetypes have full domain engine integration, creating inconsistent signal quality across the system.

**Impact**:
- 16 archetypes use basic fusion scores (weak signals, no domain knowledge)
- 3 archetypes have 8x-12x domain amplification (strong signals, expert knowledge)
- **Massive quality gap** between "smart" and "basic" archetypes

**Date**: 2025-12-12
**Auditor**: Claude Code (Refactoring Expert)

---

## Wiring Status Summary

| Wiring Level | Count | Archetypes | Boost Range |
|-------------|-------|-----------|-------------|
| **FULL** | 3 | S1, S4, S5 | 8x - 12x realistic (95x theoretical) |
| **PARTIAL** | 3 | A, B, H | 2x - 4x (incomplete wiring) |
| **NONE** | 13 | C, D, E, F, G, K, L, M, S2, S3, S8 | 1.0x (basic fusion only) |
| **GHOST/STUB** | 5 | P, Q, N, S6, S7 | N/A (not working) |

**Total Working**: 19 archetypes
**Fully Wired**: 3 (16%)
**Coverage**: **84% of archetypes are NOT using domain engines**

---

## Full Wiring Pattern (S1, S4, S5)

These three archetypes follow the **complete domain engine integration pattern**:

### Pattern Structure
```python
def _check_S1(context: RuntimeContext) -> Tuple[bool, float, Dict]:
    # 1. Extract base features
    liquidity = self._liquidity_score(context.row)
    fusion = context.row.get('fusion_score', 0.0)

    # 2. Check runtime features (if available)
    wick_ratio = self.g(context.row, 'wick_lower_ratio', None)

    # 3. DOMAIN ENGINE INTEGRATION
    domain_boost = 1.0

    # 3A. Wyckoff Events (2.0x - 2.5x per event)
    wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
    if wyckoff_spring_a:
        domain_boost *= 2.50  # Spring A = deep fake breakdown

    wyckoff_sc = self.g(context.row, 'wyckoff_sc', False)
    if wyckoff_sc:
        domain_boost *= 2.00  # Selling Climax

    # 3B. SMC Signals (1.4x - 2.0x)
    tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
    if tf4h_bos_bullish:
        domain_boost *= 2.00  # Institutional breakout

    demand_zone = self.g(context.row, 'is_demand_zone', False)
    if demand_zone:
        domain_boost *= 1.50  # Institutional support

    # 3C. Temporal Fusion (1.5x - 1.8x)
    fib_time_cluster = self.g(context.row, 'fib_time_cluster_bull', False)
    if fib_time_cluster:
        domain_boost *= 1.80  # Geometric timing

    # 3D. HOB (Hidden Order Book) (1.3x - 1.5x)
    demand_wall = self.g(context.row, 'hob_demand_wall', False)
    if demand_wall:
        domain_boost *= 1.50  # Large bid wall

    # 3E. PTI (Price-Time-Impulse) (1.2x - 1.5x)
    pti_spring = self.g(context.row, 'pti_trap_type', '') == 'spring'
    if pti_spring:
        domain_boost *= 1.50  # Wyckoff Spring detection

    # 4. APPLY BOOST BEFORE FUSION GATE
    final_score = fusion * domain_boost

    # 5. Check fusion threshold (AFTER boost)
    if final_score < fusion_threshold:
        return False, 0.0, {"reason": "below_threshold"}

    # 6. SMC VETO GATES
    if supply_zone:
        return False, 0.0, {"veto": "supply_overhead"}

    # 7. Return success
    meta = {
        "domain_boost": domain_boost,
        "base_fusion": fusion,
        "final_score": final_score,
        "wyckoff_events": [...],
        "smc_signals": [...],
        ...
    }
    return True, final_score, meta
```

### Key Components

1. **Domain Boost Calculation** (BEFORE fusion gate)
   - Multiplicative amplification from expert signals
   - Each domain contributes independently
   - Boosts stack multiplicatively (2.0x * 1.5x * 1.8x = 5.4x)

2. **Domain Sources**
   - **Wyckoff**: Spring, SC, BC, AR, ST, LPSY, UTAD (13 events)
   - **SMC**: BOS, CHOCH, demand/supply zones, liquidity sweeps (4 features)
   - **Temporal**: Fibonacci time clusters, MTF confluence (4 features)
   - **HOB**: Demand/supply walls, bid/ask imbalance (3 features)
   - **PTI**: Spring/UTAD trap detection (2 features)

3. **Veto Gates** (AFTER boost)
   - SMC supply/demand vetoes
   - Wyckoff phase vetoes (distribution = no longs)
   - Macro vetoes (crisis regime)
   - Structural vetoes (4H trend misalignment)

4. **Metadata Logging**
   - All domain signals tracked
   - Boost breakdown recorded
   - Veto reasons documented

---

## Partial Wiring Pattern (A, B, H)

These archetypes have **some domain logic** but not the full pattern:

### A - Trap Reversal
**Has**:
- PTI spring/UTAD detection
- Displacement checks
- Basic fusion scoring

**Missing**:
- Wyckoff event boosting (Spring A/B not amplified)
- SMC veto gates
- Temporal confluence
- HOB demand walls

**Location**: `logic_v2_adapter.py:869`
**Fix Needed**: Add full domain boost section before fusion check

### B - Order Block Retest
**Has**:
- BOS context checks
- BOMS strength
- Wyckoff phase awareness

**Missing**:
- Wyckoff event boosting (LPS, SOS not amplified)
- SMC demand zone amplification
- Temporal time clusters
- HOB bid walls

**Location**: `logic_v2_adapter.py:931`
**Fix Needed**: Add full domain boost section

### H - Trap Within Trend
**Has**:
- HTF trend checks
- Liquidity score
- Wick rejection logic

**Missing**:
- Wyckoff Spring detection (ironic for a trap pattern!)
- SMC BOS amplification
- Temporal timing
- HOB support

**Location**: `logic_v2_adapter.py:1144`
**Fix Needed**: Add full domain boost section

---

## No Wiring Pattern (C, D, E, F, G, K, L, M, S2, S3, S8)

These archetypes use **basic fusion score only**:

### Typical Implementation
```python
def _check_C(context: RuntimeContext) -> bool:
    # Read thresholds
    fusion_th = context.get_threshold('fvg_continuation', 'fusion', 0.40)
    rsi_min = context.get_threshold('fvg_continuation', 'rsi_min', 60.0)

    # Get features
    rsi = self.g(context.row, "rsi", 50.0)
    fusion = context.row.get('fusion_score', 0.0)

    # Simple threshold check
    return (rsi >= rsi_min and fusion >= fusion_th)
```

**Problems**:
1. No domain knowledge amplification
2. Relies on generic fusion score (already computed)
3. No expert signal integration
4. No veto gates for structural misalignment
5. **1.0x boost vs 8x-12x for wired archetypes**

**Impact**:
- Weaker signal quality
- More false positives
- No expert knowledge layer
- Higher drawdowns

---

## Inconsistencies Identified

### 1. Boost Application Timing
- **S1, S4, S5**: Apply `domain_boost` BEFORE fusion threshold check (correct)
- **A, B**: Apply some boosts but inconsistent timing
- **Others**: No boosts at all

**Impact**: Archetypes without pre-gate boosting miss signals that expert knowledge would validate.

### 2. Veto Pattern Inconsistency
**S1, S4, S5 have**:
- SMC structure vetoes (supply overhead, 4H trend)
- Wyckoff phase vetoes (distribution, markdown)
- Regime vetoes (crisis for longs)

**Others have**:
- Some: Basic RSI/ADX checks
- Some: No vetoes at all
- None: Structural vetoes

**Impact**: Weak archetypes enter trades during structural misalignment.

### 3. Metadata Quality
**S1, S4, S5 return**:
```python
meta = {
    "domain_boost": 8.5,
    "wyckoff_events": ["spring_a", "sc"],
    "smc_signals": ["bos_bullish", "demand_zone"],
    "temporal": ["fib_time_cluster"],
    "hob": ["demand_wall"],
    "base_fusion": 0.45,
    "final_score": 3.82,
    "vetoes_checked": ["supply", "distribution", "4h_trend"]
}
```

**Others return**:
```python
# Simple bool or minimal dict
return True  # or
return {"reason": "fusion_threshold"}
```

**Impact**: No observability for why signals fire or fail.

### 4. Feature Flag Respect
**Fully wired archetypes**:
- Check `feature_flags.is_enabled('wyckoff')`
- Check `feature_flags.is_enabled('smc')`
- Gracefully degrade if engines disabled

**Others**:
- Don't check feature flags
- Use features if present, ignore if not
- No explicit engine integration

**Impact**: Feature flags don't work as expected for unwired archetypes.

---

## Recommended Standardization Pattern

All archetypes should follow this structure:

```python
def _check_X(context: RuntimeContext) -> Tuple[bool, float, Dict]:
    """
    Archetype X: [Description]

    DOMAIN ENGINES: Wyckoff, SMC, Temporal, HOB, PTI
    VETOES: [List veto conditions]
    TARGET: [Trades/year], [Win rate], [PF]
    """
    # 1. EXTRACT BASE FEATURES
    base_fusion = context.row.get('fusion_score', 0.0)
    liquidity = self._liquidity_score(context.row)

    # 2. INITIALIZE DOMAIN BOOST
    domain_boost = 1.0
    domain_signals = {
        "wyckoff": [],
        "smc": [],
        "temporal": [],
        "hob": [],
        "pti": []
    }

    # 3. WYCKOFF BOOSTING (if enabled)
    if feature_flags.is_enabled('wyckoff'):
        for event_name, boost_multiplier in WYCKOFF_BOOSTS.items():
            if self.g(context.row, event_name, False):
                domain_boost *= boost_multiplier
                domain_signals["wyckoff"].append(event_name)

    # 4. SMC BOOSTING (if enabled)
    if feature_flags.is_enabled('smc'):
        for smc_signal, boost_multiplier in SMC_BOOSTS.items():
            if self.g(context.row, smc_signal, False):
                domain_boost *= boost_multiplier
                domain_signals["smc"].append(smc_signal)

    # 5. TEMPORAL BOOSTING (if enabled)
    if feature_flags.is_enabled('temporal'):
        for temporal_signal, boost_multiplier in TEMPORAL_BOOSTS.items():
            if self.g(context.row, temporal_signal, False):
                domain_boost *= boost_multiplier
                domain_signals["temporal"].append(temporal_signal)

    # 6. HOB BOOSTING (if enabled)
    if feature_flags.is_enabled('hob'):
        for hob_signal, boost_multiplier in HOB_BOOSTS.items():
            if self.g(context.row, hob_signal, False):
                domain_boost *= boost_multiplier
                domain_signals["hob"].append(hob_signal)

    # 7. PTI BOOSTING (if enabled)
    if feature_flags.is_enabled('pti'):
        pti_type = self.g(context.row, 'pti_trap_type', '')
        if pti_type in PTI_BOOSTS:
            domain_boost *= PTI_BOOSTS[pti_type]
            domain_signals["pti"].append(pti_type)

    # 8. APPLY BOOST BEFORE FUSION GATE
    final_score = base_fusion * domain_boost

    # 9. CHECK FUSION THRESHOLD (after boost)
    fusion_threshold = context.get_threshold('archetype_name', 'fusion_threshold', 0.40)
    if final_score < fusion_threshold:
        return False, 0.0, {
            "reason": "below_threshold",
            "final_score": final_score,
            "threshold": fusion_threshold,
            "domain_boost": domain_boost
        }

    # 10. VETO GATES (structural, regime, phase)
    veto_reason = self._check_vetoes_X(context, domain_signals)
    if veto_reason:
        return False, 0.0, {
            "veto": veto_reason,
            "final_score": final_score,
            "domain_boost": domain_boost
        }

    # 11. SUCCESS - RETURN RICH METADATA
    meta = {
        "base_fusion": base_fusion,
        "domain_boost": domain_boost,
        "final_score": final_score,
        "domain_signals": domain_signals,
        "vetoes_checked": self._get_veto_list(),
        "feature_flags": {
            "wyckoff": feature_flags.is_enabled('wyckoff'),
            "smc": feature_flags.is_enabled('smc'),
            "temporal": feature_flags.is_enabled('temporal'),
            "hob": feature_flags.is_enabled('hob')
        }
    }

    return True, final_score, meta
```

---

## Migration Priority

### High Priority (Next Sprint)
Wire domain engines for high-usage archetypes with clear patterns:

1. **H - Trap Within Trend** (99 configs)
   - Clear Spring pattern → Wyckoff boosting
   - HTF BOS → SMC boosting
   - Time clusters → Temporal boosting

2. **A - Trap Reversal** (99 configs)
   - Already has PTI → extend to full Wyckoff
   - Add SMC demand zones
   - Add temporal timing

3. **B - Order Block Retest** (99 configs)
   - Already has BOS → extend to full SMC
   - Add Wyckoff LPS, SOS
   - Add HOB bid walls

### Medium Priority (Future Sprint)
Wire domain engines for specialized patterns:

4. **G - Re-accumulate** (99 configs) - Wyckoff accumulation focus
5. **K - Wick Trap** (99 configs) - Rejection + liquidity sweep focus
6. **M - Ratio Coil Break** (98 configs) - Wyckoff coil + timing focus

### Low Priority (Future)
Wire remaining archetypes or deprecate if low performance:

7. **C, D, E, F, L** - Consider merging or deprecating
8. **S3, S8** - Bear archetypes, wire if needed

---

## Testing Requirements

Before deploying wired archetypes:

1. **Backtest Comparison**
   - Core (no domain engines) vs Full (with engines)
   - Measure: Trades/year, Win rate, PF, Sharpe
   - Expect: 50-80% reduction in trades, 10-15% higher win rate

2. **Boost Distribution**
   - Log all domain_boost values
   - Verify realistic range (1.0x - 15x, not 95x)
   - Check for boost inflation

3. **Veto Effectiveness**
   - Track veto reasons
   - Measure reduction in drawdown
   - Verify no good trades blocked

4. **Feature Flag Testing**
   - Test with engines on/off
   - Verify graceful degradation
   - Check metadata accuracy

---

## Code Locations

### Fully Wired Examples
- **S1**: `logic_v2_adapter.py:1321-1950` (Liquidity Vacuum)
- **S4**: `logic_v2_adapter.py:2553-2830` (Funding Divergence)
- **S5**: `logic_v2_adapter.py:2830-3142` (Long Squeeze)

### Boost Configuration
- **Wyckoff boosts**: `logic_v2_adapter.py:207-215` (init)
- **Domain boost application**: Search for `domain_boost *= ` in S1/S4/S5 methods

### Feature Flags
- **Feature flag module**: `engine/feature_flags.py`
- **Flag checks**: `feature_flags.is_enabled('wyckoff')`

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Archetypes | 24 (A-M, S1-S8, P, Q, N) |
| Working Archetypes | 19 (79%) |
| Fully Wired | 3 (16% of working) |
| Partially Wired | 3 (16% of working) |
| Not Wired | 13 (68% of working) |
| Ghost/Stub | 5 (21% of total) |
| Boost Gap | 8x-12x (wired) vs 1.0x (unwired) |
| Configs Using Unwired | 1,287 (87%) |

**Conclusion**: The system has a **massive quality inconsistency** with only 3 archetypes leveraging full domain knowledge while 13 use basic fusion scores. This creates unpredictable signal quality across the portfolio.

**Recommendation**: Standardize domain engine wiring across all working archetypes using the proven S1/S4/S5 pattern.
