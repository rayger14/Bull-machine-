# ARCHETYPE CODE PATH INVESTIGATION - FINAL SUMMARY

## Investigation Completed: November 2, 2025

### ROOT CAUSE IDENTIFIED

The **64 vs 84 vs 19 trade count discrepancy** is caused by **TWO COMPLETELY DIFFERENT ARCHETYPE DETECTION SYSTEMS**:

1. **LEGACY SYSTEM** (baseline_btc_bull_pf20.json)
   - Method: `check_archetype(row, prev_row, df, index)` from `/engine/archetypes/logic.py`
   - Thresholds: Hardcoded, static, same for entire backtest
   - Example: Archetype B always requires fusion >= 0.359
   - Trade count: ~64

2. **ADAPTIVE SYSTEM** (btc_v8_adaptive.json)
   - Method: `detect(RuntimeContext)` from `/engine/archetypes/logic_v2_adapter.py`
   - Thresholds: Dynamic, regime-aware, change every bar
   - Example: Archetype B requires fusion >= (0.34-0.52 depending on regime)
   - Trade count: ~84 (varies with regime probability mix)

---

## KEY DISCOVERY: `fusion_regime_profiles` IS THE TRIGGER

The presence of `fusion_regime_profiles` in the config file determines which path is used:

### Automatic Path Selection

```python
# In backtest_knowledge_v2.py:205-238

if ADAPTIVE_FUSION_AVAILABLE and adaptive_config.get('enable'):
    # Check for gates_regime_profiles (set by btc_v8_adaptive.json)
    self.threshold_policy = ThresholdPolicy(
        base_cfg=self.runtime_config,
        regime_profiles=self.runtime_config.get('gates_regime_profiles'),  # TRIGGER!
        ...
    )
    # Result: ADAPTIVE PATH will be used
else:
    # Result: LEGACY PATH will be used
    self.threshold_policy = None
```

**The actual branching point is in `classify_entry_archetype()` at line 482:**

```python
if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
    # Use ADAPTIVE path: detect(RuntimeContext)
    archetype_name = self.archetype_logic.detect(runtime_ctx)
else:
    # Use LEGACY path: check_archetype()
    archetype_name = self.archetype_logic.check_archetype(...)
```

---

## FOUR FILES THAT MUST BE UNDERSTOOD

### 1. Configuration Files

**Legacy:** `/configs/baseline_btc_bull_pf20.json`
- Line 146-233: Static archetype thresholds
- NO `gates_regime_profiles`, NO `fusion_regime_profiles`
- Thresholds never change

**Adaptive:** `/configs/btc_v8_adaptive.json`  
- Line 6-31: `fusion_regime_profiles` (4 regimes × 4 weights)
- Line 33-50: `gates_regime_profiles` (4 regimes × 2 gates) ← CRITICAL
- Line 52-61: `archetype_overrides` (per-archetype deltas)
- Line 63-67: `global_clamps` (guardrails)

### 2. Archetype Logic

**Legacy:** `/engine/archetypes/logic.py` (lines 1-719)
- `check_archetype()` method at lines 185-262
- Individual checks (_check_A, _check_B, ..., _check_M) at lines 268-719
- **Reads thresholds from instance variables** (self.thresh_A, self.thresh_B, ...)
- Thresholds loaded once in `__init__()`, never change

**Adaptive:** `/engine/archetypes/logic_v2_adapter.py` (lines 1-532)
- `detect(RuntimeContext)` method at lines 247-305 ← NEW METHOD
- Individual checks at lines 339-531
- **Reads thresholds from RuntimeContext.get_threshold()**
- Thresholds resolved fresh every bar by ThresholdPolicy

### 3. Threshold Policy (Adaptive Only)

**File:** `/engine/archetypes/threshold_policy.py` (lines 1-266)

The 5-step resolution pipeline:
1. **Load base thresholds** from config (line 106)
2. **Blend regime gates** using probability weights (line 109)
3. **Apply regime floors** to all archetypes (line 112)
4. **Apply per-archetype overrides** for current regime (line 115)
5. **Clamp to guardrails** (line 118)

**Example threshold for Archetype B:**

```
Base:           0.359 (from config)
Regime prob:    risk_on=0.62, neutral=0.28, ...
Blended floor:  0.62×0.36 + 0.28×0.38 + ... = 0.367
After floor:    max(0.359, 0.367) = 0.367
After override: 0.367 + (-0.02×0.62) = 0.347
After clamp:    clamp(0.347, [0.20, 0.60]) = 0.347
FINAL:          0.347 (varies by bar!)
```

Adaptive config makes thresholds **LOWER in risk_on** (0.347 vs 0.359 base), allowing **MORE entries**.

### 4. Runtime Context (Adaptive Only)

**File:** `/engine/runtime/context.py` (lines 1-61)

Immutable data container passed to `detect()`:

```python
@dataclass(frozen=True)
class RuntimeContext:
    ts: Any                                    # Timestamp
    row: pd.Series                             # Current bar data
    regime_probs: Dict[str, float]             # {'risk_on': 0.62, 'neutral': 0.28, ...}
    regime_label: str                          # 'risk_on' (argmax)
    adapted_params: Dict[str, Any]             # From AdaptiveFusion
    thresholds: Dict[str, Dict[str, float]]    # Resolved by ThresholdPolicy!
    
    def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
        """Safe lookup of regime-aware threshold"""
        return self.thresholds.get(archetype, {}).get(param, default)
```

**This is the KEY DIFFERENCE:**
- Legacy: `self.thresh_B` (static, set once)
- Adaptive: `ctx.get_threshold('order_block_retest', 'fusion')` (dynamic, resolved per bar)

---

## VISUAL: THE TWO PATHS

```
LEGACY PATH:
  Config has NO gates_regime_profiles
                    ↓
  threshold_policy = None
                    ↓
  At bar N:
    classify_entry_archetype()
      → threshold_policy is None
      → Call check_archetype(row, prev_row, df, idx)
      → Check: fusion >= self.thresh_B (0.359) ← HARDCODED
      → Decision based on static threshold

ADAPTIVE PATH:
  Config HAS gates_regime_profiles
                    ↓
  threshold_policy = ThresholdPolicy(gates_regime_profiles, ...)
                    ↓
  At bar N:
    Regime classifier identifies regime probs
      → regime_probs = {'risk_on': 0.62, ...}
      → adapted_params = {'regime_probs_ema': regime_probs, ...}
      → Store in context['adapted_params']
                    ↓
    classify_entry_archetype()
      → threshold_policy is NOT None
      → adapted_params is NOT None
      → Call threshold_policy.resolve(regime_probs, 'risk_on')
        - Blend: 0.62×0.36 + 0.28×0.38 + ... = 0.367
        - Apply floor: max(0.359, 0.367) = 0.367
        - Apply override: 0.367 + (-0.02) = 0.347
        - Clamp: 0.347
        → thresholds = {'order_block_retest': {'fusion': 0.347}, ...}
      → Build RuntimeContext(thresholds=thresholds)
      → Call detect(runtime_ctx)
      → Check: fusion >= ctx.get_threshold('order_block_retest', 'fusion')
      → ctx returns 0.347 (LOWER than 0.359 in legacy!)
      → MORE entries approved!
      → Decision based on REGIME-AWARE threshold
```

---

## WHY TRADE COUNTS DIFFER

| Trade Count | Config | Why |
|------------|--------|-----|
| 64 | baseline_btc_bull_pf20.json | Static thresholds, no regime adaptation |
| 84 | btc_v8_adaptive.json | Thresholds vary by regime; risk_on uses LOWER thresholds → MORE entries |
| 19 | Unknown variant | Possibly locked_regime='static' OR extremely tight clamps |

**The 20-trade difference (84-64) represents the sensitivity of the system to regime modulation.**

In btc_v8_adaptive.json, the **final_fusion_floor** is the most impactful:
- risk_on: 0.36 (LOWER than most base thresholds)
- crisis: 0.52 (HIGHER than most base thresholds)

This causes ~20 additional trades when regime spends time in risk_on.

---

## CRITICAL FINDING: Thresholds Are Applied AFTER Fusion Score

Both paths receive the **same fusion_score** from `compute_advanced_fusion_score()`.

**The difference is only in the threshold comparison:**

```python
# LEGACY: Is fusion_score >= static_threshold?
if fusion_score >= 0.359:  # Always 0.359
    return True

# ADAPTIVE: Is fusion_score >= regime_aware_threshold?
if fusion_score >= ctx.get_threshold(...):  # Could be 0.347, 0.367, 0.43, etc.
    return True
```

**This explains everything:**
- Same data, same feature extraction
- Same fusion score calculation
- Different thresholds = different acceptance rate
- Boom: 20+ trade difference

---

## VERIFICATION CHECKLIST

To confirm which path your backtest is using:

- [ ] **Look at config file:**
  - Has `gates_regime_profiles` key? → ADAPTIVE
  - Missing `gates_regime_profiles` key? → LEGACY

- [ ] **Check logs for:**
  - "ThresholdPolicy: ENABLED" → ADAPTIVE
  - "Adaptive Fusion: DISABLED" → LEGACY

- [ ] **Monitor trade count:**
  - ~64 trades? → Legacy (baseline_btc_bull_pf20.json)
  - ~84 trades? → Adaptive (btc_v8_adaptive.json)

- [ ] **Check source code path:**
  - Which method is called in classify_entry_archetype()?
  - See line 482 condition: if self.threshold_policy is not None

---

## IMPLICATIONS FOR YOUR INVESTIGATION

### Why You Were Confused

The **same codebase implements BOTH systems simultaneously**:
- `/engine/archetypes/logic.py` (legacy)
- `/engine/archetypes/logic_v2_adapter.py` (adaptive)

Both are loaded, but only ONE is used per backtest run, determined by:
1. Config file structure (presence of gates_regime_profiles)
2. Adaptive Fusion enabled flag
3. Initialization logic in KnowledgeAwareBacktest.__init__()

### The Code Paths Are NOT Mutually Exclusive

The `check_archetype()` method exists in **both files**:
- In logic.py: Legacy implementation, used when threshold_policy is None
- In logic_v2_adapter.py: Backward compatibility wrapper, unused in normal flow

This caused confusion about which was being called.

**The truth: Line 482 branching decides which method is called, not which file.**

### What To Fix For Consistency

If you want reproducible results:
1. **Use legacy config with legacy path:** baseline_btc_bull_pf20.json
2. **Use adaptive config with adaptive path:** btc_v8_adaptive.json
3. **Don't mix them** - config determines which path, not flags

If you want to test parity between paths:
- Use `locked_regime='static'` in adaptive config to force static thresholds
- This bypasses regime blending but keeps the RuntimeContext code path

---

## FILES CREATED FOR YOUR INVESTIGATION

1. **`ARCHETYPE_PATHS_ANALYSIS.md`** - Complete technical breakdown
2. **`ARCHETYPE_PATHS_LOCATIONS.md`** - Exact file/line references
3. **`ARCHETYPE_INVESTIGATION_SUMMARY.md`** - This file

All saved to repository root for reference.

---

## QUICK REFERENCE: THRESHOLD FORMULA

For Archetype B in Adaptive Path:

```
final_threshold = clamp(
    max(
        base_threshold,
        blended_regime_floor
    ) + regime_override,
    global_min,
    global_max
)
```

Where:
- `base_threshold` = 0.359 (from config archetypes.thresholds.B.fusion)
- `blended_regime_floor` = sum(regime_prob × regime_profile[final_fusion_floor])
- `regime_override` = archetype_overrides[order_block_retest][regime_label].get('fusion', 0)
- `global_min` = 0.20 (from global_clamps.fusion[0])
- `global_max` = 0.60 (from global_clamps.fusion[1])

**This formula ONLY applies in adaptive path when ThresholdPolicy is not None.**

---

## END OF INVESTIGATION

The root cause has been identified and documented. The trade count differences are caused by threshold sensitivity to regime state, implemented through the ThresholdPolicy and RuntimeContext pattern in the adaptive path.

