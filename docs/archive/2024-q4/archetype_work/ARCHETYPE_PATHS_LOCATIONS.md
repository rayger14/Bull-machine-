# SOURCE CODE LOCATIONS: ARCHETYPE DETECTION PATHS

## Quick File Reference

### Configuration Files
- **Legacy Config:** `/configs/baseline_btc_bull_pf20.json` - Lines 146-233 (archetypes section)
- **Adaptive Config:** `/configs/btc_v8_adaptive.json` - Lines 6-67 (fusion_regime_profiles + gates_regime_profiles)

### Core Archetype Logic
- **Legacy Implementation:** `/engine/archetypes/logic.py` (lines 1-719)
  - `class ArchetypeLogic`
  - `def check_archetype()` - Lines 185-262
  - `def _check_A()` - Lines 268-298
  - `def _check_B()` - Lines 300-349
  - (etc. for C-M)

- **Adaptive Implementation:** `/engine/archetypes/logic_v2_adapter.py` (lines 1-532)
  - `class ArchetypeLogic` (v2 with adapter layer)
  - `def detect()` - Lines 247-305 (main entry point for ADAPTIVE path)
  - `def check_archetype()` - Lines 311-333 (backward compatibility wrapper)
  - `def _check_A()` - Lines 339-357
  - (etc. for B-M)

### Threshold Policy & Runtime Context
- **ThresholdPolicy:** `/engine/archetypes/threshold_policy.py` (lines 1-266)
  - `class ThresholdPolicy` - Lines 31-266
  - `def resolve()` - Lines 80-120 (5-step pipeline)
  - `def _blend_regime_gates()` - Lines 136-166 (regime blending)
  - `def _apply_regime_floors()` - Lines 168-194 (floor application)
  - `def _apply_archetype_overrides()` - Lines 196-223 (override application)

- **RuntimeContext:** `/engine/runtime/context.py` (lines 1-61)
  - `@dataclass RuntimeContext` - Lines 13-33
  - `def get_threshold()` - Lines 35-47 (safe getter for thresholds)

### Backtest Engine Integration
- **Main Backtest Script:** `/bin/backtest_knowledge_v2.py`
  - **Initialization (Adaptive Setup):** Lines 205-238
    - Adaptive Fusion initialization
    - ThresholdPolicy creation
    - Key decision point
    
  - **Archetype Method Invocation:** Lines 458-512 in `classify_entry_archetype()`
    - **BRANCHING POINT:** Line 482
    - ADAPTIVE path (RuntimeContext): Lines 483-502
    - LEGACY path (check_archetype): Lines 503-512
    
  - **Adaptive Fusion Update:** Line 1731 in `run()`
    - `adapted_params = self.adaptive_fusion.update(regime_info)`
    - Stored in context at line 1745

### Supporting Modules
- **AdaptiveFusion:** `/engine/fusion/adaptive.py`
  - Regime-aware parameter morphing
  - Blends fusion weights, gates, exits, sizing

- **RegimeClassifier:** `/engine/context/regime_classifier.py`
  - GMM-based regime classification
  - Returns regime probabilities (smoothed with EMA)

- **Runtime Liquidity Scorer:** `/engine/liquidity/score.py`
  - PR#4 runtime intelligence
  - Computes liquidity_score on the fly

---

## Code Paths - Exact Line References

### PATH SELECTION (Which method gets called?)

**File:** `/bin/backtest_knowledge_v2.py`
**Function:** `classify_entry_archetype(self, row: pd.Series, context: Dict)`
**Lines:** 458-512

```
Line 469-470: Check if archetype_logic exists
  if self.archetype_logic and self.archetype_telemetry:

Line 482: CRITICAL BRANCHING DECISION
  if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
    
    Lines 483-502: ADAPTIVE PATH
    - Get regime_probs from adapted_params (line 485)
    - Get regime_label from adapted_params (line 486)
    - Call threshold_policy.resolve() (line 489)
    - Build RuntimeContext (lines 492-499)
    - Call archetype_logic.detect(runtime_ctx) (line 502)
    
  Else:
    Lines 503-512: LEGACY PATH
    - Call archetype_logic.check_archetype() (lines 507-512)
```

---

### ADAPTIVE PATH CALL CHAIN

```
backtest_knowledge_v2.py:1731
├─ adaptive_fusion.update(regime_info)
│  └─ Returns: adapted_params with regime_probs_ema, regime_label
│
backtest_knowledge_v2.py:1742
├─ compute_advanced_fusion_score(row, adapted_params)
│  └─ Uses adapted_params to compute fusion weights
│
backtest_knowledge_v2.py:1793
├─ check_entry_conditions(row, fusion_score, context)
│  └─ Calls: classify_entry_archetype(row, context)
│     │
│     backtest_knowledge_v2.py:489
│     ├─ threshold_policy.resolve(regime_probs, regime_label)
│     │  ├─ Lines 106: _build_base_map() - Start with base thresholds
│     │  ├─ Lines 109: _blend_regime_gates() - Weighted average of regime profiles
│     │  ├─ Lines 112: _apply_regime_floors() - Apply min/max floors
│     │  ├─ Lines 115: _apply_archetype_overrides() - Delta adjustments
│     │  ├─ Lines 118: _clamp() - Guardrail clamping
│     │  └─ Returns: thresholds dict
│     │
│     backtest_knowledge_v2.py:502
│     └─ archetype_logic.detect(runtime_ctx)
│        ├─ logic_v2_adapter.py:271: _check_A(ctx)
│        │  └─ ctx.get_threshold('spring', 'fusion', 0.33)
│        ├─ logic_v2_adapter.py:274: _check_B(ctx)
│        │  └─ ctx.get_threshold('order_block_retest', 'fusion', 0.374)
│        ├─ (etc. for C-M)
│        └─ Returns: (archetype_name, fusion_score, liquidity_score)
```

---

### LEGACY PATH CALL CHAIN

```
backtest_knowledge_v2.py:1793
├─ check_entry_conditions(row, fusion_score, context)
│  └─ Calls: classify_entry_archetype(row, context)
│     │
│     backtest_knowledge_v2.py:507-512
│     └─ archetype_logic.check_archetype(row_with_runtime, None, self.df, current_idx)
│        ├─ logic.py:204: Check use_archetypes flag
│        ├─ logic.py:209: Get liquidity_score via _get_liquidity_score()
│        ├─ logic.py:213: Calculate fusion_score via calculate_fusion_score()
│        │
│        ├─ logic.py:218-260: Check archetypes in order A-M
│        │  ├─ Line 219: _check_A(row, prev_row, df, index, fusion_score)
│        │  │  └─ Uses self.thresh_A.get('fusion', 0.33) - HARDCODED
│        │  ├─ Line 223: _check_B(row, prev_row, df, index, fusion_score)
│        │  │  └─ Uses self.thresh_B.get('fusion', 0.374) - HARDCODED
│        │  ├─ (etc. for C-M)
│        │  └─ All use self.thresh_X.get(param, default) - HARDCODED DEFAULTS
│        │
│        └─ Returns: (archetype_name_or_None, fusion_score, liquidity_score)
```

---

## Threshold Access Patterns

### ADAPTIVE PATH: Context-Aware Lookup

**File:** `/engine/archetypes/logic_v2_adapter.py`

```python
# Line 342 (Archetype A check)
fusion_th = ctx.get_threshold('spring', 'fusion', 0.33)

# RuntimeContext.get_threshold() implementation (runtime/context.py:35-47)
def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
    return self.thresholds.get(archetype, {}).get(param, default)

# self.thresholds comes from ThresholdPolicy.resolve()
# which blends regimes and applies overrides
```

**Result:** Threshold varies by bar based on regime probability

### LEGACY PATH: Config-Loaded Lookup

**File:** `/engine/archetypes/logic.py`

```python
# Line 50 (Initialization)
self.thresh_B = thresholds.get('B', {})

# Line 319 (Archetype B check)
if wyckoff_score < self.thresh_B.get('wyckoff', 0.35):
    return False

# self.thresh_B was loaded once in __init__ from config
# Never changes during backtest
```

**Result:** Threshold is static, same for all bars

---

## Configuration Key Differences

### LEGACY CONFIG (baseline_btc_bull_pf20.json)

```json
{
  "archetypes": {
    "use_archetypes": true,
    "thresholds": {
      "A": {"pti": 0.4, "disp_atr": 0.8, "fusion": 0.33},
      "B": {"fusion": 0.35912732076623655},
      "C": {"fusion": 0.49428998670944535},
      "H": {"fusion": 0.5443074169684492},
      "K": {"fusion": 0.43519385852689346},
      "L": {"fusion": 0.34945256176735884},
      ...
    },
    "exits": { ... }
  }
  // NO fusion_regime_profiles
  // NO gates_regime_profiles
  // NO archetype_overrides
  // NO global_clamps
}
```

**Key missing:** `fusion_regime_profiles`, `gates_regime_profiles`

### ADAPTIVE CONFIG (btc_v8_adaptive.json)

```json
{
  "fusion_regime_profiles": {
    "risk_on":   {"wyckoff": 0.443, "liquidity": 0.227, "momentum": 0.331, "temporal": 0.0},
    "neutral":   {"wyckoff": 0.42, "liquidity": 0.28, "momentum": 0.20, "temporal": 0.10},
    "risk_off":  {"wyckoff": 0.38, "liquidity": 0.32, "momentum": 0.16, "temporal": 0.14},
    "crisis":    {"wyckoff": 0.34, "liquidity": 0.38, "momentum": 0.12, "temporal": 0.16}
  },
  "gates_regime_profiles": {
    "risk_on":   {"min_liquidity": 0.18, "final_fusion_floor": 0.36},
    "neutral":   {"min_liquidity": 0.16, "final_fusion_floor": 0.38},
    "risk_off":  {"min_liquidity": 0.20, "final_fusion_floor": 0.43},
    "crisis":    {"min_liquidity": 0.26, "final_fusion_floor": 0.52}
  },
  "archetype_overrides": {
    "order_block_retest": {
      "risk_on": {"fusion": -0.02},
      "risk_off": {"fusion": 0.03}
    },
    "trap_within_trend": {
      "risk_on": {"fusion": -0.015},
      "crisis": {"fusion": 0.05}
    }
  },
  "global_clamps": {
    "fusion": [0.20, 0.60],
    "liquidity": [0.08, 0.32],
    "min_liquidity": [0.08, 0.30]
  },
  "archetypes": {
    "use_archetypes": true,
    "thresholds": { ... },  // Still present as base
    "exits": { ... }
  }
}
```

**Key additions:** `fusion_regime_profiles`, `gates_regime_profiles`, `archetype_overrides`, `global_clamps`

---

## Debug/Verification Commands

### Check which path is active

```bash
# Look for ThresholdPolicy in logs
grep -n "ThresholdPolicy" /path/to/backtest.log

# Should see either:
# "ThresholdPolicy: ENABLED (regime-aware archetype thresholds)" → ADAPTIVE
# OR
# "Adaptive Fusion: DISABLED" → LEGACY
```

### Check config has the key trigger

```bash
# Check for regime profiles
grep -c "gates_regime_profiles" config.json
# If > 0: Will use adaptive path (if other conditions met)
# If = 0: Will use legacy path
```

### Monitor threshold values during runtime

**Adaptive path logs regime-aware thresholds:**
```
DEBUG: Blended regime gates: {'final_fusion_floor': 0.37, 'min_liquidity': 0.19}
DEBUG: Resolved thresholds for regime=risk_on
```

**Legacy path doesn't show threshold resolution:**
```
No regime-related debug output
```

---

## Trade Count Attribution

### Why 64 trades (Legacy)
- Static thresholds B=0.359, C=0.494, H=0.544, K=0.435, L=0.349
- Same thresholds every bar
- More restrictive than adaptive average

### Why 84 trades (Adaptive)
- risk_on regime has lower floors (0.36) → MORE entries
- Blending increases entries when risk_on probability high
- 20 extra trades = difference between adaptive and legacy

### Why 19 trades (Extreme variant?)
- Possibly `locked_regime='static'` force entire backtest to base thresholds
- OR extremely tight global_clamps
- OR additional filters enabled

---

## Key Discovery: The Guardian Gate

The **most impactful threshold is `final_fusion_floor`** from gates_regime_profiles:

```python
# In ThresholdPolicy._apply_regime_floors() - Line 186
if final_fusion_floor is not None and 'fusion' in thresholds:
    thresholds['fusion'] = max(thresholds['fusion'], final_fusion_floor)
```

This **raises ALL archetype fusion thresholds** to the regime minimum:

| Regime | final_fusion_floor | Effect |
|--------|-------------------|--------|
| risk_on | 0.36 | LOWER - more entries |
| neutral | 0.38 | MODERATE |
| risk_off | 0.43 | HIGHER - fewer entries |
| crisis | 0.52 | VERY HIGH - almost no entries |

The fusion floor is why adaptive config finds **84 vs 64 trades** - it relaxes entry requirements in risk_on regime!

