# ARCHETYPE DETECTION CODE PATHS: LEGACY VS ADAPTIVE

## EXECUTIVE SUMMARY

There are **TWO FUNDAMENTALLY DIFFERENT CODE PATHS** for archetype detection in the backtest engine:

1. **LEGACY PATH** (baseline_btc_bull_pf20.json): Uses `check_archetype(row, prev_row, df, index)` method with **hardcoded thresholds**
2. **ADAPTIVE PATH** (btc_v8_adaptive.json): Uses `detect(RuntimeContext)` method with **regime-aware dynamic thresholds**

The **branching decision is made by the presence of `fusion_regime_profiles` in the config**, NOT by a direct flag.

---

## PATH SELECTION LOGIC

### File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py`

```python
# Lines 205-238: Adaptive Fusion Initialization Decision
if ADAPTIVE_FUSION_AVAILABLE and adaptive_config.get('enable'):
    try:
        # Load regime classifier
        self.regime_classifier = RegimeClassifier.load(...)
        
        # Initialize adaptive fusion coordinator
        self.adaptive_fusion = AdaptiveFusion(self.runtime_config)
        
        # PR#6B: Initialize ThresholdPolicy for regime-aware threshold management
        self.threshold_policy = ThresholdPolicy(
            base_cfg=self.runtime_config,
            regime_profiles=self.runtime_config.get('gates_regime_profiles'),  # KEY!
            archetype_overrides=self.runtime_config.get('archetype_overrides'),
            global_clamps=self.runtime_config.get('global_clamps'),
            locked_regime=locked_regime
        )
        logger.info("ThresholdPolicy: ENABLED (regime-aware archetype thresholds)")
    except Exception as e:
        self.regime_classifier = None
        self.adaptive_fusion = None
        self.threshold_policy = None
else:
    logger.info("Adaptive Fusion: DISABLED")
    self.threshold_policy = None
```

**KEY CONDITIONS FOR ADAPTIVE PATH:**
1. `ADAPTIVE_FUSION_AVAILABLE` = True (modules imported successfully)
2. `adaptive_config.get('enable')` = True in config
3. `runtime_config.get('gates_regime_profiles')` must exist
4. `runtime_config.get('regime_classifier')` config must exist

**LEGACY PATH TRIGGERED WHEN:**
- No `gates_regime_profiles` in config
- OR `adaptive_config.get('enable')` != True
- OR modules unavailable

---

## ARCHETYPE METHOD INVOCATION

### File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/backtest_knowledge_v2.py` Lines 458-512

```python
def classify_entry_archetype(self, row, context):
    """PR#6A: Classify entry opportunity into archetypes."""
    
    if self.archetype_logic and self.archetype_telemetry:
        # ... prepare row_with_runtime ...
        
        # BRANCHING POINT: Check for ThresholdPolicy (ADAPTIVE)
        if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
            # ===== ADAPTIVE PATH (uses RuntimeContext) =====
            adapted_params = context['adapted_params']
            regime_probs = adapted_params.get('regime_probs_ema', {'neutral': 1.0})
            regime_label = adapted_params.get('regime', 'neutral')
            
            # Resolve thresholds using ThresholdPolicy
            thresholds = self.threshold_policy.resolve(regime_probs, regime_label)
            
            # Build RuntimeContext with regime-aware thresholds
            runtime_ctx = RuntimeContext(
                ts=row.name,
                row=row_with_runtime,
                regime_probs=regime_probs,
                regime_label=regime_label,
                adapted_params=adapted_params,
                thresholds=thresholds  # REGIME-AWARE!
            )
            
            # CALL NEW ADAPTIVE METHOD
            archetype_name, fusion_score, liquidity_score = self.archetype_logic.detect(runtime_ctx)
            
        else:
            # ===== LEGACY PATH (uses old API) =====
            prev_row = None
            current_idx = context.get('current_index', 0)
            
            # CALL OLD LEGACY METHOD
            archetype_name, fusion_score, liquidity_score = self.archetype_logic.check_archetype(
                row=row_with_runtime,
                prev_row=prev_row,
                df=self.df,
                index=current_idx
            )
```

---

## THRESHOLD DIFFERENCES

### LEGACY PATH: `check_archetype()` in `/engine/archetypes/logic.py`

**Hardcoded thresholds** directly in archetype check methods. Example:

```python
def _check_A(self, row, prev_row, df, index, fusion_score) -> bool:
    """A - Trap Reversal"""
    pti_score = row.get('tf1h_pti_score', 0.0)
    if pti_score < self.thresh_A.get('pti', 0.40):  # Hardcoded default
        return False
    return True
```

**Threshold sources:**
1. Config `archetypes.thresholds.A.pti` = 0.40 (from baseline_btc_bull_pf20.json)
2. If missing, fallback to hardcoded defaults in logic.py
3. **NO regime awareness** - same threshold all the time

### ADAPTIVE PATH: `detect()` in `/engine/archetypes/logic_v2_adapter.py`

**Regime-aware thresholds** resolved dynamically at runtime. Example:

```python
def _check_A(self, ctx: RuntimeContext) -> bool:
    """A - Trap Reversal (PTI spring/UTAD + displacement)."""
    # Get regime-aware fusion threshold from policy
    fusion_th = ctx.get_threshold('spring', 'fusion', 0.33)  # Resolved by ThresholdPolicy!
    
    pti_trap = self.g(ctx.row, "pti_trap_type", '')
    if pti_trap not in ['spring', 'utad']:
        return False
    
    fusion = ctx.row.get('fusion_score', 0.0)
    return fusion >= fusion_th  # Uses resolved threshold
```

**Threshold sources:**
1. Base config `archetypes.thresholds.A.fusion` = 0.33
2. Blended with regime profiles from `gates_regime_profiles`
3. Applied overrides from `archetype_overrides`
4. Clamped to `global_clamps`
5. **REGIME-AWARE** - different at each bar based on regime probability

---

## THRESHOLD RESOLUTION PIPELINE (ADAPTIVE PATH ONLY)

### File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/archetypes/threshold_policy.py` Lines 80-120

```python
def resolve(self, regime_probs, regime_label):
    """
    5-STEP THRESHOLD RESOLUTION PIPELINE
    """
    # Step 1: Build base map from config
    final = self._build_base_map()
    
    # Step 2: Blend regime gates (weighted average across regimes)
    blended_gates = self._blend_regime_gates(regime_probs)
    #   Input: regime_probs = {'risk_on': 0.62, 'neutral': 0.28, ...}
    #   Input: gates_regime_profiles['risk_on'] = {'final_fusion_floor': 0.36, 'min_liquidity': 0.18}
    #   Output: blended_gates = {'final_fusion_floor': 0.37, 'min_liquidity': 0.19}
    
    # Step 3: Apply regime floors to ALL archetypes
    self._apply_regime_floors(final, blended_gates)
    #   Enforces: archetype.fusion >= blended_gates['final_fusion_floor']
    #   Enforces: archetype.liquidity >= blended_gates['min_liquidity']
    
    # Step 4: Apply per-archetype regime overrides
    self._apply_archetype_overrides(final, regime_label)
    #   Example override: order_block_retest[risk_on] += {'fusion': -0.02}
    
    # Step 5: Clamp to global guardrails
    self._clamp(final)
    #   Ensures: 0.20 <= fusion <= 0.65
    #   Ensures: 0.08 <= liquidity <= 0.35
    
    return final
```

---

## EXAMPLE: THRESHOLD DIFFERENCE FOR ARCHETYPE B

### LEGACY PATH (baseline_btc_bull_pf20.json)

```json
{
  "archetypes": {
    "thresholds": {
      "B": {
        "fusion": 0.35912732076623655  // Static, always this value
      }
    }
  }
  // NO gates_regime_profiles, NO archetype_overrides
}
```

**At every bar:** `fusion >= 0.359`

### ADAPTIVE PATH (btc_v8_adaptive.json)

```json
{
  "gates_regime_profiles": {
    "risk_on":   {"final_fusion_floor": 0.36, "min_liquidity": 0.18},
    "neutral":   {"final_fusion_floor": 0.38, "min_liquidity": 0.16},
    "risk_off":  {"final_fusion_floor": 0.43, "min_liquidity": 0.20},
    "crisis":    {"final_fusion_floor": 0.52, "min_liquidity": 0.26}
  },
  "archetype_overrides": {
    "order_block_retest": {
      "risk_on": {"fusion": -0.02},
      "risk_off": {"fusion": +0.03}
    }
  },
  "global_clamps": {
    "fusion": [0.20, 0.60]
  }
}
```

**Bar-by-bar resolution example:**

```
Scenario 1: risk_on regime (prob = 1.0)
  Base:              0.359
  Regime floor:      max(0.359, 0.36) = 0.36
  Override:          0.36 + (-0.02) = 0.34
  Clamp [0.20,0.60]: 0.34
  FINAL:             0.34

Scenario 2: crisis regime (prob = 1.0)
  Base:              0.359
  Regime floor:      max(0.359, 0.52) = 0.52
  Override:          0.52 + 0.00 = 0.52
  Clamp [0.20,0.60]: 0.52
  FINAL:             0.52

Scenario 3: blended (risk_on=0.6, neutral=0.4)
  Base:              0.359
  Blended floor:     0.6*0.36 + 0.4*0.38 = 0.368
  Regime floor:      max(0.359, 0.368) = 0.368
  Override:          0.368 + (-0.02*0.6 + 0.00*0.4) = 0.356
  Clamp [0.20,0.60]: 0.356
  FINAL:             0.356
```

**Result:** Threshold **varies from 0.34 to 0.52** depending on regime!

---

## RUNTIME CONTEXT: THE ADAPTIVE PATH'S KEY DIFFERENCE

### File: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/engine/runtime/context.py`

```python
@dataclass(frozen=True)
class RuntimeContext:
    """Immutable context object passed to archetype.detect()"""
    ts: Any
    row: pd.Series
    regime_probs: Dict[str, float]      # Current regime probability distribution
    regime_label: str                   # Argmax regime (e.g., 'risk_on')
    adapted_params: Dict[str, Any]      # From AdaptiveFusion.update()
    thresholds: Dict[str, Dict[str, float]]  # RESOLVED BY THRESHOLDPOLICY!
    
    def get_threshold(self, archetype: str, param: str, default: float = 0.0) -> float:
        """Safely retrieve regime-aware threshold"""
        return self.thresholds.get(archetype, {}).get(param, default)
```

**Archetype checks now access thresholds via:**
```python
fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)
```

**Legacy checks access thresholds via:**
```python
fusion_th = self.thresh_B.get('fusion', 0.374)
```

---

## FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│ BACKTEST LOOP: for each bar                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ Adaptive Fusion Update (if enabled)   │
        │ regime_classifier.classify()          │
        │ adaptive_fusion.update()              │
        └──────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
           YES  │                   │ NO
      (enabled) │                   │ (disabled)
                │                   │
                ▼                   ▼
    ┌──────────────────────┐  ┌──────────────┐
    │ adapted_params set   │  │ adapted_params
    │ regime_probs filled  │  │ = None
    └──────────────────────┘  └──────────────┘
                │                   │
                ▼                   │
    ┌──────────────────────┐        │
    │ ThresholdPolicy.     │        │
    │ resolve()            │        │
    │ - Blend regimes      │        │
    │ - Apply floors       │        │
    │ - Apply overrides    │        │
    │ - Clamp guardrails   │        │
    └──────────────────────┘        │
                │                   │
                │ thresholds        │
                │ (regime-aware)    │
                │                   │
                ▼                   ▼
    ┌──────────────────────┐  ┌──────────────────┐
    │ RuntimeContext()     │  │ No RuntimeContext│
    │ + thresholds         │  │ threshold_policy │
    │                      │  │ = None           │
    └──────────────────────┘  └──────────────────┘
                │                   │
                ▼                   ▼
    ┌──────────────────────┐  ┌──────────────────┐
    │ archetype.detect()   │  │ archetype.       │
    │ (ADAPTIVE PATH)      │  │ check_archetype()│
    │ Uses ctx.get_        │  │ (LEGACY PATH)    │
    │ threshold()          │  │ Uses hardcoded   │
    │                      │  │ thresholds       │
    └──────────────────────┘  └──────────────────┘
                │                   │
                │ (archetype_name)  │
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                  Trade Entry/Exit Logic
```

---

## DECISION TREE: WHICH PATH WILL BE USED?

```
Config loaded: baseline_btc_bull_pf20.json
├─ fusion_regime_profiles? NO
├─ gates_regime_profiles? NO
├─ archetype_overrides? NO
├─ regime_classifier config? NO
└─ Result: LEGACY PATH
   - threshold_policy = None
   - Uses check_archetype() with hardcoded thresholds
   - All archetypes use same threshold every bar

Config loaded: btc_v8_adaptive.json
├─ fusion_regime_profiles? YES
├─ gates_regime_profiles? YES ← KEY TRIGGER
├─ archetype_overrides? YES
├─ regime_classifier config? YES
├─ adaptive_config.enable? YES
├─ ADAPTIVE_FUSION_AVAILABLE? YES
└─ Result: ADAPTIVE PATH
   - threshold_policy = ThresholdPolicy(gates_regime_profiles, ...)
   - Uses detect(RuntimeContext) with regime-aware thresholds
   - Thresholds vary by bar based on regime probability
```

---

## WHY TRADE COUNTS DIFFER (64 vs 84 vs 19)

The three different trade counts suggest **threshold sensitivity is EXTREME**:

### 64 Trades (baseline_btc_bull_pf20.json - LEGACY)
- Hardcoded fusion thresholds: B=0.359, C=0.494, H=0.544, K=0.435, L=0.349, etc.
- No regime modulation
- Consistent through entire backtest

### 84 Trades (btc_v8_adaptive.json - ADAPTIVE with default blending)
- Thresholds vary by regime probability
- Risk_on regime uses LOWER floors (0.36) → MORE entries
- Regime blending creates intermediate values
- Different number of entries when risk_on vs crisis

### 19 Trades (locked_regime='static'?)
- OR some other extreme config variant
- Possibly extremely tight thresholds or additional filters

**Key insight:** The regime classification is **changing thresholds enough to swing trades by 65+ count!**

---

## CRITICAL IMPLEMENTATION NOTES

### 1. ArchetypeLogic Versioning

- **logic.py** (legacy): `check_archetype(row, prev_row, df, index)`
  - Hardcoded thresholds in self.thresh_A, self.thresh_B, etc.
  - No regime awareness
  - Signature shows it's for old API

- **logic_v2_adapter.py** (adaptive): `detect(RuntimeContext)`
  - Uses ctx.get_threshold() which reads from thresholds dict
  - Regime-aware via ThresholdPolicy
  - Includes alias mappings for feature name variations

### 2. ThresholdPolicy Is The Lock/Switch

The **ONLY** place where regime-awareness is applied is ThresholdPolicy.resolve():
- It's where base thresholds become regime-aware
- It's where floors/overrides/clamps are applied
- Without it, all archetypes use hardcoded values

### 3. adapted_params Is The Trigger

In classify_entry_archetype():
```python
if self.threshold_policy and 'adapted_params' in context and context['adapted_params']:
    # Use RuntimeContext path
else:
    # Use legacy path
```

The presence of **non-None adapted_params** determines which path is taken.

### 4. AdaptiveFusion.update() Sets adapted_params

```python
# In backtest loop (line 1731)
adapted_params = self.adaptive_fusion.update(regime_info)
# adapted_params is stored in context (line 1745)
context['adapted_params'] = adapted_params
```

If adaptive_fusion is None, adapted_params stays None.

---

## VALIDATION CHECKLIST

To identify which path is active:

- [ ] Look for `fusion_regime_profiles` in config → If YES, adaptive path will activate
- [ ] Look for `gates_regime_profiles` in config → If YES, ThresholdPolicy will be created
- [ ] Check logs for "ThresholdPolicy: ENABLED" or "DISABLED"
- [ ] Monitor archetype detection logs - adaptive shows threshold values, legacy doesn't
- [ ] Run with `log_level=DEBUG` to see regime classification (risk_on, neutral, risk_off, crisis)
- [ ] Check trade counts - they should differ significantly between configs

---

## SUMMARY TABLE

| Aspect | LEGACY PATH | ADAPTIVE PATH |
|--------|------------|----------------|
| **Config File** | baseline_btc_bull_pf20.json | btc_v8_adaptive.json |
| **Method** | check_archetype() | detect() |
| **Thresholds** | Hardcoded/static | Regime-aware dynamic |
| **Threshold Source** | self.thresh_A, thresh_B, ... | RuntimeContext.thresholds |
| **Regime Awareness** | NONE | FULL (via ThresholdPolicy) |
| **ThresholdPolicy** | None | Initialized + used |
| **RuntimeContext** | Not used | Built per bar |
| **AdaptiveFusion** | None | Initialized + updated |
| **RegimeClassifier** | None | Loaded + used |
| **Threshold Variation** | Same all bars | Changes per bar |
| **Expected Trades** | ~64 | ~84 (varies by regime mix) |
| **Configuration Keys** | NO fusion_regime_profiles | fusion_regime_profiles + gates_regime_profiles |

