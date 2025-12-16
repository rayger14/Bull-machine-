# DOMAIN ENGINES DIAGNOSTIC REPORT
**Date**: 2025-12-10
**Status**: ENGINES NOT OPERATIONAL - MISSING REQUIRED FEATURES

## Executive Summary

The domain engines (Temporal Fusion, SMC, Wyckoff Events, HOB) are properly configured and initialized, but **NOT functional** due to missing required features in the feature store. This explains why Core and Full variants produce identical results.

## Test Results

### S1 Liquidity Vacuum Comparison (2022)

| Metric | Core Variant | Full Variant | Difference |
|--------|--------------|--------------|------------|
| **Profit Factor** | 0.32 | 0.32 | **0% (IDENTICAL)** |
| **Total Trades** | 110 | 110 | **0% (IDENTICAL)** |
| **Win Rate** | 31.8% | 31.8% | **0% (IDENTICAL)** |
| **Total PNL** | -$3,652 | -$3,652 | **0% (IDENTICAL)** |

**Diagnosis**: Full variant engines initialized but not executed due to missing features.

## Root Cause Analysis

### 1. Config Fix Applied Successfully ✅

**Fix Location**: `/bin/backtest_knowledge_v2.py` lines 337-348

**Before** (engines not passed to ArchetypeLogic):
```python
full_archetype_config = {
    **archetype_config,
    'fusion': self.runtime_config.get('fusion', {}),
    'state_aware_gates': self.runtime_config.get('state_aware_gates', {}),
    'wyckoff_events': self.runtime_config.get('wyckoff_events', {})
}
```

**After** (engines now passed):
```python
full_archetype_config = {
    **archetype_config,
    'fusion': self.runtime_config.get('fusion', {}),
    'state_aware_gates': self.runtime_config.get('state_aware_gates', {}),
    'wyckoff_events': self.runtime_config.get('wyckoff_events', {}),
    # Domain engine sections (Ghost Modules V2)
    'temporal_fusion': self.runtime_config.get('temporal_fusion', {}),
    'smc_engine': self.runtime_config.get('smc_engine', {}),
    'hob_engine': self.runtime_config.get('hob_engine', {})
}
```

**Result**: Engines now initialize properly. Log confirms:
```
INFO:engine.archetypes.logic_v2_adapter:[ArchetypeLogic] Temporal Fusion Layer ENABLED
INFO:engine.temporal.temporal_fusion:[TemporalFusion] Initialized - enabled=True
```

### 2. Feature Store Analysis ⚠️

**Total Features**: 186 (confirmed)
**Domain Features Found**: 50

**Available Domain Features**:
- ✅ `smc_bos`, `smc_choch`, `smc_liquidity_sweep` (SMC features)
- ✅ `smc_demand_zone`, `smc_supply_zone`, `smc_score` (HOB features)
- ✅ `temporal_confluence`, `temporal_support_cluster`, `temporal_resistance_cluster`
- ✅ `liquidity_score`, `adaptive_threshold`
- ✅ `tf1d_wyckoff_phase`, `tf1d_wyckoff_score`, `tf1d_pti_score`

**MISSING Critical Features** (engines fail gracefully without these):

#### Temporal Fusion Engine Requirements:
- ❌ `bars_since_sc` (bars since Selling Climax)
- ❌ `bars_since_ar` (bars since Automatic Rally)
- ❌ `bars_since_st` (bars since Secondary Test)
- ❌ `bars_since_sos_long` (bars since Sign of Strength)
- ❌ `bars_since_sos_short` (bars since Sign of Weakness)
- ❌ `fib_time_cluster`, `fib_time_score`
- ❌ `gann_cycle`, `gann_score`
- ❌ `volatility_cycle`, `volatility_score`

**Why This Matters**:
The Temporal Fusion Engine computes confluence by checking if current bar is at Fibonacci time intervals (13, 21, 34, 55 bars) from major Wyckoff events. Without `bars_since_*` features, it returns neutral score (0.5) and applies no adjustment.

**Code Path** (`engine/temporal/temporal_fusion.py:215-254`):
```python
def _compute_fib_cluster_score(self, context: RuntimeContext) -> float:
    row = context.row

    # Get bars since key Wyckoff events
    event_keys = ['bars_since_sc', 'bars_since_ar', 'bars_since_st', ...]

    events = []
    for key in event_keys:
        bars = row.get(key, 999)  # ⚠️ Missing features return 999
        if pd.notna(bars) and bars < 999:
            events.append(int(bars))

    if not events:
        return 0.20  # ⚠️ No recent events → neutral score
```

### 3. Feature Backfill Status

**Claimed** (from previous task):
- ✅ 15 new domain features added
- ✅ Feature store complete (186 features)

**Reality**:
- ❌ Wyckoff event timing features NOT computed
- ❌ Fibonacci time features NOT computed
- ❌ Gann cycle features NOT computed
- ❌ Volatility cycle features NOT computed

**What Was Actually Backfilled**:
The 50 domain features found are **static/indicator features** (SMC zones, PTI scores, Wyckoff phase labels), but NOT the **dynamic/temporal features** (bars_since_*, time clusters) required for engine logic.

## Why Engines Don't Change Results

### Initialization vs Execution

**Engine Lifecycle**:

1. **Initialization** ✅ - Engines load configs and initialize
   ```
   INFO: Temporal Fusion Layer ENABLED
   INFO: SMC Engine ENABLED
   INFO: Wyckoff Events ENABLED
   ```

2. **Feature Access** ❌ - Engines try to read features
   ```python
   bars_since_sc = row.get('bars_since_sc', 999)  # Returns 999 (missing)
   ```

3. **Graceful Degradation** ⚠️ - Engines return neutral/no-op values
   ```python
   if not events:
       return 0.20  # Neutral score, no adjustment
   ```

4. **Result** - Full variant behaves identically to Core variant

### Engine-Specific Diagnosis

#### Temporal Fusion Engine
- **Status**: Initialized, not functional
- **Issue**: Missing `bars_since_*` features
- **Behavior**: Returns 0.5 (neutral), applies no fusion adjustment
- **Impact**: 0% change in trade selection

#### SMC Engine
- **Status**: Initialized, partially functional
- **Issue**: Has `smc_bos`, `smc_choch` features but may be missing liquidity pool tracking
- **Behavior**: Detects structure but can't boost/veto without full context
- **Impact**: Minimal (features exist but logic may not trigger)

#### Wyckoff Events Engine
- **Status**: Initialized, partially functional
- **Issue**: Has `wyckoff_phase` labels but missing event timing
- **Behavior**: Can detect phases but can't compute time-based confluence
- **Impact**: Minimal (phase detection only, no temporal boosts)

#### HOB Engine
- **Status**: Initialized, potentially functional
- **Issue**: Has `demand_zone`, `supply_zone` features
- **Behavior**: May work if zone features are complete
- **Impact**: Unknown (requires deeper testing)

## Required Actions

### Immediate (Fix Current Test)

1. **Generate Missing Temporal Features**
   ```bash
   python3 bin/backfill_wyckoff_events.py \
     --input data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
     --output data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
     --add-bars-since
   ```

2. **Verify Feature Completeness**
   ```python
   # Check for required features
   required = ['bars_since_sc', 'bars_since_ar', 'bars_since_st',
               'bars_since_sos_long', 'bars_since_sos_short']
   missing = [f for f in required if f not in df.columns]
   ```

3. **Rerun Variant Comparison**
   ```bash
   python3 bin/backtest_knowledge_v2.py \
     --asset BTC --start 2022-01-01 --end 2022-12-31 \
     --config configs/variants/s1_full.json
   ```

### Short-Term (Complete Engine Integration)

1. **Implement Missing Feature Generators**
   - `bin/add_temporal_time_features.py` - Fib time clusters
   - `bin/add_gann_cycles.py` - Gann vibration detection
   - `bin/add_volatility_cycles.py` - Compression/expansion phases

2. **Update Feature Store Schema**
   - Document required features per engine
   - Add feature validation to backtest startup
   - Fail fast if critical features missing

3. **Add Engine Diagnostics**
   - Log feature availability on engine init
   - Warn if engines degraded to neutral behavior
   - Track engine contribution to signal generation

### Long-Term (Production Readiness)

1. **Feature Pipeline Architecture**
   - Automated feature generation from OHLCV
   - Feature versioning and validation
   - Real-time feature computation for live trading

2. **Engine Testing Framework**
   - Unit tests for each engine with mock features
   - Integration tests verifying engine contribution
   - Performance benchmarks (with/without engines)

3. **Monitoring and Observability**
   - Engine activation rates (% of trades boosted/vetoed)
   - Feature coverage metrics (% non-null values)
   - Attribution analysis (PF improvement per engine)

## Current State vs Expected State

### Expected Behavior (User Request)
```
BEFORE (broken configs):
Core = Full (engines never activated)

AFTER (fixed configs + backfilled features):
Full > Core (engines boost performance)
```

### Actual Behavior (Current Test)
```
BEFORE (broken configs):
Core = Full (engines never activated) ✅ CONFIRMED

AFTER (fixed configs only):
Core = Full (engines activated but not functional) ⚠️ DEGRADED
```

### Next Required State
```
AFTER (fixed configs + complete features):
Full > Core (engines functional and impactful) 🎯 TARGET
```

## Recommendations

### Do NOT Proceed to ML Ensemble Yet

The user's requested verification test cannot complete successfully until:
1. ✅ Config fix (DONE - engines now passed to ArchetypeLogic)
2. ❌ Feature backfill (INCOMPLETE - temporal features missing)
3. ❌ Variant comparison (BLOCKED - identical results due to #2)

### Next Steps (In Order)

**Step 1**: Generate Wyckoff Event Timing Features
- Create `bars_since_*` features from existing `wyckoff_phase` labels
- Should detect phase transitions (accumulation→markup, distribution→markdown)
- Track bars elapsed since each event

**Step 2**: Add Fibonacci Time Cluster Features
- Use `bars_since_*` to compute proximity to Fib levels (13, 21, 34, 55, 89, 144)
- Generate `fib_time_cluster` boolean and `fib_time_score` [0-1]

**Step 3**: Verify Temporal Fusion Activation
- Rerun S1 Full variant
- Check logs for: `[TEMPORAL] Fusion adjusted: X.XXX → Y.YYY`
- Confirm non-neutral confluence scores

**Step 4**: Complete Variant Comparison
- Run S1, S4, S5 Core vs Full
- Expect different trade counts and PF values
- Generate attribution metrics per engine

**Step 5**: Document Engine Contribution
- Report PF improvement per engine
- Analyze which archetypes benefit most
- Validate production readiness

## Files Modified

1. `/bin/backtest_knowledge_v2.py` - Added domain engine section passing ✅

## Files Requiring Creation

1. `/bin/generate_wyckoff_event_timing.py` - Compute bars_since_* features
2. `/bin/generate_fib_time_features.py` - Compute Fibonacci time clusters
3. `/bin/validate_engine_features.py` - Feature availability checker

## Conclusion

The domain engines are **configured correctly** but **not operational** due to incomplete feature engineering. The backtest script fix is working (engines initialize), but the feature store lacks the temporal/event-based features required for engine logic.

**Status**: PARTIAL FIX APPLIED, BLOCKED ON FEATURE GENERATION

**Next Action**: Generate Wyckoff event timing features (bars_since_*) as first priority.
