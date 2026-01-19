# Variant Configs Complete Index

**Master Index for All Variant Configurations and Documentation**

---

## Quick Navigation

### Running Variants
- **Quick Start:** Run single config for testing
- **Full Suite:** Run all 9 variants sequentially
- **Comparison:** Analyze results across variants

### Documentation Files
1. **VARIANT_CONFIGS_SUMMARY.md** - Detailed technical specifications
2. **VARIANT_TESTING_GUIDE.md** - Testing instructions and analysis
3. **CONFIG_VARIANTS_QUICK_REF.md** - Quick reference lookup
4. **VARIANT_CONFIGS_INDEX.md** - This file (master index)

---

## Configuration Files

### Location
```
/configs/variants/
```

### Complete List (9 Files)

#### S1 Liquidity Vacuum
```
s1_core.json                     (4.8 KB)  Wyckoff only
s1_core_plus_time.json           (5.0 KB)  Wyckoff + Temporal
s1_full.json                     (4.3 KB)  All engines (production)
```

#### S4 Funding Divergence
```
s4_core.json                     (4.2 KB)  Funding only
s4_core_plus_macro.json          (4.5 KB)  Funding + Macro routing
s4_full.json                     (3.7 KB)  All features (production)
```

#### S5 Long Squeeze
```
s5_core.json                     (4.2 KB)  Funding + RSI
s5_core_plus_wyckoff.json        (4.2 KB)  + Wyckoff distribution
s5_full.json                     (3.6 KB)  All features (production)
```

---

## Configuration Structure Overview

### S1 Liquidity Vacuum Variants

**s1_core.json**
- Engines: 1/6 (Wyckoff)
- Regime Filter: Disabled
- Temporal: Disabled
- Expected: 100-150 trades/year
- Purpose: Test raw capitulation pattern quality

**s1_core_plus_time.json**
- Engines: 2/6 (Wyckoff + Temporal)
- Regime Filter: Disabled
- Temporal: Enabled (0-15 UTC, 20% weight boost)
- Expected: 60-80 trades/year
- Purpose: Test time-of-day filtering impact

**s1_full.json**
- Engines: 6/6 (All engines)
- Regime Filter: Enabled (risk_off, crisis)
- Routing: Weighted (crisis=2.0, bear=1.5, neutral=1.0, bull=0.5)
- Expected: 40-60 trades/year (production)
- Purpose: Production reference

---

### S4 Funding Divergence Variants

**s4_core.json**
- Engines: 1/6 (Funding logic)
- Regime Filter: Disabled
- Macro: Disabled
- Expected: 30-40 trades/year (all regimes)
- Purpose: Test raw funding divergence quality

**s4_core_plus_macro.json**
- Engines: 2/6 (Funding + Macro routing)
- Regime Filter: Enabled
- Routing: Weighted (crisis=1.5, bear=1.0, neutral=1.0, bull=0.5)
- Expected: 15-20 trades/year
- Purpose: Test regime discrimination impact

**s4_full.json**
- Engines: 6/6 (All optimized)
- Regime Filter: Enabled
- Routing: Production (crisis=1.5, bear=1.0)
- Expected: 12 trades/year in bear markets
- Profit Factor: 2.22 (optimized)
- Purpose: Production reference (bear specialist)

---

### S5 Long Squeeze Variants

**s5_core.json**
- Engines: 2/6 (Funding + RSI)
- Wyckoff: Disabled
- Macro: Disabled
- Expected: 20-30 trades/year (all regimes)
- Purpose: Test simple momentum setup

**s5_core_plus_wyckoff.json**
- Engines: 3/6 (Funding + RSI + Wyckoff)
- Wyckoff Distribution: Enabled
- Macro: Disabled
- Expected: 12-15 trades/year
- Purpose: Test structural pattern detection

**s5_full.json**
- Engines: 6/6 (All optimized)
- Regime Filter: Enabled
- Routing: Bear specialist (crisis=2.5, bear=2.2, neutral=0.5, bull=0.0)
- Expected: 9 trades/year in bear markets
- Profit Factor: 1.86 (optimized)
- Purpose: Production reference (bear specialist)

---

## Complexity Progression

### S1 (Liquidity Vacuum)
```
Core              1 engine  → Pure pattern
    ↓ (Add Temporal)
Core+Time         2 engines → Pattern + time context
    ↓ (Add Macro + other engines)
Full              6 engines → Production (regime-aware)

Expected Trades:  150 → 70 → 50/year
Filtering Impact: None → Time → Full
```

### S4 (Funding Divergence)
```
Core              1 engine  → Raw divergence
    ↓ (Add Macro routing)
Core+Macro        2 engines → Divergence + regime context
    ↓ (Full optimization)
Full              6 engines → Production (bear specialist)

Expected Trades:  35 → 17 → 12/year
Filtering Impact: None → Regime → Full
```

### S5 (Long Squeeze)
```
Core              2 engines → Momentum only
    ↓ (Add Wyckoff patterns)
Core+Wyckoff      3 engines → Momentum + structure
    ↓ (Full optimization)
Full              6 engines → Production (bear specialist)

Expected Trades:  25 → 13 → 9/year
Filtering Impact: Minimal → Patterns → Full
```

---

## Domain Engine Reference

### Which Engines in Each Variant

| Engine | S1 Core | S1 C+T | S1 Full | S4 Core | S4 C+M | S4 Full | S5 Core | S5 C+W | S5 Full |
|--------|---------|--------|---------|---------|--------|---------|---------|--------|---------|
| Wyckoff | X | X | X | - | - | - | - | X | X |
| SMC | - | - | X | - | - | - | - | - | X |
| Temporal | - | X | X | - | - | - | - | - | X |
| HOB | - | - | X | - | - | - | - | - | X |
| Fusion | - | - | X | - | - | - | - | - | X |
| Macro | - | - | X | - | X | X | - | - | X |

---

## Feature Flags by Variant

### S1 Variants

**s1_core.json:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**s1_core_plus_time.json:**
```json
{
  "enable_wyckoff": true,
  "enable_temporal": true,
  "enable_smc": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false,
  "use_temporal_confluence": true
}
```

**s1_full.json:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_fusion": true,
  "enable_macro": true,
  "use_temporal_confluence": true,
  "use_fusion_layer": true,
  "use_macro_regime": true
}
```

### S4 Variants

**s4_core.json:**
```json
{
  "enable_wyckoff": false,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**s4_core_plus_macro.json:**
```json
{
  "enable_macro": true,
  "use_macro_regime": true,
  (others false)
}
```

**s4_full.json:**
```json
{
  "enable_macro": true,
  "use_macro_regime": true,
  (minimal other flags)
}
```

### S5 Variants

**s5_core.json:**
```json
{
  "enable_wyckoff": false,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**s5_core_plus_wyckoff.json:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_fusion": false,
  "enable_macro": false
}
```

**s5_full.json:**
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_fusion": true,
  "enable_macro": true,
  "use_temporal_confluence": true,
  "use_fusion_layer": true,
  "use_macro_regime": true
}
```

---

## Regime Routing by Variant

### S1 Variants

**Core (Flat):**
```json
"routing": {
  "risk_on": {"weights": {"liquidity_vacuum": 1.0}},
  "neutral": {"weights": {"liquidity_vacuum": 1.0}},
  "risk_off": {"weights": {"liquidity_vacuum": 1.0}},
  "crisis": {"weights": {"liquidity_vacuum": 1.0}}
}
```

**Core+Time (Flat):**
```json
(identical to core, but temporal confluence adds context)
```

**Full (Weighted):**
```json
"routing": {
  "risk_on": {"weights": {"liquidity_vacuum": 0.5}},
  "neutral": {"weights": {"liquidity_vacuum": 1.0}},
  "risk_off": {"weights": {"liquidity_vacuum": 1.5}},
  "crisis": {"weights": {"liquidity_vacuum": 2.0}}
}
```

### S4 Variants

**Core (Flat):**
```json
"routing": {
  "risk_on": {"weights": {"funding_divergence": 1.0}},
  "neutral": {"weights": {"funding_divergence": 1.0}},
  "risk_off": {"weights": {"funding_divergence": 1.0}},
  "crisis": {"weights": {"funding_divergence": 1.0}}
}
```

**Core+Macro (Weighted):**
```json
"routing": {
  "risk_on": {"weights": {"funding_divergence": 0.5}},
  "neutral": {"weights": {"funding_divergence": 1.0}},
  "risk_off": {"weights": {"funding_divergence": 1.0}},
  "crisis": {"weights": {"funding_divergence": 1.5}}
}
```

**Full (Weighted):**
```json
(same as core+macro in this case)
```

### S5 Variants

**Core (Flat):**
```json
"routing": {
  "risk_on": {"weights": {"long_squeeze": 1.0}},
  "neutral": {"weights": {"long_squeeze": 1.0}},
  "risk_off": {"weights": {"long_squeeze": 1.0}},
  "crisis": {"weights": {"long_squeeze": 1.0}}
}
```

**Core+Wyckoff (Flat):**
```json
(identical to core, but Wyckoff detection adds selectivity)
```

**Full (Weighted, Bear Specialist):**
```json
"routing": {
  "risk_on": {"weights": {"long_squeeze": 0.0}},     // DISABLED
  "neutral": {"weights": {"long_squeeze": 0.5}},
  "risk_off": {"weights": {"long_squeeze": 2.2}},
  "crisis": {"weights": {"long_squeeze": 2.5}}
}
```

---

## Expected Performance by Variant

### Trade Frequency

| Variant | Expected Trades/Year | Context |
|---------|----------------------|---------|
| S1 Core | 100-150 | All regimes, no filters |
| S1 Core+Time | 60-80 | Time filtering added |
| S1 Full | 40-60 | Production (regime-aware) |
| S4 Core | 30-40 | All regimes |
| S4 Core+Macro | 15-20 | Regime aware |
| S4 Full | 12 | Bear specialist |
| S5 Core | 20-30 | All regimes |
| S5 Core+Wyckoff | 12-15 | Pattern filtering |
| S5 Full | 9 | Bear specialist |

### Win Rate & Profit Factor

| Variant | Win Rate | Profit Factor | Notes |
|---------|----------|---------------|-------|
| S1 Core | Unknown | Unknown | Baseline test |
| S1 Core+Time | Unknown | Unknown | Mid-complexity |
| S1 Full | 50-60% | Unknown | Production |
| S4 Core | Unknown | Unknown | Baseline test |
| S4 Core+Macro | Unknown | Unknown | Mid-complexity |
| S4 Full | 55.7% | 2.22 | Optimized (2022) |
| S5 Core | Unknown | Unknown | Baseline test |
| S5 Core+Wyckoff | Unknown | Unknown | Mid-complexity |
| S5 Full | 55.6% | 1.86 | Optimized (2022) |

---

## How to Use These Variants

### Step 1: Run One Variant
```bash
python bin/backtest_knowledge_v2.py configs/variants/s1_core.json
```

### Step 2: Capture Results
```
Trade Frequency: ?/year
Win Rate: ?%
Profit Factor: ?
Sharpe: ?
Max Drawdown: -?%
```

### Step 3: Run All Variants (for comparison)
```bash
for cfg in configs/variants/s1*.json configs/variants/s4*.json configs/variants/s5*.json; do
  python bin/backtest_knowledge_v2.py "$cfg"
done
```

### Step 4: Build Comparison Table
```
Archetype | Variant | Trades | Win% | PF | Complexity
S1        | core    |  ?     | ?    | ?  | 1/6
S1        | core+T  |  ?     | ?    | ?  | 2/6
S1        | full    |  ?     | ?    | ?  | 6/6
...
```

### Step 5: Analyze Tradeoffs
- Plot complexity vs performance
- Identify diminishing returns
- Find optimal configuration
- Make deployment recommendation

---

## Documentation Cross-Reference

### For Different Use Cases:

**"I need to understand what each variant does"**
→ Read: /CONFIG_VARIANTS_QUICK_REF.md

**"I need detailed technical specifications"**
→ Read: /VARIANT_CONFIGS_SUMMARY.md

**"I need to run backtests and analyze results"**
→ Read: /VARIANT_TESTING_GUIDE.md

**"I need a master index"**
→ Read: This file

---

## Key Insights

1. **Core variants isolate pattern quality** - Remove all filtering to test raw detection
2. **Core+ variants test single feature impact** - Add one key feature (time/macro/pattern)
3. **Full variants serve as production baselines** - Reference point for comparison
4. **Thresholds unchanged** - Only filtering/routing varies, allowing clean ablation
5. **Regime behavior should scale** - Core fires all regimes, full is selective
6. **Trade frequency should decrease** - More filters = fewer but higher quality trades

---

## Success Metrics

Variant testing succeeds when:
- All 9 configs run without errors
- Trade frequencies scale with expected complexity
- Regime behavior matches design expectations
- Performance tradeoff is clearly visible
- Clear recommendation emerges for next steps
- Operators understand justification for complexity

---

## Files Summary

```
Variant Configs (9):
  /configs/variants/s1_core.json
  /configs/variants/s1_core_plus_time.json
  /configs/variants/s1_full.json
  /configs/variants/s4_core.json
  /configs/variants/s4_core_plus_macro.json
  /configs/variants/s4_full.json
  /configs/variants/s5_core.json
  /configs/variants/s5_core_plus_wyckoff.json
  /configs/variants/s5_full.json

Documentation (4):
  /VARIANT_CONFIGS_SUMMARY.md (detailed specs)
  /VARIANT_TESTING_GUIDE.md (testing instructions)
  /CONFIG_VARIANTS_QUICK_REF.md (quick lookup)
  /VARIANT_CONFIGS_INDEX.md (this file)

Total: 13 files ready for backtesting and analysis
```

---

**Status:** Complete and validated. Ready for backtesting.
