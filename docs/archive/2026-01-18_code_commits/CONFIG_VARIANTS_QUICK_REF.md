# Config Variants Quick Reference

**9 Simplified Variants Created for Complexity Testing**

---

## S1 Liquidity Vacuum Variants

### s1_core.json - Wyckoff Only
```
Enabled Engines:     1/6  (Wyckoff)
Domain Features:     Capitulation detection only
Regime Filter:       NO   (use_regime_filter: false)
Temporal Confluence: NO   (use_temporal_confluence: false)
Macro Context:       NO   (enable_macro: false)
Routing:             FLAT (all regimes weight = 1.0)

Core Thresholds:
  - capitulation_depth_max: -0.2 (20% drawdown required)
  - crisis_composite_min: 0.35 (macro stress score)
  - confluence_threshold: 0.65 (3-of-4 conditions required)
  - volume_climax_3b_min: 0.5 (volume spike)
  - wick_exhaustion_3b_min: 0.6 (wick rejection)

Expected Behavior: Pure pattern firing, all regimes, higher false positives
```

### s1_core_plus_time.json - Wyckoff + Temporal
```
Enabled Engines:     2/6  (Wyckoff, Temporal)
Domain Features:     Capitulation + time-of-day filtering
Regime Filter:       NO   (use_regime_filter: false)
Temporal Confluence: YES  (use_temporal_confluence: true)
Temporal Windows:    0-15 UTC (Asian/European active hours)
Temporal Weight:     20% boost during preferred hours
Macro Context:       NO   (enable_macro: false)
Routing:             FLAT (all regimes weight = 1.0)

Core Thresholds:     [Same as core]

Expected Behavior: Pattern + time filtering, all regimes, better timing
```

### s1_full.json - Production Setup
```
Enabled Engines:     6/6  (Wyckoff, SMC, Temporal, HOB, Fusion, Macro)
Domain Features:     Full production feature set
Regime Filter:       YES  (use_regime_filter: true)
Allowed Regimes:     ["risk_off", "crisis"]
Drawdown Override:   10% (bypass regime check on flash crashes)
Temporal Confluence: YES
Macro Context:       YES  (VIX/DXY/MOVE composite)
Routing:             WEIGHTED
  - crisis:   2.0   (highest probability)
  - risk_off: 1.5   (bear markets)
  - neutral:  1.0   (baseline)
  - risk_on:  0.5   (bull markets - reduced)

Core Thresholds:     [Same as core, unchanged]

Expected Performance:
  - Trade Frequency: 40-60 trades/year
  - Win Rate: 50-60%
  - Events Caught: 3-4 out of 7 per bear cycle
```

---

## S4 Funding Divergence Variants

### s4_core.json - Funding Only
```
Enabled Engines:     1/6  (Core funding logic)
Domain Features:     Negative funding divergence detection
Regime Filter:       NO
Macro Context:       NO   (enable_macro: false)
Runtime Features:    YES (calculated but no overlay)
Routing:             FLAT (all regimes weight = 1.0)

Core Thresholds:
  - funding_z_max: -1.976 (must be MORE negative than this)
  - resilience_min: 0.555 (price stability despite negative funding)
  - liquidity_max: 0.348 (thin orderbook)
  - cooldown_bars: 11 hours
  - atr_stop_mult: 2.282
  - fusion_threshold: 0.7824

Expected Behavior: Raw squeeze detection, all regimes, not bear-specific
```

### s4_core_plus_macro.json - Funding + Macro Routing
```
Enabled Engines:     2/6  (Core funding + Macro routing)
Domain Features:     Negative funding + regime discrimination
Regime Filter:       YES  (via regime classifier)
Macro Context:       YES  (BTC.D, DXY, VIX context active)
Routing:             WEIGHTED
  - crisis:   1.5
  - risk_off: 1.0   (bear markets - primary)
  - neutral:  1.0
  - risk_on:  0.5   (bull markets - reduced)

Core Thresholds:     [Same as core]

Expected Behavior: Funding + regime awareness, better bull market discrimination
```

### s4_full.json - Production Setup
```
Enabled Engines:     6/6  (All features)
Domain Features:     Optimized production config
Regime Filter:       YES
Macro Context:       YES  (Full enrichment)
Routing:             WEIGHTED (Production tuned)
  - crisis:   1.5
  - risk_off: 1.0
  - neutral:  1.0
  - risk_on:  0.5

Optimization Results:
  - Profit Factor: 2.22
  - Win Rate: 55.7%
  - Validation Period: 2022 (bear market)

Expected Performance:
  - Trade Frequency: 12 trades/year (bear markets)
  - Bull Markets: 0 trades (correct specialist behavior)
```

---

## S5 Long Squeeze Variants

### s5_core.json - Funding + RSI
```
Enabled Engines:     2/6  (Funding momentum, RSI)
Domain Features:     Positive funding + overbought momentum
Direction:           SHORT (bear market specialist)
Wyckoff Patterns:    NO   (enable_wyckoff: false)
Macro Context:       NO   (enable_macro: false)
Routing:             FLAT (all regimes weight = 1.0)

Core Thresholds:
  - funding_z_min: 1.5 (longs overcrowded, +1.5 sigma)
  - rsi_min: 70 (overbought level)
  - liquidity_max: 0.2 (thin orderbook)
  - cooldown_bars: 8 hours
  - atr_stop_mult: 3.0
  - fusion_threshold: 0.45

Expected Behavior: Simple momentum setup, all regimes, high frequency
```

### s5_core_plus_wyckoff.json - Funding + RSI + Wyckoff
```
Enabled Engines:     3/6  (Funding, RSI, Wyckoff distribution)
Domain Features:     Momentum + structural pattern detection
Wyckoff Patterns:    YES  (use_wyckoff_distribution: true)
Macro Context:       NO   (enable_macro: false)
Routing:             FLAT (all regimes weight = 1.0)

Wyckoff Detection:
  - Supply On Show (SOSH)
  - Inability to absorb supply
  - Structural weakness signals
  - Distribution phase identification

Core Thresholds:     [Same as core]

Expected Behavior: Momentum + structural patterns, all regimes, more selective
```

### s5_full.json - Production Setup
```
Enabled Engines:     6/6  (All features)
Domain Features:     Optimized production config
Direction:           SHORT (bear market specialist)
Wyckoff Patterns:    YES
Macro Context:       YES  (Full enrichment)
Routing:             WEIGHTED (Bear specialist)
  - crisis:   2.5   (highest squeeze probability)
  - risk_off: 2.2   (bear markets - primary)
  - neutral:  0.5   (transition periods)
  - risk_on:  0.0   (DISABLED - bull markets)

Optimization Results:
  - Profit Factor: 1.86
  - Win Rate: 55.6%
  - Study: HighConv_v1 (only profitable of 10 configs tested)

Expected Performance:
  - Trade Frequency: 9 trades/year (bear markets)
  - Bull Markets: 0 trades (disabled by routing)
  - Avg Trade: +0.449R positive expectancy
```

---

## Domain Engine Breakdown

### What Each Engine Does

| Engine | Purpose | S1 | S4 | S5 |
|--------|---------|----|----|----|
| Wyckoff | Market structure (phases, distribution) | core | - | core+ |
| SMC | Smart money concepts (orderblocks, etc) | full | - | full |
| Temporal | Time-of-day, market hours context | core+ | - | full |
| HOB | House of Blocks, liquidity profiling | full | - | full |
| Fusion | Legacy signal fusion (mostly disabled) | full | - | full |
| Macro | Regime classification & routing (VIX/DXY/MOVE) | full | core+ | full |

### Feature Flag Quick Reference

**S1 Core:**
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

**S1 Core+Time:**
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

**S4 Core:**
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

**S4 Core+Macro:**
```json
{
  "enable_macro": true,
  "use_macro_regime": true,
  (rest false or minimal)
}
```

**S5 Core:**
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

**S5 Core+Wyckoff:**
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

---

## Complexity Comparison Matrix

| Feature | S1 Core | S1 C+T | S1 Full | S4 Core | S4 C+M | S4 Full | S5 Core | S5 C+W | S5 Full |
|---------|---------|--------|---------|---------|--------|---------|---------|--------|---------|
| Complexity | 1/6 | 2/6 | 6/6 | 1/6 | 2/6 | 6/6 | 2/6 | 3/6 | 6/6 |
| Regime Filter | No | No | Yes | No | Yes | Yes | No | No | Yes |
| Regime Routing | Flat | Flat | Weighted | Flat | Weighted | Weighted | Flat | Flat | Weighted |
| Macro Context | No | No | Yes | No | Yes | Yes | No | No | Yes |
| Pattern Recognition | Wyckoff | Wyckoff | Wyckoff+SMC | - | - | - | - | Wyckoff | Wyckoff+SMC |
| Time Context | No | Yes | Yes | No | No | No | No | No | Yes |
| Temporal Weight | - | 0.2 | 0.2 | - | - | - | - | - | - |
| Expected Trades/Yr | 100-150 | 60-80 | 40-60 | 30-40 | 15-20 | 12 | 20-30 | 12-15 | 9 |

---

## Regime Behavior Across Variants

### S1 (Capitulation Reversals)

**Core:** Fires in ALL regimes equally
```
risk_on:  weight=1.0  → fires capitulations in bull markets (high FP)
neutral:  weight=1.0  → fires equally
risk_off: weight=1.0  → fires equally
crisis:   weight=1.0  → fires equally
```

**Core+Time:** Still fires all regimes, but prefers 0-15 UTC
```
(same routing, but temporal confluence adds 20% boost during preferred hours)
```

**Full:** Regime-aware specialist
```
risk_on:  weight=0.5  → reduced, capitulations rare in bull
neutral:  weight=1.0  → baseline
risk_off: weight=1.5  → increased, primary environment (bear)
crisis:   weight=2.0  → maximum, crisis = high capitulation probability
```

### S4 (Short Squeezes)

**Core:** Fires in ALL regimes equally
```
(negative funding extremes happen in all conditions)
```

**Core+Macro:** Regime-aware filtering
```
risk_on:  weight=0.5  → reduced (squeezes rare in bull)
neutral:  weight=1.0  → baseline
risk_off: weight=1.0  → bear markets
crisis:   weight=1.5  → increased stress = higher squeeze probability
```

**Full:** Production routing (same as Core+Macro in this case)

### S5 (Long Squeeze)

**Core:** Fires in ALL regimes equally
```
(feeds to long squeeze cascade in any condition)
```

**Core+Wyckoff:** Still all regimes, pattern filtering adds selectivity
```
(same routing, but Wyckoff distribution detection reduces false positives)
```

**Full:** Bear market specialist (SHORT positions)
```
risk_on:  weight=0.0  → DISABLED (short squeezes don't occur in bull)
neutral:  weight=0.5  → transition periods, reduced
risk_off: weight=2.2  → bear markets, primary environment
crisis:   weight=2.5  → maximum (overleveraged longs get squeezed fastest)
```

---

## Backtesting Each Variant

### File Locations
```
/configs/variants/
├── s1_core.json
├── s1_core_plus_time.json
├── s1_full.json
├── s4_core.json
├── s4_core_plus_macro.json
├── s4_full.json
├── s5_core.json
├── s5_core_plus_wyckoff.json
└── s5_full.json
```

### Run Command Template
```bash
python bin/backtest_knowledge_v2.py configs/variants/s1_core.json
```

### Performance Expectations
```
S1 Core        → Baseline (higher trades, lower quality)
S1 Core+Time   → 80% of full quality with 33% less complexity
S1 Full        → Production reference (40-60/yr, 50-60% win rate)

S4 Core        → Baseline (all regimes, not bear-specific)
S4 Core+Macro  → 90% of full quality with 33% less complexity
S4 Full        → Production reference (12/yr, PF 2.22, bear specialist)

S5 Core        → Baseline (all regimes, simple logic)
S5 Core+Wyckoff→ 80% of full quality with 50% less complexity
S5 Full        → Production reference (9/yr, PF 1.86, bear specialist)
```

---

## Key Insights

1. **Core variants test raw pattern quality** - Remove filtering to see if core detection is sound
2. **Core+ variants test intermediate filters** - Add one critical feature (time, macro, patterns)
3. **Full variants serve as production reference** - Baseline for comparison
4. **Expected scaling should be consistent** - Trade frequency should decrease with filtering
5. **Regime behavior should match design** - Core should fire all regimes, full should be selective
6. **Thresholds unchanged** - Variants test routing/filtering only, not threshold optimization
