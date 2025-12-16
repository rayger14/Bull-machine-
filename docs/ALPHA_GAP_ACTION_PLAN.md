# ALPHA GAP ACTION PLAN
**Immediate Steps to Capture Missing 40% Alpha**

---

## TL;DR

**Current Alpha Coverage:** 33% of visual map claims
**Immediate Opportunity:** +40% alpha boost with 4 hours of wiring work
**Blocker:** NONE - test current system first, wire during optimization

---

## CRITICAL FINDING

```
VISUAL MAP WAS ASPIRATIONAL, NOT ACTUAL

What Visual Map Claims:
  "50 features wired across S1/S4/S5"
  "Wyckoff FULLY WIRED"
  "SMC PARTIALLY WIRED"

Reality:
  16 features actually working (32%)
  11 features exist but unwired (22%) ← WIRE THIS
  23 features are ghosts/missing (46%) ← IGNORE

GOOD NEWS: What's working is solid (liquidity, macro, volume)
OPPORTUNITY: 11 features sitting unused in feature store
```

---

## IMMEDIATE ACTION ITEMS (4 Hours Total)

### 1. Wire SMC BOS Signals (2 hours) ⚡ HIGHEST PRIORITY

**Features Available:**
- `tf1h_bos_bearish` ✅ Exists in store
- `tf1h_bos_bullish` ✅ Exists in store

**Where to Wire:**

**S1 (Liquidity Vacuum):**
```python
# File: engine/strategies/archetypes/bear/liquidity_vacuum_runtime.py
# Method: _compute_liquidity_vacuum_fusion()

# Add BOS bearish check (capitulation confirmation)
bos_bearish = df.get('tf1h_bos_bearish', False).astype(float)

# Update fusion calculation
fusion = (
    0.25 * liquidity_vacuum +
    0.20 * volume_panic +
    0.15 * bos_bearish +      # ← NEW
    0.15 * wick_lower +
    0.10 * funding_reversal +
    0.10 * crisis_context +
    0.05 * oversold
)
```

**S4 (Funding Divergence):**
```python
# File: engine/strategies/archetypes/bear/funding_divergence_runtime.py
# Method: _compute_s4_fusion()

# Add BOS bearish (short squeeze setup confirmation)
bos_bearish = df.get('tf1h_bos_bearish', False).astype(float)

# Update fusion calculation
fusion = (
    0.40 * funding_norm +
    0.25 * resilience_norm +
    0.10 * bos_bearish +       # ← NEW
    0.15 * vol_quiet_norm +
    0.10 * liquidity_inv
)
```

**S5 (Long Squeeze):**
```python
# File: engine/strategies/archetypes/bear/long_squeeze_runtime.py
# Method: _compute_s5_fusion()

# Add BOS bullish (exit signal for shorts)
bos_bullish = df.get('tf1h_bos_bullish', False).astype(float)

# Update fusion calculation
fusion = (
    0.35 * funding_norm +
    0.20 * oi_norm +
    0.15 * bos_bullish +       # ← NEW (exit warning)
    0.15 * rsi_ob +
    0.15 * liquidity_inv
)
```

**Expected Impact:** +20-30% signal quality (BOS detects institutional order flow)

---

### 2. Wire Temporal Fusion Scores (1 hour)

**Features Available:**
- `tf4h_fusion_score` ✅ Exists in store
- `tf1h_fusion_score` ✅ Exists in store

**Where to Wire:**

**S1 (Liquidity Vacuum):**
```python
# Add multi-timeframe confluence check
tf4h_fusion = df.get('tf4h_fusion_score', 0.5)

# Boost signal if 4H timeframe confirms
mtf_boost = np.where(tf4h_fusion > 0.40, 1.10, 1.0)
fusion = base_fusion * mtf_boost
```

**S4 (Funding Divergence):**
```python
# Add 4H trend context
tf4h_fusion = df.get('tf4h_fusion_score', 0.5)

# Only trade if 4H isn't strongly bullish (avoid fighting trend)
trend_filter = tf4h_fusion < 0.60
fusion = base_fusion * trend_filter
```

**Expected Impact:** +10-15% confluence quality (filters false signals)

---

### 3. Wire Wyckoff PTI Features (1 hour)

**Features Available:**
- `tf1h_pti_score` ✅ Exists in store
- `tf1h_pti_confidence` ✅ Exists in store
- `tf1h_pti_trap_type` ✅ Exists in store

**Where to Wire:**

**S1 (Liquidity Vacuum):**
```python
# Add PTI reversal confirmation
pti_score = df.get('tf1h_pti_score', 0.0)
pti_conf = df.get('tf1h_pti_confidence', 0.0)

# Boost if PTI detects bull trap reversal
pti_boost = np.where(
    (pti_score > 0.60) & (pti_conf > 0.65),
    1.15,  # +15% boost
    1.0
)
fusion = base_fusion * pti_boost
```

**S5 (Long Squeeze):**
```python
# Add PTI trap-within-trend detection
pti_trap = df.get('tf1h_pti_trap_type', '').str.contains('bull_trap')

# Boost if PTI confirms trap setup
trap_boost = np.where(pti_trap, 1.20, 1.0)
fusion = base_fusion * trap_boost
```

**Expected Impact:** +10% reversal timing precision

---

## TESTING PROTOCOL

### Phase 1: Baseline (Now)
```bash
# Test current system WITHOUT new wiring
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v1.json \
  --start 2022-01-01 --end 2022-12-31 \
  --output results/baseline_no_bos_pti.json
```

**Expected Results:**
- S1: ~5-8 trades, PF ~1.8-2.2
- S4: ~3-5 trades, PF ~1.5-2.0
- S5: ~6-9 trades, PF ~1.6-2.0

### Phase 2: Wire BOS/PTI/Fusion (4 hours work)

1. Make code changes listed above
2. Test runtime enrichment:
```bash
python -c "
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import LiquidityVacuumRuntimeFeatures
import pandas as pd

df = pd.read_parquet('data/features_mtf/BTC_1H_2022_ENRICHED.parquet')
enricher = LiquidityVacuumRuntimeFeatures()
df_enriched = enricher.enrich_dataframe(df)

# Verify BOS column used
print('BOS bearish available:', 'tf1h_bos_bearish' in df_enriched.columns)
print('BOS bearish events:', (df_enriched['tf1h_bos_bearish'] == True).sum())
"
```

### Phase 3: Re-test with Wiring
```bash
# Test WITH new wiring
python bin/backtest_knowledge_v2.py \
  --config configs/mvp/mvp_bear_market_v1.json \
  --start 2022-01-01 --end 2022-12-31 \
  --output results/enhanced_with_bos_pti.json
```

**Expected Results:**
- S1: ~8-12 trades (+40% more signals), PF ~2.0-2.5 (+15% better quality)
- S4: ~5-8 trades (+50% more), PF ~1.8-2.3 (+20% better)
- S5: ~8-12 trades (+30% more), PF ~1.8-2.4 (+15% better)

### Phase 4: Compare Results
```bash
python bin/compare_backtest_results.py \
  --baseline results/baseline_no_bos_pti.json \
  --enhanced results/enhanced_with_bos_pti.json \
  --output reports/alpha_uplift_measurement.md
```

---

## DON'T WASTE TIME ON THESE (Ghosts)

### Wyckoff Events (DON'T BUILD YET)
```
❌ wyckoff_spring_a
❌ wyckoff_spring_b
❌ wyckoff_ps (Preliminary Support)
❌ wyckoff_sow (Sign of Weakness)
❌ wyckoff_utad (Upthrust After Distribution)

WHY: Visual map was aspirational - these were NEVER implemented
WHEN: Only build AFTER measuring BOS/PTI alpha uplift
      (may not be needed if BOS gives enough alpha)
```

### HOB Engine (DON'T BUILD YET)
```
❌ hob_demand_zone

WHY: Engine never built, unclear if different from SMC order blocks
WHEN: Research overlap with SMC before building (likely duplicate)
```

### SMC Composite Score (DON'T BUILD YET)
```
❌ smc_score

WHY: Individual signals (BOS) are more powerful when wired separately
WHEN: Create composite AFTER testing individual BOS signals
      (composite is just cleaner code, not more alpha)
```

---

## DECISION TREE

```
START: Test current system (baseline)
  ↓
MEASURE: Current PF/WR/trade count
  ↓
WIRE: BOS + PTI + Fusion (4 hours)
  ↓
RE-TEST: Measure uplift
  ↓
IF uplift > 30%: SHIP IT ✅
  ↓
IF uplift 10-30%: Consider Wyckoff events
  ↓
IF uplift < 10%: Debug wiring or try different weights
```

---

## SUCCESS CRITERIA

**Minimum Viable Alpha (Ready to Trade):**
- S1 PF > 2.0, WR > 55%, 8+ trades/year
- S4 PF > 1.8, WR > 50%, 5+ trades/year
- S5 PF > 1.8, WR > 55%, 8+ trades/year

**Stretch Goal (Exceptional System):**
- S1 PF > 2.5, WR > 60%, 12+ trades/year
- S4 PF > 2.3, WR > 55%, 8+ trades/year
- S5 PF > 2.2, WR > 60%, 10+ trades/year

---

## FINAL RECOMMENDATION

```
ACTION SEQUENCE:

1. READ FULL REPORT:
   ALPHA_COMPLETENESS_VERIFICATION_REPORT.md

2. TEST BASELINE (NOW):
   Run current system to measure starting point
   No blockers - system is stable

3. WIRE ALPHA (4 HOURS):
   Add BOS/PTI/Fusion to S1/S4/S5 runtime code
   Low risk - just adding signal boosters

4. RE-TEST (1 HOUR):
   Measure actual alpha uplift from wiring
   Compare against baseline

5. DECIDE:
   If uplift > 30%: Ship to production
   If uplift < 30%: Investigate Wyckoff events

DO NOT BUILD WYCKOFF EVENTS UNTIL YOU MEASURE BOS ALPHA
Visual map was aspirational - test reality first
```

---

**Total Time Investment:** 6 hours (2H baseline test + 4H wiring)
**Expected Alpha Gain:** +30-40% more/better trades
**Risk Level:** LOW (additive changes only, no breaking)

---

**Generated:** 2025-12-11
**Next Review:** After baseline test completes
